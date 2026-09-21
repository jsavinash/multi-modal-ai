"""Layout-aware PDF parser.

Strategy:
- pdfminer.six performs page-by-page layout analysis, providing text lines with
  y-coordinates (header/footer boundary tagging) and image bounding boxes.
- pypdf extracts embedded image bytes and XObject dimensions.
Both libraries process page-by-page, so large documents are never fully in RAM.
"""

import hashlib
import logging
import uuid
from pathlib import Path

from pdfminer.high_level import extract_pages
from pdfminer.layout import LAParams, LTImage, LTTextContainer, LTTextLine
from pypdf import PdfReader

from app.config import settings
from app.schemas import MediaReference, TextPayload

logger = logging.getLogger(__name__)

HEADER_TAG = "header"
FOOTER_TAG = "footer"
PARAGRAPH_TAG = "paragraph"
MISSING_TEXT_LAYER_TAG = "missing_text_layer"


class CorruptedPDFError(ValueError):
    """Raised when the PDF header/xref structure is unreadable."""


def validate_pdf_header(file_path: Path) -> None:
    with file_path.open("rb") as fh:
        head = fh.read(1024)
    if not head.startswith(b"%PDF-"):
        raise CorruptedPDFError(f"File '{file_path.name}' lacks a valid %PDF- header.")


def count_pages(file_path: Path) -> int:
    """Read the page count via pypdf without loading page content."""
    try:
        reader = PdfReader(str(file_path), strict=False)
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception as exc:  # noqa: BLE001
                raise CorruptedPDFError(f"'{file_path.name}' is encrypted.") from exc
        return len(reader.pages)
    except CorruptedPDFError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise CorruptedPDFError(f"Unable to read page count of '{file_path.name}': {exc}") from exc


def _classify_line(line: LTTextLine, page_height: float) -> str:
    margin = settings.header_footer_margin_fraction * page_height
    if line.y1 > page_height - margin:
        return HEADER_TAG
    if line.y0 < margin:
        return FOOTER_TAG
    return PARAGRAPH_TAG


def _extract_page_text(page_layout) -> list[tuple[str, str]]:
    """Return (text, structural_tag) tuples for one page layout object."""
    page_height = page_layout.height or 792.0
    out: list[tuple[str, str]] = []
    for element in page_layout:
        if not isinstance(element, LTTextContainer):
            continue
        for line in element:
            if not isinstance(line, LTTextLine):
                continue
            text = line.get_text().strip()
            if text:
                out.append((text, _classify_line(line, page_height)))
    return out


def _extract_page_images(
    page_layout, reader_page, page_number: int
) -> tuple[list[MediaReference], dict[str, bytes]]:
    """Extract image bounding boxes (pdfminer) and raw bytes (pypdf) for one page."""
    references: list[MediaReference] = []
    media: dict[str, bytes] = {}
    try:
        images = list(reader_page.images)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Image extraction failed", extra={
            "page_number": page_number, "reason": str(exc),
        })
        return references, media

    lt_images = [el for el in page_layout if isinstance(el, LTImage)]
    for idx, img in enumerate(images):
        try:
            media_id = uuid.uuid4().hex
            media[media_id] = img.data
            bbox = [0.0, 0.0, 0.0, 0.0]
            if idx < len(lt_images):
                b = lt_images[idx].bbox
                bbox = [round(float(b[0]), 2), round(float(b[1]), 2),
                        round(float(b[2]), 2), round(float(b[3]), 2)]
            references.append(MediaReference(
                media_id=media_id, page_number=page_number,
                spatial_bounding_box=bbox,
            ))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping corrupt embedded image", extra={
                "page_number": page_number, "reason": str(exc),
            })
    return references, media


def _coalesce_paragraphs(payloads: list[TextPayload]) -> list[TextPayload]:
    """Re-chunk consecutive same-page paragraph payloads to the target chunk size."""
    from app.parser.chunker import chunk_text

    result: list[TextPayload] = []
    chunk_id = 0
    buffer: list[TextPayload] = []

    def flush() -> None:
        nonlocal chunk_id
        if not buffer:
            return
        merged = "\n".join(p.content for p in buffer)
        for c in (chunk_text(merged) or [""]):
            result.append(TextPayload(
                chunk_id=chunk_id, page_number=buffer[0].page_number,
                content=c, structural_tag=buffer[0].structural_tag,
            ))
            chunk_id += 1
        buffer.clear()

    for payload in payloads:
        if payload.structural_tag in (HEADER_TAG, FOOTER_TAG, MISSING_TEXT_LAYER_TAG):
            flush()
            result.append(payload.model_copy(update={"chunk_id": chunk_id}))
            chunk_id += 1
        elif buffer and buffer[0].page_number == payload.page_number:
            buffer.append(payload)
        else:
            flush()
            buffer.append(payload)
    flush()
    return result


def compute_document_id(file_path: Path) -> str:
    """Deterministic content-hash document id (SHA-256 of first MB + file size)."""
    hasher = hashlib.sha256()
    with file_path.open("rb") as fh:
        hasher.update(fh.read(1 << 20))
    hasher.update(str(file_path.stat().st_size).encode())
    return hasher.hexdigest()


def parse_pdf(
    file_path: Path, document_id: str
) -> tuple[list[TextPayload], list[MediaReference], dict[str, bytes], int]:
    """Parse a PDF into canonical payloads.

    Returns:
        (payloads, media_references, raw_image_bytes_by_media_id, total_pages)

    Raises:
        CorruptedPDFError: On unreadable structure.
    """
    validate_pdf_header(file_path)
    total_pages = count_pages(file_path)
    logger.info("Parsing PDF", extra={"document_id": document_id, "total_pages": total_pages})

    reader = PdfReader(str(file_path), strict=False)
    if reader.is_encrypted:
        reader.decrypt("")

    raw_lines: list[TextPayload] = []
    media_refs: list[MediaReference] = []
    media_bytes: dict[str, bytes] = {}

    # Stream page-by-page; only one page layout lives in memory at a time.
    for page_number, page_layout in enumerate(
        extract_pages(str(file_path), laparams=LAParams()), start=1
    ):
        page_texts = _extract_page_text(page_layout)
        if not page_texts:
            logger.warning("Page missing text layer (likely scanned image)",
                           extra={"document_id": document_id, "page_number": page_number})
            raw_lines.append(TextPayload(
                chunk_id=0, page_number=page_number, content="",
                structural_tag=MISSING_TEXT_LAYER_TAG,
            ))
        for text, tag in page_texts:
            raw_lines.append(TextPayload(chunk_id=0, page_number=page_number,
                                         content=text, structural_tag=tag))

        page_refs, page_media = _extract_page_images(
            page_layout, reader.pages[page_number - 1], page_number
        )
        media_refs.extend(page_refs)
        media_bytes.update(page_media)

    payloads = _coalesce_paragraphs(raw_lines)
    logger.info("PDF parse complete", extra={
        "document_id": document_id, "payload_count": len(payloads),
        "media_count": len(media_refs),
    })
    return payloads, media_refs, media_bytes, total_pages
