"""Parsers for plain text (TXT/MD) and DOCX documents. All operate on disk paths."""

import logging
from pathlib import Path

from docx import Document as DocxDocument  # python-docx

from app.parser.chunker import chunk_text
from app.schemas import TextPayload

logger = logging.getLogger(__name__)

PARAGRAPH_TAG = "paragraph"
HEADING_TAG = "header"


class CorruptedDocumentError(ValueError):
    """Raised when a document container cannot be opened (corrupted header/zip)."""


def parse_txt(file_path: Path, document_id: str, total_pages: int = 1) -> list[TextPayload]:
    """Stream-read a TXT/MD file in bounded windows; page 1 is a logical unit."""
    payloads: list[TextPayload] = []
    chunk_id = 0
    buffer: list[str] = []
    window_size = 1 << 20  # 1MB read window
    try:
        with file_path.open("r", encoding="utf-8", errors="replace") as fh:
            while window := fh.read(window_size):
                buffer.append(window)
                if sum(len(b) for b in buffer) >= window_size:
                    _flush(buffer, payloads, chunk_id, total_pages)
                    chunk_id = len(payloads)
                    buffer = []
    except OSError as exc:
        raise CorruptedDocumentError(f"Unable to read '{file_path.name}': {exc}") from exc
    _flush(buffer, payloads, chunk_id, total_pages)
    return payloads


def _flush(buffer: list[str], payloads: list[TextPayload], start_id: int, page: int) -> None:
    for chunk in chunk_text("".join(buffer)):
        payloads.append(TextPayload(chunk_id=start_id + len(payloads), page_number=page,
                                    content=chunk, structural_tag=PARAGRAPH_TAG))


def parse_docx(file_path: Path, document_id: str) -> tuple[list[TextPayload], int]:
    """Parse DOCX paragraph-by-paragraph, tagging headings vs body text."""
    try:
        doc = DocxDocument(str(file_path))
    except Exception as exc:  # noqa: BLE001
        raise CorruptedDocumentError(
            f"'{file_path.name}' is not a valid DOCX container: {exc}"
        ) from exc

    chunk_id = 0
    payloads: list[TextPayload] = []
    pending: list[str] = []

    def flush() -> None:
        nonlocal chunk_id
        merged = "\n".join(pending)
        for chunk in chunk_text(merged):
            payloads.append(TextPayload(chunk_id=chunk_id, page_number=1,
                                        content=chunk, structural_tag=PARAGRAPH_TAG))
            chunk_id += 1
        pending.clear()

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        if para.style and para.style.name and para.style.name.lower().startswith("heading"):
            flush()
            payloads.append(TextPayload(chunk_id=chunk_id, page_number=1,
                                        content=text, structural_tag=HEADING_TAG))
            chunk_id += 1
        else:
            pending.append(text)
            if sum(len(t) for t in pending) > 1 << 20:
                flush()
    flush()
    return payloads, 1
