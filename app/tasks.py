"""Celery tasks: background document parsing pipelines."""

import json
import logging
from pathlib import Path

from app.celery_app import celery_app
from app.config import settings
from app.magic import SupportedFormat, detect_format
from app.parser.chunker import chunk_text
from app.parser.docx_text_parser import parse_docx, parse_txt
from app.parser.pdf_parser import compute_document_id, parse_pdf
from app.schemas import CanonicalDocument, DocumentMetadata, TextPayload
from app.storage import cleanup, save_result

logger = logging.getLogger(__name__)


class ParsingFailure(RuntimeError):
    """Raised when the parsing pipeline fails; surfaced as task failure."""


def _build_canonical(
    document_id: str,
    filename: str,
    file_size_bytes: int,
    total_pages: int,
    payloads: list[TextPayload],
    media_references: list,
) -> CanonicalDocument:
    return CanonicalDocument(
        document_id=document_id,
        metadata=DocumentMetadata(
            filename=filename,
            file_size_bytes=file_size_bytes,
            total_pages=total_pages,
        ),
        payloads=payloads,
        extracted_media_references=media_references,
    )


def _parse_document(path: Path, fmt: SupportedFormat, document_id: str) -> CanonicalDocument:
    file_size = path.stat().st_size
    if fmt is SupportedFormat.PDF:
        payloads, media_refs, _media_bytes, total_pages = parse_pdf(path, document_id)
    elif fmt is SupportedFormat.DOCX:
        payloads, total_pages = parse_docx(path, document_id)
        media_refs = []
    elif fmt in (SupportedFormat.TXT, SupportedFormat.MD):
        payloads = parse_txt(path, document_id)
        media_refs, total_pages = [], 1
    else:  # pragma: no cover - guarded by magic-byte router
        raise ParsingFailure(f"Unsupported format '{fmt}' reached the parser.")
    return _build_canonical(document_id, path.name, file_size, total_pages, payloads, media_refs)


@celery_app.task(
    name="ingest.parse_document",
    bind=True,
    max_retries=2,
    default_retry_delay=5,
    autoretry_for=(OSError,),
)
def parse_document_task(self, file_path_str: str, filename: str) -> dict:
    """Parse a stored upload into the canonical structure and persist the result JSON."""
    path = Path(file_path_str)
    if not path.exists():
        raise ParsingFailure(f"Stored file not found: {file_path_str}")

    try:
        with path.open("rb") as fh:
            head = fh.read(4096)
        fmt = detect_format(head, filename)
        document_id = compute_document_id(path)
        canonical = _parse_document(path, fmt, document_id)
    except Exception as exc:  # noqa: BLE001
        logger.error("Document parsing failed", extra={
            "file_name": filename, "error": str(exc),
        })
        cleanup(path)
        raise ParsingFailure(str(exc)) from exc
    finally:
        cleanup(path)

    raw_json = canonical.model_dump_json()
    save_result(canonical.document_id, raw_json)
    logger.info("Ingestion task complete", extra={
        "document_id": canonical.document_id,
        "payloads": len(canonical.payloads),
        "media_references": len(canonical.extracted_media_references),
    })
    return json.loads(raw_json)
