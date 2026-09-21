"""Phase 2 Celery tasks: async media processing on the dedicated 'media' queue."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from app.celery_app import celery_app
from app.config import settings
from app.media.pipeline import (
    MediaProcessingTimeout,
    UnsupportedMediaError,
    detect_media_format,
    process_media,
)
from app.storage import cleanup, save_result

logger = logging.getLogger(__name__)


class MediaTaskFailure(RuntimeError):
    """Raised when media processing fails; surfaced as task failure."""


@celery_app.task(
    name="media.process",
    bind=True,
    max_retries=1,
    default_retry_delay=5,
    autoretry_for=(OSError,),
    soft_time_limit=settings.media_task_timeout_s,
    time_limit=settings.media_task_timeout_s + 60,
    queue=settings.media_task_queue,
)
def process_media_task(self, file_path_str: str, filename: str, parent_document_id: str) -> dict:
    """Process a stored media file into unified multi-modal payloads."""
    path = Path(file_path_str)
    if not path.exists():
        raise MediaTaskFailure(f"Stored media not found: {file_path_str}")
    try:
        with path.open("rb") as fh:
            head = fh.read(4096)
        signature = detect_media_format(head, filename)
        payloads = process_media(path, signature, parent_document_id)
    except Exception as exc:  # noqa: BLE001
        logger.error("Media processing failed", extra={"file_name": filename, "error": str(exc)})
        raise MediaTaskFailure(str(exc)) from exc
    finally:
        cleanup(path)

    result = {
        "parent_document_id": parent_document_id,
        "filename": filename,
        "payloads": [json.loads(p.model_dump_json()) for p in payloads],
    }
    save_result(f"media_{parent_document_id}_{path.stem}", json.dumps(result))
    logger.info("Media task complete", extra={
        "parent_document_id": parent_document_id, "payload_count": len(payloads),
    })
    return result
