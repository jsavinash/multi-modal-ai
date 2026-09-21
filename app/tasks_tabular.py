"""Phase 3 Celery task: async tabular indexing on the dedicated 'index' queue."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from app.celery_app import celery_app
from app.config import settings
from app.storage import cleanup, save_result
from app.tabular.indexing import run_indexing
from app.tabular.parsers import TabularFormat, detect_tabular_format

logger = logging.getLogger(__name__)


class IndexingTaskFailure(RuntimeError):
    """Raised when tabular indexing fails; surfaced as task failure."""


@celery_app.task(
    name="tabular.index",
    bind=True,
    max_retries=2,
    default_retry_delay=5,
    autoretry_for=(OSError,),
    queue=settings.index_task_queue,
)
def index_tabular_task(self, file_path_str: str, filename: str) -> dict:
    """Parse, serialize, normalize, embed and index one tabular file."""
    path = Path(file_path_str)
    if not path.exists():
        raise IndexingTaskFailure(f"Stored file not found: {file_path_str}")
    try:
        with path.open("rb") as fh:
            head = fh.read(4096)
        fmt = detect_tabular_format(head, filename)
        summary = run_indexing(path, fmt, filename)
    except Exception as exc:  # noqa: BLE001
        logger.error("Tabular indexing failed", extra={"file_name": filename, "error": str(exc)})
        raise IndexingTaskFailure(str(exc)) from exc
    finally:
        cleanup(path)

    save_result(f"index_{summary['source_document_id']}", json.dumps(summary))
    logger.info("Tabular indexing task complete", extra={
        "document_id": summary["source_document_id"],
        "rows_indexed": summary["rows_indexed"],
        "store_mode": summary["store_mode"],
    })
    return summary
