"""FastAPI ingestion router: multipart upload -> magic-byte validation -> Celery dispatch."""

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, File, HTTPException, UploadFile, status

from app.config import settings
from app.logging_config import configure_logging
from app.magic import UnsupportedFormatError, detect_format
from app.resources import (
    detect_system_resources,
    stack_container_budget_mb,
    validate_capacity_plan,
)
from app.storage import (
    DiskFullError,
    FileTooLargeError,
    StorageWriteError,
    ensure_dirs,
    save_upload_stream,
)
from app.tasks import parse_document_task
from app.tasks_media import process_media_task
from app.tasks_tabular import index_tabular_task
from app.tasks_reasoning import reasoning_task
from app.media.pipeline import UnsupportedMediaError as UnsupportedMediaFormatError
from app.media.pipeline import detect_media_format
from app.tabular.parsers import UnsupportedTabularError
from app.tabular.parsers import detect_tabular_format
from app.reasoning.context_manager import ContextOverflowError

configure_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    ensure_dirs()
    resources = detect_system_resources()
    stack_total_mb = stack_container_budget_mb(settings)
    logger.info("Ingestion engine starting", extra={
        "environment": settings.environment,
        "host_total_memory_mb": resources.total_memory_mb,
        "host_cpu_count": resources.cpu_count,
        "host_available_disk_mb": resources.available_disk_mb,
        "worker_memory_budget_mb": settings.worker_memory_budget_mb,
        "total_container_budget_mb": settings.total_container_budget_mb,
    })
    # Assert the real deployment invariant (concurrency x child limit per queue,
    # and the summed container caps) against the host we actually booted on.
    validate_capacity_plan(settings, resources)
    if stack_total_mb > resources.total_memory_mb:
        logger.warning(
            "Container caps exceed detected host RAM; the stack is oversubscribed",
            extra={"stack_total_mb": stack_total_mb,
                   "host_total_memory_mb": resources.total_memory_mb},
        )
    yield


app = FastAPI(
    title="Multimodal AI Ingestion Engine",
    version="1.0.0",
    description=(
        "Four-phase pipeline: P1 document ingestion, P2 vision/audio/OCR, "
        "P3 tabular telemetry + vector indexing, P4 reasoning core. "
        "Capacity guardrails are clamped to the detected host at startup."
    ),
    lifespan=lifespan,
)


@app.get("/health", tags=["ops"])
def health() -> dict[str, str]:
    return {"status": "ok", "service": settings.service_name}


@app.post("/api/v1/documents/ingest", status_code=status.HTTP_202_ACCEPTED, tags=["ingestion"])
async def ingest_document(file: UploadFile = File(...)) -> dict[str, str | int]:
    """Accept a PDF/TXT/MD/DOCX upload, validate magic bytes, enqueue async parsing."""
    filename = file.filename or "upload.bin"
    try:
        stored_path = await _validate_and_store(file, filename)
    except FileTooLargeError as exc:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=str(exc)) from exc
    except UnsupportedFormatError as exc:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=str(exc)) from exc
    except DiskFullError as exc:
        raise HTTPException(status.HTTP_507_INSUFFICIENT_STORAGE, detail=str(exc)) from exc
    except (StorageWriteError, OSError) as exc:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

    try:
        task = parse_document_task.delay(str(stored_path), filename)
    except Exception as exc:  # noqa: BLE001 - broker unreachable
        logger.error("Failed to enqueue ingestion task", extra={"reason": str(exc)})
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Task queue unavailable; try again shortly.",
        ) from exc

    logger.info("Document accepted for ingestion", extra={
        "file_name": filename, "task_id": task.id, "stored_path": str(stored_path),
    })
    return {"task_id": task.id, "status": "queued", "filename": filename,
            "file_size_bytes": stored_path.stat().st_size}


async def _validate_and_store(file: UploadFile, filename: str):
    """Stream the upload to disk while validating magic bytes and the size ceiling."""
    header = await file.read(4096)
    if not header:
        raise UnsupportedFormatError(f"File '{filename}' is empty.")
    signature = detect_format(header, filename)
    logger.info("Format validated via magic bytes", extra={
        "file_name": filename, "format": signature.fmt.value, "mime": signature.mime_type,
    })

    stream = _PrefixedStream(header, file)
    return save_upload_stream(stream, filename)


@app.post("/api/v1/media/ingest", status_code=status.HTTP_202_ACCEPTED, tags=["ingestion"])
async def ingest_media(
    file: UploadFile = File(...),
    parent_document_id: str = "",
) -> dict[str, str | int]:
    """Accept an image/video/audio upload, validate magic bytes, enqueue media processing."""
    filename = file.filename or "upload.bin"
    try:
        stored_path = await _validate_and_store_media(file, filename)
    except FileTooLargeError as exc:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=str(exc)) from exc
    except UnsupportedMediaFormatError as exc:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=str(exc)) from exc
    except DiskFullError as exc:
        raise HTTPException(status.HTTP_507_INSUFFICIENT_STORAGE, detail=str(exc)) from exc
    except (StorageWriteError, OSError) as exc:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

    try:
        task = process_media_task.delay(
            str(stored_path), filename, parent_document_id or "standalone"
        )
    except Exception as exc:  # noqa: BLE001 - broker unreachable
        logger.error("Failed to enqueue media task", extra={"reason": str(exc)})
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Task queue unavailable; try again shortly.",
        ) from exc

    logger.info("Media accepted for processing", extra={
        "file_name": filename, "task_id": task.id,
    })
    return {"task_id": task.id, "status": "queued", "filename": filename,
            "file_size_bytes": stored_path.stat().st_size}


async def _validate_and_store_media(file: UploadFile, filename: str):
    """Stream-validate a media upload (magic bytes + size ceiling) and persist it."""
    header = await file.read(4096)
    if not header:
        raise UnsupportedMediaFormatError(f"File '{filename}' is empty.")
    signature = detect_media_format(header, filename)
    logger.info("Media format validated via magic bytes", extra={
        "file_name": filename, "format": signature.fmt.value, "mime": signature.mime_type,
    })
    stream = _PrefixedStream(header, file)
    return save_upload_stream(stream, filename)


@app.post("/api/v1/tabular/ingest", status_code=status.HTTP_202_ACCEPTED, tags=["ingestion"])
async def ingest_tabular(file: UploadFile = File(...)) -> dict[str, str | int]:
    """Accept a CSV/XLSX/JSON upload, validate magic bytes, enqueue vector indexing."""
    filename = file.filename or "upload.bin"
    try:
        stored_path = await _validate_and_store_tabular(file, filename)
    except FileTooLargeError as exc:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=str(exc)) from exc
    except UnsupportedTabularError as exc:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=str(exc)) from exc
    except DiskFullError as exc:
        raise HTTPException(status.HTTP_507_INSUFFICIENT_STORAGE, detail=str(exc)) from exc
    except (StorageWriteError, OSError) as exc:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

    try:
        task = index_tabular_task.delay(str(stored_path), filename)
    except Exception as exc:  # noqa: BLE001 - broker unreachable
        logger.error("Failed to enqueue tabular task", extra={"reason": str(exc)})
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Task queue unavailable; try again shortly.",
        ) from exc

    logger.info("Tabular file accepted for indexing", extra={
        "file_name": filename, "task_id": task.id,
    })
    return {"task_id": task.id, "status": "queued", "filename": filename,
            "file_size_bytes": stored_path.stat().st_size}


async def _validate_and_store_tabular(file: UploadFile, filename: str):
    """Stream-validate a tabular upload (magic bytes + size ceiling) and persist it."""
    header = await file.read(4096)
    if not header:
        raise UnsupportedTabularError(f"File '{filename}' is empty.")
    fmt = detect_tabular_format(header, filename)
    logger.info("Tabular format validated via magic bytes", extra={
        "file_name": filename, "format": fmt.value,
    })
    stream = _PrefixedStream(header, file)
    return save_upload_stream(stream, filename)


@app.post("/api/v1/reason/query", status_code=status.HTTP_202_ACCEPTED, tags=["reasoning"])
async def reason_query(
    file: UploadFile = File(...),
    query: str = "",
) -> dict[str, str | int]:
    """Accept a JSON retrieval context bundle, enqueue Phase 4 reasoning.

    The bundle must be JSON of the form {"items": [{"modality": "text_chunk"|
    "ocr_block"|"telemetry"|"image_ref", "text": str, "similarity_score": float,
    "document_id": str, "timestamp": str|null, "image_path": str|null}, ...]}.
    """
    query = query.strip() or "Summarize the provided context."
    if file.content_type not in (None, "application/json", "text/plain") and not (
        file.filename or "").endswith(".json"):
        # Reasoning only consumes JSON bundles; validate cheaply up front.
        header = await file.read(64)
        await file.seek(0)
        if not header.lstrip().startswith((b"{", b"[")):
            raise HTTPException(
                status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="Reasoning query requires a JSON context bundle.",
            )
    try:
        stored_path = save_upload_stream(file.file, filename=(file.filename or "context.json"))
    except FileTooLargeError as exc:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=str(exc)) from exc
    except DiskFullError as exc:
        raise HTTPException(status.HTTP_507_INSUFFICIENT_STORAGE, detail=str(exc)) from exc
    except (StorageWriteError, OSError) as exc:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

    try:
        task = reasoning_task.delay(str(stored_path), query)
    except Exception as exc:  # noqa: BLE001 - broker unreachable
        logger.error("Failed to enqueue reasoning task", extra={"reason": str(exc)})
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Task queue unavailable; try again shortly.",
        ) from exc

    logger.info("Reasoning query accepted", extra={"task_id": task.id})
    return {"task_id": task.id, "status": "queued", "query": query}


class _PrefixedStream:
    """Replays the already-consumed header bytes, then proxies the upload stream."""

    def __init__(self, header: bytes, upload: UploadFile) -> None:
        self._header = header
        self._upload = upload

    def read(self, size: int = -1) -> bytes:
        if self._header:
            head, self._header = self._header, b""
            return head
        return self._upload.file.read(size)
