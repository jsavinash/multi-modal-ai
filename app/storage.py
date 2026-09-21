"""Disk-backed upload storage with strict size enforcement and streamed writes."""

import logging
import shutil
import uuid
from pathlib import Path

from app.config import settings

logger = logging.getLogger(__name__)


class FileTooLargeError(ValueError):
    """Raised when an upload exceeds the 100MB ceiling."""


class DiskFullError(RuntimeError):
    """Raised when free disk space is below the configured safety floor."""


class StorageWriteError(RuntimeError):
    """Raised when the upload cannot be persisted to disk."""


def ensure_dirs() -> None:
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.result_dir.mkdir(parents=True, exist_ok=True)


def save_upload_stream(stream, filename: str) -> Path:
    """Persist an upload to disk in bounded chunks, aborting past the size ceiling.

    Never buffers the whole file in RAM — critical for the 16GB memory guardrail.
    """
    ensure_dirs()
    if available_disk_headroom_bytes(settings.upload_dir) < settings.min_free_disk_mb * 1024 * 1024:
        raise DiskFullError(
            f"Free disk space below the {settings.min_free_disk_mb}MB safety floor; "
            "refusing new uploads until space is reclaimed."
        )
    safe_name = Path(filename).name or "upload.bin"
    dest = settings.upload_dir / f"{uuid.uuid4().hex}__{safe_name}"
    total = 0
    try:
        with dest.open("wb") as out:
            while chunk := stream.read(settings.stream_chunk_size_bytes):
                total += len(chunk)
                if total > settings.max_file_size_bytes:
                    out.close()
                    dest.unlink(missing_ok=True)
                    raise FileTooLargeError(
                        f"File '{filename}' exceeds the "
                        f"{settings.max_file_size_bytes // (1024 * 1024)}MB upload ceiling."
                    )
                out.write(chunk)
    except FileTooLargeError:
        raise
    except OSError as exc:
        dest.unlink(missing_ok=True)
        raise StorageWriteError(f"Failed to persist upload '{filename}': {exc}") from exc
    logger.info("Upload persisted", extra={"file_name": safe_name,
                                           "stored_path": str(dest), "file_size_bytes": total})
    return dest


def save_result(document_id: str, raw_json: str) -> Path:
    ensure_dirs()
    dest = settings.result_dir / f"{document_id}.json"
    dest.write_text(raw_json, encoding="utf-8")
    return dest


def cleanup(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:  # noqa: BLE001
        logger.warning("Failed to clean up temp file", extra={"path": str(path), "reason": str(exc)})


def available_disk_headroom_bytes(path: Path) -> int:
    return shutil.disk_usage(path).free
