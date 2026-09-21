"""Vector Storage Handlers: pooled Qdrant client with retries and batch upserts.

Offloads all distance computation to the dedicated Vector Database instance —
no local similarity matrices are ever held in active memory.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from app.config import settings
from app.vector_schema import IndexPayload, validate_payload_for_collection

logger = logging.getLogger(__name__)

try:  # qdrant-client is an optional runtime dependency
    from qdrant_client import QdrantClient
    from qdrant_client import models as qm
    _QDRANT_IMPORT_OK = True
except ImportError:  # pragma: no cover
    _QDRANT_IMPORT_OK = False


class VectorStoreUnavailableError(RuntimeError):
    """Raised when Qdrant stays unreachable after all retry attempts."""


def _get_client():
    """Pooled Qdrant connection (module-level singleton; client manages its pool)."""
    global _client
    if not _QDRANT_IMPORT_OK:
        raise VectorStoreUnavailableError("qdrant-client is not installed.")
    if _client is None:
        _client = QdrantClient(url=settings.qdrant_url, timeout=settings.qdrant_timeout_s)
    return _client


_client = None


def _with_retries(op_name: str, fn):
    """Robust retry logic under network failure states (exponential backoff)."""
    last_exc: Exception | None = None
    for attempt in range(settings.qdrant_max_retries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - network errors are heterogeneous
            last_exc = exc
            delay = settings.qdrant_retry_base_delay_s * (2 ** attempt)
            logger.warning("Qdrant op failed; retrying", extra={
                "operation": op_name, "attempt": attempt + 1,
                "retry_in_s": round(delay, 2), "reason": str(exc),
            })
            time.sleep(delay)
    raise VectorStoreUnavailableError(
        f"Qdrant '{op_name}' failed after {settings.qdrant_max_retries} retries: {last_exc}"
    ) from last_exc


def ensure_collections() -> None:
    """Create one collection per modality with payload indexes (idempotent)."""
    client = _get_client()

    def _create():
        existing = {c.name for c in client.get_collections().collections}
        from app.vector_schema import COLLECTIONS
        for modality, name in COLLECTIONS.items():
            if name not in existing:
                client.create_collection(
                    collection_name=name,
                    vectors_config=qm.VectorParams(
                        size=settings.embedding_dim, distance=qm.Distance.COSINE,
                    ),
                )
                logger.info("Qdrant collection created", extra={
                    "collection": name, "modality": modality, "dim": settings.embedding_dim,
                })

    _with_retries("ensure_collections", _create)

    def _indexes():
        from app.vector_schema import COLLECTIONS
        for name in COLLECTIONS.values():
            schema = client.get_collection(name).payload_schema or {}
            existing_fields = set(schema.keys())
            for field_name in ("document_id", "modality", "timestamp"):
                if field_name not in existing_fields:
                    client.create_payload_index(
                        collection_name=name, field_name=field_name,
                        field_schema=qm.PayloadSchema.KEYWORD,
                    )

    _with_retries("create_payload_indexes", _indexes)


def upsert_batch(payloads: list[IndexPayload], document_id: str) -> int:
    """Validate then upsert in efficient batches of `upsert_batch_size` (64)."""
    for p in payloads:
        validate_payload_for_collection(p)
    if not payloads:
        return 0
    client = _get_client()
    written = 0

    def _upsert():
        nonlocal written
        from app.vector_schema import COLLECTIONS
        for start in range(0, len(payloads), settings.upsert_batch_size):
            batch = payloads[start:start + settings.upsert_batch_size]
            points = [
                qm.PointStruct(id=p.point_id, vector=p.vector, payload=p.payload)
                for p in batch
            ]
            client.upsert(collection_name=batch[0].collection, points=points, wait=True)
            written += len(batch)

    _with_retries("upsert_batch", _upsert)
    logger.info("Batch upsert complete", extra={
        "vectors": written, "document_id": document_id,
        "batch_size": settings.upsert_batch_size,
    })
    return written


def spool_offline(payloads: list[IndexPayload], document_id: str) -> Path:
    """When Qdrant is unreachable, spool validated payloads to local JSON."""
    for p in payloads:
        validate_payload_for_collection(p)
    spool_dir = settings.result_dir / settings.spool_dir_name
    spool_dir.mkdir(parents=True, exist_ok=True)
    dest = spool_dir / f"{document_id}.json"
    dest.write_text(json.dumps(
        [{"collection": p.collection, "point_id": p.point_id,
          "vector": p.vector, "payload": p.payload} for p in payloads],
    ), encoding="utf-8")
    logger.warning("Vector store unavailable; payloads spooled locally", extra={
        "spooled": len(payloads), "spool_path": str(dest),
    })
    return dest
