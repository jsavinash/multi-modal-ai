"""Phase 3 indexing pipeline: tabular file -> serialized rows -> embeddings -> Qdrant."""

from __future__ import annotations

import logging
from pathlib import Path

from app.config import settings
from app.embeddings import embedding_engine
from app.tabular.parsers import (
    TabularFormat,
    detect_tabular_format,
    new_source_document_id,
    parse_table,
    serialize_rows,
)
from app.tabular.telemetry import process_telemetry
from app.vector_schema import IndexPayload, make_payload
from app.vector_store import (
    VectorStoreUnavailableError,
    ensure_collections,
    spool_offline,
    upsert_batch,
)

logger = logging.getLogger(__name__)


class IndexingError(RuntimeError):
    """Raised when the indexing pipeline fails."""


def run_indexing(path: Path, fmt: TabularFormat, filename: str) -> dict:
    """Full Phase 3 pipeline for one tabular file.

    Returns a summary dict: source_document_id, rows indexed, store mode
    ("qdrant" | "spooled"), spool_path (when spooled).
    """
    document_id = new_source_document_id(path)
    table = parse_table(path, fmt, filename)
    rows = serialize_rows(table)
    batch = process_telemetry(table, rows)

    # Embed in bounded sub-batches (RAM guardrail), attach normalized values.
    texts = [nr.row.text for nr in batch.rows]
    index_payloads: list[IndexPayload] = []
    step = settings.max_embed_batch_rows
    offset = 0
    for start in range(0, len(texts), step):
        chunk_texts = texts[start:start + step]
        vectors = embedding_engine.embed_batch(chunk_texts)
        for i, vector in enumerate(vectors):
            nr = batch.rows[start + i]
            extra: dict = {
                "row_index": nr.row.row_index,
                "source_document_id": document_id,
                "source_filename": filename,
                "normalized_values": nr.scaled_values,
            }
            if batch.is_gps:
                extra["gps_record"] = True
            payload = make_payload(
                collection_key="tabular",
                document_id=document_id,
                text=nr.row.text,
                timestamp=nr.row.timestamp,
                extra=extra,
            )
            payload.vector = vector
            index_payloads.append(payload)
        offset += len(vectors)

    # Offload to the dedicated Vector DB; spool offline when unreachable.
    store_mode = "qdrant"
    spool_path = None
    try:
        ensure_collections()
        written = upsert_batch(index_payloads, document_id)
        logger.info("Indexing complete via Qdrant", extra={
            "document_id": document_id, "vectors": written, "rows": len(batch.rows),
        })
    except VectorStoreUnavailableError as exc:
        spool_path = str(spool_offline(index_payloads, document_id))
        store_mode = "spooled"
        logger.warning("Qdrant unreachable; indexed payloads spooled", extra={
            "document_id": document_id, "vectors": len(index_payloads),
            "reason": str(exc),
        })

    return {
        "source_document_id": document_id,
        "source_filename": filename,
        "table_shape": {"rows": table.frame.height, "columns": table.frame.width},
        "rows_indexed": len(index_payloads),
        "store_mode": store_mode,
        "spool_path": spool_path,
        "scaling_ranges": {
            col: [lo, hi] for col, (lo, hi) in batch.scaling_ranges.items()
        },
        "is_gps": batch.is_gps,
        "embedding_backend": "minilm" if embedding_engine.available else "hash_fallback",
    }
