# Phase 3 — Telemetry, Tabular Parser & Vector Database Indexer (Specification)

> Phase 3 of the Multimodal AI Ingestion Engine: structured data ingestion
> (CSV/XLSX/JSON), telemetry normalization, semantic serialization, local
> embedding, and offloaded vector storage.
>
> **Adapted to the actual host system** (8 GB RAM / 8-core Apple M1, no discrete
> GPU; Qdrant/Milvus not running locally). Deviations marked **[ADAPTED]**.

## 0. System Analysis → Prompt Modifications

| Original constraint | Actual system finding | Adaptation |
|---|---|---|
| High-throughput CPU/RAM processing | 8 GB RAM / 8-core M1 | **[ADAPTED]** Polars over Pandas (streaming, low memory); row-batch caps keep RSS bounded; Phase 1 `resources.py` clamps still apply |
| Dedicated Vector DB (Qdrant or Milvus) | Neither is running on this host; Docker available | **[ADAPTED]** Qdrant selected (lighter footprint); client is lazily initialized with pooled connections + retry logic; when Qdrant is unreachable, index payloads are spooled to local JSON so nothing is lost — the same graceful-degradation pattern used in Phase 2 |
| `all-MiniLM-L6-v2` / quantized CLIP | MiniLM (~80 MB) fits the CPU budget; CLIP is out of memory envelope | **[ADAPTED]** MiniLM primary; deterministic hash-based fallback embedder keeps tests/offline runs deterministic |

## 1. Functional Requirements

1. **Structured Data Engine:** parse CSV, XLSX and JSON tables with **Polars**
   (streaming-friendly); magic-byte format detection (XLSX = PK/OPC container,
   JSON = decodable text starting `{`/`[`, CSV = delimited text).
2. **Semantic Matrix Serialization:** raw rows are never embedded. Each row becomes a
   declarative sentence; telemetry-shaped rows (timestamp + sensor + value) render as
   *"At timestamp `<ts>`, the `<sensor>` sensor recorded a telemetry value of `<value>` units"*;
   generic rows render as `"k1 is v1; k2 is v2; ..."`.
3. **Telemetry Stream Processor:** accepts simulated time-series logs and GPS coordinate
   arrays; applies **min-max scaling** (bounded to `[0, 1]`) to numeric vectors, tracked
   per column.
4. **Embedding Model Pipeline:** local `sentence-transformers/all-MiniLM-L6-v2`
   (384-dim), lazily loaded, normalized vectors; deterministic fallback embedder when
   the model stack is absent.
5. **Vector Storage Handlers:** Qdrant connection pool with **retry + exponential
   backoff**; one **collection per modality** (`tabular_telemetry`, plus
   `document_text`, `vision`, `audio`); payload indexes on `document_id`, `modality`,
   `timestamp`; **batch upserts of 64 vectors**.

## 2. Coding Standards

- Connection pooling + robust retry logic (exponential backoff, capped attempts) on all
  Qdrant operations.
- `validate_payload_for_collection()` verifies that embedded vector dimensionality and
  payload field types map to the target collection schema before any write.
- JSON logging only (shared `JsonFormatter`); `try...finally` model cleanup like Phase 2.

## 3. Canonical Output

```json
{
  "source_document_id": "str",
  "source_filename": "str",
  "table_shape": {"rows": int, "columns": int},
  "index_payloads": [
    {
      "point_id": "str (uuid)",
      "collection": "tabular_telemetry",
      "vector": [float],
      "payload": {
        "document_id": "str", "modality": "tabular",
        "text": "str (serialized sentence)",
        "row_index": int, "timestamp": "str|null",
        "normalized_values": {"col": float}
      }
    }
  ]
}
```

## Revision History

| Date | Version | Change |
|---|---|---|
| 2026-09-21 | 3.0 | Original prompt adapted to host (Polars over Pandas, Qdrant with offline spooling, MiniLM with fallback embedder). |
