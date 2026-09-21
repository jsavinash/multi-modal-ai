# Phase 3 — Technical Documentation

Telemetry, Tabular Data Parser & Vector Database Indexer of the Multimodal AI
Ingestion Engine.

- Runtime: Python 3.10+; CPU-first, high-throughput (8 GB host budget respected)
- Stack: Polars, qdrant-client, sentence-transformers `all-MiniLM-L6-v2` (optional, lazy)
- Spec & prompt adaptations: [`docs/PHASE3_SPEC.md`](PHASE3_SPEC.md)

---

## 1. Architecture

```
 HTTP upload (CSV/XLSX/JSON)
        │  /api/v1/tabular/ingest  (magic-byte sniff + streamed storage)
        ▼
 Celery queue "index"  ──►  index-worker (x1 child, 1g cap)
        │
        ▼
 app/tabular/indexing.run_indexing()
   1. parse_table()          Polars CSV / XLSX / JSON -> DataFrame
   2. serialize_rows()       Semantic Matrix Serialization (rows -> sentences)
   3. process_telemetry()    min-max scaling of numeric vectors (GPS-aware)
   4. embedding_engine       all-MiniLM-L6-v2 (or hash fallback), 384-dim, batched ≤256
   5. vector_store           Qdrant: pooled client, retries/backoff,
                             batches of 64, collections per modality
                             ↳ unreachable? spool to results/spool/*.json
```

## 2. Module Reference

| Module | Lines | Responsibility |
|---|---|---|
| `app/tabular/parsers.py` | 177 | Magic-byte detection (CSV/XLSX/JSON), Polars parsing, Semantic Matrix Serialization |
| `app/tabular/telemetry.py` | 74 | Min-max scaling of numeric columns, GPS detection, scaling-range tracking |
| `app/tabular/indexing.py` | 102 | Full pipeline orchestration: parse → serialize → scale → embed → store/spool |
| `app/embeddings.py` | 110 | Lazy MiniLM (384-dim, normalized); configurable device (`INGEST_EMBEDDING_DEVICE`, CPU default); deterministic hash fallback; 256-row batch cap |
| `app/vector_schema.py` | 80 | Modality collections map, `IndexPayload`, `validate_payload_for_collection()` |
| `app/vector_store.py` | 142 | Pooled Qdrant client, exponential-backoff retries, collection+index init, 64-batch upserts, offline spool |
| `app/tasks_tabular.py` | 52 | Celery task `tabular.index` on the `index` queue |

## 3. API Reference

### `POST /api/v1/tabular/ingest` → `202`
Multipart field: `file` (required). Formats validated by magic bytes:

| Format | Signature | Parser |
|---|---|---|
| CSV | decodable text with delimiter + newline | `pl.read_csv` (ragged-line tolerant) |
| XLSX | `PK\x03\x04` + `xl/` / `spreadsheetml` | `pl.read_excel` (sheet 1) |
| JSON | decodable text starting `{`/`[` | records array or `{"rows"|"data"|"records"|"items": [...]}` |

Error codes: `413` size ceiling, `415` non-tabular bytes, `507` disk floor, `503` broker down.

## 4. Semantic Matrix Serialization (FR-2)

Raw rows are **never embedded**. Column-name heuristics detect table shape
(case-insensitive):

- **Telemetry-shaped** (has `timestamp`/`time`/`datetime` + `sensor`/`device` +
  `value`/`reading`):
  `Timestamp: 2026-09-21T10:00:00, Sensor: Temp, Value: 23.5` →
  *"At timestamp 2026-09-21T10:00:00, the Temp sensor recorded a telemetry value of 23.5 units"*
- **Generic rows**: `Event: login; Region: eu-west; Score: 0.98`

## 5. Telemetry Stream Processor (FR-3)

- Numeric columns are scaled per column: `scaled = (v - min) / (max - min)`, clamped
  to `[0, 1]`; degenerate ranges (min == max) map to `0.0`.
- Scaling ranges are persisted in the task summary for reproducibility.
- **GPS arrays** (columns matching `lat|latitude` AND `lon|lng|longitude`) are detected
  and flagged (`is_gps`, payload `gps_record: true`), with Latitude/Longitude scaled
  independently.

## 6. Embedding Pipeline (FR-4)

- Model: `sentence-transformers/all-MiniLM-L6-v2`, 384-dim, normalized vectors, CPU.
- Lazy singleton per worker process; `try...finally` device-cache cleanup via
  `DeviceManager.inference_session()` (inherited from Phase 2).
- **Deterministic hash fallback** (seeded SHA-256 stream, normalized) when the model
  stack is absent — tests and offline runs stay reproducible.
- RAM guardrail: embed calls capped at `INGEST_MAX_EMBED_BATCH_ROWS` (256).

## 7. Vector Storage Handlers (FR-5)

- **Collections per modality** (`app/vector_schema.COLLECTIONS`):
  `tabular_telemetry`, `document_text`, `vision`, `audio` — cosine distance,
  384-dim, created idempotently.
- **Payload indexes**: `document_id`, `modality` (keyword), `timestamp` — created if
  missing on every `ensure_collections()` call.
- **Connection pooling**: single module-level `QdrantClient` (manages its own HTTP pool);
  all operations wrapped in `_with_retries()` — up to 5 attempts with exponential
  backoff (0.5s, 1s, 2s, 4s, 8s).
- **Batch upserts**: 64 vectors per upsert (`INGEST_UPSERT_BATCH_SIZE`), `wait=True`.
- **Offline spool**: when Qdrant is unreachable after retries, validated payloads are
  spooled to `results/spool/{document_id}.json` so no data is lost; replay is a Phase 4 hook.
- **Validation function**: `validate_payload_for_collection()` verifies vector
  dimensionality == 384, collection name ∈ COLLECTIONS, and required payload fields
  (`document_id: str`, `modality: str`, `timestamp: str|null`, `text: str`) — runs
  before *every* upsert/spool write.

## 8. Configuration (`INGEST_` prefix)

| Variable | Default | Purpose |
|---|---|---|
| `INGEST_INDEX_TASK_QUEUE` | `index` | Dedicated Phase 3 queue |
| `INGEST_EMBEDDING_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `INGEST_EMBEDDING_DIM` | 384 | Collection vector size |
| `INGEST_UPSERT_BATCH_SIZE` | 64 | Vectors per batch upsert |
| `INGEST_QDRANT_URL` | `http://localhost:6333` | Vector DB endpoint |
| `INGEST_QDRANT_TIMEOUT_S` | 10.0 | Client timeout |
| `INGEST_QDRANT_MAX_RETRIES` | 5 | Retry attempts |
| `INGEST_QDRANT_RETRY_BASE_DELAY_S` | 0.5 | Backoff base (exponential) |
| `INGEST_MAX_EMBED_BATCH_ROWS` | 256 | RAM guardrail per embed call |

## 9. Test Coverage (`tests/test_tabular.py`)

| Test | Covers |
|---|---|
| `test_detect_tabular_formats` | CSV/JSON/XLSX magic-byte routing + binary rejection |
| `test_parse_and_serialize_telemetry` | Telemetry sentence rendering + numeric extraction |
| `test_serialize_generic_rows` | Generic `k is v; ...` serialization |
| `test_min_max_scaling` | Min-max math incl. degenerate range; `(0, 0.5, 1.0)` on (10,20,30) |
| `test_gps_detection` | GPS column detection |
| `test_embedding_fallback_deterministic` | 384-dim, deterministic, normalized fallback |
| `test_vector_schema_validation` | Dim/type/missing-field/unknown-collection rejection |
| `test_batch_upsert_splits_at_64` | 130 vectors → upserts of 64, 64, 2 |
| `test_retry_logic_backoff` | 5 attempts then failure; recovery after transient blips |
| `test_offline_spool` | Validated JSON spool on Qdrant outage |
| `test_run_indexing_end_to_end_spooled` | Full pipeline, `store_mode="spooled"` |

## 10. Running the Service

```bash
docker compose up --build   # redis + api + ingestion/media/index workers + qdrant:6333

# dedicated index worker (local dev)
celery -A app.celery_app.celery_app worker -l INFO --queues=index --concurrency=1

curl -F "file=@telemetry.csv" localhost:8000/api/v1/tabular/ingest

# validation
pytest -q                                     # 33 tests (14 P1 + 8 P2 + 11 P3)
PYTHONPATH=. python scripts/smoke_tabular.py  # CSV/GPS -> serialize -> scale -> spool
```

## 11. Known Limitations & Phase 4 Hooks

- **Spooled payloads are not auto-replayed** — a Phase 4 sync job should drain
  `results/spool/` into Qdrant once it is reachable.
- **XLSX multi-sheet**: only the first sheet is parsed (`sheet_id=1`).
- **Serialization heuristics** are column-name based; a schema-inference layer (Phase 4)
  could support arbitrary domain templates.
- **Timestamp payload index** uses KEYWORD; switch to a datetime range index when
  time-window search is needed.
- **sentence-transformers is optional** — install it to switch from the hash fallback
  to real MiniLM vectors (hash vectors are NOT semantically meaningful).
- **Phase 4 delivered the reasoning core**; a dedicated multi-modal *retrieval/search*
  API (querying all four collections with filters on `document_id` / `modality` /
  `timestamp` to produce context bundles) remains a hardening-phase item — Phase 4
  currently consumes bundles produced by the caller.

## Revision History

| Date | Version | Change |
|---|---|---|
| 2026-09-21 | 3.0 | Initial Phase 3 implementation: Polars tabular engine, semantic serialization, min-max telemetry scaling, MiniLM embedding pipeline (fallback), pooled Qdrant store with retries, 64-batch upserts, schema validation, offline spooling, 11 new tests (33 total), smoke script. |
| 2026-09-21 | 3.1 | Doc refresh: corrected module line counts to match source; added system index [`docs/INDEX.md`](INDEX.md); cross-linked endpoints in the Phase 1 doc API section. |
| 2026-09-21 | 4.0 | Phase 4 reasoning core consumes Phase 3 index payloads; retrieval/search API noted as hardening-phase item. |
