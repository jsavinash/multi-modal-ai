# Phase 1 — Technical Documentation

Multimodal AI Ingestion Engine: **Document Decomposition & Structural Text Ingestion** microservice.

- Runtime: Python 3.10+ (container pins 3.12)
- Frameworks: FastAPI, Celery, Redis, pypdf, pdfminer.six, python-docx
- Requirements traceability: see [`docs/PHASE1_SPEC.md`](PHASE1_SPEC.md)

---

## 1. Architecture

```
                    ┌──────────────────────────────────────────────┐
                    │                  API Service                 │
  HTTP multipart    │  ┌────────────┐  ┌──────────┐  ┌──────────┐  │
 ──────────────────►│  │ main.py    │─►│ magic.py │─►│ storage  │  │
   PDF/TXT/MD/DOCX  │  │ (FastAPI)  │  │ (sniff)  │  │ (.py)    │  │
                    │  └────────────┘  └──────────┘  └────┬─────┘  │
                    │              202 + task_id          │ file   │
                    └─────────────────────────────────────┼────────┘
                                                          ▼
                                              ┌───────────────────┐
                                              │  Redis (broker /  │
                                              │  result backend)  │
                                              └─────────┬─────────┘
                                                        ▼
                    ┌──────────────────────────────────────────────┐
                    │               Celery Worker                  │
                    │  ┌──────────┐   ┌──────────────────────────┐ │
                    │  │ tasks.py │──►│ parsers (page-streamed)  │ │
                    │  └──────────┘   │  pdf_parser (pdfminer +  │ │
                    │        │        │  pypdf)                  │ │
                    │        │        │  docx_text_parser        │ │
                    │        │        │  chunker (512t/10% ovl)  │ │
                    │        ▼        └──────────────────────────┘ │
                    │  CanonicalDocument JSON → results/           │
                    └──────────────────────────────────────────────┘
```

**Data flow (happy path):**
1. Client `POST`s a multipart file to `/api/v1/documents/ingest`.
2. `main.py` reads the **first 4 KB**, `magic.py` identifies the true format (`_PrefixedStream` replays those bytes so no data is lost during persistence).
3. `storage.py` streams the upload to `uploads/` in 1 MB windows, enforcing the 100 MB ceiling (exceeding → delete partial + HTTP 413).
4. API dispatches `parse_document_task.delay(stored_path, filename)` and returns `202 {task_id, status: queued}`.
5. Celery worker re-sniffs magic bytes, computes a deterministic `document_id` (SHA-256 of first MB + file size), parses, chunks, validates against `CanonicalDocument`, writes `results/{document_id}.json`, and cleans up the temp upload.

## 2. Module Reference

| Module | Lines | Responsibility |
|---|---|---|
| `app/main.py` | 115 | FastAPI routing, lifespan startup (resource logging), format validation, Celery dispatch |
| `app/magic.py` | 80 | Magic-byte format detection; raises `UnsupportedFormatError` (→ 415) |
| `app/storage.py` | 81 | Streamed persistence, 100 MB ceiling, disk-floor guard, result persistence, temp cleanup |
| `app/schemas.py` | 38 | `CanonicalDocument` / `TextPayload` / `MediaReference` Pydantic contract |
| `app/tasks.py` | 93 | Celery task: parse → chunk → validate → persist; retries on `OSError` |
| `app/parser/pdf_parser.py` | 211 | Layout-aware PDF decomposition (structural tagging + image extraction) |
| `app/parser/docx_text_parser.py` | 83 | DOCX heading/body parsing; TXT/MD windowed stream reading |
| `app/parser/chunker.py` | 92 | Recursive character splitter, 512-token target, 10% overlap |
| `app/celery_app.py` | 60 | Celery config: queues, time limits, memory guardrails clamped to host RAM + declared budget |
| `app/config.py` | 122 | Typed settings via `pydantic-settings` (`INGEST_` env prefix); single source of truth for the capacity plan |
| `app/logging_config.py` | 44 | Single-line JSON `Formatter`, third-party logger silencing |
| `app/resources.py` | 238 | Runtime host probe (RAM/CPU/disk); budget-aware clamping + per-queue capacity plan validation |

**Host adaptation:** guardrails are no longer hard-coded to a target spec. At startup,
`app/resources.py` probes total RAM (macOS `sysctl hw.memsize`, Linux `/proc/meminfo`),
CPU count and free disk; `clamp_worker_memory` caps the per-child Celery limit to the
**lower** of the declared budget (`INGEST_WORKER_MEMORY_BUDGET_MB`, i.e. this queue's
container cap) and ~50 % of detected RAM (floor 256 MB).
`validate_capacity_plan()` then asserts `concurrency × child_limit ≤ container cap` for
all four queues and that the summed caps fit `INGEST_TOTAL_CONTAINER_BUDGET_MB`
(6400 MB), logging each verdict — call it from the API lifespan and the Celery
bootstrap. The current development host (8 GB / 8-core Apple M1 / ~346 GB free disk,
measured) therefore runs 2 × 1024 MB ingestion, 2 × 384 MB media, 1 × 768 MB index and
1 × 512 MB reasoning children inside a 6.25 GB total stack envelope, and the same image
scales up unchanged on 16 GB+ production machines.

## 3. API Reference

> Phase 2 and Phase 3 add further endpoints on the same app:
> `POST /api/v1/media/ingest` (image/video/audio — see
> [`docs/PHASE2_DOCUMENTATION.md`](PHASE2_DOCUMENTATION.md)) and
> `POST /api/v1/tabular/ingest` (CSV/XLSX/JSON — see
> [`docs/PHASE3_DOCUMENTATION.md`](PHASE3_DOCUMENTATION.md)). A cross-phase map
> lives in [`docs/INDEX.md`](INDEX.md).

### `GET /health` → `200`
Liveness probe. `{"status": "ok", "service": "..."}`

### `POST /api/v1/documents/ingest`
Multipart field: `file` (required).

| Response | Meaning |
|---|---|
| `202` | Accepted. Body: `{"task_id", "status": "queued", "filename", "file_size_bytes"}` |
| `413` | File exceeds 100 MB ceiling |
| `415` | Magic bytes don't match PDF/DOCX/TXT/MD (empty or binary file) |
| `500` | Disk/storage failure while persisting upload |
| `503` | Redis/broker unreachable — task could not be enqueued |
| `507` | Free disk below the 2 GB safety floor (`INGEST_MIN_FREE_DISK_MB`) |

## 4. Canonical Output Contract

Validated by `app.schemas.CanonicalDocument` (Pydantic v2, strict):

- `document_id: str` — SHA-256 content hash (first 1 MB + file size), deterministic.
- `metadata`: `filename`, `file_size_bytes ≥ 0`, `total_pages ≥ 0`.
- `payloads[]`: `chunk_id` (sequential int), `page_number ≥ 1`, `modality="text"`,
  `content`, `structural_tag ∈ {"header", "paragraph", "footer", "missing_text_layer"}`.
  - PDF: headers/footers are one payload each (never chunk-merged); consecutive same-page
    paragraphs are coalesced then re-split to the 512-token target.
  - TXT/MD/DOCX: `page_number = 1` (logical unit); DOCX headings get `structural_tag="header"`.
- `extracted_media_references[]`: `media_id` (UUID hex), `page_number`,
  `spatial_bounding_box = [x0, y0, x1, y1]` in PDF points (exactly 4 floats, enforced),
  `modality="raw_image"`. Raw bytes are held per-page during the parse loop only; the
  JSON contract intentionally carries references — persist bytes to object storage in Phase 2.

## 5. PDF Parsing Details

- **Text**: `pdfminer.six` `extract_pages(..., laparams=LAParams())` — exactly one page
  layout lives in memory at any time (stream chunking rule).
- **Structural tagging**: a text line is `header` if `y1 > page_height − 8%`, `footer`
  if `y0 < 8%` (configurable: `INGEST_HEADER_FOOTER_MARGIN_FRACTION`), else `paragraph`.
- **Missing text layer**: pages with zero text elements get a single
  `structural_tag="missing_text_layer"` payload (OCR hook for Phase 2).
- **Images**: raw bytes via `pypdf` `page.images`; spatial bbox via pdfminer `LTImage.bbox`
  (zero bbox when pdfminer did not emit a matching layout object). Corrupt individual
  images are logged and skipped, never fail the page.

## 6. Chunking Algorithm

`app/parser/chunker.py` — `recursive_split`:
1. Normalize whitespace (`[ \t]+ → " "`, collapse 3+ newlines).
2. Token estimate: `len(text) / chars_per_token` (default 4.0, configurable).
3. Descend separators `["\n\n", "\n", ". ", " ", ""]`; the final `""` hard-splits per character.
4. Chunks accumulate to ~512 tokens; when a chunk closes, the tail segments totalling
   ~10% of the target (≈51 tokens) are carried into the next chunk as overlap.
5. Oversized segments are recursed; empty text yields zero chunks.

## 7. Configuration

All settings are environment-driven with the `INGEST_` prefix (see `app/config.py`):

| Variable | Default | Purpose |
|---|---|---|
| `INGEST_MAX_FILE_SIZE_BYTES` | 104857600 (100 MB) | Upload ceiling |
| `INGEST_WORKER_CHILD_MEMORY_LIMIT_MB` | 1024 | Per-Celery-child memory cap (clamped to host RAM at startup) |
| `INGEST_WORKER_MEMORY_BUDGET_MB` | 2048 | Ingestion worker footprint budget (= its 2g container cap); enforced alongside host-RAM clamping |
| `INGEST_WORKER_MAX_TASKS_PER_CHILD` | 20 | Proactive child recycling |
| `INGEST_MIN_FREE_DISK_MB` | 2048 | Refuse uploads below 2GB free disk (HTTP 507) |
| `INGEST_UPLOAD_DIR` / `INGEST_RESULT_DIR` | `/tmp/multimodal-ingestion/{uploads,results}` | Storage paths |
| `INGEST_REDIS_URL` | `redis://localhost:6379/0` | Broker + result backend |
| `INGEST_CELERY_TASK_QUEUE` | `ingestion` | Queue name |
| `INGEST_CELERY_TASK_TIME_LIMIT_S` | 1200 | Hard task time limit |
| `INGEST_CHUNK_TARGET_TOKENS` | 512 | Chunk target |
| `INGEST_CHUNK_OVERLAP_FRACTION` | 0.10 | Overlap factor |
| `INGEST_CHARS_PER_TOKEN` | 4.0 | Token estimate divisor |
| `INGEST_HEADER_FOOTER_MARGIN_FRACTION` | 0.08 | PDF header/footer margin |
| `INGEST_ENVIRONMENT` | `development` | Stamped into every log line |

## 8. Observability

- All logs are single-line JSON (`JsonFormatter`): `timestamp`, `level`, `logger`,
  `message`, `service`, `environment`, plus structured `extra` fields
  (`document_id`, `file_name`, `page_number`, `stored_path`, `task_id`, ...).
- **Startup resource logs**: the API lifespan emits a `System resources detected`
  event (`total_memory_mb`, `cpu_count`, `available_disk_mb`, `platform`) and warns when
  host RAM is below the deployment-spec target. The Celery worker logs its effective
  `worker_max_memory_per_child` after clamping.
- Note: log extras must avoid reserved `LogRecord` attribute names (`filename`,
  `message`, ...) — the service uses `file_name` instead.
- Noisy third-party loggers (`pdfminer`, `urllib3`, `kombu`) are clamped to `WARNING`.
- No `print` statements exist in `app/` or `scripts/`.

## 9. Error Handling Matrix

| Failure | Raised As | HTTP / Task Outcome |
|---|---|---|
| Empty or binary upload | `UnsupportedFormatError` | 415 |
| Upload > 100 MB | `FileTooLargeError` | 413 (partial file deleted) |
| Fake `%PDF-` header / broken xref / encrypted PDF | `CorruptedPDFError` | Task fails, upload cleaned, error logged |
| Invalid DOCX container | `CorruptedDocumentError` | Task fails, upload cleaned |
| Page with no text layer | — (warning + `missing_text_layer` tag) | Continues |
| Corrupt embedded image | — (warning, image skipped) | Continues |
| Broker down | — | 503 at enqueue time |
| Storage write failure | `StorageWriteError` | 500 |
| Free disk below safety floor | `DiskFullError` | 507 |

## 10. Running the Service

```bash
# Local
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
docker compose up -d redis
uvicorn app.main:app --reload                              # API :8000
celery -A app.celery_app.celery_app worker -l INFO         # worker
curl -F "file=@sample.pdf" localhost:8000/api/v1/documents/ingest

# Full stack
docker compose up --build     # redis + api + 4 workers + qdrant (caps total 6.25g)

# Validation
pytest -q                                    # 33 tests (all phases)
PYTHONPATH=. python scripts/smoke_pdf.py     # end-to-end PDF parse (no Redis needed)
```

Interactive API docs: `http://localhost:8000/docs` (Swagger UI).

## 11. Test Suite (`tests/test_ingestion.py`)

| Test | Covers |
|---|---|
| `test_detect_{pdf,docx,txt_and_md}` | Magic-byte routing (FR-1) |
| `test_reject_binary` | Corrupted/unsupported header rejection |
| `test_chunk_short_text_single_chunk` | Chunker small-input path |
| `test_chunk_large_text_target_size_and_overlap` | 512-token target + 10% overlap (FR-4) |
| `test_canonical_schema_validates` / `_rejects_bad_bbox` | Contract strictness (FR-5) |
| `test_health`, `test_ingest_rejects_unsupported`, `test_ingest_accepts_txt` | API surface incl. 415/202 |
| `test_detect_system_resources` | Host probe sanity (RAM/CPU bounds, platform tag) |
| `test_clamp_reduces_when_host_smaller_than_config` | Guardrail clamping: 16 GB config vs 8 GB host → ~50 % budget |
| `test_clamp_floor` | 256 MB clamp floor on tiny hosts |

The API tests monkeypatch `upload_dir` and stub `parse_document_task.delay`, so the suite
runs **without Redis**.

## 12. Known Limitations & Phase 2 Hooks

- **No OCR** — scanned pages are tagged `missing_text_layer`, not transcribed.
- **Image bytes not persisted** — `extracted_media_references` carry `media_id` only;
  Phase 2 should stream bytes to object storage keyed by `media_id`.
- **Token count is a heuristic** (chars/4); swap in a real tokenizer before Phase 2 embedding.
- **XLSX parser deferred** — `openpyxl` is already a dependency.
- **No auth/rate-limiting** on the ingest endpoint — add API-key middleware / gateway policy.

---

## Revision History

| Date | Version | Change |
|---|---|---|
| 2026-09-21 | 1.0 | Initial Phase 1 implementation: ingestion router, Celery/Redis pipeline, layout-aware PDF parsing, recursive chunking, canonical schema, JSON logging, docs. |
| 2026-09-21 | 1.1 | System-resource adaptation: added `app/resources.py` host probe; worker guardrails (1 GB/child, 4 GB budget) sized and auto-clamped for the 8 GB / 8-core Apple M1 dev host; disk safety floor (2 GB) with HTTP 507; compose `mem_limit` caps; 3 new tests (14 total); docs updated. |
| 2026-09-21 | 2.0 | Phase 2 node added alongside Phase 1 (see [`docs/PHASE2_DOCUMENTATION.md`](PHASE2_DOCUMENTATION.md)): media ingestion API, `media` queue isolation, dedicated media-worker in compose. Phase 1 modules unchanged; test suite now 22 tests. |
| 2026-09-21 | 3.0 | Phase 3 added (see [`docs/PHASE3_DOCUMENTATION.md`](PHASE3_DOCUMENTATION.md)): tabular ingestion API, `index` queue + index-worker + qdrant service in compose. Test suite now 33 tests. |
| 2026-09-21 | 4.0 | Phase 4 added (see [`docs/PHASE4_DOCUMENTATION.md`](PHASE4_DOCUMENTATION.md)): reasoning core, `reasoning` queue + worker in compose. Test suite now 46 tests. |
| 2026-09-21 | 4.1 | Resource realignment: compose caps trimmed 11g → **6.25g** (ingestion 4g→2g, media 3g→1g, index 2g→1g, api 1g→512m, reasoning 1g→512m, qdrant 1g, redis 256m); ingestion concurrency 4→2; per-worker child limits clamped (media 384 MB, index 768 MB); qdrant/redis now capped. Invariant: caps ≤ ~75 % of host RAM. |
