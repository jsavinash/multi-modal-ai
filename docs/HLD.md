# HLD — High-Level Design & Execution Flows

High-level design of the Multimodal AI Ingestion Engine: the components, and the
**step-by-step flow of every execution the system can perform**.

Diagrams are [Mermaid](https://mermaid.js.org/) and render automatically on GitHub.
A plain-text version of the top-level view is included for terminals and printers.

> Plain-English companion: [`EXPLAINER.md`](EXPLAINER.md) · architecture map:
> [`INDEX.md`](INDEX.md)

---

## 1. System at a glance

```mermaid
flowchart LR
    U["Caller<br/>curl / SDK / UI"]

    subgraph APIB["api container 512m - FastAPI, I/O bound"]
        V["validate magic bytes<br/>100 MB cap, 2 GB disk floor<br/>stream 1 MB chunks to disk"]
    end

    subgraph QB["redis 256m - broker + result backend"]
        Q1(["ingestion queue conc 2"])
        Q2(["media queue conc 2"])
        Q3(["index queue conc 1"])
        Q4(["reasoning queue conc 1"])
    end

    subgraph WB["Celery workers - one container per queue"]
        W1["worker 2g<br/>P1 documents"]
        W2["media-worker 1g<br/>P2 vision + audio"]
        W3["index-worker 1g<br/>P3 tabular + vectors"]
        W4["reasoning-worker 512m<br/>P4 reasoning"]
    end

    QD[("qdrant 1g<br/>4 collections")]
    FS[("results<br/>JSON + MD reports")]

    U -->|upload| V
    V -->|enqueue| Q1
    V -->|enqueue| Q2
    V -->|enqueue| Q3
    V -->|enqueue| Q4
    Q1 --> W1
    Q2 --> W2
    Q3 --> W3
    Q4 --> W4
    W3 -->|vectors 64 per batch| QD
    W4 -.->|retrieval hook| QD
    W1 --> FS
    W2 --> FS
    W3 --> FS
    W4 --> FS
```

Same view as plain text, for terminals:

```
  Caller
    POST /api/v1/{documents|media|tabular}/ingest      POST /api/v1/reason/query
    │
    ▼
  ┌─ api container 512m  -  FastAPI, I/O bound ─────────────────────────────────────────────────┐
  │ verify magic bytes (real signature, never the filename)  |  100 MB cap  |  2 GB disk floor  │
  │ stream to disk in 1 MB chunks, then answer 202 Accepted + task_id                           │
  │                                                                                             │
  └──────────────────────────────────────────┬──────────────────────────────────────────────────┘
                                             │ enqueue Celery ticket
                                             ▼
  ┌─ redis 256m  -  broker + result backend ────────────────────────────────────────────────────┐
  │ ingestion conc 2        media conc 2        index conc 1        reasoning conc 1            │
  │                                                                                             │
  └─────────┬───────────────────────┬───────────────────────┬───────────────────────┬───────────┘
            ▼                       ▼                       ▼                       ▼
  ┌─ 1  worker ────────┐  ┌─ 2  media ─────────┐  ┌─ 3  index ─────────┐  ┌─ 4  reason ────────┐
  │ 2g                 │  │ 1g                 │  │ 1g                 │  │ 512m               │
  │ P1 documents       │  │ P2 vision          │  │ P3 tabular         │  │ P4 reasoning       │
  │                    │  │ P2 audio           │  │                    │  │                    │
  └────────────────────┘  └────────────────────┘  └────────────────────┘  └────────────────────┘
            └───────────────────────┴───────────┬───────────┴───────────────────────┘
                                                │
                                                ▼
  ┌─ results/  *.json  +  reports/*.md ─────────────────────────────────────────────────────────┐
  └─────────────────────────────────────────────────────────────────────────────────────────────┘
```

Legend: **qdrant 1g** holds the 384-dim vectors — index-worker writes them (64 per batch)
and reasoning-worker reads retrieval context back. Phases 1-4 run in separate containers so
one heavy video can never block a small text file.


---

## 2. Every execution the system can perform

| # | Execution | Trigger | Ticket (queue) | Executor | Result |
|---|---|---|---|---|---|
| 1 | Document ingestion | `POST /api/v1/documents/ingest` | `ingestion` | worker 2g, conc 2 | `results/<document_id>.json` |
| 2 | Media processing | `POST /api/v1/media/ingest` | `media` | media-worker 1g, conc 2 | `results/media_<id>_<name>.json` |
| 3 | Tabular indexing | `POST /api/v1/tabular/ingest` | `index` | index-worker 1g, conc 1 | `results/index_<id>.json` (+ Qdrant) |
| 4 | Reasoning query | `POST /api/v1/reason/query` | `reasoning` | reasoning-worker 512m, conc 1 | `results/reasoning_<qid>.json` + `reports/report_<qid>.md` |
| 5 | Health check | `GET /health` | — (inline) | api container | `{"status": "ok"}` |

**Universal pattern for 1–4:** the API validates and stores the file, puts a ticket on a
queue, and answers `202 Accepted` immediately. A worker picks the ticket up and does the
slow work. Nothing slow ever happens inside the HTTP request.

---

## 3. Execution 1 — upload a document

```mermaid
sequenceDiagram
    autonumber
    actor U as Caller
    participant API as api container
    participant R as redis
    participant W as worker, ingestion queue conc 2
    participant FS as results

    U->>API: POST /api/v1/documents/ingest + file
    API->>API: read first 4 KB and verify magic bytes (PDF/DOCX/TXT/MD)
    API->>FS: stream to disk in 1 MB chunks, reject above 100 MB
    API->>R: enqueue ingest.parse_document(file, filename)
    API-->>U: 202 Accepted with task_id
    R->>W: deliver ticket
    W->>W: re-verify magic bytes and compute document_id
    W->>W: PDF - layout analysis page by page, header and footer tags
    W->>W: DOCX/TXT/MD - headings and body text, windowed reads
    W->>W: chunk to 512 tokens with a 10 percent overlap
    W->>W: validate CanonicalDocument
    W->>FS: write results/document_id.json
    W->>FS: delete the upload in a finally block
```

**Produces:** `document_id`, `metadata`, `payloads[]` (chunk_id, page_number, modality,
content, structural_tag) and `extracted_media_references[]`.

**Failure paths:** unsupported bytes → `415` at the API · too large → `413` with the
partial file deleted · unreadable PDF → task fails with `ParsingFailure`, and the upload
is still cleaned up.

---

## 4. Execution 2 — upload media (image, video, audio)

```mermaid
sequenceDiagram
    autonumber
    actor U as Caller
    participant API as api container
    participant R as redis
    participant W as media-worker, media queue conc 2
    participant FS as results

    U->>API: POST /api/v1/media/ingest + file
    API->>API: magic bytes (JPEG PNG WebP BMP MP4 AVI MKV WAV MP3)
    API->>R: enqueue media.process(file, filename, parent_document_id)
    API-->>U: 202 Accepted with task_id
    R->>W: deliver ticket
    alt image
        W->>W: downsample to 512 px longest edge
        W->>W: OCR with EasyOCR, confidence gate 0.35
    else video
        W->>W: sample exactly 1 frame per second
        W->>W: OCR each sampled frame at 512 px
    else audio
        W->>W: resample to 16 kHz mono
        W->>W: slice into strict 30 second windows
        W->>W: transcribe with whisper tiny int8, keep start and end times
    end
    W->>W: build UnifiedMediaPayload with notes markers
    W->>FS: write results/media_parent_name.json
    W->>FS: delete the upload in a finally block
```

**Produces:** one payload per image/frame/window: `processed_text_content`,
`media_metadata` (resolution, sample rate, duration, temporal anchors) and `notes[]` such
as `ocr_unavailable`, `asr_unavailable`, `image_decode_failed`.

**Failure paths:** a missing OCR/ASR model yields empty text plus a note, never a crash ·
`ffmpeg` missing means MP3 reports a clear message while WAV still works · per-item work
runs under a 300 s timeout, and `ffmpeg` itself under 120 s.

---

## 5. Execution 3 — upload tabular data (CSV, XLSX, JSON)

```mermaid
sequenceDiagram
    autonumber
    actor U as Caller
    participant API as api container
    participant R as redis
    participant W as index-worker, index queue conc 1
    participant Q as qdrant
    participant FS as results

    U->>API: POST /api/v1/tabular/ingest + file
    API->>API: magic bytes (CSV, XLSX zip package, JSON)
    API->>R: enqueue tabular.index(file, filename)
    API-->>U: 202 Accepted with task_id
    R->>W: deliver ticket
    W->>W: parse with Polars, first sheet for XLSX
    W->>W: rewrite each row as a sentence
    W->>W: min-max scale numeric columns to 0..1 and detect GPS columns
    W->>W: embed with MiniLM, 384 dims, batches of 256 rows maximum
    W->>W: validate vector dimension and required payload fields
    alt qdrant reachable
        W->>Q: ensure collections and payload indexes
        W->>Q: upsert 64 vectors per batch into tabular_telemetry
    else qdrant unreachable
        W->>FS: spool validated payloads to results/spool/id.json
    end
    W->>FS: write results/index_id.json summary
    W->>FS: delete the upload in a finally block
```

**Produces:** a summary (rows indexed, table shape, per-column scaling ranges, `is_gps`,
store mode `qdrant` or `spooled`, embedding backend) plus searchable vectors.

**Failure paths:** Qdrant down → 5 retries with exponential backoff → validated offline
spool, so no data is lost · `sentence-transformers` absent → deterministic hash fallback
and the pipeline still completes · unknown collection or wrong dimension →
`SchemaValidationError` before any write reaches the database.

---

## 6. Execution 4 — ask a question

```mermaid
sequenceDiagram
    autonumber
    actor U as Caller
    participant API as api container
    participant R as redis
    participant W as reasoning-worker, conc 1
    participant LLM as reasoning backbone
    participant FS as results

    U->>API: POST /api/v1/reason/query?query=... + JSON context bundle
    API->>API: cheap JSON sniff and stream the bundle to disk
    API->>R: enqueue reasoning.query(bundle_path, query)
    API-->>U: 202 Accepted with task_id
    R->>W: deliver ticket
    W->>W: load items and estimate tokens per item
    W->>W: rank by similarity and prune to the budget 8k local / 32k cloud
    W->>W: keep at least one item per modality that was present
    W->>W: enforce the hard 2 MB byte cap and assemble the prompt
    alt local GGUF enabled
        W->>LLM: llama.cpp, never on this 8 GB host
    else cloud key configured and breaker closed
        W->>LLM: OpenAI or Anthropic, 60 s timeout, 429 aware
    else nothing reachable
        W->>W: deterministic offline fallback summary
    end
    W->>W: split answer from ACTIONS and validate ActionBlocks
    W->>FS: write reports/report_qid.md
    W->>FS: write results/reasoning_qid.json and delete the bundle
```

**Produces:** `query_id`, `backend_used`, `answer_text`, validated `actions[]`,
`report_path`, `pruning` statistics and `context_bytes`.

**Failure paths:** provider 5xx or timeout → 3 failures open the circuit breaker for 30 s ·
rate limit `429` honoured with `Retry-After` · oversized context → `ContextOverflowError`
instead of shipping gigabytes · malformed action lines become `action: "unparsed"` rather
than failing the whole report.

---

## 7. Execution 5 — health check

```mermaid
sequenceDiagram
    autonumber
    actor P as Probe or load balancer
    participant API as api container

    P->>API: GET /health
    API-->>P: 200 status ok
```

Handled inline — no queue, no worker, no disk access.

---

## 8. What every execution shares

All four async executions are the same shape, so the behaviour is predictable:

| Stage | What happens | Where |
|---|---|---|
| 1. Validate | Real byte signature checked (never the filename or the client's content type) | `app/magic.py`, `app/media/pipeline.py`, `app/tabular/parsers.py` |
| 2. Persist | Streamed to disk in 1 MB chunks, 100 MB ceiling, HTTP 507 below 2 GB free | `app/storage.py` |
| 3. Enqueue | One Celery task onto that phase's dedicated queue | `app/main.py` → `app/tasks*.py` |
| 4. Ack | `202 Accepted` with a `task_id` — never blocks on the work | `app/main.py` |
| 5. Process | Worker does the slow work, lazily loading models | `app/tasks*.py` → phase module |
| 6. Persist result | Validated JSON written to `results/` | `app/storage.py::save_result` |
| 7. Clean up | Upload deleted in a `finally` block, even on failure | `app/tasks*.py` |
| 8. Report | Structured JSON log line per stage (no print statements) | `app/logging_config.py` |

---

## 9. Startup execution (runs once, before any work)

```mermaid
flowchart TB
    S["process start"] --> D["probe host: RAM, CPU count, free disk"]
    D --> C["clamp worker child memory to min(declared budget, 50 percent host RAM)"]
    C --> P["build queue capacity plan"]
    P --> V{"concurrency x child limit <= container cap for every queue?"}
    V -->|no| E["log ERROR for the offending queue"]
    V -->|yes| OK["log INFO with per-queue headroom"]
    E --> T{"sum of container caps <= 6.25 GB envelope?"}
    OK --> T
    T -->|no| E2["log ERROR: stack oversubscribed"]
    T -->|yes| R["log INFO: host headroom"]
    R --> READY["ready to accept work"]
    E2 --> READY
```

Verified by `tests/test_capacity.py`, which also fails the build if `docker-compose.yml`
ever drifts from `app/config.py`.

---

## 10. Where each execution lives in code

| Execution | HTTP entry | Task | Core logic | Contract |
|---|---|---|---|---|
| 1. Documents | `app/main.py::ingest_document` | `app/tasks.py::parse_document_task` | `app/parser/` (pdf, docx, chunker) | `app/schemas.py` |
| 2. Media | `app/main.py::ingest_media` | `app/tasks_media.py::process_media_task` | `app/media/` (vision, video, audio) | `app/schemas_media.py` |
| 3. Tabular | `app/main.py::ingest_tabular` | `app/tasks_tabular.py::index_tabular_task` | `app/tabular/` + `app/embeddings.py` + `app/vector_store.py` | `app/vector_schema.py` |
| 4. Reasoning | `app/main.py::reason_query` | `app/tasks_reasoning.py::reasoning_task` | `app/reasoning/` (context, prompt, backends, compiler) | `app/reasoning/output_compiler.py` |
| 5. Health | `app/main.py::health` | — | — | — |

Shared infrastructure: `app/config.py` (settings) · `app/resources.py` (capacity) ·
`app/celery_app.py` (queues) · `app/storage.py` (files) · `app/logging_config.py`
(JSON logs) · `app/media/device_manager.py` (CPU/MPS/CUDA policy).
