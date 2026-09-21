# Running Locally (Developer Guide)

Step-by-step guide to run the Multimodal AI Ingestion Engine on your machine.
Defaults are sized for the current development host: **8GB RAM / 8-core Apple M1 /
~346GB free disk**. Container caps sum to **6.25GB** (~1.75GB headroom for macOS +
Docker); the worker clamps its memory guardrails to the detected host at startup
(`app/resources.py`) and validates the whole queue plan
(`concurrency × child_limit ≤ container cap`), so the same setup deploys safely on
larger machines.

There are two ways to run the stack:

1. **Hybrid (recommended for development)** — Redis (+ optionally Qdrant) in Docker,
   API and Celery workers on the host with live-reload.
2. **Full Docker stack** — everything in containers via `docker compose up --build`.

---

## 1. Prerequisites

| Requirement | Version / Notes |
|---|---|
| Python | 3.12 (the container image uses `python:3.12-slim`) |
| Docker + Docker Compose | for Redis, Qdrant, and the full-stack mode |
| ffmpeg | optional — audio resample fallback for non-WAV inputs (native WAV parsing works without it) |

## 2. Setup

```bash
git clone https://github.com/jsavinash/multi-modal-ai.git
cd multi-modal-ai

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Optional (dev only): enables MPS/CUDA acceleration for MiniLM embeddings and EasyOCR.
# Torch is ~590MB on disk and is deliberately NOT in the container image (the media
# child limit is 384MB, the index cap is 1g). On Apple Silicon this installs MPS builds.
pip install torch
```

## 3. Environment variables

Copy the template and fill in what you need — `.env` is gitignored, never commit real keys:

```bash
cp .env.example .env
```

Key variables (all prefixed `INGEST_`):

| Variable | Default | Purpose |
|---|---|---|
| `INGEST_REDIS_URL` | `redis://localhost:6379/0` | Celery broker + result backend |
| `INGEST_QDRANT_URL` | `http://localhost:6333` | Phase 3 vector DB |
| `INGEST_MAX_FILE_SIZE_BYTES` | `104857600` (100MB) | hard upload ceiling |
| `INGEST_EMBEDDING_DEVICE` | `auto` | `auto \| cpu \| mps \| cuda`; `auto` = CUDA > MPS (16GB+ hosts) > CPU |
| `INGEST_REASONING_BACKEND` | `auto` | `auto \| local \| openai \| anthropic` (Phase 4) |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` | empty | cloud reasoning backends (opt-in) |
| `INGEST_LOCAL_LLM_ENABLED` | `false` | opt-in local GGUF backbone (needs llama-cpp-python + a `.gguf` model) |

Note on device policy: `auto` prefers CUDA, allows MPS only on hosts ≥
`INGEST_ACCELERATOR_MIN_HOST_MEMORY_MB` (16 GB), else CPU. On the 8GB dev host,
`auto` correctly keeps CPU — MPS measures ~3–4× slower than CPU for MiniLM-scale
matmuls. Re-measure with `PYTHONPATH=. python scripts/smoke_device.py`.
## 4. Option A — Hybrid run (recommended for dev)

Start only the infrastructure containers, run the API and workers on the host:

```bash
docker compose up -d redis        # broker + result backend
docker compose up -d qdrant       # optional: only needed for Phase 3 indexing
```

Then, in separate terminals (with `.venv` activated):

```bash
# Terminal 1 — API (live reload)
uvicorn app.main:app --reload

# Terminal 2 — document ingestion worker (queue: ingestion, concurrency 2)
celery -A app.celery_app.celery_app worker -l INFO --queues=ingestion --concurrency=2

# Terminal 3 — media worker (queue: media, concurrency 2)
celery -A app.celery_app.celery_app worker -l INFO --queues=media --concurrency=2

# Terminal 4 — index worker (queue: index, concurrency 1)
celery -A app.celery_app.celery_app worker -l INFO --queues=index --concurrency=1

# Terminal 5 — reasoning worker (queue: reasoning, concurrency 1)
celery -A app.celery_app.celery_app worker -l INFO --queues=reasoning --concurrency=1
```

The API is now at http://localhost:8000 — interactive docs at `/docs`, OpenAPI at `/openapi.json`.


## 5. Option B — Full Docker stack

```bash
docker compose up --build
```

Container layout (see `docker-compose.yml`):

| Service | Queue | `mem_limit` | Notes |
|---|---|---|---|
| `api` | — | 512m | FastAPI, I/O bound; uploads stream to disk |
| `worker` | ingestion | 2g | 2 children × 1024MB child limit |
| `media-worker` | media | 1g | 2 children; whisper tiny int8 + 512px frame buffers |
| `index-worker` | index | 1g | 1 child; MiniLM ~90MB |
| `reasoning-worker` | reasoning | 512m | cloud I/O bound; local GGUF must NOT run on this host |
| `redis` | — | 256m | broker + result backend |
| `qdrant` | — | 1g | persistent volume `qdrant_storage` |

The API and all workers share the named volume `ingestion_data` mounted at
`/data` (`INGEST_UPLOAD_DIR=/data/uploads`, `INGEST_RESULT_DIR=/data/results` in
the image) — the API streams uploads there and workers read them, so the volume
must stay mounted on every service that runs tasks.

Concurrency and per-child memory limits are declared once in `app/config.py`;
`docker-compose.yml` only consumes them (`${INGEST_*:-default}`) and
`app/resources.py` asserts `concurrency × child_limit ≤ container cap` at startup.

Stop with `docker compose down` (add `-v` to also drop the Qdrant volume).

## 6. Using the API

| Endpoint | Method | Accepts |
|---|---|---|
| `/health` | GET | liveness probe |
| `/api/v1/documents/ingest` | POST | PDF / DOCX / TXT / MD (magic-byte validated, 100MB ceiling) |
| `/api/v1/media/ingest` | POST | images (JPEG/PNG/WebP/BMP), video (MP4/AVI/MKV), audio (WAV/MP3) |
| `/api/v1/tabular/ingest` | POST | CSV / XLSX / JSON |
| `/api/v1/reason/query` | POST | multimodal reasoning query (Phase 4) |

All ingest endpoints return `202 Accepted` with a `task_id` (async Celery dispatch).
Progress and results are visible in the matching worker's logs; results are also
written to the result directory (`INGEST_RESULT_DIR`, default
`/tmp/multimodal-ingestion/results` on the host; `/data/results` inside
containers). Uploaded files land in `INGEST_UPLOAD_DIR`.

```bash
# Health
curl localhost:8000/health

# Document ingestion (PDF / DOCX / TXT / MD)
curl -F "file=@sample.pdf" localhost:8000/api/v1/documents/ingest

# Media ingestion (image / video / audio)
curl -F "file=@photo.jpg" localhost:8000/api/v1/media/ingest

# Tabular ingestion (CSV / XLSX / JSON)
curl -F "file=@telemetry.csv" localhost:8000/api/v1/tabular/ingest
```

Each ingest call returns `202 Accepted` with
`{"task_id": "...", "status": "queued", "filename": ..., "file_size_bytes": ...}`.
All endpoints were verified live; the matching worker then completes the task
(check the worker's logs for `Task ... succeeded`).

Quick self-test with a generated sample CSV:

```bash
printf 'timestamp,lat,lon,temperature\n2026-01-01T00:00:00Z,12.97,77.59,24.5\n' > /tmp/sample.csv
curl -F "file=@/tmp/sample.csv" localhost:8000/api/v1/tabular/ingest
# For media use any real JPEG/PNG; for documents any real PDF/TXT/MD.
```

Note: `curl -F "file=@..."` reads the file client-side — if the path doesn't
exist, curl exits with code 26 without sending the request.

## 7. Smoke scripts

End-to-end pipeline checks without the API (run with the venv active):

```bash
PYTHONPATH=. python scripts/smoke_pdf.py          # PDF decomposition
PYTHONPATH=. python scripts/smoke_media.py        # vision / OCR / audio pipeline
PYTHONPATH=. python scripts/smoke_tabular.py      # tabular engine + embeddings + Qdrant
PYTHONPATH=. python scripts/smoke_reasoning.py    # reasoning core
PYTHONPATH=. python scripts/smoke_device.py       # torch/MPS detection + CPU-vs-MPS bench
```

## 8. Tests

```bash
pytest -q
```

- `tests/test_capacity.py` — capacity contract: budget-aware clamping, per-queue
  memory plan, and a drift guard that fails the build if `docker-compose.yml`
  values ever diverge from `app/config.py`.
- `tests/test_devices.py` — accelerator policy (CUDA-first priority, 16GB MPS
  threshold, explicit-request fallback, CUDA-only OCR gate). Live MPS checks skip
  automatically when torch is absent.

## 9. Troubleshooting

| Symptom | Fix |
|---|---|
| `curl` exits with code 26 before any request | The local file doesn't exist (`file=@sample.pdf` is read client-side). Check the path/extension. |
| API returns `503 Task queue unavailable` | Redis isn't reachable — `docker compose up -d redis`, check `INGEST_REDIS_URL`. |
| Tasks stay `queued` forever | No worker on the target queue — start the matching `celery ... --queues=<queue>` worker. |
| `415 Unsupported Media Type` | Magic bytes didn't match an accepted format; renamed files are rejected by design. |
| Worker logs `Received unregistered task of type ...` | Task modules aren't loaded — the Celery app must declare `include=["app.tasks", "app.tasks_media", "app.tasks_tabular", "app.tasks_reasoning"]` (fixed in `app/celery_app.py`; don't remove it). |
| Worker raises `Stored file not found: /data/uploads/...` (full Docker mode) | The API and workers share the `ingestion_data` named volume mounted at `/data` in `docker-compose.yml` — make sure every service mounts it. |
| Task fails with `Unsupported format ... reached the parser` | The parser dispatch expects the `SupportedFormat` enum; `app/tasks.py` passes `fmt.fmt` from the `FormatSignature`. If reworking, keep the `.fmt` unwrap. |
| `413 Request Entity Too Large` | File exceeds `INGEST_MAX_FILE_SIZE_BYTES` (100MB default). |
| OCR notes `ocr_unavailable` | EasyOCR is optional and lazily imported — `pip install easyocr` if you need it. |
| Embeddings fall back to hashes | `sentence-transformers` is optional — `pip install sentence-transformers` for MiniLM. |
| Qdrant unavailable during indexing | The vector store spools offline and retries; ensure `docker compose up -d qdrant` runs and `INGEST_QDRANT_URL` matches. |
| Startup warning "Container caps exceed detected host RAM" | Host RAM is below the 6.25GB container budget; lower `INGEST_*_CONCURRENCY` / child limits via `.env`. |

