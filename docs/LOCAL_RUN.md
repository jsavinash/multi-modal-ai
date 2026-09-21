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

```bash
# Health
curl localhost:8000/health

# Document ingestion
curl -F "file=@sample.pdf" localhost:8000/api/v1/documents/ingest
# -> {"task_id": "...", "status": "queued", "filename": "sample.pdf", ...}

# Media ingestion
curl -F "file=@photo.jpg" localhost:8000/api/v1/media/ingest

# Tabular ingestion
curl -F "file=@telemetry.csv" localhost:8000/api/v1/tabular/ingest
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
| API returns `503 Task queue unavailable` | Redis isn't reachable — `docker compose up -d redis`, check `INGEST_REDIS_URL`. |
| Tasks stay `queued` forever | No worker on the target queue — start the matching `celery ... --queues=<queue>` worker. |
| `415 Unsupported Media Type` | Magic bytes didn't match an accepted format; renamed files are rejected by design. |
| `413 Request Entity Too Large` | File exceeds `INGEST_MAX_FILE_SIZE_BYTES` (100MB default). |
| OCR notes `ocr_unavailable` | EasyOCR is optional and lazily imported — `pip install easyocr` if you need it. |
| Embeddings fall back to hashes | `sentence-transformers` is optional — `pip install sentence-transformers` for MiniLM. |
| Qdrant unavailable during indexing | The vector store spools offline and retries; ensure `docker compose up -d qdrant` runs and `INGEST_QDRANT_URL` matches. |
| Startup warning "Container caps exceed detected host RAM" | Host RAM is below the 6.25GB container budget; lower `INGEST_*_CONCURRENCY` / child limits via `.env`. |

```

Task results are written to the result directory (`INGEST_RESULT_DIR`, default
`/tmp/multimodal-ingestion/results`; `/data/results` inside containers). Uploaded
files land in `INGEST_UPLOAD_DIR`.
