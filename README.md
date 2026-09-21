# Multimodal AI Ingestion Engine — Phase 1 & 2

Document decomposition, structural text ingestion (Phase 1) and the Vision/Audio/OCR processing node (Phase 2).

- **Phase 1 API**: FastAPI (`/api/v1/documents/ingest`) with magic-byte validation (PDF/DOCX/TXT/MD), strict 100MB ceiling, streamed persistence.
- **Phase 2 API**: FastAPI (`/api/v1/media/ingest`) for images (JPEG/PNG/WebP/BMP), video (MP4/AVI/MKV), audio (WAV/MP3) with the same magic-byte + ceiling guardrails.
- **Queue**: Celery + Redis; four isolated queues — `ingestion` (conc. 2), `media` (conc. 2), `index` (conc. 1), `reasoning` (conc. 1) — sized to a 6.25 GB total container budget on the 8 GB host. Concurrency and per-child memory limits are declared once in `app/config.py`; `docker-compose.yml` only consumes them (`${INGEST_*:-default}`) and `app/resources.py` asserts `concurrency × child_limit ≤ container cap` at startup.
- **PDF**: `pdfminer.six` layout analysis + `pypdf` image extraction, page-by-page streaming.
- **Vision (P2)**: strict 512px longest-edge bilinear downsample guardrail; EasyOCR routing (lazy, optional — degrades gracefully with an `ocr_unavailable` note).
- **Video (P2)**: OpenCV, strict 1 frame/second sampling; sampled frames feed the OCR router.
- **Audio (P2)**: 16kHz mono resample (numpy linear interpolation; native WAV parsing, ffmpeg fallback with a 120s timeout), strict 30-second windows, faster-whisper `int8` transcription with start/end timestamps.
- **Phase 3**: Polars tabular engine (`/api/v1/tabular/ingest`, CSV/XLSX/JSON) with Semantic Matrix Serialization (rows → declarative sentences), min-max telemetry scaling (GPS-aware), MiniLM 384-dim embeddings, and a pooled Qdrant store (retries + backoff, 64-vector batch upserts, per-modality collections, schema validation, offline spooling).
- **Phase 4**: Reasoning core (`/api/v1/reason/query`) — context-window manager (8k local / 32k cloud budgets), similarity-ranked context pruning with modality reservation, 2MB hard payload cap, tag-interleaved multimodal prompts, cloud-hybrid routing (OpenAI/Anthropic, opt-in local GGUF, deterministic fallback) behind a strict circuit breaker, and a structured output compiler (Pydantic `ActionBlock`s + downloadable Markdown reports).
- **Payloads**: Pydantic-validated `CanonicalDocument` (P1), `UnifiedMediaPayload` (P2), Qdrant index payloads (P3), and `ReasoningResult` (P4).
- **Ops**: JSON logging via `logging`, no print statements; guardrails auto-clamped to detected host RAM (`app/resources.py`); `try...finally` device-cache cleanup around all inference; explicit timeouts (ffmpeg 120s, per-item 300s, cloud 60s).
- **Accelerators (optional)**: `pip install torch` enables the MPS/CUDA path. Torch is **not** in the container image — it needs ~590 MB disk and its runtime overhead exceeds the 384 MB media child limit — so containers stay CPU-only by design. Device selection is `INGEST_EMBEDDING_DEVICE=auto|cpu|mps|cuda`: `auto` prefers CUDA, allows MPS only on hosts ≥ `INGEST_ACCELERATOR_MIN_HOST_MEMORY_MB` (16 GB), else CPU. Measured here: MPS is **~3–4× slower** than CPU for MiniLM-scale matmuls (≈0.039 vs ≈0.010 ms/op, kernel-launch bound), so `auto` correctly keeps CPU on this 8 GB host. Run `scripts/smoke_device.py` to re-measure on your machine.

Specs & docs: [Plain-English explainer](docs/EXPLAINER.md) · [HLD & execution flows](docs/HLD.md) · [System Index](docs/INDEX.md) · [`docs/PHASE1_SPEC.md`](docs/PHASE1_SPEC.md) · [`docs/PHASE1_DOCUMENTATION.md`](docs/PHASE1_DOCUMENTATION.md) · [`docs/PHASE2_SPEC.md`](docs/PHASE2_SPEC.md) · [`docs/PHASE2_DOCUMENTATION.md`](docs/PHASE2_DOCUMENTATION.md) · [`docs/PHASE3_SPEC.md`](docs/PHASE3_SPEC.md) · [`docs/PHASE3_DOCUMENTATION.md`](docs/PHASE3_DOCUMENTATION.md) · [`docs/PHASE4_SPEC.md`](docs/PHASE4_SPEC.md) · [`docs/PHASE4_DOCUMENTATION.md`](docs/PHASE4_DOCUMENTATION.md)

## Run locally

Sized for the current development host: **8GB RAM / 8-core Apple M1 / ~346GB free disk**.
Container caps sum to **6.25GB** (~1.75GB headroom for macOS + Docker); the worker
clamps its memory guardrails to the detected host at startup (`app/resources.py`) and
validates the whole queue plan (`concurrency × child_limit ≤ container cap`), so
the same image deploys safely on larger machines.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install torch                                  # optional: enables MPS/CUDA (dev only)
docker compose up -d redis
uvicorn app.main:app --reload                      # API
celery -A app.celery_app.celery_app worker -l INFO --queues=ingestion --concurrency=2
curl -F "file=@sample.pdf" localhost:8000/api/v1/documents/ingest
```

Full stack: `docker compose up --build` — worker 2g, media/index/qdrant 1g each,
api/reasoning 512m, redis 256m (see `docker-compose.yml` and `docs/INDEX.md`).

## Tests

```bash
pytest -q
```

`tests/test_capacity.py` encodes the capacity contract: budget-aware clamping, the
per-queue memory plan, and a drift guard that fails the build if `docker-compose.yml`
concurrency/child-limit/`mem_limit` values ever diverge from `app/config.py`.
`tests/test_devices.py` covers the accelerator policy (CUDA-first priority, the
16 GB MPS threshold, explicit-request fallback, the CUDA-only OCR gate) and includes
live MPS checks that skip automatically when torch is absent.

```bash
PYTHONPATH=. python scripts/smoke_device.py   # torch/MPS detection + CPU-vs-MPS bench
```
