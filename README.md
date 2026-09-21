# Multimodal AI Ingestion Engine — Phase 1 & 2

Document decomposition, structural text ingestion (Phase 1) and the Vision/Audio/OCR processing node (Phase 2).

- **Phase 1 API**: FastAPI (`/api/v1/documents/ingest`) with magic-byte validation (PDF/DOCX/TXT/MD), strict 100MB ceiling, streamed persistence.
- **Phase 2 API**: FastAPI (`/api/v1/media/ingest`) for images (JPEG/PNG/WebP/BMP), video (MP4/AVI/MKV), audio (WAV/MP3) with the same magic-byte + ceiling guardrails.
- **Queue**: Celery + Redis; Phase 1 on the `ingestion` queue (concurrency 4), Phase 2 on a dedicated `media` queue (concurrency 2).
- **PDF**: `pdfminer.six` layout analysis + `pypdf` image extraction, page-by-page streaming.
- **Vision (P2)**: strict 512px longest-edge bilinear downsample guardrail; EasyOCR routing (lazy, optional — degrades gracefully with an `ocr_unavailable` note).
- **Video (P2)**: OpenCV, strict 1 frame/second sampling; sampled frames feed the OCR router.
- **Audio (P2)**: 16kHz mono resample (numpy linear interpolation; native WAV parsing, ffmpeg fallback with a 120s timeout), strict 30-second windows, faster-whisper `int8` transcription with start/end timestamps.
- **Phase 3**: Polars tabular engine (`/api/v1/tabular/ingest`, CSV/XLSX/JSON) with Semantic Matrix Serialization (rows → declarative sentences), min-max telemetry scaling (GPS-aware), MiniLM 384-dim embeddings, and a pooled Qdrant store (retries + backoff, 64-vector batch upserts, per-modality collections, schema validation, offline spooling).
- **Phase 4**: Reasoning core (`/api/v1/reason/query`) — context-window manager (8k local / 32k cloud budgets), similarity-ranked context pruning with modality reservation, 2MB hard payload cap, tag-interleaved multimodal prompts, cloud-hybrid routing (OpenAI/Anthropic, opt-in local GGUF, deterministic fallback) behind a strict circuit breaker, and a structured output compiler (Pydantic `ActionBlock`s + downloadable Markdown reports).
- **Payloads**: Pydantic-validated `CanonicalDocument` (P1), `UnifiedMediaPayload` (P2), Qdrant index payloads (P3), and `ReasoningResult` (P4).
- **Ops**: JSON logging via `logging`, no print statements; guardrails auto-clamped to detected host RAM (`app/resources.py`); `try...finally` device-cache cleanup around all inference; explicit timeouts (ffmpeg 120s, per-item 300s, cloud 60s).

Specs & docs: [System Index](docs/INDEX.md) · [`docs/PHASE1_SPEC.md`](docs/PHASE1_SPEC.md) · [`docs/PHASE1_DOCUMENTATION.md`](docs/PHASE1_DOCUMENTATION.md) · [`docs/PHASE2_SPEC.md`](docs/PHASE2_SPEC.md) · [`docs/PHASE2_DOCUMENTATION.md`](docs/PHASE2_DOCUMENTATION.md) · [`docs/PHASE3_SPEC.md`](docs/PHASE3_SPEC.md) · [`docs/PHASE3_DOCUMENTATION.md`](docs/PHASE3_DOCUMENTATION.md) · [`docs/PHASE4_SPEC.md`](docs/PHASE4_SPEC.md) · [`docs/PHASE4_DOCUMENTATION.md`](docs/PHASE4_DOCUMENTATION.md)

## Run locally

Sized for the current development host: **8GB RAM / 8-core Apple M1 / ~348GB free disk**.
The worker clamps its memory guardrails to the detected host at startup
(`app/resources.py`), so the same image deploys safely on larger machines.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
docker compose up -d redis
uvicorn app.main:app --reload                      # API
celery -A app.celery_app.celery_app worker -l INFO # worker
curl -F "file=@sample.pdf" localhost:8000/api/v1/documents/ingest
```

Or full stack: `docker compose up --build` (worker container capped at 4GB, API at 1GB).

## Tests

```bash
pytest -q
```
