# Multimodal AI Ingestion Engine — System Index

> **New here? Start with [`EXPLAINER.md`](EXPLAINER.md)** — the whole project in plain
> English, no jargon.

Cross-phase architecture map for the three implemented phases. Detailed
specifications and technical docs live in the per-phase files below.

## Phase Overview

| Phase | Scope | API | Celery Queue | Key Docs |
|---|---|---|---|---|
| **1** | Document decomposition & structural text ingestion (PDF/DOCX/TXT/MD) | `POST /api/v1/documents/ingest` | `ingestion` (conc. 2) | [SPEC](PHASE1_SPEC.md) · [DOC](PHASE1_DOCUMENTATION.md) |
| **2** | Vision, audio & OCR processing node (image/video/audio) | `POST /api/v1/media/ingest` | `media` (conc. 2) | [SPEC](PHASE2_SPEC.md) · [DOC](PHASE2_DOCUMENTATION.md) |
| **3** | Telemetry, tabular parser & vector DB indexer (CSV/XLSX/JSON) | `POST /api/v1/tabular/ingest` | `index` (conc. 1) | [SPEC](PHASE3_SPEC.md) · [DOC](PHASE3_DOCUMENTATION.md) |
| **4** | Central reasoning core & output optimization | `POST /api/v1/reason/query` | `reasoning` (conc. 1) | [SPEC](PHASE4_SPEC.md) · [DOC](PHASE4_DOCUMENTATION.md) |

## End-to-End Data Flow

```
                 ┌──────────────────────┐
  documents ────►│ Phase 1: ingestion   │──► CanonicalDocument (payloads + media refs)
  media     ────►│ Phase 2: vision/audio│──► UnifiedMediaPayload (OCR text, transcripts)
  tables    ────►│ Phase 3: tabular     │──► IndexPayload (embeddings + normalized rows)
                 └──────────┬───────────┘
                            ▼
              Qdrant (per-modality collections):
              tabular_telemetry · document_text · vision · audio
              payload indexes: document_id / modality / timestamp
                            ▼
              Phase 4: reasoning core — context pruning (8k/32k budgets),
              cloud-hybrid router w/ circuit breaker -> Markdown reports
              + validated ActionBlocks (results/reports/)
```

- **Phase 1 → 2**: `extracted_media_references[].media_id` identifies image bytes
  for OCR routing.
- **Phase 1/2/3 → Qdrant**: every write passes `validate_payload_for_collection()`
  (384-dim check + required payload fields) before batch upserts (64/batch).
- **Resilience**: Qdrant unreachable → exponential-backoff retries (5×) → validated
  offline spool (`results/spool/`); replay is a Phase 4 hook.

## Shared Infrastructure

| Concern | Implementation |
|---|---|
| Config | `app/config.py` (`INGEST_` env prefix), host-resource clamping via `app/resources.py` |
| Uploads | `app/storage.py` — streamed 1MB chunks, 100 MB ceiling, 2 GB disk floor (HTTP 507) |
| Queues | Redis broker; queues `ingestion` / `media` / `index` / `reasoning`; concurrency + child limits declared once in `app/config.py` and consumed by `docker-compose.yml` via `${INGEST_*:-default}` |
| Logging | `app/logging_config.py` — single-line JSON, no print statements |
| Device mgmt | `app/media/device_manager.py` — resolves **CUDA → MPS → CPU** (CUDA first: dedicated VRAM beats shared unified memory); `try/finally` cache cleanup. Policy `select_inference_device()` keeps CPU on hosts below `INGEST_ACCELERATOR_MIN_HOST_MEMORY_MB` (16 GB), so MPS never competes with the container budget; EasyOCR is CUDA-gated (no Metal backend) |

## Host Sizing (current development machine)

8 GB unified RAM · 8-core Apple M1 · ~346 GB free disk (measured). Container caps
sum to **6.25 GB** (`INGEST_TOTAL_CONTAINER_BUDGET_MB=6400`, ~1.75 GB headroom for
macOS + the Docker daemon):

| Container | Cap | Concurrency | Worst case | Rationale |
|---|---|---|---|---|
| `worker` (ingestion) | 2g | 2 | 2 × 1024 = 2048 MB | page-streamed PDF parsing keeps RSS low |
| `media-worker` | 1g | 2 | 2 × 384 = 768 MB | whisper `tiny` int8 + 512px frames |
| `index-worker` | 1g | 1 | 1 × 768 = 768 MB | MiniLM ~90 MB; vector math offloaded to Qdrant |
| `qdrant` | 1g | — | — | dedicated vector DB |
| `api` | 512m | — | — | I/O bound; uploads stream to disk |
| `reasoning-worker` | 512m | 1 | 1 × 512 = 512 MB | cloud I/O bound; local GGUF must not run on this host |
| `redis` | 256m | — | — | broker + result backend |

Cap, concurrency and per-child limits live **once** in `app/config.py`; compose only
consumes them (`--concurrency=${INGEST_*_CONCURRENCY:-N}`), and
`tests/test_capacity.py` fails the build if the two ever drift apart.

All guardrails auto-clamp via `app/resources.py` at startup: each queue's child
limit is clamped by both its declared budget and ~50 % of detected host RAM, then
`validate_capacity_plan()` asserts `concurrency × child_limit ≤ container cap` and
that the summed caps stay inside the 6.25 GB envelope (the API lifespan and the
Celery bootstrap both call it). **Warning:** enabling the local GGUF backbone
(`INGEST_LOCAL_LLM_ENABLED=true`) on this host would violate the budget — it
requires a 16 GB+ machine. Do not raise caps without raising host RAM
proportionally.

## Revision History

| Date | Version | Change |
|---|---|---|
| 2026-09-21 | 1.0 | System index created after Phase 3 completion; consolidates cross-phase flow, shared infrastructure and host sizing. |
| 2026-09-21 | 2.0 | Phase 4 added (see [`docs/PHASE4_DOCUMENTATION.md`](PHASE4_DOCUMENTATION.md)): `reasoning` queue + worker, cloud-hybrid router, structured output compiler. End-to-end flow updated through the reasoning core. |
| 2026-09-21 | 2.1 | Resource realignment: compose caps trimmed 11g → **6.25g** (ingestion 4g→2g, media 3g→1g, index 2g→1g, api 1g→512m, reasoning 1g→512m, qdrant 1g, redis 256m); ingestion concurrency 4→2; per-worker child limits clamped; Host Sizing table made the deployment invariant. |
| 2026-09-21 | 2.3 | **Accelerator path made real + capacity-guarded.** `torch 2.14.0` installed in the dev venv (590 MB disk; MPS now genuinely available on this M1 and verified by an actual 64×64 MPS matmul) but kept **out** of `requirements.txt`/the image, where the 384 MB media child limit could not host it. `resolve_compute_device()` now prefers **CUDA → MPS → CPU** (previously MPS was checked first, so a dual-GPU host would never use its discrete card). New `DeviceManager.select_inference_device()` policy: `INGEST_EMBEDDING_DEVICE=auto` (default) allows MPS only on hosts ≥ `INGEST_ACCELERATOR_MIN_HOST_MEMORY_MB` (16 GB) because unified-memory accelerators compete with the 6.25 GB container budget — on this 8 GB host `auto` resolves to CPU, and explicit `mps` now genuinely selects MPS instead of silently degrading. Fixed a latent bug the install exposed: `OcrRouter` passed `gpu=True` for MPS, but EasyOCR has no Metal backend — now gated by `ocr_gpu_enabled()` (CUDA only). Measured: MPS is **~3–4× slower** than CPU for MiniLM-scale matmuls (≈0.039 vs ≈0.010 ms/op), justifying the CPU default. Added `tests/test_devices.py` (16 tests) and `scripts/smoke_device.py`. **Capacity-contract hardening (same session):** `worker_memory_budget_mb` 4096→**2048** (the old value exceeded the 2g ingestion cap and was never enforced — `clamp_worker_memory()` now honours the declared budget as well as host RAM); concurrency moved into `app/config.py` as the single source of truth (compose reads `${INGEST_*:-default}`) — `index` documented concurrency corrected 2→**1**; added `QueueCapacity`/`build_queue_capacity_plan`/`validate_capacity_plan` asserted at API + Celery startup; new `tests/test_capacity.py` (13 tests) guards the plan and compose/config drift; removed duplicated `_PrefixedStream` body and stale "16GB spec" warning in `app/main.py`; embedding device now explicit/configurable (`INGEST_EMBEDDING_DEVICE`); **reasoning-worker child limit pinned to 512 MB** (`INGEST_REASONING_CHILD_MEMORY_LIMIT_MB`) — it previously inherited the 1024 MB ingestion limit inside a 512m cap, so a child could be OOM-killed; stale phase docs corrected (index 2GB/2-children → 1g/1-child, reasoning 1g → 512m, PHASE4_SPEC `1g+4g+3g+2g` → the real 6.25g stack). |
