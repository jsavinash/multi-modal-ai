# Multimodal AI Ingestion Engine — System Index

Cross-phase architecture map for the three implemented phases. Detailed
specifications and technical docs live in the per-phase files below.

## Phase Overview

| Phase | Scope | API | Celery Queue | Key Docs |
|---|---|---|---|---|
| **1** | Document decomposition & structural text ingestion (PDF/DOCX/TXT/MD) | `POST /api/v1/documents/ingest` | `ingestion` (conc. 4) | [SPEC](PHASE1_SPEC.md) · [DOC](PHASE1_DOCUMENTATION.md) |
| **2** | Vision, audio & OCR processing node (image/video/audio) | `POST /api/v1/media/ingest` | `media` (conc. 2) | [SPEC](PHASE2_SPEC.md) · [DOC](PHASE2_DOCUMENTATION.md) |
| **3** | Telemetry, tabular parser & vector DB indexer (CSV/XLSX/JSON) | `POST /api/v1/tabular/ingest` | `index` (conc. 2) | [SPEC](PHASE3_SPEC.md) · [DOC](PHASE3_DOCUMENTATION.md) |
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
| Queues | Redis broker; queues `ingestion` / `media` / `index`; per-worker compose caps (4g/3g/2g) |
| Logging | `app/logging_config.py` — single-line JSON, no print statements |
| Device mgmt | `app/media/device_manager.py` — cpu/mps/cuda resolution, `try/finally` cache cleanup |

## Host Sizing (current development machine)

8 GB unified RAM · 8-core Apple M1 · ~348 GB free disk. Total container budget:
API 1g + ingestion 4g + media 3g + index 2g + Redis/Qdrant ≈ within host RAM.
All guardrails auto-clamp via `app/resources.py` at startup.

## Revision History

| Date | Version | Change |
|---|---|---|
| 2026-09-21 | 1.0 | System index created after Phase 3 completion; consolidates cross-phase flow, shared infrastructure and host sizing. |
| 2026-09-21 | 2.0 | Phase 4 added (see [`docs/PHASE4_DOCUMENTATION.md`](PHASE4_DOCUMENTATION.md)): `reasoning` queue + worker, cloud-hybrid router, structured output compiler. End-to-end flow updated through the reasoning core. |
