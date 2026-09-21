# Phase 1 — Requirements Specification

> Source: original engineering prompt, preserved verbatim in intent. This is the authoritative
> statement of *what* the Phase 1 microservice must do and under which constraints.

## 1. Mission Statement

Build the **Phase 1 microservice of an enterprise-grade Multimodal AI Ingestion Engine**, focused on:

- **Document Decomposition** — breaking source documents into structured, machine-consumable units.
- **Structural Text Ingestion** — preserving layout/structural context (headers, paragraphs, footers) during text extraction.

## 2. System Capacity Constraints (Moderate Scale Guardrails)

| Constraint | Value | Enforcement Point |
|---|---|---|
| System RAM | Max **16 GB** per service worker *(deployment-spec target; dev host has 8 GB — guardrails auto-clamp to detected RAM at startup)* | `app/resources.py` probe + Celery `worker_max_memory_per_child` (1 GB/child, 4 GB budget) + child recycling (`worker_max_tasks_per_child=20`) |
| Processing rule | **Never read an entire large document into memory at once** — use stream chunking | 1 MB read windows in `storage.py`, `docx_text_parser.py`; page-by-page iteration in `pdf_parser.py` |
| Per-file ceiling | Strict **100 MB** max upload | Streamed size accounting in `storage.save_upload_stream` → HTTP 413 |
| Dependency policy | Reliable open-source parsing libraries (e.g., `pypdf`, `python-docx`, `openpyxl`) | `requirements.txt`; `pdfminer.six` added for layout coordinates |

## 3. Functional Requirements

### FR-1: Ingestion File Router
- FastAPI endpoint accepting **multipart form** file uploads.
- Supported formats: **PDF, TXT, DOCX, MD**.
- Format validated by inspecting **magic bytes / MIME types** — the client-supplied `Content-Type` header is never trusted.

### FR-2: Async Task Delegation
- **Celery** task queue backed by **Redis**.
- On acceptance, the API passes the **stored file path** to an asynchronous Celery worker; parsing happens entirely in the background (API responds `202 Accepted` immediately).

### FR-3: Layout-Aware PDF Parser
- Extract raw text characters **page-by-page**.
- Maintain structural context by tagging: **headers**, **standard paragraphs**, and **footer boundaries**.
- Extract **embedded image frames** as separate byte strings **along with their spatial page coordinates**.

### FR-4: Text Chunking & Normalization
- Segment extracted text with a **recursive character text splitter**.
- Target chunk size: **512 tokens**, overlap factor: **10%**.

### FR-5: Standardized Canonical Output
The parser must emit a strictly **Pydantic-validated** structure:

```json
{
  "document_id": "str",
  "metadata": {
    "filename": "str",
    "file_size_bytes": "int",
    "total_pages": "int"
  },
  "payloads": [
    {
      "chunk_id": "int",
      "page_number": "int",
      "modality": "text",
      "content": "str",
      "structural_tag": "str"
    }
  ],
  "extracted_media_references": [
    {
      "media_id": "str",
      "page_number": "int",
      "spatial_bounding_box": ["float", "float", "float", "float"],
      "modality": "raw_image"
    }
  ]
}
```

## 4. Coding Standards

- Production-ready, clean, **type-hinted Python 3.10+**.
- Robust **error handling** for: corrupted headers, missing text layers, structural anomalies.
- **No `print` statements** — Python native `logging` module, configured for **JSON logging**.

## 5. Acceptance Criteria

| # | Criterion | Verified By |
|---|---|---|
| AC-1 | Uploads of PDF/DOCX/TXT/MD are accepted; binary/corrupt files rejected with 415 | `test_ingest_*`, `tests/test_ingestion.py` |
| AC-2 | Files > 100 MB are rejected before disk pressure occurs | `save_upload_stream` size guard |
| AC-3 | Parsing runs async; API never blocks on parsing | Celery `delay()` dispatch, 202 response |
| AC-4 | PDF headers/footers/paragraphs are structurally tagged from layout geometry | `scripts/smoke_pdf.py` (tags: `header`, `paragraph`, `footer`) |
| AC-5 | Chunks respect 512-token target with 10% overlap | `test_chunk_large_text_target_size_and_overlap` |
| AC-6 | Output conforms to `CanonicalDocument` schema, serialized as JSON | `test_canonical_schema_*` |
| AC-7 | All operational logs are structured JSON, no print statements | `JsonFormatter`; `grep` sweep in CI |

## 6. Out of Scope (Deferred to Later Phases)

- OCR for scanned/image-only pages (flagged via `missing_text_layer` tag in Phase 1).
- Persistence of raw image bytes to object storage (Phase 1 returns `media_id` references only).
- XLSX ingestion (library reserved in requirements; parser deferred).
- Embedding generation, vector indexing, and downstream retrieval.

---

## Revision History

| Date | Version | Change |
|---|---|---|
| 2026-09-21 | 1.0 | Baseline specification from engineering prompt. |
| 2026-09-21 | 1.1 | Constraint clarified: 16 GB is the deployment-spec target; guardrails auto-clamp to detected host RAM at runtime (`app/resources.py`). Development host: 8 GB RAM / 8-core Apple M1. Added disk safety floor (2 GB, HTTP 507) as an additional guardrail. |
