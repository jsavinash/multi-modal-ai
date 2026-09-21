# Phase 4 — Technical Documentation

Central Reasoning Core & Output Optimization Layer of the Multimodal AI Ingestion Engine.

- Runtime: Python 3.10+; cloud-first routing, optional local GGUF backbone
- Stack: httpx (cloud endpoints), pydantic (action blocks), llama.cpp (optional local)
- Spec & prompt adaptations: [`docs/PHASE4_SPEC.md`](PHASE4_SPEC.md) · system map: [`docs/INDEX.md`](INDEX.md)

---

## 1. Architecture

```
 retrieval bundles (JSON, from Phase 1-3 stores / Qdrant queries)
        │  POST /api/v1/reason/query
        ▼
 Celery queue "reasoning" ──► reasoning-worker (x1, 1g cap)
        │
        ▼
 app/reasoning/orchestrator.run_reasoning()
   1. prune_context()       token footprint -> budget (8k local / 32k cloud)
                            rank by similarity score; drop lowest; keep ≥1 per modality
   2. enforce_byte_guardrail()  HARD CAP 2MB — refuse oversized payloads
   3. assemble_prompt()     delimited, tag-interleaved multimodal prompt
   4. route_query()         cloud-hybrid router w/ circuit breaker:
                            openai -> anthropic -> (opt-in local GGUF) -> fallback
   5. compile_output()      parse ACTIONS JSON lines (Pydantic ActionBlock)
                            -> Markdown report -> results/reports/report_*.md
```

## 2. Module Reference

| Module | Lines | Responsibility |
|---|---|---|
| `app/reasoning/context_manager.py` | 122 | `ContextItem` token footprint, byte guardrail, similarity-ranked pruning w/ modality reservation |
| `app/reasoning/prompt_assembler.py` | 71 | Tagged, delimited multimodal prompt (image refs first, then text/OCR/telemetry) |
| `app/reasoning/backends.py` | 212 | `CircuitBreaker` (CLOSED/OPEN/HALF_OPEN), OpenAI + Anthropic clients, optional llama.cpp, deterministic fallback |
| `app/reasoning/output_compiler.py` | 119 | ACTIONS parsing (`ActionBlock` Pydantic), Markdown report rendering, persistence |
| `app/reasoning/orchestrator.py` | 82 | Pipeline orchestration + retrieval-bundle loading |
| `app/tasks_reasoning.py` | 55 | Celery task `reasoning.query` on the `reasoning` queue |

## 3. API Reference

### `POST /api/v1/reason/query` → `202`
Multipart `file` (JSON context bundle) + `query` string. Bundle schema:
`{"items": [{"modality": "text_chunk"|"ocr_block"|"telemetry"|"image_ref", "text", "similarity_score", "document_id", "timestamp", "image_path"}, ...]}`

| Response | Meaning |
|---|---|
| `202` | Queued; body `{task_id, status, query}` |
| `413` | Upload exceeds the 100 MB ceiling |
| `415` | Bundle is not JSON |
| `507` | Disk floor tripped |
| `503` | Broker unreachable |

## 4. Context Window & Pruning (FR-1, FR-2)

- **Token footprint**: `len(text) / chars_per_token` (4.0 heuristic) per item, summed.
- **Budgets**: 8,000 tokens when the local backbone is selected, 32,000 for cloud.
- **Pruning**: rank by `similarity_score` desc; keep items until budget; dropped
  items are reported. **Reservation pass**: at least one item per input modality is
  always retained (no silent modality erasure).
- **Absolute guardrails** (raise `ContextOverflowError` *before* any network call):
  combined raw bytes ≤ `INGEST_MAX_CONTEXT_BYTES` (2 MB) and the assembled prompt
  is re-checked after assembly.

## 5. Multimodal Prompt Assembler (FR-3)

System prompt pins the backbone to the provided context with `[tag]` citations.
User prompt sections, image references first, then similarity-ordered blocks:

```
--- [BEGIN DOCUMENT TEXT | text_chunk#1] (source: doc-1)
the boiler pressure rose sharply
--- [END DOCUMENT TEXT | text_chunk#1]
--- [BEGIN TELEMETRY RECORD | telemetry#1] (source: doc-2, ts: 2026-09-21T10:00:00)
At timestamp 2026-09-21T10:00:00, the Temp sensor recorded ...
--- [END TELEMETRY RECORD | telemetry#1]
--- [BEGIN IMAGE REFERENCE | image_ref#1] (source: doc-3)
<<image: /data/img/boiler_plant_1.png>>
--- [END IMAGE REFERENCE | image_ref#1]
```

The instructions require the backbone to end with `### ACTIONS` followed by
JSON action lines (consumed by the compiler).

## 6. Backends & Circuit Breaking (coding standard)

| Backend | Trigger | Notes |
|---|---|---|
| `openai` | key configured | GPT-4o chat completions |
| `anthropic` | key configured | Claude 3.5 Sonnet messages API |
| `local` | `INGEST_LOCAL_LLM_ENABLED=true` + `.gguf` path | llama.cpp quantized model, n_ctx = local budget |
| `fallback_summary` | all backends failed/refused | deterministic extractive answer |

- **Toggle**: `INGEST_REASONING_BACKEND=auto|local|openai|anthropic`. `auto` prefers
  local only when explicitly enabled, else openai → anthropic.
- **Circuit breaker**: CLOSED → OPEN after 3 failures
  (`INGEST_BREAKER_FAILURE_THRESHOLD`), cooldown 30 s → HALF_OPEN probe → success
  closes. HTTP 429 honors `Retry-After`; 5xx and transport errors count as failures;
  2xx resets. The pipeline is **never answerless** — the fallback summary keeps the
  system useful during total cloud outage.

## 7. Structured Output Compiler (FR-4)

- `### ACTIONS` section: each JSON line is validated into
  `ActionBlock {action: str, target: str|null, parameters: dict}`. Malformed lines
  become `action="unparsed"` entries with raw text preserved — the report is never lost.
- Markdown report: header (query, backend, pruning stats), answer body with inline
  `[tag]` citations, enumerated structured actions. Persisted to
  `results/reports/report_{query_id}.md` (downloadable).
- `ReasoningResult` (Pydantic) is the validated wire contract:
  `{query_id, backend_used, answer_text, actions, report_path, pruning, context_bytes}`.

## 8. Configuration (`INGEST_` prefix)

| Variable | Default | Purpose |
|---|---|---|
| `INGEST_REASONING_TASK_QUEUE` | `reasoning` | Dedicated queue |
| `INGEST_REASONING_BACKEND` | `auto` | `auto\|local\|openai\|anthropic` |
| `INGEST_LOCAL_TOKEN_BUDGET` | 8000 | Local context threshold |
| `INGEST_CLOUD_TOKEN_BUDGET` | 32000 | Cloud context threshold |
| `INGEST_MAX_CONTEXT_BYTES` | 2097152 (2 MB) | Hard uncompressed-payload cap |
| `INGEST_LOCAL_LLM_ENABLED` | false | Opt-in local GGUF backbone |
| `INGEST_LOCAL_LLM_MODEL_PATH` | "" | Path to quantized model |
| `INGEST_OPENAI_API_KEY` / `INGEST_ANTHROPIC_API_KEY` | "" | Cloud credentials |
| `INGEST_REASONING_CLOUD_MODEL` | `gpt-4o` | Cloud model id |
| `INGEST_BREAKER_FAILURE_THRESHOLD` | 3 | Breaker trips after N failures |
| `INGEST_BREAKER_COOLDOWN_S` | 30.0 | OPEN → HALF_OPEN cooldown |
| `INGEST_REASONING_REQUEST_TIMEOUT_S` | 60.0 | Cloud request timeout |

## 9. Test Coverage (`tests/test_reasoning.py`)

| Test | Covers |
|---|---|
| `test_token_footprint_estimation` | 500 chars → 125 tokens; summed footprint (FR-1) |
| `test_enforce_byte_guardrail_rejects_oversized` | Hard byte cap raises `ContextOverflowError` |
| `test_prune_respects_local_budget` | 8k budget; highest scores survive; dropped accounting |
| `test_prune_cloud_budget_higher_than_local` | 32k cloud budget, no drops |
| `test_prune_keeps_every_modality_represented` | Modality reservation pass |
| `test_prompt_assembler_interleaves_and_tags` | Delimited tagged sections, `<<image:>>`, ACTIONS instructions (FR-3) |
| `test_circuit_breaker_states` | CLOSED → OPEN → HALF_OPEN → CLOSED lifecycle |
| `test_cloud_call_429_records_failure` | Rate-limit handling (429) |
| `test_cloud_call_success_resets_breaker` | 2xx resets the breaker |
| `test_route_query_falls_back_when_all_backends_fail` | Fallback summary path |
| `test_output_compiler_parses_actions_and_report` | Pydantic `ActionBlock`, unparsed fallback, Markdown report (FR-4) |
| `test_load_context_items_bundle` | Retrieval bundle loading |
| `test_run_reasoning_end_to_end` | Full pipeline offline → compiled report |

## 10. Running the Service

```bash
docker compose up --build   # adds reasoning-worker; export OPENAI_API_KEY to enable cloud

# dedicated reasoning worker (local dev)
celery -A app.celery_app.celery_app worker -l INFO --queues=reasoning --concurrency=1

curl -F "file=@context_bundle.json" \
     -F "query=What happened to the boiler?" localhost:8000/api/v1/reason/query

# validation
pytest -q                                        # 46 tests (all phases)
PYTHONPATH=. python scripts/smoke_reasoning.py   # prune -> assemble -> fallback -> report
```

## 11. Known Limitations & Post-Phase-4 Hooks

- **Token counting is a heuristic** (chars/4); swap in `tiktoken` for exact provider
  budgeting before production cloud routing.
- **Local GGUF is opt-in and untested on this host** — enable only on machines with
  adequate unified memory; llama-cpp-python is not installed by default.
- **Image paths are prompt references only** — multimodal vision payloads require a
  provider that accepts image inputs (GPT-4o vision / Claude vision) with base64
  attachment; current assembler emits path placeholders.
- **No streaming** — cloud calls are single-shot; SSE streaming is a natural upgrade.
- **Spool replay (Phase 3) & report delivery** — wire reports into a notification/
  download API in a Phase 5 hardening pass.
- **Action execution** — `ActionBlock`s are validated but not executed; downstream
  systems must subscribe to `results/` outputs.

## Revision History

| Date | Version | Change |
|---|---|---|
| 2026-09-21 | 4.0 | Initial Phase 4 implementation: context manager + pruning engine, multimodal prompt assembler, cloud-hybrid router with circuit breaker (OpenAI/Anthropic/optional local GGUF/fallback), structured output compiler (ActionBlock + Markdown reports), `reasoning` queue + worker, 13 new tests (46 total), smoke script. |
