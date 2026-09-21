# Phase 4 — Central Reasoning Core & Output Optimization Layer (Specification)

> Final component: consumes retrieval results from the Phase 1–3 stores, manages the
> context window, assembles multimodal prompts, routes to a reasoning backbone, and
> compiles structured outputs.
>
> **Adapted to the actual host system** (8 GB unified RAM, 8-core Apple M1, no discrete
> GPU). Deviations from the original prompt marked **[ADAPTED]**.

## 0. System Analysis → Prompt Modifications

| Original constraint | Actual system finding | Adaptation |
|---|---|---|
| Local quantized Llama-3.2-Vision / Mistral-7B (4/8-bit via vLLM or bitsandbytes) | No discrete GPU; 8 GB unified RAM already budgeted to workers (1g+4g+3g+2g) | **[ADAPTED]** Local backbone is an **optional, lazily-loaded GGUF backend** (llama.cpp); never auto-loaded on this host. Cloud is the default reasoning path |
| Cloud-hybrid toggle | Host has HTTPS access to managed endpoints | Kept, made concrete: `INGEST_REASONING_BACKEND=auto\|local\|openai\|anthropic`; `auto` = local if explicitly enabled and context fits, else cloud |
| 8k local / 32k cloud token thresholds | — | Kept exactly; tokens estimated (chars/token heuristic) with optional tiktoken refinement |
| "Never send multi-GB payloads" | 8 GB host | **[ADAPTED]** hard byte cap on the assembled context (`INGEST_MAX_CONTEXT_BYTES`, default 2 MB) + token-budget enforcement *before* any network call |
| Circuit breaking for rate limits | — | Concrete breaker: CLOSED → OPEN after N failures → HALF_OPEN after cooldown; honors `Retry-After` on 429 |

## 1. Functional Requirements

1. **Dynamic Context-Window Manager:** accept a multimodal query plus retrieved items
   (text chunks, OCR blocks, serialized telemetry sentences); compute the token
   footprint of every item and of the combined context.
2. **Context Pruning Engine:** if the combined footprint exceeds the active budget
   (8,000 local / 32,000 cloud), rank items by **vector-database similarity score**
   (descending) and drop the lowest-relevance items until the context fits; pruning
   always keeps at least one item per modality when available.
3. **Multimodal Prompt Assembler:** interleave `<<image: path>>` references, text
   blocks, and structured records into a clearly delimited system+user prompt format.
4. **Structured Output Compiler:** parse the backbone's raw text into
   (a) a clean **Markdown report** (downloadable file) and (b) **structured JSON action
   blocks** validated by Pydantic (`ActionBlock`).

## 2. Coding Standards

- Absolute guardrails: byte-cap + token-budget checks raise `ContextOverflowError`
  before any backbone call; no raw uncompressed payload ever leaves the process.
- Cloud connections feature a strict circuit breaker (CLOSED/OPEN/HALF_OPEN) with
  Retry-After awareness and capped timeouts.
- JSON logging only; all backends lazily initialized; `try...finally` cleanup for
  local model memory (inherited `DeviceManager`).

## 3. Canonical Output

```json
{
  "query_id": "str",
  "backend_used": "openai|anthropic|local|fallback_summary",
  "pruning": {"input_items": int, "kept_items": int, "tokens_before": int, "tokens_after": int, "budget": int},
  "context_bytes": int,
  "report_path": "str (markdown file)",
  "answer_text": "str",
  "actions": [{"action": "str", "target": "str|null", "parameters": {}}]
}
```

## Revision History

| Date | Version | Change |
|---|---|---|
| 2026-09-21 | 4.0 | Original prompt adapted to host (optional local GGUF; cloud-first toggle; hard byte/token guardrails). |
