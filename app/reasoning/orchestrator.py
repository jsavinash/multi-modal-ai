"""Phase 4 orchestrator: query + retrieved items -> compiled structured output."""

from __future__ import annotations

import logging
from pathlib import Path

from app.config import settings
from app.reasoning.backends import route_query
from app.reasoning.context_manager import (
    ContextItem,
    ContextOverflowError,
    Modality,
    enforce_byte_guardrail,
    prune_context,
)
from app.reasoning.output_compiler import compile_output
from app.reasoning.prompt_assembler import assemble_prompt

logger = logging.getLogger(__name__)


def run_reasoning(
    query: str,
    items: list[ContextItem],
    query_id: str | None = None,
) -> dict:
    """Full Phase 4 pipeline: footprint -> prune -> assemble -> route -> compile."""
    # Decide the effective backend early so the pruning budget matches it.
    backend = settings.reasoning_backend
    effective = "local" if backend == "local" else "cloud"

    kept, pruning_report = prune_context(items, effective)
    context_bytes = enforce_byte_guardrail(kept)

    system, user = assemble_prompt(query, kept)
    prompt_bytes = len(system.encode("utf-8")) + len(user.encode("utf-8"))
    if prompt_bytes > settings.max_context_bytes:
        raise ContextOverflowError(
            f"Assembled prompt is {prompt_bytes} bytes; hard cap is "
            f"{settings.max_context_bytes}."
        )

    answer_text, backend_used = route_query(system, user, query, kept)

    pruning = {
        "input_items": pruning_report.input_items,
        "kept_items": pruning_report.kept_items,
        "tokens_before": pruning_report.tokens_before,
        "tokens_after": pruning_report.tokens_after,
        "budget": pruning_report.budget,
        "dropped": len(pruning_report.dropped),
    }
    result = compile_output(
        query=query, backend_used=backend_used, answer_text=answer_text,
        pruning=pruning, context_bytes=context_bytes + prompt_bytes,
        query_id=query_id,
    )
    logger.info("Reasoning complete", extra={
        "query_id": result.query_id, "backend_used": backend_used,
        "actions": len(result.actions),
    })
    return result.model_dump()


def load_context_items(path: Path) -> list[ContextItem]:
    """Load ContextItems from a JSON retrieval bundle (produced by Phase 1-3 stores)."""
    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    raw_items = data.get("items", data if isinstance(data, list) else [])
    items: list[ContextItem] = []
    for raw in raw_items:
        items.append(ContextItem(
            modality=Modality(raw.get("modality", "text_chunk")),
            text=raw.get("text", ""),
            similarity_score=float(raw.get("similarity_score", 0.0)),
            source_document_id=raw.get("document_id", raw.get("source_document_id", "")),
            timestamp=raw.get("timestamp"),
            image_path=raw.get("image_path"),
        ))
    return items
