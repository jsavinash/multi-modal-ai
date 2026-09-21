"""Phase 4 reasoning core: context window management & pruning."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from app.config import settings

logger = logging.getLogger(__name__)


class ContextOverflowError(ValueError):
    """Raised when a context payload would violate the hard byte/token guardrails."""


class Modality(str, Enum):
    TEXT_CHUNK = "text_chunk"      # Phase 1 payloads
    OCR_BLOCK = "ocr_block"        # Phase 2 vision payloads
    TELEMETRY = "telemetry"        # Phase 3 serialized rows
    IMAGE_REF = "image_ref"        # image path references (prompt-only)


@dataclass
class ContextItem:
    """One retrieved item destined for the reasoning context."""
    modality: Modality
    text: str
    similarity_score: float = 0.0   # vector-DB relevance, higher = keep
    source_document_id: str = ""
    timestamp: str | None = None
    image_path: str | None = None   # only for IMAGE_REF
    token_footprint: int = 0

    def compute_tokens(self) -> int:
        self.token_footprint = max(1, round(len(self.text) / settings.chars_per_token))
        return self.token_footprint


@dataclass
class PruningReport:
    input_items: int
    kept_items: int
    tokens_before: int
    tokens_after: int
    budget: int
    dropped: list[tuple[str, float]] = field(default_factory=list)  # (modality, score)


def _budget_for(backend: str) -> int:
    return settings.local_token_budget if backend == "local" else settings.cloud_token_budget


def compute_footprint(items: list[ContextItem]) -> int:
    return sum(item.compute_tokens() for item in items)


def enforce_byte_guardrail(items: list[ContextItem]) -> int:
    """Absolute guardrail: refuse payloads above the hard byte cap before any send."""
    total_bytes = sum(len(item.text.encode("utf-8")) for item in items)
    if total_bytes > settings.max_context_bytes:
        raise ContextOverflowError(
            f"Context payload is {total_bytes} bytes; hard cap is "
            f"{settings.max_context_bytes}. Refusing to send an uncompressed "
            "oversized payload to the reasoning backbone."
        )
    return total_bytes


def prune_context(items: list[ContextItem], backend: str) -> tuple[list[ContextItem], PruningReport]:
    """Prioritize by similarity score and drop low-relevance items until within budget.

    Guarantees at least one item per modality survives when the modality was present
    in the input (so no modality is silently erased).
    """
    budget = _budget_for(backend)
    tokens_before = compute_footprint(items)
    byte_total = enforce_byte_guardrail(items)

    ranked = sorted(items, key=lambda it: it.similarity_score, reverse=True)
    kept: list[ContextItem] = []
    tokens_after = 0
    dropped: list[tuple[str, float]] = []

    # Pass 1: keep highest-scoring items within budget.
    for item in ranked:
        if tokens_after + item.token_footprint <= budget:
            kept.append(item)
            tokens_after += item.token_footprint
        else:
            dropped.append((item.modality.value, item.similarity_score))

    # Pass 2: ensure modality representation (reserve one item per input modality).
    present_modalities = {it.modality for it in items}
    kept_modalities = {it.modality for it in kept}
    for modality in present_modalities - kept_modalities:
        best = max(
            (it for it in ranked if it.modality is modality),
            key=lambda it: it.similarity_score,
            default=None,
        )
        if best is not None:
            kept.append(best)
            tokens_after += best.token_footprint
            if (best.modality.value, best.similarity_score) in dropped:
                dropped.remove((best.modality.value, best.similarity_score))

    report = PruningReport(
        input_items=len(items), kept_items=len(kept),
        tokens_before=tokens_before, tokens_after=tokens_after,
        budget=budget, dropped=dropped,
    )
    logger.info("Context assembled", extra={
        "backend": backend, "input_items": len(items), "kept_items": len(kept),
        "tokens_before": tokens_before, "tokens_after": tokens_after,
        "budget": budget, "context_bytes": byte_total,
        "dropped": len(dropped),
    })
    return kept, report
