"""Multimodal Prompt Assembler: interleaves image refs, text, and structured records."""

from __future__ import annotations

import logging

from app.reasoning.context_manager import ContextItem, Modality

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are the reasoning core of a multimodal ingestion engine. You receive "
    "retrieved context blocks from three modalities: document text chunks, OCR "
    "extractions from images, and serialized telemetry records. Answer the user "
    "query using ONLY the provided context. Cite blocks by their [tag]. If the "
    "context is insufficient, say so explicitly."
)

_DELIMITERS = {
    Modality.TEXT_CHUNK: "DOCUMENT TEXT",
    Modality.OCR_BLOCK: "OCR EXTRACTION",
    Modality.TELEMETRY: "TELEMETRY RECORD",
    Modality.IMAGE_REF: "IMAGE REFERENCE",
}


def assemble_prompt(query: str, items: list[ContextItem]) -> tuple[str, str]:
    """Build (system_prompt, user_prompt) with clearly delimited, interleaved blocks.

    Ordering: image references first, then text chunks / OCR / telemetry
    interleaved in descending similarity order. Every block carries a
    `[tag: modality#n]` reference so the backbone can cite its sources.
    """
    counters: dict[Modality, int] = {}
    sections: list[str] = []

    ordered = sorted(
        items,
        key=lambda it: (0 if it.modality is Modality.IMAGE_REF else 1, -it.similarity_score),
    )
    for item in ordered:
        counters[item.modality] = counters.get(item.modality, 0) + 1
        tag = f"{item.modality.value}#{counters[item.modality]}"
        if item.modality is Modality.IMAGE_REF and item.image_path:
            body = f"<<image: {item.image_path}>>"
        else:
            body = item.text
        meta = f" (source: {item.source_document_id}" + (
            f", ts: {item.timestamp})" if item.timestamp else ")"
        )
        label = _DELIMITERS[item.modality]
        sections.append(
            f"--- [BEGIN {label} | {tag}]{meta}\n"
            f"{body}\n"
            f"--- [END {label} | {tag}]"
        )

    user_prompt = (
        f"## QUERY\n{query}\n\n"
        f"## CONTEXT ({len(sections)} blocks)\n" + "\n\n".join(sections) + "\n\n"
        "## INSTRUCTIONS\n"
        "1. Answer the query grounded in the tagged context blocks.\n"
        "2. Cite blocks inline as [tag].\n"
        "3. End with a line '### ACTIONS' followed by zero or more JSON lines "
        "of the form {\"action\": str, \"target\": str|null, \"parameters\": {}}."
    )
    logger.info("Prompt assembled", extra={
        "blocks": len(sections),
        "modalities": sorted(m.value for m in counters),
    })
    return SYSTEM_PROMPT, user_prompt
