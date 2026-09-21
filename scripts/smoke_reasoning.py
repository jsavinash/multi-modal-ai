"""End-to-end Phase 4 smoke: retrieval bundle -> prune -> assemble -> fallback compile."""
import logging
import tempfile
from pathlib import Path

from app.logging_config import configure_logging

configure_logging()

from app.config import settings  # noqa: E402
from app.reasoning.context_manager import ContextItem, Modality  # noqa: E402
from app.reasoning.orchestrator import run_reasoning  # noqa: E402

items = [
    ContextItem(modality=Modality.TEXT_CHUNK,
                text="Incident report: the boiler pressure rose sharply after the "
                     "valve replacement on the east line.",
                similarity_score=0.94, source_document_id="doc-text-1"),
    ContextItem(modality=Modality.OCR_BLOCK,
                text="SCANNED MAINTENANCE LOG: valve 7 replaced on 2026-09-20",
                similarity_score=0.87, source_document_id="doc-text-2"),
    ContextItem(modality=Modality.TELEMETRY,
                text="At timestamp 2026-09-21T10:00:00, the Temp sensor recorded "
                     "a telemetry value of 23.5 units",
                similarity_score=0.86, source_document_id="doc-tele",
                timestamp="2026-09-21T10:00:00"),
    ContextItem(modality=Modality.IMAGE_REF, text="",
                similarity_score=0.70, source_document_id="doc-text-2",
                image_path="/data/img/boiler_plant_1.png"),
]

# Force the offline fallback (no API keys on this host).
settings.reasoning_backend = "openai"
settings.openai_api_key = ""
settings.anthropic_api_key = ""

result = run_reasoning("What happened to the boiler pressure?", items, query_id="smoke-q1")
logging.getLogger("smoke").info(
    "result: backend=%s pruning=%s report=%s", result["backend_used"],
    result["pruning"], result["report_path"],
)
assert result["pruning"]["kept_items"] == 4  # all items fit the 32k cloud budget
assert result["pruning"]["budget"] == settings.cloud_token_budget
assert result["context_bytes"] < settings.max_context_bytes
report = Path(result["report_path"])
assert report.exists() and "# Reasoning Report" in report.read_text(encoding="utf-8")
logging.getLogger("smoke").info("report preview: %s",
                                report.read_text(encoding="utf-8")[:220])
logging.getLogger("smoke").info("PHASE 4 SMOKE OK")
