"""Phase 4 Celery task: async reasoning on the dedicated 'reasoning' queue."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from app.celery_app import celery_app
from app.config import settings
from app.reasoning.context_manager import ContextOverflowError
from app.reasoning.orchestrator import load_context_items, run_reasoning
from app.storage import cleanup, save_result

logger = logging.getLogger(__name__)


class ReasoningTaskFailure(RuntimeError):
    """Raised when the reasoning pipeline fails; surfaced as task failure."""


@celery_app.task(
    name="reasoning.query",
    bind=True,
    max_retries=1,
    default_retry_delay=10,
    autoretry_for=(OSError,),
    queue=settings.reasoning_task_queue,
    soft_time_limit=300,
    time_limit=360,
)
def reasoning_task(
    self, context_bundle_path: str, query: str, query_id: str | None = None
) -> dict:
    """Run the reasoning core over a stored retrieval context bundle."""
    path = Path(context_bundle_path)
    if not path.exists():
        raise ReasoningTaskFailure(f"Context bundle not found: {context_bundle_path}")
    try:
        items = load_context_items(path)
        result = run_reasoning(query, items, query_id)
    except ContextOverflowError as exc:
        logger.error("Context guardrail tripped", extra={"reason": str(exc)})
        raise ReasoningTaskFailure(str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.error("Reasoning task failed", extra={"error": str(exc)})
        raise ReasoningTaskFailure(str(exc)) from exc
    finally:
        cleanup(path)

    save_result(f"reasoning_{result['query_id']}", json.dumps(result))
    logger.info("Reasoning task complete", extra={
        "query_id": result["query_id"], "backend_used": result["backend_used"],
    })
    return result
