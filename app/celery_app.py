"""Celery application wired to Redis with memory-safety limits clamped to host resources."""

import logging

from celery import Celery
from kombu import Queue

from app.config import settings
from app.resources import (
    clamp_worker_memory,
    detect_system_resources,
    validate_capacity_plan,
)

logger = logging.getLogger(__name__)

_resources = detect_system_resources()
# Child limit clamped by BOTH the declared worker budget and the detected host RAM.
_effective_child_limit_mb = clamp_worker_memory(
    settings.worker_child_memory_limit_mb, _resources, settings.worker_memory_budget_mb
)

celery_app = Celery(
    "multimodal_ingestion",
    broker=settings.redis_url,
    backend=settings.redis_url,
)

celery_app.conf.update(
    task_queues=(
        Queue(settings.celery_task_queue),
        Queue(settings.media_task_queue),
        Queue(settings.index_task_queue),
        Queue(settings.reasoning_task_queue),
    ),
    task_default_queue=settings.celery_task_queue,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_time_limit=settings.celery_task_time_limit_s,
    task_soft_time_limit=settings.celery_task_soft_time_limit_s,
    # Memory guardrails, clamped to the detected host RAM.
    worker_max_tasks_per_child=settings.worker_max_tasks_per_child,
    worker_max_memory_per_child=_effective_child_limit_mb * 1024,  # kB units
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    result_expires=86400,
)

logger.info(
    "Celery worker memory guardrails configured",
    extra={
        "host_total_memory_mb": _resources.total_memory_mb,
        "worker_memory_budget_mb": settings.worker_memory_budget_mb,
        "effective_child_memory_limit_mb": _effective_child_limit_mb,
    },
)

# Assert the deployment invariant for every queue on the detected host.
validate_capacity_plan(settings, _resources)
