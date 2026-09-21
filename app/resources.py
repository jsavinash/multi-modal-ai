"""Runtime system-resource detection.

Guardrails must reflect the hardware the service actually runs on, not a
hard-coded target spec. This module probes total RAM, CPU count and free disk
at startup so the Celery worker can clamp its memory limits accordingly.
Supported probes: macOS (sysctl) and Linux (/proc/meminfo).
"""

import logging
import os
import subprocess
import sys
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_MB = 1024 * 1024


@dataclass(frozen=True)
class SystemResources:
    """Snapshot of the host resources available to this service."""

    total_memory_mb: int
    cpu_count: int
    available_disk_mb: int
    platform: str

    @property
    def safe_worker_budget_mb(self) -> int:
        """Recommended total worker memory: ~50% of host RAM, min 512MB."""
        return max(512, int(self.total_memory_mb * 0.5))


def _probe_total_memory_mb() -> int:
    try:
        if sys.platform == "darwin":
            out = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, timeout=5, check=True,
            )
            return int(out.stdout.strip()) // _MB
        # Linux: /proc/meminfo MemTotal
        with open("/proc/meminfo", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) // 1024  # kB -> MB
    except Exception as exc:  # noqa: BLE001
        logger.warning("Memory probe failed; using conservative fallback", extra={"reason": str(exc)})
    # Fallback heuristic when the platform probe is unavailable.
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = os.sysconf("SC_PHYS_PAGES")
        return (page_size * page_count) // _MB
    except (ValueError, OSError, AttributeError):
        return 4096  # last-resort assumption


def detect_system_resources() -> SystemResources:
    cpu = os.cpu_count() or 1
    try:
        usage = os.statvfs("/")
        disk_mb = (usage.f_bavail * usage.f_frsize) // _MB
    except OSError:
        disk_mb = 0
    resources = SystemResources(
        total_memory_mb=_probe_total_memory_mb(),
        cpu_count=cpu,
        available_disk_mb=disk_mb,
        platform=f"{sys.platform}/{os.uname().machine}",
    )
    logger.info(
        "System resources detected",
        extra={
            "total_memory_mb": resources.total_memory_mb,
            "cpu_count": resources.cpu_count,
            "available_disk_mb": resources.available_disk_mb,
            "platform": resources.platform,
        },
    )
    return resources


def clamp_worker_memory(
    configured_mb: int, resources: SystemResources, budget_mb: int | None = None
) -> int:
    """Clamp a configured worker memory limit to what the host can actually provide.

    Two ceilings apply, and the lower one wins:

    * ``budget_mb`` - the operator-declared budget (e.g. the target queue's
      container cap). This makes the configured deployment invariant
      authoritative instead of decorative.
    * ``resources.safe_worker_budget_mb`` - ~50% of detected host RAM, so the
      same image degrades safely on smaller machines.
    """
    host_cap = max(256, resources.safe_worker_budget_mb)
    effective_budget = host_cap if budget_mb is None else max(256, min(budget_mb, host_cap))
    if configured_mb > effective_budget:
        logger.warning(
            "Configured worker memory exceeds the effective budget; clamping",
            extra={
                "configured_mb": configured_mb,
                "clamped_mb": effective_budget,
                "declared_budget_mb": budget_mb,
                "host_safe_budget_mb": host_cap,
                "host_total_memory_mb": resources.total_memory_mb,
            },
        )
        return effective_budget
    return configured_mb


@dataclass(frozen=True)
class QueueCapacity:
    """Worst-case memory envelope for one queue's worker container.

    Celery runs ``concurrency`` child processes, each bounded by its own
    ``--max-memory-per-child``, so the container cap must cover all of them:
    the invariant is ``concurrency * child_limit_mb <= container_cap_mb``.
    """

    name: str
    concurrency: int
    child_limit_mb: int
    container_cap_mb: int

    @property
    def worst_case_mb(self) -> int:
        return self.concurrency * self.child_limit_mb

    @property
    def headroom_mb(self) -> int:
        return self.container_cap_mb - self.worst_case_mb

    @property
    def fits(self) -> bool:
        return self.worst_case_mb <= self.container_cap_mb


def build_queue_capacity_plan(settings, resources: SystemResources) -> list[QueueCapacity]:
    """Derive every queue's memory envelope from configured concurrency + child limits.

    Child limits are clamped to the host first, so the plan reflects what will
    actually run rather than what was merely configured.
    """
    ingestion_child = clamp_worker_memory(
        settings.worker_child_memory_limit_mb, resources, settings.worker_memory_budget_mb
    )
    media_child = clamp_worker_memory(
        settings.media_child_memory_limit_mb, resources, settings.media_container_cap_mb
    )
    index_child = clamp_worker_memory(
        settings.index_child_memory_limit_mb, resources, settings.index_container_cap_mb
    )
    reasoning_child = clamp_worker_memory(
        settings.reasoning_child_memory_limit_mb, resources, settings.reasoning_container_cap_mb
    )
    return [
        QueueCapacity(
            name=settings.celery_task_queue,
            concurrency=settings.ingestion_concurrency,
            child_limit_mb=ingestion_child,
            container_cap_mb=settings.ingestion_container_cap_mb,
        ),
        QueueCapacity(
            name=settings.media_task_queue,
            concurrency=settings.media_concurrency,
            child_limit_mb=media_child,
            container_cap_mb=settings.media_container_cap_mb,
        ),
        QueueCapacity(
            name=settings.index_task_queue,
            concurrency=settings.index_concurrency,
            child_limit_mb=index_child,
            container_cap_mb=settings.index_container_cap_mb,
        ),
        QueueCapacity(
            name=settings.reasoning_task_queue,
            concurrency=settings.reasoning_concurrency,
            child_limit_mb=reasoning_child,
            container_cap_mb=settings.reasoning_container_cap_mb,
        ),
    ]


def stack_container_budget_mb(settings) -> int:
    """Sum of every container cap - the host-level envelope, not a worker budget."""
    return sum((
        settings.ingestion_container_cap_mb,
        settings.media_container_cap_mb,
        settings.index_container_cap_mb,
        settings.reasoning_container_cap_mb,
        settings.api_container_cap_mb,
        settings.qdrant_container_cap_mb,
        settings.redis_container_cap_mb,
    ))


def validate_capacity_plan(settings, resources: SystemResources) -> list[QueueCapacity]:
    """Log and return the queue plan, flagging any queue that cannot fit its container.

    Called from both the API lifespan and the Celery bootstrap so the runtime
    capacity contract is asserted once per process, on the real detected host.
    """
    plan = build_queue_capacity_plan(settings, resources)
    for queue in plan:
        extra = {
            "queue": queue.name,
            "concurrency": queue.concurrency,
            "child_limit_mb": queue.child_limit_mb,
            "worst_case_mb": queue.worst_case_mb,
            "container_cap_mb": queue.container_cap_mb,
        }
        if queue.fits:
            logger.info("Queue capacity verified", extra={**extra, "headroom_mb": queue.headroom_mb})
        else:
            logger.error("Queue capacity violation: worst case exceeds container cap", extra=extra)

    stack_total = stack_container_budget_mb(settings)
    if stack_total > settings.total_container_budget_mb:
        logger.error(
            "Container caps exceed the host stack budget",
            extra={
                "stack_total_mb": stack_total,
                "total_container_budget_mb": settings.total_container_budget_mb,
                "host_total_memory_mb": resources.total_memory_mb,
            },
        )
    else:
        logger.info(
            "Host stack budget verified",
            extra={
                "stack_total_mb": stack_total,
                "total_container_budget_mb": settings.total_container_budget_mb,
                "host_total_memory_mb": resources.total_memory_mb,
                "host_headroom_mb": resources.total_memory_mb - stack_total,
            },
        )
    return plan

