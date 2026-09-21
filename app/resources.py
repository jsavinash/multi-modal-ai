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


def clamp_worker_memory(configured_mb: int, resources: SystemResources) -> int:
    """Clamp a configured worker memory limit to what the host can actually provide."""
    cap = max(256, resources.safe_worker_budget_mb)
    if configured_mb > cap:
        logger.warning(
            "Configured worker memory exceeds host capacity; clamping",
            extra={"configured_mb": configured_mb, "clamped_mb": cap,
                   "host_total_memory_mb": resources.total_memory_mb},
        )
        return cap
    return configured_mb
