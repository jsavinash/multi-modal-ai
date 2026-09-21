"""Device abstraction + cache cleanup for the shared-compute guardrail.

Original prompt assumed an enterprise GPU; actual host is an Apple M1 with 8GB
unified RAM and no discrete GPU. We run CPU-first inference, enabling an
accelerator only when torch is installed *and* the host can afford it
(`select_inference_device()`): unified-memory accelerators (MPS) allocate from
the same RAM pool the container caps are carved from. "VRAM cleanup" is
generalized to device-cache cleanup inside try...finally blocks.
"""

from __future__ import annotations

import gc
import logging
from contextlib import contextmanager
from typing import Iterator

from app.config import settings
from app.resources import SystemResources, detect_system_resources

logger = logging.getLogger(__name__)


def resolve_compute_device(torch_module) -> tuple[str, str]:
    """Pick the best (device, backend) pair from an optional torch module.

    Priority is **CUDA -> MPS -> CPU**, evaluated in that order because a machine
    can expose both: a discrete GPU has its own VRAM and is always the better
    default than Apple MPS, which allocates out of the shared unified memory.
    """
    if torch_module is None:
        return "cpu", "cpu"
    cuda = getattr(torch_module, "cuda", None)
    if cuda is not None and cuda.is_available():
        return "cuda", "torch-cuda"
    mps = getattr(getattr(torch_module, "backends", None), "mps", None)
    if mps is not None and mps.is_available():
        return "mps", "torch-mps"
    return "cpu", "cpu"


class DeviceManager:
    """Resolves the best available compute device and provides cleanup hooks."""

    def __init__(self, resources: SystemResources | None = None) -> None:
        self._torch = None
        self.torch_version: str | None = None
        self._resources = resources or detect_system_resources()
        try:  # optional dependency - graceful degradation
            import torch  # type: ignore

            self._torch = torch
            self.torch_version = getattr(torch, "__version__", "unknown")
        except ImportError:
            pass
        self.device, self.backend = resolve_compute_device(self._torch)
        logger.info("Compute device resolved", extra={
            "device": self.device,
            "backend": self.backend,
            "torch_version": self.torch_version or "not-installed",
            "host_total_memory_mb": self._resources.total_memory_mb,
        })

    @property
    def torch_available(self) -> bool:
        """True when torch is importable (required for MPS/CUDA acceleration)."""
        return self._torch is not None

    @property
    def host_memory_mb(self) -> int:
        """Detected host RAM - the input to the capacity-aware device policy."""
        return self._resources.total_memory_mb

    def select_inference_device(self, requested: str = "auto") -> str:
        """Capacity-aware device policy for model inference.

        * ``cpu`` / ``mps`` / ``cuda`` - explicit request, honoured only when that
          device is actually available (otherwise CPU + a warning).
        * ``auto`` - the best available accelerator, but only when the host can
          afford it: unified-memory accelerators allocate from the same RAM pool
          that the container budget is carved from, so constrained hosts stay on
          CPU. CUDA always wins when present (dedicated VRAM).
        """
        want = (requested or "auto").strip().lower()
        if want in ("cuda", "mps"):
            if want == self.device:
                return want
            logger.warning(
                "Requested inference device unavailable; falling back to CPU",
                extra={"requested_device": want, "available_device": self.device,
                       "torch_version": self.torch_version or "not-installed"},
            )
            return "cpu"
        if want == "cpu":
            return "cpu"
        if want != "auto":
            logger.warning(
                "Unknown inference device request; using the auto policy",
                extra={"requested_device": want},
            )
        if self.device == "cuda":
            return "cuda"
        if (self.device == "mps"
                and self.host_memory_mb >= settings.accelerator_min_host_memory_mb):
            return "mps"
        if self.device != "cpu":
            logger.info(
                "Accelerator skipped by the auto policy: host RAM below the threshold",
                extra={"available_device": self.device,
                       "host_total_memory_mb": self.host_memory_mb,
                       "required_host_memory_mb": settings.accelerator_min_host_memory_mb},
            )
        return "cpu"

    def release_memory(self) -> None:
        """try/finally cleanup hook: free device caches and collect garbage."""
        if self._torch is not None:
            try:
                if self.device == "mps" and hasattr(self._torch.mps, "empty_cache"):
                    self._torch.mps.empty_cache()
                elif self.device == "cuda" and self._torch.cuda.is_available():
                    self._torch.cuda.empty_cache()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Device cache release failed", extra={"reason": str(exc)})
        gc.collect()

    @contextmanager
    def inference_session(self) -> Iterator["DeviceManager"]:
        """Context manager ensuring cleanup even when model inference raises."""
        try:
            yield self
        finally:
            self.release_memory()


device_manager = DeviceManager()
