"""Device abstraction + cache cleanup for the shared-compute guardrail.

Original prompt assumed an enterprise GPU; actual host is an Apple M1 with 8GB
unified RAM and no discrete GPU. We run CPU-first inference with optional Apple
MPS acceleration when PyTorch is installed, and generalize "VRAM cleanup" to
device-cache cleanup inside try...finally blocks.
"""

from __future__ import annotations

import gc
import logging
from contextlib import contextmanager
from typing import Iterator

logger = logging.getLogger(__name__)


class DeviceManager:
    """Resolves the best available compute device and provides cleanup hooks."""

    def __init__(self) -> None:
        self.device: str = "cpu"
        self.backend: str = "cpu"
        self._torch = None
        try:  # optional dependency - graceful degradation
            import torch  # type: ignore

            self._torch = torch
            if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                self.device, self.backend = "mps", "torch-mps"
            elif torch.cuda.is_available():
                self.device, self.backend = "cuda", "torch-cuda"
        except ImportError:
            pass
        logger.info("Compute device resolved", extra={
            "device": self.device, "backend": self.backend,
        })

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
