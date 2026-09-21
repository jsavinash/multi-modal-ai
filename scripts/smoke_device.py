"""Device smoke test: torch/MPS detection, capacity-aware policy, CPU-vs-accelerator bench.

Proves the accelerator path is live (not dead code behind an ImportError) and
reports measured timings so the `auto` policy can be justified on real numbers.
"""
import logging
import time

from app.config import settings
from app.embeddings import EmbeddingEngine
from app.logging_config import configure_logging
from app.media.device_manager import device_manager
from app.media.vision import ocr_gpu_enabled

configure_logging()
log = logging.getLogger("smoke")

log.info(
    "torch=%s available=%s | device=%s backend=%s",
    device_manager.torch_version or "not-installed", device_manager.torch_available,
    device_manager.device, device_manager.backend,
)
log.info(
    "policy: INGEST_EMBEDDING_DEVICE=%s -> embedding device=%s (threshold %dMB, host %dMB)",
    settings.embedding_device, EmbeddingEngine._resolve_device(),
    settings.accelerator_min_host_memory_mb, device_manager.host_memory_mb,
)
log.info("ocr: gpu_enabled=%s (EasyOCR is CUDA-only; MPS hosts stay on CPU)",
         ocr_gpu_enabled())


def _sync(torch_mod, device: str) -> None:
    if device == "mps" and hasattr(torch_mod, "mps"):
        torch_mod.mps.synchronize()
    elif device == "cuda":
        torch_mod.cuda.synchronize()


if device_manager.torch_available:
    import torch

    def bench(device: str, n: int = 200, warm: int = 20, m: int = 32, k: int = 384) -> float:
        """MiniLM-shaped matmul: 32x384 @ 384x384, warm-up then timed loop."""
        left, right = torch.randn(m, k, device=device), torch.randn(k, k, device=device)
        for _ in range(warm):
            left @ right
        _sync(torch, device)
        started = time.perf_counter()
        for _ in range(n):
            left @ right
        _sync(torch, device)
        return (time.perf_counter() - started) * 1000.0 / n

    cpu_ms = bench("cpu")
    log.info("matmul 32x384@384x384: cpu=%.4f ms/op", cpu_ms)
    if device_manager.device in ("mps", "cuda"):
        accel_ms = bench(device_manager.device)
        speedup = (cpu_ms / accel_ms) if accel_ms else 0.0
        log.info("matmul 32x384@384x384: %s=%.4f ms/op (speedup x%.2f)",
                 device_manager.device, accel_ms, speedup)
        if speedup < 1.0:
            log.info(
                "note: accelerator loses at MiniLM scale (kernel-launch bound); "
                "the auto policy keeps CPU on constrained hosts",
            )
    # Prove the accelerator computes correct values, not just that it is advertised.
    if device_manager.device == "mps":
        a = torch.randn(64, 64)
        mps_out = (a.to("mps") @ a.to("mps").T).to("cpu")
        assert torch.allclose(mps_out, a @ a.T, atol=1e-4), "MPS matmul mismatch"
        log.info("mps correctness: 64x64 matmul matches CPU within 1e-4")
else:
    log.info("torch absent: MPS/CUDA paths inactive, CPU-only inference (pip install torch)")

log.info("DEVICE SMOKE OK")
