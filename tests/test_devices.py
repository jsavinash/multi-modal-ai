"""Device tests: torch/MPS detection, CUDA-first priority, capacity-aware selection.

`torch` is an optional dev-only dependency; these tests exercise the real one
when installed and never require it (capability-based assertions).
"""

import platform
import sys
from types import SimpleNamespace

import pytest

from app.config import settings
from app.embeddings import EmbeddingEngine
from app.media.device_manager import (
    DeviceManager,
    device_manager,
    resolve_compute_device,
)
from app.media.vision import ocr_gpu_enabled
from app.resources import SystemResources

HOST_8GB = SystemResources(total_memory_mb=8192, cpu_count=8,
                           available_disk_mb=346000, platform="darwin/arm64")
HOST_32GB = SystemResources(total_memory_mb=32768, cpu_count=10,
                            available_disk_mb=346000, platform="darwin/arm64")


class _FakeBackend:
    def __init__(self, available: bool) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available


class _FakeTorch:
    """Minimal torch surface used by resolve_compute_device()."""

    def __init__(self, cuda: bool = False, mps: bool = True, version: str = "9.9.9") -> None:
        self.cuda = _FakeBackend(cuda)
        self.backends = SimpleNamespace(mps=_FakeBackend(mps))
        self.__version__ = version


# ---------- Priority resolution ----------

def test_cuda_wins_over_mps_when_both_available():
    assert resolve_compute_device(_FakeTorch(cuda=True, mps=True)) == ("cuda", "torch-cuda")


def test_mps_used_when_cuda_absent():
    assert resolve_compute_device(_FakeTorch(cuda=False, mps=True)) == ("mps", "torch-mps")


def test_cpu_without_torch_or_accelerator():
    assert resolve_compute_device(None) == ("cpu", "cpu")
    assert resolve_compute_device(_FakeTorch(cuda=False, mps=False)) == ("cpu", "cpu")


def test_manager_handles_torch_presence_and_absence():
    mgr = DeviceManager(resources=HOST_8GB)
    if mgr.torch_available:
        assert mgr.torch_version and mgr.device in ("cpu", "mps", "cuda")
    else:
        assert mgr.device == "cpu" and mgr.torch_version is None


# ---------- Capacity-aware policy ----------

def _manager(device: str, resources: SystemResources) -> DeviceManager:
    mgr = DeviceManager(resources=resources)
    mgr.device = device  # simulate the detected device
    mgr.backend = f"torch-{device}"
    return mgr


def test_auto_policy_keeps_cpu_on_memory_constrained_host():
    # 8GB host: unified-memory accelerator shares the pool the containers use.
    assert _manager("mps", HOST_8GB).select_inference_device("auto") == "cpu"


def test_auto_policy_enables_mps_on_roomy_host():
    assert _manager("mps", HOST_32GB).select_inference_device("auto") == "mps"


def test_auto_policy_prefers_cuda_regardless_of_host_ram():
    assert _manager("cuda", HOST_8GB).select_inference_device("auto") == "cuda"


def test_explicit_cpu_always_cpu():
    assert _manager("mps", HOST_32GB).select_inference_device("cpu") == "cpu"
    assert _manager("cuda", HOST_32GB).select_inference_device("cpu") == "cpu"


def test_explicit_request_falls_back_when_device_absent():
    assert _manager("cpu", HOST_32GB).select_inference_device("mps") == "cpu"
    assert _manager("mps", HOST_32GB).select_inference_device("cuda") == "cpu"


def test_unknown_request_uses_auto_policy():
    assert _manager("cuda", HOST_8GB).select_inference_device("gpu0") == "cuda"
    assert _manager("mps", HOST_8GB).select_inference_device("") == "cpu"


def test_threshold_boundary_is_inclusive():
    exact = SystemResources(total_memory_mb=settings.accelerator_min_host_memory_mb,
                            cpu_count=8, available_disk_mb=1, platform="test")
    assert _manager("mps", exact).select_inference_device("auto") == "mps"


# ---------- Engine + OCR wiring ----------

def test_embedding_engine_delegates_to_shared_policy(monkeypatch):
    for value in ("auto", "cpu", "mps", "cuda"):
        monkeypatch.setattr("app.config.settings.embedding_device", value)
        assert EmbeddingEngine._resolve_device() == device_manager.select_inference_device(value)


def test_ocr_gpu_gate_is_cuda_only(monkeypatch):
    monkeypatch.setattr(device_manager, "device", "cuda")
    assert ocr_gpu_enabled() is True
    for dev in ("cpu", "mps"):
        monkeypatch.setattr(device_manager, "device", dev)
        assert ocr_gpu_enabled() is False


# ---------- Live accelerator checks (skip when unavailable) ----------

def test_live_mps_detection_matches_platform():
    torch = pytest.importorskip("torch")
    is_apple_silicon = sys.platform == "darwin" and platform.machine() == "arm64"
    if is_apple_silicon:
        assert torch.backends.mps.is_built() and torch.backends.mps.is_available()
        assert device_manager.device == "mps"


def test_live_mps_computes_correct_values():
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available on this host")
    a = torch.randn(64, 64)
    on_mps = (a.to("mps") @ a.to("mps").T).to("cpu")
    assert torch.allclose(on_mps, a @ a.T, atol=1e-4)


def test_live_mps_cache_release_is_safe():
    pytest.importorskip("torch")
    device_manager.release_memory()  # must never raise, whatever the device
