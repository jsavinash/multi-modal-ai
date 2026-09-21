"""Capacity tests: host clamping, per-queue memory plan, and compose/config agreement.

These encode the deployment invariant for this host: 8 GB RAM / 8 cores, four
queues whose worst-case footprint (concurrency x child limit) must fit inside
their container cap, and a 6.25 GB stack envelope.
"""

import re
from pathlib import Path
from types import SimpleNamespace

from app.config import settings
from app.embeddings import EmbeddingEngine
from app.media.device_manager import device_manager
from app.resources import (
    SystemResources,
    build_queue_capacity_plan,
    clamp_worker_memory,
    stack_container_budget_mb,
    validate_capacity_plan,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

HOST_8GB = SystemResources(total_memory_mb=8192, cpu_count=8,
                           available_disk_mb=346000, platform="darwin/arm64")
HOST_16GB = SystemResources(total_memory_mb=16384, cpu_count=8,
                            available_disk_mb=346000, platform="linux/x86_64")
HOST_TINY = SystemResources(total_memory_mb=2048, cpu_count=2,
                            available_disk_mb=1000, platform="linux/x86_64")


# ---------- Budget-aware clamping ----------

def test_declared_budget_wins_over_larger_host():
    # A 16GB host must not let the ingestion worker exceed its 2g container cap.
    assert clamp_worker_memory(8192, HOST_16GB, settings.worker_memory_budget_mb) == 2048
    # Configuration already inside the budget is left untouched.
    assert clamp_worker_memory(1024, HOST_16GB, settings.worker_memory_budget_mb) == 1024


def test_host_capacity_wins_over_larger_budget():
    # 2GB host -> safe budget is 1024MB even though 4096MB was declared.
    assert clamp_worker_memory(4096, HOST_TINY, 4096) == HOST_TINY.safe_worker_budget_mb == 1024


def test_clamp_without_declared_budget_keeps_legacy_behaviour():
    assert clamp_worker_memory(16384, HOST_8GB) == HOST_8GB.safe_worker_budget_mb


def test_ingestion_budget_matches_container_cap():
    # The documented budget must be derivable from the container cap (2 children x 1GB).
    assert settings.worker_memory_budget_mb == settings.ingestion_container_cap_mb
    assert (settings.ingestion_concurrency * settings.worker_child_memory_limit_mb
            == settings.worker_memory_budget_mb)


# ---------- Per-queue plan ----------

def test_plan_covers_all_four_queues_and_fits():
    plan = validate_capacity_plan(settings, HOST_8GB)
    assert [q.name for q in plan] == ["ingestion", "media", "index", "reasoning"]
    assert all(q.fits for q in plan), [(q.name, q.worst_case_mb, q.container_cap_mb) for q in plan]
    ingestion = plan[0]
    assert ingestion.worst_case_mb == 2048 and ingestion.headroom_mb == 0
    media = plan[1]
    assert media.worst_case_mb == 768 and media.headroom_mb == 256
    assert plan[2].worst_case_mb == 768
    assert plan[3].worst_case_mb == 512  # child pinned to the 512m reasoning cap


def test_every_child_limit_fits_its_container_cap():
    # Regression guard: the reasoning worker used to inherit the 1024MB ingestion
    # child limit inside a 512m container.
    for queue in build_queue_capacity_plan(settings, HOST_8GB):
        assert queue.child_limit_mb <= queue.container_cap_mb, (
            f"{queue.name}: child {queue.child_limit_mb}MB > cap {queue.container_cap_mb}MB"
        )


def test_plan_child_limits_are_clamped_on_small_host():
    plan = build_queue_capacity_plan(settings, HOST_TINY)
    assert all(q.child_limit_mb <= HOST_TINY.safe_worker_budget_mb for q in plan)
    assert all(q.fits for q in plan), "clamped children must still fit their container"


def test_violation_is_flagged_when_concurrency_outgrows_container():
    oversubscribed = SimpleNamespace(
        celery_task_queue="ingestion", media_task_queue="media",
        index_task_queue="index", reasoning_task_queue="reasoning",
        worker_child_memory_limit_mb=1024, worker_memory_budget_mb=2048,
        media_child_memory_limit_mb=384, index_child_memory_limit_mb=768,
        reasoning_child_memory_limit_mb=512,
        ingestion_concurrency=4,  # 4 x 1024MB > 2048MB cap -> violation
        media_concurrency=2, index_concurrency=1, reasoning_concurrency=1,
        ingestion_container_cap_mb=2048, media_container_cap_mb=1024,
        index_container_cap_mb=1024, reasoning_container_cap_mb=512,
        api_container_cap_mb=512, qdrant_container_cap_mb=1024, redis_container_cap_mb=256,
        total_container_budget_mb=6400,
    )
    plan = build_queue_capacity_plan(oversubscribed, HOST_8GB)
    assert plan[0].fits is False and plan[0].worst_case_mb == 4096


# ---------- Host stack envelope ----------

def test_stack_budget_within_host_and_envelope():
    stack_total = stack_container_budget_mb(settings)
    assert stack_total == 6400  # 6.25 GiB caps
    assert stack_total <= settings.total_container_budget_mb
    assert settings.total_container_budget_mb < HOST_8GB.total_memory_mb  # host keeps headroom


# ---------- Embedding device alignment ----------

def test_embedding_device_defaults_to_cpu_on_unified_memory_host():
    # Default is "auto", but this 8GB host is below the 16GB accelerator threshold,
    # so the resolved device must be CPU (MPS shares the container RAM pool).
    assert settings.embedding_device == "auto"
    assert settings.accelerator_min_host_memory_mb == 16384
    assert EmbeddingEngine._resolve_device() == "cpu"


def test_embedding_device_falls_back_when_accelerator_absent(monkeypatch):
    monkeypatch.setattr("app.config.settings.embedding_device", "mps")
    if device_manager.device == "mps":
        assert EmbeddingEngine._resolve_device() == "mps"  # torch installed, MPS real
    else:
        assert EmbeddingEngine._resolve_device() == "cpu"  # graceful fallback, never a crash


# ---------- Compose <-> config agreement (drift guard) ----------

def _compose_defaults(key: str) -> list[str]:
    text = (REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    return re.findall(rf"\$\{{{key}:-(\d+)\}}", text)


def test_compose_concurrency_defaults_match_config():
    assert _compose_defaults("INGEST_INGESTION_CONCURRENCY") == [str(settings.ingestion_concurrency)]
    assert _compose_defaults("INGEST_MEDIA_CONCURRENCY") == [str(settings.media_concurrency)]
    assert _compose_defaults("INGEST_INDEX_CONCURRENCY") == [str(settings.index_concurrency)]
    assert _compose_defaults("INGEST_REASONING_CONCURRENCY") == [str(settings.reasoning_concurrency)]


def test_compose_child_limits_match_config():
    assert _compose_defaults("INGEST_WORKER_CHILD_MEMORY_LIMIT_MB") == [
        str(settings.worker_child_memory_limit_mb)
    ]
    assert _compose_defaults("INGEST_MEDIA_CHILD_MEMORY_LIMIT_MB") == [
        str(settings.media_child_memory_limit_mb)
    ]
    assert _compose_defaults("INGEST_INDEX_CHILD_MEMORY_LIMIT_MB") == [
        str(settings.index_child_memory_limit_mb)
    ]
    assert _compose_defaults("INGEST_REASONING_CHILD_MEMORY_LIMIT_MB") == [
        str(settings.reasoning_child_memory_limit_mb)
    ]
    assert _compose_defaults("INGEST_WORKER_MEMORY_BUDGET_MB") == [
        str(settings.worker_memory_budget_mb)
    ]
    assert _compose_defaults("INGEST_MEDIA_MAX_TASKS_PER_CHILD") == [
        str(settings.media_max_tasks_per_child)
    ]


def test_compose_container_caps_match_config():
    text = (REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    to_mb = {"512m": 512, "1g": 1024, "2g": 2048, "256m": 256}
    caps = [to_mb[m] for m in re.findall(r"mem_limit:\s*(\d+[mg])", text)]
    assert sorted(caps) == sorted([
        settings.redis_container_cap_mb, settings.api_container_cap_mb,
        settings.ingestion_container_cap_mb, settings.media_container_cap_mb,
        settings.index_container_cap_mb, settings.qdrant_container_cap_mb,
        settings.reasoning_container_cap_mb,
    ])
    assert sum(caps) == stack_container_budget_mb(settings)
