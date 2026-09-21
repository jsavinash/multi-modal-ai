"""Documentation contract tests: diagram structure, link integrity, doc-vs-code numbers.

During development the Mermaid diagrams were validated with the real Mermaid parser
(`mermaid.parse` via node). These tests keep them honest in plain Python with no extra
dependencies: fence/structure balance, resolvable links, and every documented figure
checked against `app/config.py` and the actual Celery task names.
"""

import re
from pathlib import Path

import pytest

from app.config import settings

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs"
MARKDOWN = sorted(DOCS.glob("*.md")) + [REPO / "README.md"]

MERMAID_TYPES = {
    "flowchart", "graph", "sequenceDiagram", "classDiagram", "stateDiagram-v2",
    "erDiagram", "journey", "gantt", "pie", "mindmap", "timeline", "quadrantChart",
}
_ARROW = re.compile(r"^\s*(\w+)\s*(?:--?>>?|--?x|--?\))\s*(\w+)\s*:")


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _mermaid_blocks(text: str) -> list[str]:
    return re.findall(r"```mermaid\n(.*?)```", text, re.S)


# ---------- Structure ----------

@pytest.mark.parametrize("path", MARKDOWN, ids=lambda p: p.name)
def test_code_fences_are_balanced(path):
    assert _text(path).count("```") % 2 == 0, f"{path.name}: unbalanced code fences"


@pytest.mark.parametrize("path", MARKDOWN, ids=lambda p: p.name)
def test_mermaid_blocks_are_structurally_sound(path):
    for i, block in enumerate(_mermaid_blocks(_text(path)), start=1):
        kind = block.strip().split()[0]
        where = f"{path.name} diagram {i} ({kind})"
        assert kind in MERMAID_TYPES, f"{where}: unknown diagram type"
        ends = len(re.findall(r"^\s*end\s*$", block, re.M))
        if kind in {"flowchart", "graph"}:
            opens = len(re.findall(r"^\s*subgraph\b", block, re.M))
            assert opens == ends, f"{where}: {opens} subgraph vs {ends} end"
        elif kind == "sequenceDiagram":
            alts = len(re.findall(r"^\s*alt\b", block, re.M))
            assert alts == ends, f"{where}: {alts} alt vs {ends} end"
            declared = set(re.findall(r"^\s*(?:participant|actor)\s+(\w+)", block, re.M))
            assert declared, f"{where}: no participants declared"
            for line in block.splitlines():
                match = _ARROW.match(line)
                if match:
                    for side in match.groups():
                        assert side in declared, (
                            f"{where}: '{side}' used before declaration - {line.strip()}"
                        )


@pytest.mark.parametrize("path", MARKDOWN, ids=lambda p: p.name)
def test_relative_links_resolve(path):
    for target in re.findall(r"\]\(([^)#]+?)\)", _text(path)):
        if target.startswith(("http://", "https://", "mailto:")):
            continue
        assert (path.parent / target).exists(), f"{path.name}: broken link -> {target}"


def test_hld_covers_every_execution():
    hld = _text(DOCS / "HLD.md")
    for endpoint in ("/api/v1/documents/ingest", "/api/v1/media/ingest",
                     "/api/v1/tabular/ingest", "/api/v1/reason/query", "/health"):
        assert endpoint in hld, f"HLD missing execution for {endpoint}"
    for marker in ("Execution 1", "Execution 2", "Execution 3", "Execution 4",
                   "Execution 5"):
        assert marker in hld, f"HLD missing '{marker}'"


# ---------- Documentation vs configuration ----------

def test_hld_queue_names_match_config():
    hld = _text(DOCS / "HLD.md")
    for queue in (settings.celery_task_queue, settings.media_task_queue,
                  settings.index_task_queue, settings.reasoning_task_queue):
        assert f"{queue} queue" in hld or f"`{queue}`" in hld or f"{queue} (" in hld, (
            f"HLD does not document queue '{queue}'"
        )


def test_hld_task_names_match_registered_celery_tasks():
    from app.tasks import parse_document_task
    from app.tasks_media import process_media_task
    from app.tasks_reasoning import reasoning_task
    from app.tasks_tabular import index_tabular_task

    hld = _text(DOCS / "HLD.md")
    for task in (parse_document_task, process_media_task, index_tabular_task, reasoning_task):
        assert task.name in hld, f"HLD documents no flow for task '{task.name}'"


def test_hld_container_caps_match_config():
    hld = _text(DOCS / "HLD.md")
    expected = [
        "api container 512m",
        "redis 256m",
        f"worker {settings.ingestion_container_cap_mb // 1024}g",
        f"media-worker {settings.media_container_cap_mb // 1024}g",
        f"index-worker {settings.index_container_cap_mb // 1024}g",
        f"reasoning-worker {settings.reasoning_container_cap_mb}m",
        f"qdrant {settings.qdrant_container_cap_mb // 1024}g",
    ]
    for item in expected:
        assert item in hld, f"HLD missing container cap '{item}'"
    assert "6.25 GB" in hld


def test_hld_concurrency_matches_config():
    hld = _text(DOCS / "HLD.md")
    assert f"ingestion queue conc {settings.ingestion_concurrency}" in hld
    assert f"media queue conc {settings.media_concurrency}" in hld
    assert f"index queue conc {settings.index_concurrency}" in hld
    assert f"reasoning conc {settings.reasoning_concurrency}" in hld


def test_hld_guardrails_match_config():
    """The numbers quoted in the HLD flows must be the numbers the code enforces."""
    hld = _text(DOCS / "HLD.md")
    assert f"{settings.max_file_size_bytes // (1024 * 1024)} MB cap" in hld
    assert f"{settings.min_free_disk_mb // 1024} GB disk floor" in hld
    assert f"{settings.chunk_target_tokens} tokens" in hld
    assert f"{settings.max_image_edge_px} px" in hld
    assert f"{settings.audio_target_sample_rate_hz // 1000} kHz" in hld
    assert f"{int(settings.audio_window_seconds)} second windows" in hld
    assert f"{int(settings.video_sample_fps)} frame per second" in hld
    assert f"{settings.embedding_dim} dims" in hld
    assert f"{settings.max_embed_batch_rows} rows" in hld
    assert f"{settings.upsert_batch_size} vectors per batch" in hld
    assert f"{settings.local_token_budget // 1000}k local" in hld
    assert f"{settings.cloud_token_budget // 1000}k cloud" in hld
    assert f"{settings.max_context_bytes // (1024 * 1024)} MB byte cap" in hld
    assert f"{settings.breaker_failure_threshold} failures" in hld
    assert f"{int(settings.breaker_cooldown_s)} s" in hld
    assert f"{int(settings.media_task_timeout_s)} s timeout" in hld
    assert f"{settings.ffmpeg_timeout_s} s" in hld
    assert f"{int(settings.reasoning_request_timeout_s)} s timeout" in hld
    assert f"{settings.ocr_confidence_threshold}" in hld


def test_hld_names_the_real_vector_collections():
    from app.vector_schema import COLLECTIONS

    hld = _text(DOCS / "HLD.md")
    assert f"{len(COLLECTIONS)} collections" in hld
    assert COLLECTIONS["tabular"] in hld


def test_explainer_numbers_match_config():
    """The plain-English guide quotes live guardrails; keep the two in step."""
    doc = _text(DOCS / "EXPLAINER.md")
    stack_mb = (settings.ingestion_container_cap_mb + settings.media_container_cap_mb
                + settings.index_container_cap_mb + settings.reasoning_container_cap_mb
                + settings.api_container_cap_mb + settings.qdrant_container_cap_mb
                + settings.redis_container_cap_mb)
    assert stack_mb == settings.total_container_budget_mb == 6400
    assert "6.25 GB" in doc and f"{settings.total_container_budget_mb}" not in doc
    assert "100 MB" in doc and settings.max_file_size_bytes == 100 * 1024 * 1024
    assert "512 pixels" in doc and settings.max_image_edge_px == 512
    assert "1 picture per second" in doc and settings.video_sample_fps == 1.0
    assert "16 kHz" in doc and settings.audio_target_sample_rate_hz == 16000
    assert "30-second windows" in doc and settings.audio_window_seconds == 30.0
    assert "512 tokens" in doc and settings.chunk_target_tokens == 512
    assert "10 % overlap" in doc and settings.chunk_overlap_fraction == 0.10
    assert "384 numbers" in doc and settings.embedding_dim == 384
    assert "256 rows" in doc and settings.max_embed_batch_rows == 256
    assert "8,000 tokens" in doc and settings.local_token_budget == 8000
    assert "32,000" in doc and settings.cloud_token_budget == 32000
    assert "2 MB" in doc and settings.max_context_bytes == 2 * 1024 * 1024
    assert "3 failures" in doc and "30 seconds" in doc
    assert (settings.breaker_failure_threshold, settings.breaker_cooldown_s) == (3, 30.0)
    assert f"{settings.ingestion_concurrency} \u00d7 1 GB = 2 GB" in doc
    assert f"{settings.media_concurrency} \u00d7 384 MB = 768 MB" in doc
    assert "16 GB+" in doc and settings.accelerator_min_host_memory_mb == 16384
    assert "(`auto`)" in doc and settings.embedding_device == "auto"
    assert str(settings.result_dir) in doc


def test_explainer_and_hld_cross_link_each_other():
    explainer = _text(DOCS / "EXPLAINER.md")
    hld = _text(DOCS / "HLD.md")
    assert "HLD.md" in explainer or "INDEX.md" in explainer
    assert "EXPLAINER.md" in hld and "INDEX.md" in hld
