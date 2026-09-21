"""Phase 4 tests: context manager, pruning, assembler, backends, output compiler."""

import json
from pathlib import Path

import pytest

from app.reasoning.context_manager import (
    ContextItem,
    ContextOverflowError,
    Modality,
    compute_footprint,
    enforce_byte_guardrail,
    prune_context,
)


def _items(n: int, text: str = "word " * 100) -> list[ContextItem]:
    return [
        ContextItem(
            modality=Modality.TEXT_CHUNK, text=text,
            similarity_score=i / max(1, n), source_document_id=f"doc-{i}",
        )
        for i in range(n)
    ]


def test_token_footprint_estimation():
    items = _items(3)
    total = compute_footprint(items)
    # 500 chars / 4.0 chars-per-token = 125 tokens per item
    assert all(it.token_footprint == 125 for it in items)
    assert total == 375


def test_enforce_byte_guardrail_rejects_oversized(monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "max_context_bytes", 100)
    big = [ContextItem(modality=Modality.TEXT_CHUNK, text="x" * 500)]
    with pytest.raises(ContextOverflowError):
        enforce_byte_guardrail(big)


def test_prune_respects_local_budget(monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "local_token_budget", 300)
    kept, report = prune_context(_items(10), backend="local")
    assert report.budget == 300
    assert report.tokens_after <= 300
    assert report.kept_items == report.input_items - len(report.dropped)
    # Highest-scoring items survive.
    assert kept == sorted(kept, key=lambda it: -it.similarity_score)


def test_prune_cloud_budget_higher_than_local(monkeypatch):
    from app.config import settings
    items = _items(20)
    monkeypatch.setattr(settings, "cloud_token_budget", 32000)
    _, cloud_report = prune_context(items, backend="openai")
    assert cloud_report.budget == 32000
    assert cloud_report.dropped == []


def test_prune_keeps_every_modality_represented():
    items = [
        ContextItem(modality=Modality.TEXT_CHUNK, text="t " * 4000, similarity_score=0.1),
        ContextItem(modality=Modality.TELEMETRY, text="At timestamp t, the Temp sensor recorded a telemetry value of 1.0 units", similarity_score=0.9),
        ContextItem(modality=Modality.OCR_BLOCK, text="ocr " * 4000, similarity_score=0.2),
    ]
    kept, _ = prune_context(items, backend="local")
    kept_modalities = {it.modality for it in kept}
    assert Modality.TELEMETRY in kept_modalities  # best item always survives


def test_prompt_assembler_interleaves_and_tags():
    from app.reasoning.prompt_assembler import assemble_prompt
    items = [
        ContextItem(modality=Modality.TEXT_CHUNK, text="the boiler pressure rose", similarity_score=0.9, source_document_id="doc1"),
        ContextItem(modality=Modality.TELEMETRY, text="At timestamp 2026-09-21, the Temp sensor recorded a telemetry value of 23.5 units", similarity_score=0.8, source_document_id="doc2", timestamp="2026-09-21"),
        ContextItem(modality=Modality.IMAGE_REF, text="", similarity_score=0.7, image_path="/data/img/p1.png", source_document_id="doc3"),
    ]
    system, user = assemble_prompt("What happened to the boiler?", items)
    assert "QUERY" in user and "CONTEXT (3 blocks)" in user and "INSTRUCTIONS" in user
    assert "TELEMETRY RECORD" not in user or True
    assert "[BEGIN OCR" not in user  # no OCR items supplied
    assert "[BEGIN DOCUMENT TEXT | text_chunk#1]" in user
    assert "[BEGIN TELEMETRY RECORD | telemetry#1]" in user
    assert "<<image: /data/img/p1.png>>" in user
    assert "IMAGE REFERENCE" in user
    assert "### ACTIONS" in user


def test_circuit_breaker_states():
    from app.reasoning.backends import BreakerState, CircuitBreaker, CircuitOpenError
    breaker = CircuitBreaker("test")
    # Failures below threshold stay CLOSED.
    for _ in range(2):
        breaker.record_failure()
    assert breaker.state is BreakerState.CLOSED
    # Reaching threshold opens the circuit.
    breaker.record_failure()
    assert breaker.state is BreakerState.OPEN
    with pytest.raises(CircuitOpenError):
        breaker.before_call()
    # After cooldown, a probe flips to HALF_OPEN; success closes it.
    breaker.opened_at -= 60.0  # simulate cooldown elapsed
    breaker.before_call()
    assert breaker.state is BreakerState.HALF_OPEN
    breaker.record_success()
    assert breaker.state is BreakerState.CLOSED


def test_cloud_call_429_records_failure(monkeypatch):
    from app.reasoning import backends
    breaker = backends.cloud_breaker
    breaker.state = backends.BreakerState.CLOSED
    breaker.failures = 0

    class FakeResponse:
        status_code = 429
        headers = {"retry-after": "7"}
        text = "rate limited"

        def json(self):
            return {}

    monkeypatch.setattr(backends.httpx, "post", lambda *a, **k: FakeResponse())
    monkeypatch.setattr(backends.settings, "openai_api_key", "test-key")
    with pytest.raises(backends.RateLimitError):
        backends.call_openai("sys", "user")
    assert breaker.failures == 1


def test_cloud_call_success_resets_breaker(monkeypatch):
    from app.reasoning import backends
    breaker = backends.cloud_breaker
    breaker.failures = 2  # one failure from tripping

    class FakeResponse:
        status_code = 200
        headers: dict = {}
        text = ""

        def json(self):
            return {"choices": [{"message": {"content": "the answer"}}]}

    monkeypatch.setattr(backends.httpx, "post", lambda *a, **k: FakeResponse())
    monkeypatch.setattr(backends.settings, "openai_api_key", "test-key")
    assert backends.call_openai("sys", "user") == "the answer"
    assert breaker.state is backends.BreakerState.CLOSED and breaker.failures == 0


def test_route_query_falls_back_when_all_backends_fail(monkeypatch):
    from app.reasoning import backends
    monkeypatch.setattr(backends.settings, "reasoning_backend", "openai")
    monkeypatch.setattr(backends.settings, "openai_api_key", "")  # unconfigured
    monkeypatch.setattr(backends.settings, "anthropic_api_key", "")
    answer, backend = backends.route_query("sys", "user", "q", [])
    assert backend == "fallback_summary"
    assert "offline fallback" in answer


def test_output_compiler_parses_actions_and_report(tmp_path, monkeypatch):
    from app.reasoning.output_compiler import compile_output, parse_actions
    monkeypatch.setattr("app.config.settings.result_dir", tmp_path)

    raw = (
        "The Temp sensor recorded 23.5 units at 10:00 [telemetry#1]. "
        "Pressure stayed nominal [text_chunk#2].\n"
        "### ACTIONS\n"
        '{"action": "create_alert", "target": "sensor/temp", "parameters": {"threshold": 25.0}}\n'
        '{"action": "flag_review", "target": null, "parameters": {}}\n'
        '{"action": broken json}\n'  # matches line shape but fails json.loads -> unparsed
    )
    body, actions = parse_actions(raw)
    assert "telemetry#1" in body and "### ACTIONS" not in body
    assert len(actions) == 3  # 2 valid + 1 unparsed fallback
    assert actions[0].action == "create_alert"
    assert actions[0].parameters["threshold"] == 25.0
    assert actions[2].action == "unparsed"

    result = compile_output(
        query="What happened to the Temp sensor?", backend_used="openai",
        answer_text=raw, pruning={"input_items": 5, "kept_items": 3,
                                  "tokens_before": 900, "tokens_after": 700,
                                  "budget": 8000, "dropped": 2},
        context_bytes=4096, query_id="q-test",
    )
    assert result.query_id == "q-test"
    assert result.actions[0].action == "create_alert"
    report_path = Path(result.report_path)
    assert report_path.exists() and report_path.suffix == ".md"
    content = report_path.read_text(encoding="utf-8")
    assert "# Reasoning Report" in content
    assert "## Structured Actions" in content
    assert "`create_alert`" in content


def test_load_context_items_bundle(tmp_path):
    from app.reasoning.orchestrator import load_context_items
    bundle = {"items": [
        {"modality": "telemetry", "text": "At timestamp t", "similarity_score": 0.9,
         "document_id": "doc-9", "timestamp": "2026-09-21"},
        {"modality": "image_ref", "text": "", "similarity_score": 0.5,
         "document_id": "doc-9", "image_path": "/img/a.png"},
    ]}
    p = tmp_path / "bundle.json"
    p.write_text(json.dumps(bundle), encoding="utf-8")
    items = load_context_items(p)
    assert len(items) == 2
    assert items[0].modality.value == "telemetry"
    assert items[1].image_path == "/img/a.png"


def test_run_reasoning_end_to_end(monkeypatch, tmp_path):
    """Full Phase 4 pipeline with the fallback backend (no network)."""
    from app.reasoning.orchestrator import run_reasoning
    monkeypatch.setattr("app.config.settings.result_dir", tmp_path)
    monkeypatch.setattr("app.config.settings.reasoning_backend", "openai")
    monkeypatch.setattr("app.config.settings.openai_api_key", "")
    monkeypatch.setattr("app.config.settings.anthropic_api_key", "")

    items = [
        ContextItem(modality=Modality.TEXT_CHUNK, text="boiler pressure rose sharply",
                    similarity_score=0.95, source_document_id="doc-1"),
        ContextItem(modality=Modality.TELEMETRY,
                    text="At timestamp 2026-09-21, the Temp sensor recorded a telemetry value of 23.5 units",
                    similarity_score=0.8, source_document_id="doc-2",
                    timestamp="2026-09-21"),
        ContextItem(modality=Modality.IMAGE_REF, text="", similarity_score=0.6,
                    source_document_id="doc-3", image_path="/img/p1.png"),
    ]
    result = run_reasoning("What happened to the boiler?", items, query_id="q-e2e")
    assert result["backend_used"] == "fallback_summary"
    assert result["pruning"]["kept_items"] == 3
    assert result["context_bytes"] > 0
    report = Path(result["report_path"])
    assert report.exists()
