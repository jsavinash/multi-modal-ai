"""Reasoning backends: cloud-hybrid router with a strict circuit breaker.

Backends:
- openai / anthropic: managed cloud endpoints (httpx, lazy client).
- local: optional quantized GGUF backbone (llama.cpp); never auto-loaded.
- fallback_summary: deterministic extractive summary when no backbone is
  reachable (keeps the pipeline answerable offline).
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from enum import Enum

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


class CircuitOpenError(RuntimeError):
    """Raised when the circuit breaker is OPEN and calls are refused."""


class BreakerState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Strict circuit breaker for cloud endpoints and rate limiting."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.state = BreakerState.CLOSED
        self.failures = 0
        self.opened_at = 0.0

    def before_call(self) -> None:
        if self.state is BreakerState.OPEN:
            elapsed = time.monotonic() - self.opened_at
            if elapsed >= settings.breaker_cooldown_s:
                self.state = BreakerState.HALF_OPEN
                logger.info("Circuit HALF_OPEN; probing", extra={"breaker": self.name})
            else:
                raise CircuitOpenError(
                    f"Circuit '{self.name}' is OPEN; retry in "
                    f"{round(settings.breaker_cooldown_s - elapsed, 1)}s."
                )

    def record_success(self) -> None:
        self.failures = 0
        if self.state is not BreakerState.CLOSED:
            logger.info("Circuit CLOSED; recovered", extra={"breaker": self.name})
        self.state = BreakerState.CLOSED

    def record_failure(self) -> None:
        self.failures += 1
        if self.state is BreakerState.HALF_OPEN or self.failures >= settings.breaker_failure_threshold:
            self.state = BreakerState.OPEN
            self.opened_at = time.monotonic()
            logger.warning("Circuit OPEN", extra={
                "breaker": self.name, "failures": self.failures,
            })


cloud_breaker = CircuitBreaker("cloud-reasoning")


class RateLimitError(RuntimeError):
    """Raised on HTTP 429; the caller should back off (Retry-After aware)."""


def _cloud_call(url: str, headers: dict, body: dict, provider: str) -> str:
    """Single cloud completion call with breaker + Retry-After handling."""
    cloud_breaker.before_call()
    try:
        response = httpx.post(
            url, headers=headers, json=body,
            timeout=settings.reasoning_request_timeout_s,
        )
    except httpx.HTTPError as exc:
        cloud_breaker.record_failure()
        raise RuntimeError(f"{provider} transport failure: {exc}") from exc

    if response.status_code == 429:
        cloud_breaker.record_failure()
        retry_after = response.headers.get("retry-after", "1")
        raise RateLimitError(
            f"{provider} rate limited; Retry-After: {retry_after}s"
        )
    if response.status_code >= 500:
        cloud_breaker.record_failure()
        raise RuntimeError(f"{provider} server error {response.status_code}")
    if response.status_code >= 400:
        # Client error (bad key, bad request) is not transient: open the breaker
        # but surface the body for diagnosis.
        cloud_breaker.record_failure()
        raise RuntimeError(f"{provider} client error {response.status_code}: {response.text[:200]}")

    cloud_breaker.record_success()
    data = response.json()
    if provider == "openai":
        return data["choices"][0]["message"]["content"]
    return data["content"][0]["text"]  # anthropic


def call_openai(system: str, user: str) -> str:
    """OpenAI chat completion (GPT-4o class multimodal endpoint)."""
    if not settings.openai_api_key:
        raise RuntimeError("OpenAI backend not configured (missing API key).")
    return _cloud_call(
        "https://api.openai.com/v1/chat/completions",
        {"Authorization": f"Bearer {settings.openai_api_key}",
         "Content-Type": "application/json"},
        {
            "model": settings.reasoning_cloud_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": 2048,
        },
        "openai",
    )


def call_anthropic(system: str, user: str) -> str:
    """Anthropic Messages API (Claude class multimodal endpoint)."""
    if not settings.anthropic_api_key:
        raise RuntimeError("Anthropic backend not configured (missing API key).")
    return _cloud_call(
        "https://api.anthropic.com/v1/messages",
        {"x-api-key": settings.anthropic_api_key,
         "anthropic-version": "2023-06-01", "Content-Type": "application/json"},
        {
            "model": "claude-3-5-sonnet-latest",
            "max_tokens": 2048,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        },
        "anthropic",
    )


def call_local(system: str, user: str) -> str:
    """Optional quantized GGUF backbone via llama.cpp; never auto-loaded."""
    if not (settings.local_llm_enabled and settings.local_llm_model_path):
        raise RuntimeError("Local LLM backend not enabled/configured.")
    try:
        from llama_cpp import Llama  # optional heavy dependency
    except ImportError as exc:
        raise RuntimeError("llama-cpp-python not installed.") from exc
    from app.media.device_manager import device_manager
    try:
        with device_manager.inference_session():
            llm = Llama(
                model_path=settings.local_llm_model_path, n_ctx=settings.local_token_budget,
                verbose=False,
            )
            out = llm.create_chat_completion(
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                max_tokens=2048,
            )
            return out["choices"][0]["message"]["content"]
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Local LLM inference failed: {exc}") from exc


def fallback_summary(query: str, items) -> str:
    """Deterministic extractive answer when every backbone is unavailable."""
    lines = [f"(offline fallback summary — no reasoning backbone reachable)", ""]
    lines.append(f"Query: {query}")
    for it in sorted(items, key=lambda i: -i.similarity_score)[:5]:
        snippet = it.text[:200].replace("\n", " ")
        lines.append(f"- [{it.modality.value}] (score {it.similarity_score:.3f}) {snippet}")
    return "\n".join(lines)


def route_query(system: str, user: str, query: str, items) -> tuple[str, str]:
    """Cloud-hybrid router. Returns (answer_text, backend_used)."""
    order: list[tuple[str, callable]] = []
    backend = settings.reasoning_backend
    if backend == "openai":
        order = [("openai", call_openai), ("anthropic", call_anthropic)]
    elif backend == "anthropic":
        order = [("anthropic", call_anthropic), ("openai", call_openai)]
    elif backend == "local":
        order = [("local", call_local)]
    else:  # auto: local only when explicitly enabled; else cloud, then fallback
        if settings.local_llm_enabled:
            order = [("local", call_local), ("openai", call_openai), ("anthropic", call_anthropic)]
        else:
            order = [("openai", call_openai), ("anthropic", call_anthropic)]

    for name, fn in order:
        try:
            return fn(system, user), name
        except CircuitOpenError as exc:
            logger.warning("Backend refused by breaker", extra={"backend": name, "reason": str(exc)})
        except RateLimitError as exc:
            logger.warning("Backend rate limited", extra={"backend": name, "reason": str(exc)})
        except RuntimeError as exc:
            logger.warning("Backend failed", extra={"backend": name, "reason": str(exc)})

    return fallback_summary(query, items), "fallback_summary"
