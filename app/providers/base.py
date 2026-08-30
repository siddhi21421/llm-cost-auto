"""
Unified provider interface. Every provider module exposes a single
`send_request(prompt, model_config, **kwargs) -> Response` function.
Callers never branch on provider type -- that's the whole point of
the abstraction.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from app.models.registry import ModelConfig


@dataclass
class Response:
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost: float
    model_id: str
    provider: str
    raw_error: Optional[str] = None  # set if the call failed and we degraded gracefully

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def ok(self) -> bool:
        return self.raw_error is None


class ProviderError(RuntimeError):
    """Raised when a provider call fails in a way callers should know about."""


def timed(fn):
    """Decorator that measures wall-clock latency in ms and injects it into
    the returned Response. Keeps latency measurement out of every provider
    implementation."""

    def wrapper(*args, **kwargs) -> Response:
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        result.latency_ms = elapsed_ms
        return result

    return wrapper


def estimate_tokens(text: str) -> int:
    """Cheap fallback token estimator (~4 chars/token for English) used only
    when a provider doesn't return exact usage numbers. Real providers
    (OpenAI, Anthropic) return exact counts in their response payloads --
    prefer those whenever available."""
    return max(1, len(text) // 4)
