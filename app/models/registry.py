"""
Model registry: defines every model the router is allowed to pick from,
along with the cost/latency/quality metadata the router needs to make
a decision.

Prices are USD per token (not per 1K/1M) so cost math elsewhere never
has to remember a multiplier. Update PRICING_SOURCE_DATE when you refresh
these numbers -- provider pricing changes often and stale numbers will
quietly corrupt your "cost saved" headline metric.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional


PRICING_SOURCE_DATE = "2026-08-30"  # update this whenever you refresh prices below


class QualityTier(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Provider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"


@dataclass(frozen=True)
class ModelConfig:
    provider: Provider
    model_id: str                 # the exact string the provider API expects
    display_name: str
    cost_per_input_token: float   # USD
    cost_per_output_token: float  # USD
    avg_latency_ms: int           # rough baseline, refined by live stats later
    quality_tier: QualityTier
    context_window: int = 128_000
    notes: str = ""

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self.cost_per_input_token
            + output_tokens * self.cost_per_output_token
        )


# ---------------------------------------------------------------------------
# Registry. Prices are $/token. Ollama is local so cost is ~electricity;
# we set it to 0.0 and let the dashboard call that out explicitly rather
# than pretend it's free-free.
# ---------------------------------------------------------------------------
MODEL_REGISTRY: Dict[str, ModelConfig] = {
    "gpt-4o": ModelConfig(
        provider=Provider.OPENAI,
        model_id="gpt-4o",
        display_name="GPT-4o",
        cost_per_input_token=2.50 / 1_000_000,
        cost_per_output_token=10.00 / 1_000_000,
        avg_latency_ms=1800,
        quality_tier=QualityTier.HIGH,
        context_window=128_000,
    ),
    "gpt-4o-mini": ModelConfig(
        provider=Provider.OPENAI,
        model_id="gpt-4o-mini",
        display_name="GPT-4o mini",
        cost_per_input_token=0.15 / 1_000_000,
        cost_per_output_token=0.60 / 1_000_000,
        avg_latency_ms=900,
        quality_tier=QualityTier.MEDIUM,
        context_window=128_000,
    ),
    "claude-sonnet": ModelConfig(
        provider=Provider.ANTHROPIC,
        model_id="claude-sonnet-4-6",
        display_name="Claude Sonnet",
        cost_per_input_token=3.00 / 1_000_000,
        cost_per_output_token=15.00 / 1_000_000,
        avg_latency_ms=1600,
        quality_tier=QualityTier.HIGH,
        context_window=200_000,
    ),
    "claude-haiku": ModelConfig(
        provider=Provider.ANTHROPIC,
        model_id="claude-haiku-4-5-20251001",
        display_name="Claude Haiku",
        cost_per_input_token=0.80 / 1_000_000,
        cost_per_output_token=4.00 / 1_000_000,
        avg_latency_ms=700,
        quality_tier=QualityTier.LOW,
        context_window=200_000,
    ),
    "llama-local": ModelConfig(
        provider=Provider.OLLAMA,
        model_id="llama3.1:8b",
        display_name="Llama 3.1 8B (local)",
        cost_per_input_token=0.0,
        cost_per_output_token=0.0,
        avg_latency_ms=1200,
        quality_tier=QualityTier.LOW,
        context_window=128_000,
        notes="Cost is $0 in token terms but not free: factor in your own compute/electricity.",
    ),
}


def get_model(name: str) -> ModelConfig:
    try:
        return MODEL_REGISTRY[name]
    except KeyError as e:
        raise KeyError(
            f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}"
        ) from e


def models_by_tier(tier: QualityTier) -> Dict[str, ModelConfig]:
    return {k: v for k, v in MODEL_REGISTRY.items() if v.quality_tier == tier}
