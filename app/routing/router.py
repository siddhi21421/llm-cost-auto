"""
The router: takes a prompt, classifies its complexity, and returns the
ModelConfig it should be sent to -- plus the reasoning, so the API layer
can tell the caller *why* a given model was picked (useful for debugging
and for the portfolio demo).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import yaml

from app.classifier.predict import classify_with_confidence
from app.models.registry import ModelConfig, get_model

CONFIG_PATH = Path(__file__).resolve().parent / "routing_config.yaml"


@dataclass
class RoutingDecision:
    model_config: ModelConfig
    tier: int
    raw_tier: int
    confidence: float
    escalated_for_confidence: bool
    reason: str


class Router:
    def __init__(self, config_path: Path = CONFIG_PATH):
        self.config_path = config_path
        self._load_config()

    def _load_config(self):
        with self.config_path.open() as f:
            self.config = yaml.safe_load(f)
        self.routing_map: Dict[int, str] = {
            int(k): v for k, v in self.config["routing_map"].items()
        }
        self.verifier_model_name = self.config["verifier_model"]
        self.low_confidence_threshold = float(self.config["low_confidence_threshold"])
        self.escalate_on_low_confidence = bool(self.config["escalate_on_low_confidence"])

    def reload(self):
        """Reload config from disk -- called after PUT /v1/routing-config."""
        self._load_config()

    def route(self, prompt: str) -> RoutingDecision:
        raw_tier, confidence = classify_with_confidence(prompt)
        tier = raw_tier
        escalated = False

        if (
            self.escalate_on_low_confidence
            and confidence < self.low_confidence_threshold
            and raw_tier < 3
        ):
            tier = raw_tier + 1
            escalated = True

        model_name = self.routing_map.get(tier)
        if model_name is None:
            raise ValueError(f"No model configured for tier {tier} in routing_config.yaml")

        model_config = get_model(model_name)

        reason = (
            f"classified as tier {raw_tier} (confidence {confidence:.0%})"
            + (f", escalated to tier {tier} due to low confidence" if escalated else "")
            + f" -> routed to {model_config.display_name}"
        )

        return RoutingDecision(
            model_config=model_config,
            tier=tier,
            raw_tier=raw_tier,
            confidence=confidence,
            escalated_for_confidence=escalated,
            reason=reason,
        )

    def get_verifier_model(self) -> ModelConfig:
        return get_model(self.verifier_model_name)


# Module-level singleton -- config is cheap to hold in memory, reload() on write.
router = Router()
