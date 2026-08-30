"""
Phase 3: async quality verification.

After the cheap model's response has already gone back to the user, this
module re-runs the prompt against the verifier model (the highest-tier
model in routing_config.yaml) and scores agreement using an LLM-as-judge
prompt. If the cheap model's answer diverges too much, it's logged as a
routing failure, optionally auto-escalated, and fed back to the classifier
as a training example.

This is designed to run as a background task (FastAPI BackgroundTasks or
a separate worker process reading a queue) -- see app/api/main.py for how
it's wired into the request lifecycle.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

from app.db import database as db
from app.models.registry import ModelConfig
from app.providers import send_request
from app.routing.router import router

# Quality bar below which we consider the cheap model's answer a "failure".
# On a 1-5 LLM-as-judge scale, 4/5 matches the spec's summarization bar;
# we use it as the default for all task types in v1.
QUALITY_PASS_THRESHOLD = 4.0

JUDGE_PROMPT_TEMPLATE = """You are grading whether two AI responses to the same \
prompt are substantively equivalent in correctness and completeness. Ignore \
differences in style or verbosity -- focus only on whether a user would be \
equally well served by response A instead of the reference response B.

Prompt:
{prompt}

Response A (candidate, cheaper model):
{candidate}

Response B (reference, high-quality model):
{reference}

Score response A from 1 to 5, where:
5 = fully equivalent, no meaningful gaps
4 = minor stylistic differences only, same substance
3 = mostly equivalent but missing a minor detail
2 = meaningfully incomplete or contains an error the reference avoids
1 = wrong, contradicts the reference, or fails the task

Respond with ONLY a JSON object: {{"score": <1-5>, "rationale": "<one sentence>"}}
"""


@dataclass
class VerificationResult:
    quality_score: float
    passed: bool
    rationale: str


def _parse_judge_response(text: str) -> VerificationResult:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        # Judge call malformed -- fail safe by treating as a failure so a
        # human/dashboard notices rather than silently assuming success.
        return VerificationResult(quality_score=0.0, passed=False, rationale="unparseable judge output")
    try:
        parsed = json.loads(match.group(0))
        score = float(parsed["score"])
        return VerificationResult(
            quality_score=score,
            passed=score >= QUALITY_PASS_THRESHOLD,
            rationale=parsed.get("rationale", ""),
        )
    except (json.JSONDecodeError, KeyError, ValueError):
        return VerificationResult(quality_score=0.0, passed=False, rationale="malformed judge JSON")


def verify_and_maybe_escalate(
    request_id: int,
    prompt: str,
    candidate_text: str,
    candidate_model: ModelConfig,
    candidate_cost: float,
    latency_budget_ms: Optional[float] = None,
) -> None:
    """
    Runs the full Phase 3 loop:
      1. Get the reference answer from the verifier (highest-tier) model.
      2. Score agreement with an LLM-as-judge call.
      3. If it fails, log the failure, optionally escalate, and record
         feedback for classifier retraining.

    This function does all its own logging to SQLite -- callers just fire
    it (typically via BackgroundTasks) and move on.
    """
    verifier_model = router.get_verifier_model()

    reference_resp = send_request(prompt, verifier_model)
    if not reference_resp.ok:
        # Can't verify without a working reference call -- log and bail.
        db.log_verification(
            request_id=request_id,
            verifier_model=verifier_model.model_id,
            quality_score=0.0,
            passed=False,
            escalated=False,
        )
        return

    judge_prompt = JUDGE_PROMPT_TEMPLATE.format(
        prompt=prompt, candidate=candidate_text, reference=reference_resp.text
    )
    judge_resp = send_request(judge_prompt, verifier_model)
    result = _parse_judge_response(judge_resp.text) if judge_resp.ok else VerificationResult(
        0.0, False, "judge call failed"
    )

    escalated = False
    escalated_model_name = None
    escalated_cost = None
    cost_delta = None

    if not result.passed:
        # Routing failure: the cheap model's answer didn't hold up.
        within_budget = latency_budget_ms is None or reference_resp.latency_ms <= latency_budget_ms
        if within_budget:
            escalated = True
            escalated_model_name = verifier_model.model_id
            escalated_cost = reference_resp.cost
            cost_delta = reference_resp.cost - candidate_cost

        # Feed the failure back to the classifier: this prompt should have
        # been routed at least one tier higher than it was.
        db.log_feedback(
            prompt=prompt,
            original_tier=_tier_for_model(candidate_model),
            corrected_tier=min(3, _tier_for_model(candidate_model) + 1),
        )

    db.log_verification(
        request_id=request_id,
        verifier_model=verifier_model.model_id,
        quality_score=result.quality_score,
        passed=result.passed,
        escalated=escalated,
        escalated_model=escalated_model_name,
        escalated_cost=escalated_cost,
        cost_delta=cost_delta,
    )


def _tier_for_model(model_config: ModelConfig) -> int:
    """Reverse lookup: what tier does routing_config.yaml currently map
    this model to? Used only for feedback labeling."""
    for tier, model_name in router.routing_map.items():
        from app.models.registry import get_model
        if get_model(model_name).model_id == model_config.model_id:
            return tier
    return 2  # sane default if the model isn't in the current routing map
