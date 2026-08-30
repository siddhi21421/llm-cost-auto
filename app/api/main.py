"""
Phase 5: the API surface.

POST /v1/completions      - main endpoint, router picks the model
GET  /v1/models            - list registry with pricing
GET  /v1/stats              - cost savings summary for the dashboard
PUT  /v1/routing-config      - hot-swap tier->model mapping
"""
from __future__ import annotations

import time
from typing import Optional

import yaml
from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.db import database as db
from app.models.registry import MODEL_REGISTRY, get_model
from app.providers import send_request
from app.routing.router import router, CONFIG_PATH
from app.verification.verifier import verify_and_maybe_escalate

app = FastAPI(
    title="LLM Cost Autopilot",
    description="Routes requests to the cheapest model capable of handling them.",
    version="1.0.0",
)


@app.on_event("startup")
def startup():
    db.init_db()


# ---------------------------------------------------------------------------
# POST /v1/completions
# ---------------------------------------------------------------------------
class CompletionRequest(BaseModel):
    prompt: str = Field(..., description="The user's prompt. The router chooses the model.")
    max_tokens: int = 1024
    verify: bool = Field(
        True, description="Whether to queue an async quality verification job."
    )


class CompletionResponse(BaseModel):
    text: str
    routed_model: str
    provider: str
    tier: int
    raw_tier: int
    classifier_confidence: float
    escalated_for_confidence: bool
    routing_reason: str
    input_tokens: int
    output_tokens: int
    cost: float
    latency_ms: float
    request_id: int


@app.post("/v1/completions", response_model=CompletionResponse)
def completions(req: CompletionRequest, background_tasks: BackgroundTasks):
    decision = router.route(req.prompt)
    resp = send_request(req.prompt, decision.model_config, max_tokens=req.max_tokens)

    if not resp.ok:
        raise HTTPException(status_code=502, detail=f"Provider error: {resp.raw_error}")

    request_id = db.log_request(
        prompt=req.prompt,
        raw_tier=decision.raw_tier,
        routed_tier=decision.tier,
        confidence=decision.confidence,
        routed_model=decision.model_config.model_id,
        provider=decision.model_config.provider.value,
        input_tokens=resp.input_tokens,
        output_tokens=resp.output_tokens,
        cost=resp.cost,
        latency_ms=resp.latency_ms,
        ok=resp.ok,
    )

    if req.verify:
        background_tasks.add_task(
            verify_and_maybe_escalate,
            request_id=request_id,
            prompt=req.prompt,
            candidate_text=resp.text,
            candidate_model=decision.model_config,
            candidate_cost=resp.cost,
        )

    return CompletionResponse(
        text=resp.text,
        routed_model=decision.model_config.display_name,
        provider=decision.model_config.provider.value,
        tier=decision.tier,
        raw_tier=decision.raw_tier,
        classifier_confidence=decision.confidence,
        escalated_for_confidence=decision.escalated_for_confidence,
        routing_reason=decision.reason,
        input_tokens=resp.input_tokens,
        output_tokens=resp.output_tokens,
        cost=resp.cost,
        latency_ms=resp.latency_ms,
        request_id=request_id,
    )


# ---------------------------------------------------------------------------
# GET /v1/models
# ---------------------------------------------------------------------------
@app.get("/v1/models")
def list_models():
    return {
        name: {
            "provider": cfg.provider.value,
            "model_id": cfg.model_id,
            "display_name": cfg.display_name,
            "cost_per_input_token": cfg.cost_per_input_token,
            "cost_per_output_token": cfg.cost_per_output_token,
            "avg_latency_ms": cfg.avg_latency_ms,
            "quality_tier": cfg.quality_tier.value,
        }
        for name, cfg in MODEL_REGISTRY.items()
    }


# ---------------------------------------------------------------------------
# GET /v1/stats
# ---------------------------------------------------------------------------
@app.get("/v1/stats")
def stats():
    with db.get_conn() as conn:
        rows = conn.execute(
            "SELECT routed_model, provider, cost, input_tokens, output_tokens, ok "
            "FROM requests"
        ).fetchall()

    total_requests = len(rows)
    total_cost = sum(r["cost"] for r in rows)
    total_input = sum(r["input_tokens"] for r in rows)
    total_output = sum(r["output_tokens"] for r in rows)

    # "What would it have cost if everything went to the most expensive model"
    ceiling_model = max(MODEL_REGISTRY.values(), key=lambda m: m.cost_per_output_token)
    ceiling_cost = ceiling_model.estimate_cost(total_input, total_output)

    savings_pct = 0.0
    if ceiling_cost > 0:
        savings_pct = max(0.0, (ceiling_cost - total_cost) / ceiling_cost * 100)

    distribution: dict[str, int] = {}
    for r in rows:
        distribution[r["routed_model"]] = distribution.get(r["routed_model"], 0) + 1

    with db.get_conn() as conn:
        v_rows = conn.execute("SELECT escalated, passed FROM verifications").fetchall()
    escalation_rate = (
        sum(v["escalated"] for v in v_rows) / len(v_rows) * 100 if v_rows else 0.0
    )
    avg_quality = None
    with db.get_conn() as conn:
        q = conn.execute("SELECT AVG(quality_score) as avg_q FROM verifications").fetchone()
        avg_quality = q["avg_q"]

    return {
        "total_requests": total_requests,
        "total_cost_usd": round(total_cost, 4),
        "ceiling_cost_usd_if_all_gpt4o": round(ceiling_cost, 4),
        "cost_savings_pct": round(savings_pct, 1),
        "routing_distribution": distribution,
        "escalation_rate_pct": round(escalation_rate, 1),
        "avg_verifier_quality_score": round(avg_quality, 2) if avg_quality else None,
    }


# ---------------------------------------------------------------------------
# PUT /v1/routing-config
# ---------------------------------------------------------------------------
class RoutingConfigUpdate(BaseModel):
    routing_map: dict[int, str]
    verifier_model: Optional[str] = None
    low_confidence_threshold: Optional[float] = None
    escalate_on_low_confidence: Optional[bool] = None


@app.put("/v1/routing-config")
def update_routing_config(update: RoutingConfigUpdate):
    for model_name in update.routing_map.values():
        try:
            get_model(model_name)
        except KeyError:
            raise HTTPException(status_code=400, detail=f"Unknown model: {model_name}")

    with CONFIG_PATH.open() as f:
        config = yaml.safe_load(f)

    config["routing_map"] = update.routing_map
    if update.verifier_model:
        get_model(update.verifier_model)  # validate
        config["verifier_model"] = update.verifier_model
    if update.low_confidence_threshold is not None:
        config["low_confidence_threshold"] = update.low_confidence_threshold
    if update.escalate_on_low_confidence is not None:
        config["escalate_on_low_confidence"] = update.escalate_on_low_confidence

    with CONFIG_PATH.open("w") as f:
        yaml.safe_dump(config, f)

    router.reload()
    return {"status": "updated", "config": config}


@app.get("/health")
def health():
    return {"status": "ok"}
