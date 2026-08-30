import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.models.registry import MODEL_REGISTRY, get_model, QualityTier
from app.routing.router import Router, CONFIG_PATH


def test_registry_has_expected_models():
    for name in ["gpt-4o", "gpt-4o-mini", "claude-sonnet", "claude-haiku", "llama-local"]:
        assert name in MODEL_REGISTRY


def test_local_model_is_free():
    llama = get_model("llama-local")
    assert llama.cost_per_input_token == 0.0
    assert llama.cost_per_output_token == 0.0


def test_estimate_cost():
    gpt4o = get_model("gpt-4o")
    cost = gpt4o.estimate_cost(input_tokens=1000, output_tokens=500)
    expected = 1000 * gpt4o.cost_per_input_token + 500 * gpt4o.cost_per_output_token
    assert abs(cost - expected) < 1e-9


def test_router_routes_to_configured_tiers():
    router = Router(config_path=CONFIG_PATH)
    decision = router.route("Extract the total from this invoice.")
    assert decision.tier in (1, 2, 3)
    assert decision.model_config.model_id is not None
    assert "routed to" in decision.reason


def test_router_reload_picks_up_config_changes(tmp_path):
    # Point at a temp copy of the config so we don't mutate the real one.
    import shutil
    temp_config = tmp_path / "routing_config.yaml"
    shutil.copy(CONFIG_PATH, temp_config)

    router = Router(config_path=temp_config)
    original_tier1_model = router.routing_map[1]

    # Mutate the file directly and reload.
    import yaml
    with temp_config.open() as f:
        cfg = yaml.safe_load(f)
    cfg["routing_map"][1] = "claude-haiku"
    with temp_config.open("w") as f:
        yaml.safe_dump(cfg, f)

    router.reload()
    assert router.routing_map[1] == "claude-haiku"
    assert router.routing_map[1] != original_tier1_model or original_tier1_model == "claude-haiku"
