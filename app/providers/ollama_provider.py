from __future__ import annotations

import os

import requests

from app.models.registry import ModelConfig
from app.providers.base import Response, estimate_tokens, timed

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


@timed
def send_request(prompt: str, model_config: ModelConfig, max_tokens: int = 1024) -> Response:
    try:
        resp = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={"model": model_config.model_id, "prompt": prompt, "stream": False},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data.get("response", "")
        # Ollama returns exact counts when available; fall back to estimation.
        input_tokens = data.get("prompt_eval_count") or estimate_tokens(prompt)
        output_tokens = data.get("eval_count") or estimate_tokens(text)
        cost = model_config.estimate_cost(input_tokens, output_tokens)  # $0 for local
        return Response(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=0.0,
            cost=cost,
            model_id=model_config.model_id,
            provider="ollama",
        )
    except Exception as e:  # noqa: BLE001
        return Response(
            text="",
            input_tokens=0,
            output_tokens=0,
            latency_ms=0.0,
            cost=0.0,
            model_id=model_config.model_id,
            provider="ollama",
            raw_error=str(e),
        )
