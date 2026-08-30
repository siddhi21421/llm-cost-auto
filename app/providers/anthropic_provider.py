from __future__ import annotations

import os

from app.models.registry import ModelConfig
from app.providers.base import Response, ProviderError, timed

try:
    import anthropic
except ImportError:  # pragma: no cover
    anthropic = None

_client = None


def _get_client():
    global _client
    if _client is None:
        if anthropic is None:
            raise ProviderError(
                "anthropic package not installed. Run: pip install anthropic"
            )
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ProviderError("ANTHROPIC_API_KEY environment variable not set")
        _client = anthropic.Anthropic(api_key=api_key)
    return _client


@timed
def send_request(prompt: str, model_config: ModelConfig, max_tokens: int = 1024) -> Response:
    try:
        client = _get_client()
        message = client.messages.create(
            model=model_config.model_id,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(
            block.text for block in message.content if getattr(block, "type", None) == "text"
        )
        input_tokens = message.usage.input_tokens
        output_tokens = message.usage.output_tokens
        cost = model_config.estimate_cost(input_tokens, output_tokens)
        return Response(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=0.0,  # filled in by @timed
            cost=cost,
            model_id=model_config.model_id,
            provider="anthropic",
        )
    except Exception as e:  # noqa: BLE001 - deliberately broad: this is a network boundary
        return Response(
            text="",
            input_tokens=0,
            output_tokens=0,
            latency_ms=0.0,
            cost=0.0,
            model_id=model_config.model_id,
            provider="anthropic",
            raw_error=str(e),
        )
