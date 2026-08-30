from __future__ import annotations

import os

from app.models.registry import ModelConfig
from app.providers.base import Response, ProviderError, timed

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None

_client = None


def _get_client():
    global _client
    if _client is None:
        if OpenAI is None:
            raise ProviderError("openai package not installed. Run: pip install openai")
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ProviderError("OPENAI_API_KEY environment variable not set")
        _client = OpenAI(api_key=api_key)
    return _client


@timed
def send_request(prompt: str, model_config: ModelConfig, max_tokens: int = 1024) -> Response:
    try:
        client = _get_client()
        completion = client.chat.completions.create(
            model=model_config.model_id,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        text = completion.choices[0].message.content or ""
        input_tokens = completion.usage.prompt_tokens
        output_tokens = completion.usage.completion_tokens
        cost = model_config.estimate_cost(input_tokens, output_tokens)
        return Response(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=0.0,
            cost=cost,
            model_id=model_config.model_id,
            provider="openai",
        )
    except Exception as e:  # noqa: BLE001
        return Response(
            text="",
            input_tokens=0,
            output_tokens=0,
            latency_ms=0.0,
            cost=0.0,
            model_id=model_config.model_id,
            provider="openai",
            raw_error=str(e),
        )
