"""
This is the single entry point the rest of the codebase calls:

    from app.providers import send_request
    response = send_request(prompt, model_config)

It dispatches to the right provider module based on model_config.provider,
so routing/verification/api code never has to know OpenAI from Anthropic
from Ollama.
"""
from __future__ import annotations

from app.models.registry import ModelConfig, Provider
from app.providers.base import Response
from app.providers import anthropic_provider, openai_provider, ollama_provider

_DISPATCH = {
    Provider.ANTHROPIC: anthropic_provider.send_request,
    Provider.OPENAI: openai_provider.send_request,
    Provider.OLLAMA: ollama_provider.send_request,
}


def send_request(prompt: str, model_config: ModelConfig, max_tokens: int = 1024) -> Response:
    fn = _DISPATCH.get(model_config.provider)
    if fn is None:
        raise ValueError(f"No provider implementation for {model_config.provider}")
    return fn(prompt, model_config, max_tokens=max_tokens)


__all__ = ["send_request", "Response"]
