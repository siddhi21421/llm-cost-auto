"""
Phase 1 validation: send the same set of prompts to every model in the
registry, log outputs/cost/latency, and confirm the abstraction layer
behaves identically regardless of provider.

Usage:
    python scripts/baseline_test.py

Requires the relevant API keys to be set as environment variables
(ANTHROPIC_API_KEY, OPENAI_API_KEY) and, for the local model, an Ollama
server running with `ollama pull llama3.1:8b` already done.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.models.registry import MODEL_REGISTRY
from app.providers import send_request

TEST_PROMPTS = [
    "Extract the name and date from: 'Meeting with Sarah on March 3rd.'",
    "Reformat this as bullet points: apples, bananas, and oranges are fruits.",
    "What is the capital of France?",
    "Summarize in one sentence: The stock market fell 2% today amid inflation fears.",
    "Classify the sentiment of this review as positive, negative, or neutral: 'It was okay, nothing special.'",
    "Compare the pros and cons of remote work vs office work in 3 bullet points each.",
    "Write a short creative opening line for a mystery novel.",
    "Given this data, is there a statistically meaningful trend? [12, 15, 14, 19, 22, 25]",
    "Explain the tradeoffs between SQL and NoSQL databases for a startup choosing its first database.",
    "Draft a two-sentence apology email for a shipping delay.",
]


def main():
    results = []
    for name, config in MODEL_REGISTRY.items():
        print(f"\n=== {config.display_name} ({config.provider.value}) ===")
        for prompt in TEST_PROMPTS:
            resp = send_request(prompt, config)
            status = "OK" if resp.ok else f"ERROR: {resp.raw_error}"
            print(f"  [{status}] cost=${resp.cost:.6f} latency={resp.latency_ms:.0f}ms")
            results.append(
                {
                    "model": name,
                    "provider": config.provider.value,
                    "prompt": prompt,
                    "output": resp.text,
                    "input_tokens": resp.input_tokens,
                    "output_tokens": resp.output_tokens,
                    "cost": resp.cost,
                    "latency_ms": resp.latency_ms,
                    "ok": resp.ok,
                    "error": resp.raw_error,
                }
            )

    out_path = Path(__file__).resolve().parent.parent / "baseline_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {len(results)} results to {out_path}")


if __name__ == "__main__":
    main()
