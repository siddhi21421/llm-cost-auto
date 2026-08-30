"""
Phase 6: realistic load test. Sends a batch of diverse prompts through the
live API (not the internal router directly) so the resulting numbers
reflect the exact same path a real caller would hit -- then prints the
cost-savings headline for the portfolio write-up.

Usage:
    # in one terminal:
    uvicorn app.api.main:app --reload

    # in another:
    python scripts/load_test.py --n 500 --url http://localhost:8000
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.generate_dataset import TIER1_TEMPLATES, TIER2_TEMPLATES, TIER3_TEMPLATES, fill

ALL_TEMPLATES = TIER1_TEMPLATES + TIER2_TEMPLATES + TIER3_TEMPLATES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--url", type=str, default="http://localhost:8000")
    parser.add_argument("--verify", action="store_true", help="Enable async verification for every request")
    args = parser.parse_args()

    ok, failed = 0, 0
    start = time.time()

    for i in range(args.n):
        prompt = fill(random.choice(ALL_TEMPLATES))
        try:
            resp = requests.post(
                f"{args.url}/v1/completions",
                json={"prompt": prompt, "verify": args.verify},
                timeout=30,
            )
            if resp.status_code == 200:
                ok += 1
            else:
                failed += 1
                print(f"[{i}] HTTP {resp.status_code}: {resp.text[:200]}")
        except requests.RequestException as e:
            failed += 1
            print(f"[{i}] request error: {e}")

        if (i + 1) % 50 == 0:
            print(f"...{i + 1}/{args.n} sent")

    elapsed = time.time() - start
    print(f"\nDone: {ok} ok, {failed} failed, {elapsed:.1f}s total")

    stats = requests.get(f"{args.url}/v1/stats", timeout=10).json()
    print("\n=== Cost Savings Report ===")
    print(f"Total requests logged: {stats['total_requests']}")
    print(f"Actual cost:           ${stats['total_cost_usd']}")
    print(f"Cost if all → GPT-4o:  ${stats['ceiling_cost_usd_if_all_gpt4o']}")
    print(f"COST SAVED:            {stats['cost_savings_pct']}%   <-- headline number")
    print(f"Routing distribution:  {stats['routing_distribution']}")
    print(f"Escalation rate:       {stats['escalation_rate_pct']}%")


if __name__ == "__main__":
    main()
