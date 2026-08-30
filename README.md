# LLM Cost Autopilot

An intelligent routing layer that sits in front of multiple LLM providers,
classifies the complexity of each incoming request, routes it to the
cheapest model capable of handling it, and continuously verifies that the
routing decision was actually correct — auto-escalating and retraining
itself when it wasn't.

> **Headline result (750-request simulated load test):**
> **64.5% cost reduction** vs. sending every request to the top-tier model,
> with a 4.14/5 average quality score from the LLM-as-judge verifier and a
> 22.4% escalation rate on requests the cheap model got meaningfully wrong.

## Why this exists

Every company running LLMs in production is over-provisioning: sending
simple extraction and reformatting tasks to the same frontier model that
handles genuinely hard reasoning. That's a business problem as much as a
technical one. This system treats cost as a first-class optimization
target — not by degrading quality, but by matching request complexity to
the cheapest model that can actually do the job, and catching it
automatically on the rare occasions that fails.

## Architecture

```
                    ┌──────────────────────────────────────────┐
                    │              POST /v1/completions          │
                    └──────────────────┬───────────────────────┘
                                        ▼
                         ┌────────────────────────────┐
                         │   Complexity Classifier      │
                         │  (RandomForest, 10 features) │
                         │  Tier 1 / 2 / 3 + confidence  │
                         └──────────────┬───────────────┘
                                        ▼
                         ┌────────────────────────────┐
                         │   Router (routing_config.yaml)│
                         │  tier → model, w/ low-confidence│
                         │  escalation                     │
                         └──────────────┬───────────────┘
                                        ▼
              ┌──────────────────────────────────────────────┐
              │       Unified Provider Interface                │
              │  OpenAI  │  Anthropic  │  Ollama (local)         │
              │        -> standardized Response object            │
              └──────────────────────┬───────────────────────┘
                                      ▼
                     response returned to caller immediately
                                      │
                                      ▼ (background task)
                     ┌─────────────────────────────────────┐
                     │   Async Quality Verifier                │
                     │  re-runs prompt on top-tier model,      │
                     │  LLM-as-judge scores agreement           │
                     │  auto-escalates on failure                │
                     └──────────────────┬────────────────────┘
                                        ▼
                     ┌─────────────────────────────────────┐
                     │  SQLite audit log (requests,             │
                     │  verifications, classifier_feedback)      │
                     └──────────────────┬────────────────────┘
                            │                        │
                            ▼                        ▼
                 Streamlit dashboard       Weekly classifier retrain
                 (cost, quality,             (folds routing failures
                 routing distribution)       back into training data)
```

## Project structure

```
app/
  models/registry.py        # ModelConfig dataclass + pricing for 5 models
  providers/                # unified send_request() across OpenAI/Anthropic/Ollama
  classifier/
    features.py              # 10 cheap heuristic features, no embeddings
    train.py                 # trains RandomForest on labeled_prompts.csv
    predict.py                # inference wrapper used by the router
    retrain.py                 # weekly feedback flywheel
    data/labeled_prompts.csv    # 210 labeled examples across 3 tiers
  routing/
    router.py                  # classifies + picks model + handles low-confidence escalation
    routing_config.yaml         # tier -> model map, hot-swappable via API
  verification/verifier.py     # async LLM-as-judge quality check + auto-escalation
  db/database.py                # SQLite schema + logging
  api/main.py                    # FastAPI service
dashboard/dashboard.py           # Streamlit cost/quality dashboard
scripts/
  generate_dataset.py             # builds the labeled training set
  baseline_test.py                 # Phase 1 sanity check across all providers
  load_test.py                      # Phase 6 load test against the live API
tests/                                # pytest unit tests for classifier + router
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in ANTHROPIC_API_KEY / OPENAI_API_KEY

# train the classifier (already checked in, but you'll want to retrain
# once you've reviewed/expanded the labeled dataset)
python -m app.classifier.train

# run the API
uvicorn app.api.main:app --reload

# in another terminal: the dashboard
streamlit run dashboard/dashboard.py
```

Or with Docker:

```bash
docker compose up --build
```

Local models (tier 1) go through [Ollama](https://ollama.com) — install it
and run `ollama pull llama3.1:8b` before sending traffic, or remap tier 1
to a hosted model in `routing_config.yaml` if you'd rather skip that.

## Using the API

```bash
curl -X POST localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Extract the invoice total from: Total due: $482.10"}'
```

The response tells you exactly what happened and why:

```json
{
  "text": "...",
  "routed_model": "Llama 3.1 8B (local)",
  "tier": 1,
  "classifier_confidence": 0.98,
  "routing_reason": "classified as tier 1 (confidence 98%) -> routed to Llama 3.1 8B (local)",
  "cost": 0.0,
  "latency_ms": 340,
  "request_id": 42
}
```

Other endpoints:
- `GET /v1/models` — registry with live pricing
- `GET /v1/stats` — cost savings summary (what the dashboard reads)
- `PUT /v1/routing-config` — hot-swap the tier → model map without redeploying

## How the classifier works

Ten cheap, synchronous features (token count, presence of reasoning/creative
/extraction language, number of constraints, output format complexity,
etc.) feed a RandomForest classifier — no embeddings or LLM calls on the
hot path, since this runs before every routed request. On the 210-example
labeled set it holds **95% held-out accuracy**, well above the 80% v1 bar.

Low-confidence predictions (below 55% by default) are escalated one tier
rather than trusted outright — a cheap insurance policy against the
classifier's own uncertainty.

## How the verification/escalation loop works

Every routed response is queued for async re-evaluation: the same prompt
is re-run against the highest-tier model, and an LLM-as-judge call scores
whether the cheap model's answer was substantively equivalent (1–5 scale,
4+ passes). Failures are:

1. Logged with the cost delta between what was spent and what the
   escalated call would have cost.
2. Optionally auto-escalated in real time if the reference call still fits
   the latency budget.
3. Fed back into `classifier_feedback` as a corrected training example.

`python -m app.classifier.retrain` (intended to run weekly via cron) folds
accumulated feedback into the training set and retrains — the flywheel
that makes the router's classifier get better as it sees more traffic.

## Results from the load test

750 diverse prompts run through the live API (simulated model responses —
see `scripts/load_test.py` for how to point this at real provider keys):

| Metric | Value |
|---|---|
| Total requests | 750 |
| Actual cost | $0.64 |
| Cost if everything went to GPT-4o | $1.80 |
| **Cost saved** | **64.5%** |
| Routing distribution | 37% local (free), 28% mid-tier, 35% top-tier |
| Escalation rate | 22.4% |
| Avg. verifier quality score | 4.14 / 5 |

## Known limitations / what a v2 would add

- The classifier uses hand-built heuristic features, not embeddings —
  good enough for a v1 routing skeleton, but a small fine-tuned classifier
  would likely push accuracy higher on genuinely ambiguous prompts.
- The verifier re-runs the full prompt on the reference model for every
  request, which itself costs money — in production you'd sample
  (e.g. verify 10–20% of traffic) rather than verify everything.
- Latency-aware escalation currently checks against a single budget; a
  real system would want per-endpoint SLAs.
