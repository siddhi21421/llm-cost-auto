"""
Generates the labeled training set for the complexity classifier.

Tier 1 (simple):   reformatting, extraction, basic Q&A from provided context
Tier 2 (moderate): summarization, classification, structured analysis
Tier 3 (complex):  multi-step reasoning, creative generation, nuanced judgment

Templates are filled with varied subjects so the dataset has real lexical
diversity rather than 200 near-duplicates of the same sentence. Output:
app/classifier/data/labeled_prompts.csv
"""
from __future__ import annotations

import csv
import itertools
import random
from pathlib import Path

random.seed(7)

SUBJECTS = [
    "the Q3 sales report", "this customer support ticket", "the attached resume",
    "our marketing email", "the product review", "this legal clause",
    "the meeting transcript", "the user's feedback", "this GitHub issue",
    "the earnings call", "the recipe", "this job posting", "the flight itinerary",
    "the invoice", "the research abstract", "this news article",
    "the onboarding doc", "the churn data", "the survey responses",
    "this API changelog",
]

NAMES_DATES = [
    "'Call with Priya on April 2nd at 3pm.'",
    "'Invoice #4521 due June 15, 2026, from Acme Corp.'",
    "'John Smith, born 1990-04-12, joined on 2021-03-01.'",
    "'Order shipped March 9th to 221B Baker Street.'",
    "'Contract signed by Maria Lopez on 2026-01-20.'",
]

TIER1_TEMPLATES = [
    "Extract the name and date from: {nd}",
    "Reformat {subj} into a bulleted list with no other changes.",
    "Convert this into title case: {subj}",
    "What is the capital of {country}?",
    "Given the following text, pull out just the email address: {nd}",
    "List the fields present in {subj} without interpreting them.",
    "Rewrite this sentence in the passive voice: 'The team shipped the release.'",
    "What is 15% of 240?",
    "Find the total dollar amount mentioned in: {nd}",
    "Convert '3pm EST' to UTC.",
]

COUNTRIES = ["France", "Japan", "Brazil", "Kenya", "Canada", "Egypt", "Chile"]

TIER2_TEMPLATES = [
    "Summarize {subj} in two sentences.",
    "Classify the sentiment of {subj} as positive, negative, or neutral.",
    "Categorize {subj} into one of: urgent, normal, low priority.",
    "Provide a structured analysis of {subj} covering key metrics and one risk.",
    "Summarize the main complaint in {subj} and suggest one fix.",
    "Tag {subj} with up to three relevant keywords.",
    "Break {subj} down into a short table of key points and values.",
    "Identify whether {subj} requires follow-up action, and why, in one paragraph.",
    "Compare two options from {subj} on a single dimension: cost.",
    "Extract and summarize the top 3 takeaways from {subj}.",
]

TIER3_TEMPLATES = [
    "Analyze {subj} and recommend a strategy, weighing at least two trade-offs.",
    "Write a short creative story opening inspired by {subj}.",
    "Compare the pros and cons of two competing approaches to {subj}, and justify a recommendation.",
    "Critique the reasoning in {subj} and propose a stronger argument.",
    "Given {subj}, design a step-by-step plan to address the underlying problem, explaining your reasoning at each step.",
    "Argue for or against a claim found in {subj}, addressing the strongest counterargument.",
    "Synthesize insights across {subj} and a competitor's data to infer a market trend.",
    "Brainstorm five creative and non-obvious solutions to the problem described in {subj}.",
    "Evaluate the trade-offs between speed and accuracy for the process in {subj}, and recommend a threshold.",
    "Explain why {subj} might be misleading, considering statistical and contextual nuance.",
]


def fill(template: str) -> str:
    return (
        template.replace("{subj}", random.choice(SUBJECTS))
        .replace("{nd}", random.choice(NAMES_DATES))
        .replace("{country}", random.choice(COUNTRIES))
    )


def generate(n_per_tier: int = 70):
    rows = []
    for tier, templates in [(1, TIER1_TEMPLATES), (2, TIER2_TEMPLATES), (3, TIER3_TEMPLATES)]:
        combos = list(itertools.product(templates, range(n_per_tier)))
        random.shuffle(combos)
        seen = set()
        count = 0
        for template, _ in combos:
            prompt = fill(template)
            if prompt in seen:
                continue
            seen.add(prompt)
            rows.append({"prompt": prompt, "tier": tier})
            count += 1
            if count >= n_per_tier:
                break
    random.shuffle(rows)
    return rows


def main():
    rows = generate(n_per_tier=70)  # 210 total, comfortably above the 200+ target
    out_path = (
        Path(__file__).resolve().parent.parent
        / "app" / "classifier" / "data" / "labeled_prompts.csv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["prompt", "tier"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} labeled prompts to {out_path}")


if __name__ == "__main__":
    main()
