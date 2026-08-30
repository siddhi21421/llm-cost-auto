"""
Feature extraction for the complexity classifier.

These features are deliberately cheap to compute (no LLM calls, no
embeddings) because they run synchronously on the hot path before
every routing decision. Fancier semantic features can be added later
as a v2, but v1 is regex/heuristics only, by design.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Dict

REASONING_WORDS = [
    "analyze", "compare", "evaluate", "argue", "justify", "critique",
    "synthesize", "why", "design", "strategy", "trade-off", "tradeoff",
    "reason", "infer", "recommend", "assess",
]
CREATIVE_WORDS = [
    "write a story", "poem", "creative", "imagine", "brainstorm",
    "invent", "compose", "narrative",
]
EXTRACTION_WORDS = [
    "extract", "reformat", "convert", "list the", "find the", "pull out",
]
CONSTRAINT_MARKERS = ["must", "should", "exactly", "no more than", "at least", "only", "format:"]


@dataclass
class PromptFeatures:
    token_count: int
    char_count: int
    has_reasoning_words: int
    has_creative_words: int
    has_extraction_words: int
    num_constraints: int
    has_context_provided: int      # e.g. prompt includes a quoted block / "given the following"
    output_format_complexity: int  # 0 plain text, 1 structured (bullets/list), 2 strict schema (json/table)
    num_sentences: int
    question_count: int

    def to_vector(self):
        return list(asdict(self).values())

    @staticmethod
    def feature_names():
        return list(PromptFeatures.__annotations__.keys())


def extract_features(prompt: str) -> PromptFeatures:
    text = prompt.lower()
    token_count = max(1, len(prompt.split()))
    char_count = len(prompt)

    has_reasoning = int(any(w in text for w in REASONING_WORDS))
    has_creative = int(any(w in text for w in CREATIVE_WORDS))
    has_extraction = int(any(w in text for w in EXTRACTION_WORDS))

    num_constraints = sum(text.count(marker) for marker in CONSTRAINT_MARKERS)

    has_context = int(
        bool(re.search(r"given the following|based on this|context:|:\s*['\"]", text))
        or ("'" in prompt and prompt.count("'") >= 2)
        or ('"' in prompt and prompt.count('"') >= 2)
    )

    if re.search(r"\bjson\b|\btable\b|\bschema\b|\bcsv\b", text):
        output_format_complexity = 2
    elif re.search(r"bullet|list|steps|numbered", text):
        output_format_complexity = 1
    else:
        output_format_complexity = 0

    num_sentences = max(1, len(re.findall(r"[.!?]+", prompt)))
    question_count = prompt.count("?")

    return PromptFeatures(
        token_count=token_count,
        char_count=char_count,
        has_reasoning_words=has_reasoning,
        has_creative_words=has_creative,
        has_extraction_words=has_extraction,
        num_constraints=num_constraints,
        has_context_provided=has_context,
        output_format_complexity=output_format_complexity,
        num_sentences=num_sentences,
        question_count=question_count,
    )
