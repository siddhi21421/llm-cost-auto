import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.classifier.features import extract_features
from app.classifier.predict import classify_complexity, classify_with_confidence


def test_feature_extraction_basic():
    f = extract_features("Extract the name from: 'John Smith, age 30.'")
    assert f.has_extraction_words == 1
    assert f.has_context_provided == 1


def test_feature_extraction_reasoning():
    f = extract_features("Analyze the trade-offs and recommend a strategy.")
    assert f.has_reasoning_words == 1


def test_classify_simple_prompt():
    tier = classify_complexity("Extract the date from: 'Meeting on May 3rd.'")
    assert tier in (1, 2)  # simple extraction should not land in tier 3


def test_classify_complex_prompt():
    tier = classify_complexity(
        "Critique the reasoning in this argument and propose a stronger "
        "counterargument, weighing at least two trade-offs."
    )
    assert tier == 3


def test_classify_with_confidence_returns_valid_range():
    tier, confidence = classify_with_confidence("What is the capital of Japan?")
    assert tier in (1, 2, 3)
    assert 0.0 <= confidence <= 1.0
