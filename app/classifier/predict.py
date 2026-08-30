"""
Inference-time wrapper around the trained classifier. This is what the
router imports -- it never touches sklearn directly.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import numpy as np

from app.classifier.features import extract_features

MODEL_PATH = Path(__file__).resolve().parent / "classifier.pkl"

_model = None


def _load_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"No trained classifier found at {MODEL_PATH}. "
                "Run `python -m app.classifier.train` first."
            )
        _model = joblib.load(MODEL_PATH)
    return _model


def classify_complexity(prompt: str) -> int:
    """Returns 1 (simple), 2 (moderate), or 3 (complex)."""
    model = _load_model()
    features = extract_features(prompt).to_vector()
    pred = model.predict(np.array([features]))[0]
    return int(pred)


def classify_with_confidence(prompt: str):
    """Returns (tier, confidence) where confidence is the model's own
    probability estimate for the predicted class. Low-confidence
    predictions are a good signal to route conservatively (bump up a tier)
    rather than trust the classifier blindly."""
    model = _load_model()
    features = np.array([extract_features(prompt).to_vector()])
    proba = model.predict_proba(features)[0]
    pred_idx = int(np.argmax(proba))
    tier = int(model.classes_[pred_idx])
    confidence = float(proba[pred_idx])
    return tier, confidence
