"""
Trains the complexity classifier (Tier 1 / 2 / 3) on the labeled dataset
and saves the fitted model to classifier.pkl.

Usage:
    python -m app.classifier.train
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.classifier.features import extract_features, PromptFeatures

DATA_PATH = Path(__file__).resolve().parent / "data" / "labeled_prompts.csv"
MODEL_PATH = Path(__file__).resolve().parent / "classifier.pkl"


def load_dataset():
    prompts, labels = [], []
    with DATA_PATH.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompts.append(row["prompt"])
            labels.append(int(row["tier"]))
    return prompts, labels


def build_feature_matrix(prompts):
    return np.array([extract_features(p).to_vector() for p in prompts])


def main():
    prompts, labels = load_dataset()
    print(f"Loaded {len(prompts)} labeled examples")

    X = build_feature_matrix(prompts)
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\nHeld-out accuracy: {acc:.2%}")
    print("\nConfusion matrix (rows=true, cols=predicted), tiers [1,2,3]:")
    print(confusion_matrix(y_test, preds, labels=[1, 2, 3]))
    print("\nClassification report:")
    print(classification_report(y_test, preds, labels=[1, 2, 3]))

    feature_importance = sorted(
        zip(PromptFeatures.feature_names(), clf.feature_importances_),
        key=lambda x: -x[1],
    )
    print("Feature importance:")
    for name, imp in feature_importance:
        print(f"  {name:28s} {imp:.3f}")

    joblib.dump(clf, MODEL_PATH)
    print(f"\nSaved model to {MODEL_PATH}")

    if acc < 0.80:
        print(
            "\nWARNING: accuracy is below the 80% v1 bar. Consider adding more "
            "distinguishing features or more labeled examples before shipping "
            "this classifier behind the router."
        )


if __name__ == "__main__":
    main()
