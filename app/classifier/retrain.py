"""
Phase 3 step 4: the feedback flywheel. Pulls accumulated routing-failure
feedback from SQLite, appends it to the labeled dataset, retrains the
classifier, and marks the feedback rows as consumed.

Intended to run on a schedule (cron / weekly), not on every request:

    0 3 * * 0 cd /app && python -m app.classifier.retrain
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.db import database as db
from app.classifier.train import DATA_PATH, main as retrain_main


def fold_in_feedback() -> int:
    feedback_rows = db.get_unused_feedback()
    if not feedback_rows:
        print("No new feedback since last retrain. Skipping.")
        return 0

    with DATA_PATH.open("a", newline="") as f:
        writer = csv.writer(f)
        for row in feedback_rows:
            writer.writerow([row["prompt"], row["corrected_tier"]])

    db.mark_feedback_used([r["id"] for r in feedback_rows])
    print(f"Folded {len(feedback_rows)} corrected examples into the training set.")
    return len(feedback_rows)


def main():
    db.init_db()
    n_new = fold_in_feedback()
    if n_new == 0:
        return
    print("Retraining classifier with updated dataset...")
    retrain_main()


if __name__ == "__main__":
    main()
