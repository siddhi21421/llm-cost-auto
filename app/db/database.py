"""
SQLite-backed audit trail. One row per request, plus a second table for
verification/escalation events and a third for classifier retraining
feedback. Kept intentionally simple (raw sqlite3, no ORM) since the
whole point of this table is to be trivially queryable for the dashboard.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).resolve().parent.parent.parent / "cost_autopilot.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    prompt_hash TEXT NOT NULL,
    raw_tier INTEGER NOT NULL,
    routed_tier INTEGER NOT NULL,
    classifier_confidence REAL NOT NULL,
    routed_model TEXT NOT NULL,
    provider TEXT NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    cost REAL NOT NULL,
    latency_ms REAL NOT NULL,
    ok INTEGER NOT NULL,
    error TEXT
);

CREATE TABLE IF NOT EXISTS verifications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id INTEGER NOT NULL,
    timestamp REAL NOT NULL,
    verifier_model TEXT NOT NULL,
    quality_score REAL NOT NULL,
    passed INTEGER NOT NULL,
    escalated INTEGER NOT NULL,
    escalated_model TEXT,
    escalated_cost REAL,
    cost_delta REAL,
    FOREIGN KEY (request_id) REFERENCES requests(id)
);

CREATE TABLE IF NOT EXISTS classifier_feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prompt TEXT NOT NULL,
    original_tier INTEGER NOT NULL,
    corrected_tier INTEGER NOT NULL,
    created_at REAL NOT NULL,
    used_in_retrain INTEGER DEFAULT 0
);
"""


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.executescript(SCHEMA)


def hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def log_request(
    prompt: str,
    raw_tier: int,
    routed_tier: int,
    confidence: float,
    routed_model: str,
    provider: str,
    input_tokens: int,
    output_tokens: int,
    cost: float,
    latency_ms: float,
    ok: bool,
    error: Optional[str] = None,
) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO requests
               (timestamp, prompt_hash, raw_tier, routed_tier, classifier_confidence,
                routed_model, provider, input_tokens, output_tokens, cost, latency_ms,
                ok, error)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                time.time(), hash_prompt(prompt), raw_tier, routed_tier, confidence,
                routed_model, provider, input_tokens, output_tokens, cost, latency_ms,
                int(ok), error,
            ),
        )
        return cur.lastrowid


def log_verification(
    request_id: int,
    verifier_model: str,
    quality_score: float,
    passed: bool,
    escalated: bool,
    escalated_model: Optional[str] = None,
    escalated_cost: Optional[float] = None,
    cost_delta: Optional[float] = None,
):
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO verifications
               (request_id, timestamp, verifier_model, quality_score, passed,
                escalated, escalated_model, escalated_cost, cost_delta)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                request_id, time.time(), verifier_model, quality_score, int(passed),
                int(escalated), escalated_model, escalated_cost, cost_delta,
            ),
        )


def log_feedback(prompt: str, original_tier: int, corrected_tier: int):
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO classifier_feedback
               (prompt, original_tier, corrected_tier, created_at)
               VALUES (?,?,?,?)""",
            (prompt, original_tier, corrected_tier, time.time()),
        )


def get_unused_feedback():
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM classifier_feedback WHERE used_in_retrain = 0"
        ).fetchall()
        return [dict(r) for r in rows]


def mark_feedback_used(ids: list[int]):
    if not ids:
        return
    with get_conn() as conn:
        conn.executemany(
            "UPDATE classifier_feedback SET used_in_retrain = 1 WHERE id = ?",
            [(i,) for i in ids],
        )
