"""
Phase 4: cost and quality dashboard.

Run with:
    streamlit run dashboard/dashboard.py

Reads directly from the SQLite audit log (same DB the API writes to),
so this reflects live traffic with no separate sync step.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.express as px
import streamlit as st

from app.db import database as db
from app.models.registry import MODEL_REGISTRY

st.set_page_config(page_title="LLM Cost Autopilot", layout="wide")
st.title("💸 LLM Cost Autopilot")
st.caption("Cost and quality dashboard for the routing layer")

db.init_db()

with db.get_conn() as conn:
    requests_df = pd.read_sql_query("SELECT * FROM requests", conn)
    verifications_df = pd.read_sql_query("SELECT * FROM verifications", conn)

if requests_df.empty:
    st.info("No requests logged yet. Send some traffic through POST /v1/completions first.")
    st.stop()

requests_df["timestamp"] = pd.to_datetime(requests_df["timestamp"], unit="s")
requests_df["date"] = requests_df["timestamp"].dt.date

# ---------------------------------------------------------------------------
# Headline metric: cost saved vs. sending everything to the most expensive model
# ---------------------------------------------------------------------------
ceiling_model = max(MODEL_REGISTRY.values(), key=lambda m: m.cost_per_output_token)
total_input = requests_df["input_tokens"].sum()
total_output = requests_df["output_tokens"].sum()
actual_cost = requests_df["cost"].sum()
ceiling_cost = ceiling_model.estimate_cost(total_input, total_output)
savings_pct = max(0.0, (ceiling_cost - actual_cost) / ceiling_cost * 100) if ceiling_cost else 0.0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total requests", f"{len(requests_df):,}")
col2.metric("Actual cost", f"${actual_cost:.4f}")
col3.metric(f"Cost if all → {ceiling_model.display_name}", f"${ceiling_cost:.4f}")
col4.metric("💰 Cost saved", f"{savings_pct:.1f}%", delta=f"-${ceiling_cost - actual_cost:.4f}")

st.divider()

# ---------------------------------------------------------------------------
# Cost over time
# ---------------------------------------------------------------------------
left, right = st.columns(2)

with left:
    st.subheader("Cost per day")
    daily_cost = requests_df.groupby("date")["cost"].sum().reset_index()
    fig = px.bar(daily_cost, x="date", y="cost", labels={"cost": "Cost ($)"})
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Routing distribution")
    dist = requests_df["routed_model"].value_counts().reset_index()
    dist.columns = ["model", "count"]
    fig = px.pie(dist, names="model", values="count", hole=0.4)
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Quality + escalation
# ---------------------------------------------------------------------------
left2, right2 = st.columns(2)

with left2:
    st.subheader("Quality score distribution (LLM-as-judge)")
    if not verifications_df.empty:
        fig = px.histogram(verifications_df, x="quality_score", nbins=5, range_x=[0, 5])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No verification data yet.")

with right2:
    st.subheader("Escalation rate over time")
    if not verifications_df.empty:
        verifications_df["timestamp"] = pd.to_datetime(verifications_df["timestamp"], unit="s")
        verifications_df["date"] = verifications_df["timestamp"].dt.date
        esc = verifications_df.groupby("date")["escalated"].mean().reset_index()
        esc["escalated"] *= 100
        fig = px.line(esc, x="date", y="escalated", markers=True, labels={"escalated": "Escalation rate (%)"})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No verification data yet.")

st.divider()

# ---------------------------------------------------------------------------
# Raw table
# ---------------------------------------------------------------------------
st.subheader("Recent requests")
st.dataframe(
    requests_df.sort_values("timestamp", ascending=False)
    .head(50)[
        ["timestamp", "routed_model", "provider", "routed_tier", "classifier_confidence",
         "input_tokens", "output_tokens", "cost", "latency_ms", "ok"]
    ],
    use_container_width=True,
)
