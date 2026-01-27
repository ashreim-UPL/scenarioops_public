from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

import pandas as pd
import plotly.express as px
import streamlit as st

from scenarioops.ui.page_utils import (
    card_grid,
    load_artifact,
    page_header,
    placeholder_section,
    resolve_run_id,
    section,
)

st.set_page_config(page_title="Clusters", page_icon="C", layout="wide")
run_id = resolve_run_id()
page_header("Clusters", run_id, subtitle="Force clustering and system dynamics")

payload = load_artifact(run_id, "clusters")
if not payload:
    placeholder_section("Cluster Summary", ["Cluster labels", "Coherence scores", "Underlying dynamics"])
    st.stop()

clusters = payload.get("clusters", []) if isinstance(payload, dict) else []
card_grid(
    [
        ("Clusters Identified", str(len(clusters))),
        ("Company", str(payload.get("company_name", ""))),
        ("Horizon (months)", str(payload.get("horizon_months", ""))),
    ]
)

if not clusters:
    st.stop()

df = pd.DataFrame(clusters)
columns = [c for c in ["cluster_id", "cluster_label", "coherence_score", "centroid_summary", "underlying_dynamic"] if c in df.columns]
section("Cluster Landscape", "Groupings of forces that move together.")
chart_df = df.copy()
if not chart_df.empty:
    chart_df["force_count"] = chart_df["force_ids"].apply(lambda items: len(items) if isinstance(items, list) else 0)
    chart_df["coherence_score"] = pd.to_numeric(chart_df.get("coherence_score"), errors="coerce")
    fig = px.treemap(
        chart_df,
        path=["cluster_label"],
        values="force_count",
        color="coherence_score",
        hover_data=["cluster_id", "centroid_summary", "underlying_dynamic"],
        color_continuous_scale="Teal",
    )
    fig.update_layout(
        height=520,
        margin=dict(t=10, l=10, r=10, b=10),
    )
    st.plotly_chart(fig, width="stretch")
st.dataframe(df[columns], height=420)
