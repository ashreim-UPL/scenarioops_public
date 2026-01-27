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

st.set_page_config(page_title="EBE Rank", page_icon="E", layout="wide")
run_id = resolve_run_id()
page_header("EBE Rank", run_id, subtitle="Evidence x Business impact x Emergence")

payload = load_artifact(run_id, "forces_ranked")
if not payload:
    placeholder_section("Ranking Summary", ["Top forces by EBE score", "Coverage by layer"])
    st.stop()

forces = payload.get("forces", []) if isinstance(payload, dict) else []
forces_catalog = load_artifact(run_id, "forces") or {}
force_rows = forces_catalog.get("forces", []) if isinstance(forces_catalog, dict) else []
force_meta = {
    item.get("force_id"): {
        "label": item.get("label"),
        "domain": item.get("domain"),
        "layer": item.get("layer"),
    }
    for item in force_rows
    if isinstance(item, dict)
}
coverage = payload.get("coverage_stats", {}) if isinstance(payload, dict) else {}
coverage_layers = coverage.get("by_layer", {}) if isinstance(coverage, dict) else {}

card_grid(
    [
        ("Ranked Forces", str(len(forces))),
        ("Linked Forces", str(coverage.get("linked_forces", ""))),
        ("Coverage Ratio", str(coverage.get("linked_ratio", ""))),
    ]
)

if coverage_layers:
    section("Coverage by Layer", "Distribution of linked forces across layers.")
    st.table(
        {
            "layer": list(coverage_layers.keys()),
            "count": list(coverage_layers.values()),
        }
    )

if not forces:
    st.stop()

df = pd.DataFrame(forces)
if not df.empty:
    df["B_business_impact"] = pd.to_numeric(df.get("B_business_impact"), errors="coerce")
    df["E_emergence"] = pd.to_numeric(df.get("E_emergence"), errors="coerce")
    df["ebe_score"] = pd.to_numeric(df.get("ebe_score"), errors="coerce")
    df["label"] = df["force_id"].map(lambda fid: (force_meta.get(fid) or {}).get("label"))
    df["domain"] = df["force_id"].map(lambda fid: (force_meta.get(fid) or {}).get("domain"))
    top_labels = (
        df.sort_values(by="ebe_score", ascending=False)
        .head(10)["label"]
        .dropna()
        .tolist()
    )
    df["label_display"] = df["label"].where(df["label"].isin(top_labels), "")
    fig = px.scatter(
        df,
        x="B_business_impact",
        y="E_emergence",
        size="ebe_score",
        color="domain",
        text="label_display",
        hover_data=["force_id", "rationale"],
        size_max=70,
    )
    fig.update_traces(textposition="top center", opacity=0.75)
    fig.update_layout(
        height=460,
        xaxis_title="Business impact (local)",
        yaxis_title="Emergence (global/structural)",
        showlegend=True,
    )
    st.plotly_chart(fig, use_container_width=True)
    top_bar = df.sort_values(by="ebe_score", ascending=False).head(10)
    bar_fig = px.bar(
        top_bar,
        x="ebe_score",
        y="label",
        color="domain",
        orientation="h",
        title="Top 10 EBE scores",
    )
    bar_fig.update_layout(height=420, yaxis_title="", xaxis_title="EBE score")
    st.plotly_chart(bar_fig, use_container_width=True)
if "ebe_score" in df.columns:
    df = df.sort_values(by="ebe_score", ascending=False)

columns = [c for c in ["force_id", "ebe_score", "B_business_impact", "E_emergence", "E_evidence_strength", "rationale"] if c in df.columns]
section("Top Forces", "Highest leverage forces ranked by EBE scoring.")
st.dataframe(df[columns].head(20), height=420)
