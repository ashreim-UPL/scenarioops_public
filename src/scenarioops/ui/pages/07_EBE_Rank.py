from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
    df = df.dropna(subset=["ebe_score"]).copy()
    df = df.sort_values(by="ebe_score", ascending=True).reset_index(drop=True)
    df["rank"] = df.index + 1
    df["percentile"] = (df["rank"] / len(df)) * 100

    label_stride = 4
    label_candidates = (
        df[["label", "percentile", "ebe_score"]]
        .dropna(subset=["label"])
        .to_dict("records")
    )

    domain_colors = {
        "economic": "#1f77b4",
        "technological": "#2ca02c",
        "political": "#d62728",
        "legal": "#9467bd",
        "environmental": "#17becf",
        "social": "#2aa198",
    }

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["percentile"],
            y=df["ebe_score"],
            mode="lines",
            line=dict(color="rgba(80,80,80,0.35)", width=2),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    for domain, group in df.groupby("domain"):
        fig.add_trace(
            go.Scatter(
                x=group["percentile"],
                y=group["ebe_score"],
                mode="markers",
                name=domain or "unknown",
                marker=dict(
                    size=8,
                    color=domain_colors.get(domain, "#6b7280"),
                    line=dict(color="white", width=0.5),
                    opacity=0.9,
                ),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "EBE: %{y:.2f}<br>"
                    "Percentile: %{x:.0f}%<extra></extra>"
                ),
                customdata=group[["label"]].fillna(""),
            )
        )

    for pct in (25, 50, 75):
        fig.add_vline(x=pct, line_width=1, line_color="rgba(120,120,120,0.2)")

    toggle = 1
    for idx, row in enumerate(label_candidates):
        if idx % label_stride != 0:
            continue
        x = float(row["percentile"])
        y = float(row["ebe_score"])
        toggle *= -1
        fig.add_annotation(
            x=x,
            y=y,
            text=str(row["label"]),
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=50 * toggle,
            font=dict(size=11, color="#1f2937"),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(0,0,0,0.08)",
            borderpad=4,
        )

    fig.update_layout(
        title="EBE ranking curve",
        height=480,
        margin=dict(l=50, r=20, t=60, b=50),
        xaxis_title="Forces by percentile (ascending EBE score)",
        yaxis_title="EBE score",
        xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.06)", zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.06)", zeroline=False),
        legend_title_text="domain",
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    st.plotly_chart(fig, width="stretch")
if "ebe_score" in df.columns:
    df = df.sort_values(by="ebe_score", ascending=False)

columns = [c for c in ["force_id", "ebe_score", "B_business_impact", "E_emergence", "E_evidence_strength", "rationale"] if c in df.columns]
section("Top Forces", "Highest leverage forces ranked by EBE scoring.")
st.dataframe(df[columns].head(20), height=420)
