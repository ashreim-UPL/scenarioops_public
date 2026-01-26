from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

import pandas as pd
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
if "ebe_score" in df.columns:
    df = df.sort_values(by="ebe_score", ascending=False)

columns = [c for c in ["force_id", "ebe_score", "B_business_impact", "E_emergence", "E_evidence_strength", "rationale"] if c in df.columns]
section("Top Forces", "Highest leverage forces ranked by EBE scoring.")
st.dataframe(df[columns].head(20), height=420)
