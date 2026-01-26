from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

import streamlit as st

from scenarioops.ui.page_utils import (
    card_grid,
    load_artifact,
    page_header,
    placeholder_section,
    resolve_run_id,
    section,
)

st.set_page_config(page_title="Uncertainty Axes", page_icon="U", layout="wide")
run_id = resolve_run_id()
page_header("Uncertainty Axes", run_id, subtitle="Primary uncertainties defining the scenario space")

payload = load_artifact(run_id, "uncertainty_axes")
if not payload:
    placeholder_section("Axes", ["Two core uncertainties", "Pole definitions", "Evidence basis"])
    st.stop()

axes = payload.get("axes", []) if isinstance(payload, dict) else []
card_grid(
    [
        ("Axes Selected", str(len(axes))),
        ("Company", str(payload.get("company_name", ""))),
        ("Horizon (months)", str(payload.get("horizon_months", ""))),
    ]
)

if not axes:
    st.stop()

for axis in axes:
    section(axis.get("axis_name", "Axis"), axis.get("independence_notes", ""))
    st.markdown(f"**Pole A:** {axis.get('pole_a', '')}")
    st.markdown(f"**Pole B:** {axis.get('pole_b', '')}")
    st.markdown(
        f"**Impact / Uncertainty:** {axis.get('impact_score', '')} / {axis.get('uncertainty_score', '')}"
    )
    changes = axis.get("what_would_change_mind", [])
    if changes:
        st.markdown("**What would change our mind**")
        st.write(changes)
