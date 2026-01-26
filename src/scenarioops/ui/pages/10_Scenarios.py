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

st.set_page_config(page_title="Scenarios", page_icon="S", layout="wide")
run_id = resolve_run_id()
page_header("Scenarios", run_id, subtitle="Narrative worlds anchored on uncertainty axes")

payload = load_artifact(run_id, "scenarios")
if not payload:
    placeholder_section("Scenario Set", ["Scenario names", "Narratives", "Implications"])
    st.stop()

scenarios = payload.get("scenarios", []) if isinstance(payload, dict) else []
card_grid(
    [
        ("Scenarios", str(len(scenarios))),
        ("Axes", ", ".join(payload.get("axes", []))),
        ("Horizon (months)", str(payload.get("horizon_months", ""))),
    ]
)

if not scenarios:
    st.stop()

for scenario in scenarios:
    section(scenario.get("name", "Scenario"), scenario.get("narrative", ""))
    implications = scenario.get("implications", [])
    if implications:
        st.markdown("**Implications**")
        st.write(implications)
    moves = scenario.get("no_regret_moves", [])
    if moves:
        st.markdown("**No-regret moves**")
        st.write(moves)
