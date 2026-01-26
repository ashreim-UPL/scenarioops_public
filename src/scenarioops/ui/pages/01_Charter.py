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
    pill_row,
    resolve_run_id,
    section,
)

st.set_page_config(page_title="Charter", page_icon="C", layout="wide")
run_id = resolve_run_id()
page_header("Charter", run_id, subtitle="Decision frame and operating intent")

charter = load_artifact(run_id, "scenario_charter")
if not charter:
    placeholder_section("Purpose", ["State the decision purpose."])
    placeholder_section("Decision Context", ["Describe the decision context."])
    placeholder_section("Scope & Horizon", ["Geography", "Time horizon", "Stakeholders"])
    placeholder_section("Constraints", ["Key constraints"])
    placeholder_section("Assumptions", ["Core assumptions"])
    placeholder_section("Success Criteria", ["Success criteria"])
    st.stop()

section(charter.get("title") or "Strategic Charter", charter.get("purpose") or "")
card_grid(
    [
        ("Decision Context", charter.get("decision_context") or "Pending"),
        ("Scope", charter.get("scope") or "Pending"),
        ("Time Horizon", charter.get("time_horizon") or "Pending"),
    ]
)

st.markdown("### Stakeholders")
pill_row(charter.get("stakeholders", []))

st.markdown("### Constraints")
pill_row(charter.get("constraints", []))

st.markdown("### Assumptions")
pill_row(charter.get("assumptions", []))

st.markdown("### Success Criteria")
pill_row(charter.get("success_criteria", []))

notes = charter.get("notes")
if notes:
    section("Notes", str(notes))
