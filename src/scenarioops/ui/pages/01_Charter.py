from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))


import streamlit as st

from scenarioops.ui.page_utils import load_artifact, page_header, placeholder_section, resolve_run_id

st.set_page_config(page_title="Charter", page_icon="C", layout="wide")
run_id = resolve_run_id()
page_header("Charter", run_id)

charter = load_artifact(run_id, "scenario_charter")
if not charter:
    placeholder_section("Purpose", ["State the decision purpose."])
    placeholder_section("Decision Context", ["Describe the decision context."])
    placeholder_section("Scope & Horizon", ["Geography", "Time horizon", "Stakeholders"])
    placeholder_section("Constraints", ["Key constraints"])
    placeholder_section("Assumptions", ["Core assumptions"])
    placeholder_section("Success Criteria", ["Success criteria"])
    st.stop()

st.subheader(charter.get("title") or "Strategic Charter")
st.markdown(f"**Purpose**: {charter.get('purpose','')}")
st.markdown(f"**Decision Context**: {charter.get('decision_context','')}")
st.markdown(f"**Scope**: {charter.get('scope','')}")
st.markdown(f"**Time Horizon**: {charter.get('time_horizon','')}")

with st.expander("Stakeholders", expanded=True):
    st.write(charter.get("stakeholders", []))
with st.expander("Constraints", expanded=True):
    st.write(charter.get("constraints", []))
with st.expander("Assumptions", expanded=True):
    st.write(charter.get("assumptions", []))
with st.expander("Success Criteria", expanded=True):
    st.write(charter.get("success_criteria", []))
notes = charter.get("notes")
if notes:
    st.markdown("**Notes**")
    st.write(notes)
