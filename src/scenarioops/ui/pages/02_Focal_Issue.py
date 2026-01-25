from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))


import streamlit as st

from scenarioops.ui.page_utils import load_artifact, page_header, placeholder_section, resolve_run_id

st.set_page_config(page_title="Focal Issue", page_icon="F", layout="wide")
run_id = resolve_run_id()
page_header("Focal Issue", run_id)

focal = load_artifact(run_id, "focal_issue")
if not focal:
    placeholder_section("Decision", ["Primary focal issue statement."])
    placeholder_section("Scope", ["Geography", "Sectors", "Time horizon"])
    placeholder_section("Exclusions", ["Out of scope items"])
    placeholder_section("Success Criteria", ["What success looks like"])
    st.stop()

st.subheader("Decision")
st.write(focal.get("focal_issue", ""))

scope = focal.get("scope", {}) if isinstance(focal.get("scope"), dict) else {}
cols = st.columns(3)
cols[0].metric("Geography", scope.get("geography", ""))
cols[1].metric("Sectors", ", ".join(scope.get("sectors", [])) if scope.get("sectors") else "")
cols[2].metric("Horizon (years)", scope.get("time_horizon_years", ""))

st.markdown(f"**Decision Type**: {focal.get('decision_type','')}")
exclusions = focal.get("exclusions", [])
if exclusions:
    st.subheader("Exclusions")
    st.write(exclusions)
criteria = focal.get("success_criteria")
if criteria:
    st.subheader("Success Criteria")
    st.write(criteria)
