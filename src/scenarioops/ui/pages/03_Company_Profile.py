from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))


import streamlit as st

from scenarioops.ui.page_utils import load_artifact, page_header, placeholder_section, resolve_run_id

st.set_page_config(page_title="Company Profile", page_icon="P", layout="wide")
run_id = resolve_run_id()
page_header("Company Profile", run_id)

profile = load_artifact(run_id, "company_profile")
if not profile:
    placeholder_section("Overview", ["Company summary", "Geography", "Horizon"])
    placeholder_section("Annual Report Highlights", ["Key risks", "Strategic priorities"])
    placeholder_section("Source Basis", ["URLs", "Internal docs", "Manual input"])
    st.stop()

cols = st.columns(3)
cols[0].metric("Company", profile.get("company_name", ""))
cols[1].metric("Geography", profile.get("geography", ""))
cols[2].metric("Horizon (months)", profile.get("horizon_months", ""))

summary = profile.get("annual_report_summary")
if summary:
    st.subheader("Annual Report Summary")
    st.write(summary)

key_risks = profile.get("key_risks", [])
if key_risks:
    st.subheader("Key Risks")
    st.write(key_risks)

priorities = profile.get("strategic_priorities", [])
if priorities:
    st.subheader("Strategic Priorities")
    st.write(priorities)

source_basis = profile.get("source_basis", {})
if source_basis:
    st.subheader("Source Basis")
    st.write(source_basis)
