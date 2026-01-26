from __future__ import annotations

import json
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

st.set_page_config(page_title="Company Profile", page_icon="P", layout="wide")
run_id = resolve_run_id()
page_header("Company Profile", run_id, subtitle="Organizational context and posture")

profile = load_artifact(run_id, "company_profile")
if not profile:
    placeholder_section("Overview", ["Company summary", "Geography", "Horizon"])
    placeholder_section("Annual Report Highlights", ["Key risks", "Strategic priorities"])
    placeholder_section("Source Basis", ["URLs", "Internal docs", "Manual input"])
    st.stop()

section(profile.get("company_name", "Company"), profile.get("summary") or profile.get("overview") or "")
card_grid(
    [
        ("Geography", profile.get("geography", "Pending")),
        ("Industry", profile.get("industry", "Pending")),
        ("Horizon (months)", profile.get("horizon_months", "Pending")),
    ]
)

annual = profile.get("annual_report_summary")
if annual:
    section("Annual Report Summary", annual)

st.markdown("### Key Risks")
pill_row(profile.get("key_risks", []))

st.markdown("### Strategic Priorities")
pill_row(profile.get("strategic_priorities", []))

source_basis = profile.get("source_basis", {})
if source_basis:
    section("Source Basis", json.dumps(source_basis, indent=2))
