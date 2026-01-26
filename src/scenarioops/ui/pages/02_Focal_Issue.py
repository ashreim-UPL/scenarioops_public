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

st.set_page_config(page_title="Focal Issue", page_icon="F", layout="wide")
run_id = resolve_run_id()
page_header("Focal Issue", run_id, subtitle="Strategic question and scope")

focal = load_artifact(run_id, "focal_issue")
if not focal:
    placeholder_section("Decision", ["Primary focal issue statement."])
    placeholder_section("Scope", ["Geography", "Sectors", "Time horizon"])
    placeholder_section("Exclusions", ["Out of scope items"])
    placeholder_section("Success Criteria", ["What success looks like"])
    st.stop()

section("Decision Focus", focal.get("focal_issue", ""))
scope = focal.get("scope", {}) if isinstance(focal.get("scope"), dict) else {}
card_grid(
    [
        ("Geography", scope.get("geography", "Pending")),
        ("Time Horizon (months)", scope.get("time_horizon_months", "Pending")),
        ("Decision Type", focal.get("decision_type", "Pending")),
    ]
)

st.markdown("### Sectors in Scope")
pill_row(scope.get("sectors", []))

st.markdown("### Exclusions")
pill_row(focal.get("exclusions", []))

section("Success Criteria", focal.get("success_criteria", ""))
