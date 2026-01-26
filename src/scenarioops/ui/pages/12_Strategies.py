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

st.set_page_config(page_title="Strategies", page_icon="T", layout="wide")
run_id = resolve_run_id()
page_header("Strategies", run_id, subtitle="Strategic options under scenario stress")

payload = load_artifact(run_id, "strategies")
if not payload:
    placeholder_section("Strategy Set", ["Strategy names", "Objectives", "Actions"])
    st.stop()

strategies = payload.get("strategies", []) if isinstance(payload, dict) else []
card_grid(
    [
        ("Strategies", str(len(strategies))),
        ("Run ID", str(payload.get("id", ""))),
        ("Status", "Ready" if strategies else "Pending"),
    ]
)

if not strategies:
    st.stop()

for strategy in strategies:
    section(strategy.get("name", "Strategy"), strategy.get("objective", ""))
    actions = strategy.get("actions", [])
    if actions:
        st.markdown("**Action Sequence**")
        st.write(actions)
    kpis = strategy.get("kpis", [])
    if kpis:
        st.markdown("**Leading KPIs**")
        st.write(kpis)
