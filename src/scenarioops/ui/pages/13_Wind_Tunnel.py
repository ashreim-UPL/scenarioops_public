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

st.set_page_config(page_title="Wind Tunnel", page_icon="W", layout="wide")
run_id = resolve_run_id()
page_header("Wind Tunnel", run_id, subtitle="Stress-testing strategies across scenarios")

payload = load_artifact(run_id, "wind_tunnel")
if not payload:
    placeholder_section("Wind Tunnel", ["Strategy x scenario matrix", "Robustness outcomes"])
    st.stop()

matrix = payload.get("matrix", []) if isinstance(payload, dict) else []
tests = payload.get("tests", []) if isinstance(payload, dict) else []

card_grid(
    [
        ("Matrix Cells", str(len(matrix))),
        ("Tests", str(len(tests))),
        ("Break Conditions", str(len(payload.get("break_conditions", [])))),
    ]
)

break_conditions = payload.get("break_conditions", [])
if break_conditions:
    section("Break Conditions", " | ".join(break_conditions))

if matrix:
    df = pd.DataFrame(matrix)
    columns = [c for c in ["strategy_id", "scenario_id", "outcome", "robustness_score"] if c in df.columns]
    section("Outcome Matrix", "Summary of outcomes by strategy and scenario.")
    st.dataframe(df[columns], height=420)

if tests:
    df_tests = pd.DataFrame(tests)
    columns = [c for c in ["id", "strategy_id", "scenario_id", "outcome", "confidence"] if c in df_tests.columns]
    section("Test Evidence", "Underlying tests supporting each outcome.")
    st.dataframe(df_tests[columns], height=420)
