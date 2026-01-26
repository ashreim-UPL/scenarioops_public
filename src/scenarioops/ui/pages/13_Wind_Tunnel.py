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
from scenarioops.ui.wind_tunnel_v2 import build_matrix, get_cell_detail

st.set_page_config(page_title="Wind Tunnel", page_icon="W", layout="wide")
run_id = resolve_run_id()
page_header("Wind Tunnel", run_id, subtitle="Stress-testing strategies across scenarios")

payload = load_artifact(run_id, "wind_tunnel")
evaluations_v2 = load_artifact(run_id, "wind_tunnel_evaluations_v2")
if not payload:
    placeholder_section("Wind Tunnel", ["Strategy x scenario matrix", "Robustness outcomes"])
    st.stop()

matrix = payload.get("matrix", []) if isinstance(payload, dict) else []
tests = payload.get("tests", []) if isinstance(payload, dict) else []
evaluations = (
    evaluations_v2.get("evaluations", [])
    if isinstance(evaluations_v2, dict)
    else []
)

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

if evaluations:
    section("Outcome Matrix (Grades)", "Each cell shows Grade and Score.")
    st.markdown(
        """
        <style>
        .wt-matrix { border-collapse: collapse; width: 100%; }
        .wt-matrix th, .wt-matrix td { border: 1px solid #e2e2e2; padding: 10px; text-align: center; }
        .wt-matrix th { background: #f6f4ef; font-weight: 700; }
        .wt-grade { font-size: 20px; font-weight: 700; }
        .wt-score { font-size: 12px; color: #666; }
        .wt-outcome { font-size: 11px; color: #888; margin-top: 4px; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    grid = build_matrix(evaluations)
    strategy_ids = list(grid["matrix"].keys())
    scenario_ids: list[str] = []
    for evaluation in evaluations:
        scenario_id = str(evaluation.get("scenario_id"))
        if scenario_id not in scenario_ids:
            scenario_ids.append(scenario_id)
    header = "".join(
        f"<th>{grid['scenario_names'].get(sid, sid)}</th>" for sid in scenario_ids
    )
    rows = []
    for strategy_id in strategy_ids:
        row_cells = []
        for scenario_id in scenario_ids:
            cell = grid["matrix"].get(strategy_id, {}).get(scenario_id)
            if not cell:
                row_cells.append("<td>-</td>")
                continue
            grade = cell.get("grade_letter", "-")
            score = cell.get("score_0_100", "-")
            outcome = cell.get("outcome_label", "")
            row_cells.append(
                f"<td><div class='wt-grade'>{grade}</div>"
                f"<div class='wt-score'>{score}</div>"
                f"<div class='wt-outcome'>{outcome}</div></td>"
            )
        rows.append(
            f"<tr><th>{grid['strategy_names'].get(strategy_id, strategy_id)}</th>{''.join(row_cells)}</tr>"
        )
    table_html = f"<table class='wt-matrix'><tr><th>Strategy</th>{header}</tr>{''.join(rows)}</table>"
    st.markdown(table_html, unsafe_allow_html=True)

    rankings = evaluations_v2.get("rankings", {}) if isinstance(evaluations_v2, dict) else {}
    overall = rankings.get("overall", []) if isinstance(rankings, dict) else []
    if overall:
        section("Strategy Comparison", "Overall ranking and robustness metrics.")
        st.dataframe(pd.DataFrame(overall), height=240)

    recommendations = (
        evaluations_v2.get("recommendations", {})
        if isinstance(evaluations_v2, dict)
        else {}
    )
    if recommendations:
        primary = recommendations.get("primary_recommended_strategy", {})
        section("Recommended Strategy Now", primary.get("rationale", ""))
        st.markdown(
            f"**Primary:** {primary.get('strategy_name', primary.get('strategy_id'))}"
        )
        st.markdown(
            "**Hardening actions:** "
            + ", ".join(recommendations.get("hardening_actions", [])[:5])
        )
        triggers = recommendations.get("triggers_to_watch", [])[:5]
        if triggers:
            st.markdown(
                "**Triggers to watch:** "
                + "; ".join([t.get("description", "") for t in triggers if isinstance(t, dict)])
            )
else:
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

if evaluations:
    section("Cell Evidence", "Select a strategy/scenario cell to view rationale and triggers.")
    strategy_ids = sorted({str(e.get("strategy_id")) for e in evaluations})
    scenario_ids = sorted({str(e.get("scenario_id")) for e in evaluations})
    selected_strategy = st.selectbox("Strategy", strategy_ids, key="wt_strategy")
    selected_scenario = st.selectbox("Scenario", scenario_ids, key="wt_scenario")
    detail = get_cell_detail(
        evaluations,
        strategy_id=selected_strategy,
        scenario_id=selected_scenario,
    )
    if detail:
        st.markdown("**Rationale**")
        st.write(detail.get("rationale_bullets", []))
        st.markdown("**Dominant forces**")
        st.write(detail.get("dominant_forces", []))
        st.markdown("**Failed assumptions**")
        st.write(detail.get("failed_assumptions", []))
        st.markdown("**Break conditions**")
        st.write(detail.get("break_conditions_triggered", []))
        st.markdown("**KPI projections**")
        st.json(detail.get("kpi_projection", {}))
        st.markdown("**Triggers**")
        st.write(detail.get("trigger_points", []))
