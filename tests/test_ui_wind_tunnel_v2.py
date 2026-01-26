from __future__ import annotations

from scenarioops.ui.wind_tunnel_v2 import build_matrix, get_cell_detail


def test_cell_detail_contains_rationale_and_triggers() -> None:
    evaluations = [
        {
            "strategy_id": "s1",
            "scenario_id": "sc1",
            "grade_letter": "B",
            "score_0_100": 76,
            "outcome_label": "Moderate Success",
            "rationale_bullets": ["Reason 1", "Reason 2", "Reason 3"],
            "trigger_points": [{"description": "Trigger 1"}],
        }
    ]
    detail = get_cell_detail(evaluations, strategy_id="s1", scenario_id="sc1")
    assert detail is not None
    assert detail.get("rationale_bullets")
    assert detail.get("trigger_points")


def test_build_matrix_maps_names() -> None:
    evaluations = [
        {
            "strategy_id": "s1",
            "scenario_id": "sc1",
            "grade_letter": "B",
            "score_0_100": 76,
            "outcome_label": "Moderate Success",
        }
    ]
    strategies = [{"id": "s1", "name": "Strategy One"}]
    scenarios = [{"id": "sc1", "name": "Scenario One"}]
    matrix = build_matrix(evaluations, strategies=strategies, scenarios=scenarios)
    assert matrix["strategy_names"]["s1"] == "Strategy One"
    assert matrix["scenario_names"]["sc1"] == "Scenario One"
