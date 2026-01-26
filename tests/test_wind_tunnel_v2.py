from __future__ import annotations

from scenarioops.graph.tools.wind_tunnel_v2 import (
    EvaluationInputs,
    baseline_score,
    build_wind_tunnel_evaluations,
    outcome_band,
)


def test_baseline_score_within_band() -> None:
    labels = [
        "Optimal Success",
        "Hegemonic Success",
        "Moderate Success",
        "Partial Integration",
        "Stalled Growth",
        "Inefficient Dominance",
        "Irrelevant Strategy",
        "Total Strategic Failure",
        "INCONCLUSIVE",
    ]
    for label in labels:
        low, high = outcome_band(label)
        score = baseline_score(label, rubric_score=0.5)
        assert low <= score <= high


def test_evaluations_complete_matrix() -> None:
    strategies = [
        {"id": f"strategy-{idx}", "name": f"Strategy {idx}", "kpis": ["Revenue"]}
        for idx in range(1, 5)
    ]
    scenarios = [
        {"id": f"scenario-{idx}", "name": f"Scenario {idx}", "narrative": "Test"}
        for idx in range(1, 5)
    ]
    payload = build_wind_tunnel_evaluations(
        EvaluationInputs(
            run_id="run-1",
            strategies=strategies,
            scenarios=scenarios,
            tests=[],
            forces=[],
            break_conditions=["Break A", "Break B"],
            triggers=[],
        )
    )
    evaluations = payload.get("evaluations", [])
    assert len(evaluations) == 16
    pairs = {(e["strategy_id"], e["scenario_id"]) for e in evaluations}
    assert len(pairs) == 16
    for evaluation in evaluations:
        assert 0 <= evaluation["score_0_100"] <= 100
        assert evaluation["grade_letter"] in {"A", "B", "C", "D", "F"}
