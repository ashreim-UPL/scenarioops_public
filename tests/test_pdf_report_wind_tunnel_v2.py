from __future__ import annotations

import json
from pathlib import Path

from scenarioops.reporting.pdf_report import build_management_report


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_management_report_contains_wind_tunnel_sections(tmp_path: Path) -> None:
    run_id = "run-123"
    artifacts_dir = tmp_path / run_id / "artifacts"
    evaluations = {
        "run_id": run_id,
        "generated_at": "2026-01-26T00:00:00Z",
        "evaluations": [
            {
                "run_id": run_id,
                "strategy_id": "s1",
                "scenario_id": "sc1",
                "outcome_label": "Moderate Success",
                "grade_letter": "B",
                "score_0_100": 76,
                "confidence_0_1": 0.7,
                "kpi_projection": {},
                "dominant_forces": [],
                "failed_assumptions": [],
                "break_conditions_triggered": [],
                "rationale_bullets": ["Outcome bullet", "Second bullet", "Third bullet"],
                "recommended_hardening_actions": ["Harden 1"],
                "trigger_points": [
                    {
                        "trigger_id": "t1",
                        "description": "Trigger 1",
                        "signal_source": "external",
                        "threshold": "test",
                        "recommended_action": "adapt",
                    }
                ],
            }
        ],
        "rankings": {
            "per_scenario": [],
            "overall": [
                {
                    "strategy_id": "s1",
                    "strategy_name": "Strategy One",
                    "overall_score": 76,
                    "min_score": 76,
                    "variance": 0,
                    "robustness_index": 76,
                    "rank": 1,
                }
            ],
            "scenario_fit": [],
        },
        "recommendations": {
            "primary_recommended_strategy": {
                "strategy_id": "s1",
                "strategy_name": "Strategy One",
                "rationale": "Best robustness index.",
            },
            "conditional_recommendations": [],
            "hardening_actions": ["Harden 1"],
            "hedge_actions": ["Hedge 1"],
            "triggers_to_watch": [
                {
                    "trigger_id": "t1",
                    "description": "Trigger 1",
                    "signal_source": "external",
                    "threshold": "test",
                    "recommended_action": "adapt",
                }
            ],
        },
    }
    _write_json(artifacts_dir / "wind_tunnel_evaluations_v2.json", evaluations)

    output_path = build_management_report(run_id, base_dir=tmp_path)
    content = output_path.read_bytes()
    assert b"Wind Tunnel Scorecard" in content
    assert b"Strategy Ranking Summary" in content
    assert b"Recommendation" in content
