from __future__ import annotations

from pathlib import Path
from typing import Any

from scenarioops.app.config import LLMConfig, ScenarioOpsSettings
from scenarioops.graph.nodes.utils import build_prompt, get_client
from scenarioops.graph.state import ScenarioOpsState, WindTunnel, WindTunnelTest
from scenarioops.graph.tools.normalization import stable_id
from scenarioops.graph.tools.schema_validate import load_schema, validate_artifact
from scenarioops.graph.tools.scoring import score_with_rubric
from scenarioops.graph.tools.storage import log_normalization, write_artifact
from scenarioops.llm.guards import ensure_dict


FEASIBILITY_THRESHOLD = 0.6


def _normalize_tests(raw_tests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for entry in raw_tests:
        rubric_inputs = entry.get("rubric_inputs", {})
        scoring = score_with_rubric(rubric_inputs)
        feasibility_score = float(entry.get("feasibility_score", 0.0))
        action = scoring.action
        if action == "KEEP" and feasibility_score < FEASIBILITY_THRESHOLD:
            raise ValueError(
                f"KEEP action requires feasibility >= {FEASIBILITY_THRESHOLD:.2f}."
            )

        payload = {
            "id": entry.get("id"),
            "strategy_id": entry.get("strategy_id"),
            "scenario_id": entry.get("scenario_id"),
            "outcome": entry.get("outcome"),
            "failure_modes": entry.get("failure_modes", []),
            "adaptations": entry.get("adaptations", []),
            "feasibility_score": feasibility_score,
            "rubric_score": scoring.normalized_total,
            "action": action,
        }
        if entry.get("rating") is not None:
            payload["rating"] = entry.get("rating")
        if entry.get("notes") is not None:
            payload["notes"] = entry.get("notes")
        normalized.append(payload)
    return normalized


def _missing_scenarios(
    tests: list[dict[str, Any]], scenario_ids: list[str]
) -> list[str]:
    covered = {scenario_id: False for scenario_id in scenario_ids}
    for test in tests:
        scenario_id = str(test.get("scenario_id"))
        if scenario_id in covered:
            covered[scenario_id] = True
    return [scenario_id for scenario_id, ok in covered.items() if not ok]


def run_wind_tunnel_node(
    *,
    run_id: str,
    state: ScenarioOpsState,
    llm_client=None,
    base_dir: Path | None = None,
    config: LLMConfig | None = None,
    settings: ScenarioOpsSettings | None = None,
) -> ScenarioOpsState:
    if state.strategies is None:
        raise ValueError("Strategies are required for the wind tunnel.")
    scenario_payload = None
    if state.logic is None:
        if state.scenarios is None:
            raise ValueError("Logic or scenarios are required for the wind tunnel.")
        scenario_payload = state.scenarios.get("scenarios", [])

    context = {
        "strategies": [strategy.__dict__ for strategy in state.strategies.strategies],
        "scenarios": [scenario.__dict__ for scenario in state.logic.scenarios]
        if state.logic is not None
        else scenario_payload,
    }
    prompt_bundle = build_prompt("wind_tunnel", context)
    prompt = prompt_bundle.text

    client = get_client(llm_client, config)
    schema = load_schema("wind_tunnel")
    response = client.generate_json(prompt, schema)
    parsed = ensure_dict(response, node_name="wind_tunnel")

    raw_tests = parsed.get("tests", [])
    normalized_tests = _normalize_tests(raw_tests)
    scenario_ids = (
        [scenario.id for scenario in state.logic.scenarios]
        if state.logic is not None
        else [
            str(entry.get("scenario_id") or entry.get("id"))
            for entry in scenario_payload or []
            if isinstance(entry, dict)
        ]
    )
    missing = _missing_scenarios(normalized_tests, scenario_ids)
    allow_fixture_coverage = (
        settings is not None and settings.sources_policy == "fixtures"
    )
    if missing and not allow_fixture_coverage:
        raise ValueError(f"Wind tunnel missing tests for scenarios: {missing}")
    if missing and allow_fixture_coverage:
        log_normalization(
            run_id=run_id,
            node_name="wind_tunnel",
            operation="fixture_missing_scenarios",
            details={"missing": missing},
            base_dir=base_dir,
        )
    tunnel_id = parsed.get("id")
    if not tunnel_id:
        tunnel_id = stable_id(
            "wind_tunnel",
            [strategy.id for strategy in state.strategies.strategies],
            scenario_ids,
        )
        log_normalization(
            run_id=run_id,
            node_name="wind_tunnel",
            operation="stable_id_assigned",
            details={"field": "id"},
            base_dir=base_dir,
        )

    matrix = parsed.get("matrix")
    if not isinstance(matrix, list):
        matrix = [
            {
                "strategy_id": test.get("strategy_id"),
                "scenario_id": test.get("scenario_id"),
                "outcome": test.get("outcome"),
                "robustness_score": test.get("rubric_score", 0.0),
            }
            for test in normalized_tests
        ]
    robustness_scores = [float(test.get("rubric_score", 0.0)) for test in normalized_tests]
    robustness = sum(robustness_scores) / max(1, len(robustness_scores))
    payload = {
        "id": tunnel_id,
        "title": parsed.get("title", "Wind Tunnel"),
        "tests": normalized_tests,
        "matrix": matrix,
        "robustness_score": robustness,
        "break_conditions": parsed.get("break_conditions", []),
        "triggers": parsed.get("triggers", []),
    }

    validate_artifact("wind_tunnel", payload)

    write_artifact(
        run_id=run_id,
        artifact_name="wind_tunnel",
        payload=payload,
        ext="json",
        input_values={"strategy_ids": [strategy.id for strategy in state.strategies.strategies]},
        prompt_values={
            "prompt_name": prompt_bundle.name,
            "prompt_sha256": prompt_bundle.sha256,
        },
        tool_versions={"wind_tunnel_node": "0.1.0"},
        base_dir=base_dir,
    )

    tests = [WindTunnelTest(**test) for test in normalized_tests]
    state.wind_tunnel = WindTunnel(
        id=payload["id"], title=payload["title"], tests=tests
    )
    return state
