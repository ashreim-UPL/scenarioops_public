from __future__ import annotations

from pathlib import Path
from typing import Any

from scenarioops.app.config import LLMConfig, ScenarioOpsSettings
from scenarioops.graph.nodes.utils import build_prompt, get_client
from scenarioops.graph.state import ScenarioOpsState, WindTunnel, WindTunnelTest
from scenarioops.graph.tools.normalization import stable_id
from scenarioops.graph.tools.schema_validate import load_schema, validate_artifact
from scenarioops.graph.tools.scoring import score_with_rubric
from scenarioops.graph.tools.storage import (
    log_normalization,
    write_artifact,
    write_trace_artifact,
)
from scenarioops.llm.guards import ensure_dict


FEASIBILITY_THRESHOLD = 0.6


def _relax_wind_tunnel_schema(schema: dict[str, Any]) -> dict[str, Any]:
    relaxed = dict(schema)
    relaxed["additionalProperties"] = True
    relaxed["required"] = []
    properties = relaxed.get("properties")
    if isinstance(properties, dict):
        tests_prop = properties.get("tests")
        if isinstance(tests_prop, dict):
            tests_prop["minItems"] = 0
            items = tests_prop.get("items")
            if isinstance(items, dict):
                items["additionalProperties"] = True
                items["required"] = []
        matrix_prop = properties.get("matrix")
        if isinstance(matrix_prop, dict):
            matrix_prop["type"] = ["array", "object"]
            items = matrix_prop.get("items")
            if isinstance(items, dict):
                items["additionalProperties"] = True
                items["required"] = []
    return relaxed

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


def _truncate(value: str, limit: int) -> str:
    if not value:
        return ""
    return value if len(value) <= limit else value[: limit - 3].rstrip() + "..."


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
    strict_schema = load_schema("wind_tunnel")
    schema = _relax_wind_tunnel_schema(strict_schema)
    response = client.generate_json(prompt, schema)
    response_raw = getattr(response, "raw", None)
    parsed = ensure_dict(response, node_name="wind_tunnel")

    raw_tests = parsed.get("tests", [])
    normalized_tests = _normalize_tests(raw_tests)
    missing_test_ids = [idx for idx, test in enumerate(normalized_tests, start=1) if not test.get("id")]
    if missing_test_ids:
        for idx, test in enumerate(normalized_tests, start=1):
            if test.get("id"):
                continue
            test["id"] = stable_id(
                "wind_tunnel_test",
                test.get("strategy_id"),
                test.get("scenario_id"),
                test.get("outcome"),
                idx,
            )
        log_normalization(
            run_id=run_id,
            node_name="wind_tunnel",
            operation="imputed_test_ids",
            details={"count": len(missing_test_ids), "tests": missing_test_ids[:5]},
            base_dir=base_dir,
        )
    missing_outcomes = [
        test.get("id") or f"test-{idx}"
        for idx, test in enumerate(normalized_tests, start=1)
        if not test.get("outcome")
    ]
    if missing_outcomes:
        for test in normalized_tests:
            if not test.get("outcome"):
                test["outcome"] = "INCONCLUSIVE"
        log_normalization(
            run_id=run_id,
            node_name="wind_tunnel",
            operation="imputed_test_outcomes",
            details={"count": len(missing_outcomes), "tests": missing_outcomes[:5]},
            base_dir=base_dir,
        )
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
    def _matrix_from_tests() -> list[dict[str, Any]]:
        return [
            {
                "strategy_id": test.get("strategy_id"),
                "scenario_id": test.get("scenario_id"),
                "outcome": test.get("outcome"),
                "robustness_score": test.get("rubric_score", 0.0),
            }
            for test in normalized_tests
        ]

    if isinstance(matrix, dict):
        rebuilt: list[dict[str, Any]] = []
        for strategy_id, scenarios in matrix.items():
            if not isinstance(scenarios, dict):
                continue
            for scenario_id, score in scenarios.items():
                rebuilt.append(
                    {
                        "strategy_id": str(strategy_id),
                        "scenario_id": str(scenario_id),
                        "outcome": "MIXED",
                        "robustness_score": float(score) if score is not None else 0.0,
                    }
                )
        matrix = rebuilt or _matrix_from_tests()
        log_normalization(
            run_id=run_id,
            node_name="wind_tunnel",
            operation="matrix_rebuilt_from_mapping",
            details={"entries": len(matrix)},
            base_dir=base_dir,
        )
    elif not isinstance(matrix, list):
        matrix = _matrix_from_tests()
    else:
        required_keys = {"strategy_id", "scenario_id", "outcome", "robustness_score"}
        if any(
            not isinstance(entry, dict) or not required_keys.issubset(entry.keys())
            for entry in matrix
        ):
            matrix = _matrix_from_tests()
            log_normalization(
                run_id=run_id,
                node_name="wind_tunnel",
                operation="matrix_rebuilt_from_tests",
                details={"reason": "missing_required_fields"},
                base_dir=base_dir,
            )
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

    if missing_outcomes or (response_raw and matrix is None):
        trace_payload = {
            "node": "wind_tunnel",
            "missing_outcomes": missing_outcomes,
            "response_raw_excerpt": _truncate(str(response_raw), 2000),
            "parsed_keys": sorted(parsed.keys()),
        }
        write_trace_artifact(
            run_id=run_id,
            artifact_name="wind_tunnel_debug",
            payload=trace_payload,
            input_values={"missing_outcomes": len(missing_outcomes)},
            prompt_values={
                "prompt_name": prompt_bundle.name,
                "prompt_sha256": prompt_bundle.sha256,
            },
            tool_versions={"wind_tunnel_node": "0.1.0"},
            base_dir=base_dir,
        )

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
