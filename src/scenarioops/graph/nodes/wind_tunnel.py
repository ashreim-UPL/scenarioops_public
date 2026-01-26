from __future__ import annotations

from pathlib import Path
from typing import Any
import os

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
from scenarioops.graph.tools.wind_tunnel_v2 import (
    EvaluationInputs,
    build_wind_tunnel_evaluations,
)
from scenarioops.llm.guards import ensure_dict


FEASIBILITY_THRESHOLD = 0.6


def _max_test_pairs() -> int:
    raw = os.environ.get("SCENARIOOPS_WIND_TUNNEL_MAX_PAIRS", "8")
    try:
        value = int(raw)
    except ValueError:
        return 16
    return max(1, value)


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
        raw_feasibility = entry.get("feasibility_score", 0.0)
        try:
            feasibility_score = float(raw_feasibility)
        except (TypeError, ValueError):
            feasibility_score = 0.0
        if feasibility_score > 1.0:
            if feasibility_score <= 100.0:
                feasibility_score = feasibility_score / 100.0
            else:
                feasibility_score = 1.0
        elif feasibility_score < 0.0:
            feasibility_score = 0.0
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


def _compact_strategy(strategy: Any) -> dict[str, Any]:
    return {
        "id": getattr(strategy, "id", None),
        "name": _truncate(str(getattr(strategy, "name", "") or ""), 120),
        "objective": _truncate(str(getattr(strategy, "objective", "") or ""), 240),
        "actions": [
            _truncate(str(item), 160)
            for item in (getattr(strategy, "actions", []) or [])[:4]
        ],
        "kpis": [
            _truncate(str(item), 120)
            for item in (getattr(strategy, "kpis", []) or [])[:4]
        ],
    }


def _compact_scenario_from_logic(scenario: Any) -> dict[str, Any]:
    return {
        "id": getattr(scenario, "id", None),
        "name": _truncate(str(getattr(scenario, "name", "") or ""), 120),
        "summary": _truncate(str(getattr(scenario, "summary", "") or ""), 400),
        "logic": _truncate(str(getattr(scenario, "logic", "") or ""), 500),
    }


def _compact_scenario_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "scenario_id": payload.get("scenario_id") or payload.get("id"),
        "name": _truncate(str(payload.get("name", "") or ""), 120),
        "axis_states": payload.get("axis_states"),
        "narrative": _truncate(str(payload.get("narrative", "") or ""), 600),
    }


def _full_strategy(strategy: Any) -> dict[str, Any]:
    return {
        "id": getattr(strategy, "id", None),
        "name": getattr(strategy, "name", None),
        "objective": getattr(strategy, "objective", None),
        "actions": list(getattr(strategy, "actions", []) or []),
        "kpis": list(getattr(strategy, "kpis", []) or []),
    }


def _full_scenario_from_logic(scenario: Any) -> dict[str, Any]:
    return {
        "id": getattr(scenario, "id", None),
        "name": getattr(scenario, "name", None),
        "summary": getattr(scenario, "summary", None),
        "logic": getattr(scenario, "logic", None),
    }


def _full_scenario_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": payload.get("scenario_id") or payload.get("id"),
        "name": payload.get("name"),
        "narrative": payload.get("narrative"),
        "summary": payload.get("summary"),
        "logic": payload.get("logic"),
    }


def _strategy_chunks(
    strategies: list[dict[str, Any]],
    scenario_count: int,
    max_pairs: int,
) -> list[list[dict[str, Any]]]:
    if scenario_count <= 0 or not strategies:
        return [strategies]
    pair_count = len(strategies) * scenario_count
    if pair_count <= max_pairs:
        return [strategies]
    per_chunk = max(1, max_pairs // scenario_count)
    return [strategies[idx : idx + per_chunk] for idx in range(0, len(strategies), per_chunk)]


def _chunk_list(items: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    if not items:
        return []
    if size <= 0:
        return [items]
    return [items[idx : idx + size] for idx in range(0, len(items), size)]


def _pair_chunks(
    strategies: list[dict[str, Any]],
    scenarios: list[dict[str, Any]],
    max_pairs: int,
) -> list[tuple[list[dict[str, Any]], list[dict[str, Any]]]]:
    if not strategies or not scenarios:
        return [(strategies, scenarios)]
    strategy_chunks = _strategy_chunks(strategies, len(scenarios), max_pairs)
    chunks: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = []
    for chunk in strategy_chunks:
        scenario_chunk_size = max(1, max_pairs // max(1, len(chunk)))
        for scenario_chunk in _chunk_list(scenarios, scenario_chunk_size):
            chunks.append((chunk, scenario_chunk))
    return chunks


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

    strategies_compact = [
        _compact_strategy(strategy) for strategy in state.strategies.strategies
    ]
    strategies_full = [
        _full_strategy(strategy) for strategy in state.strategies.strategies
    ]
    if state.logic is not None:
        scenarios_compact = [
            _compact_scenario_from_logic(scenario) for scenario in state.logic.scenarios
        ]
        scenarios_full = [
            _full_scenario_from_logic(scenario) for scenario in state.logic.scenarios
        ]
    else:
        scenarios_compact = [
            _compact_scenario_payload(entry)
            for entry in (scenario_payload or [])
            if isinstance(entry, dict)
        ]
        scenarios_full = [
            _full_scenario_payload(entry)
            for entry in (scenario_payload or [])
            if isinstance(entry, dict)
        ]

    client = get_client(llm_client, config)
    strict_schema = load_schema("wind_tunnel")
    schema = _relax_wind_tunnel_schema(strict_schema)

    scenario_count = len(scenarios_compact)
    max_pairs = _max_test_pairs()
    chunks = _pair_chunks(strategies_compact, scenarios_compact, max_pairs)
    if len(chunks) > 1:
        log_normalization(
            run_id=run_id,
            node_name="wind_tunnel",
            operation="chunked_requests",
            details={
                "chunks": len(chunks),
                "max_pairs": max_pairs,
                "strategy_count": len(strategies_compact),
                "scenario_count": scenario_count,
            },
            base_dir=base_dir,
        )

    prompt_bundle = None
    response_raw = None
    parsed = {}
    collected_tests: list[dict[str, Any]] = []
    collected_break_conditions: list[str] = []
    collected_triggers: list[str] = []

    for idx, (strategy_chunk, scenario_chunk) in enumerate(chunks, start=1):
        context = {
            "strategies": strategy_chunk,
            "scenarios": scenario_chunk,
        }
        prompt_bundle = build_prompt("wind_tunnel", context)
        prompt = (
            f"{prompt_bundle.text}\n\n"
            "Output constraints:\n"
            "- Keep break_conditions and triggers concise (max 3 each).\n"
            "- Keep each break_condition/trigger <= 12 words.\n"
            "- Keep failure_modes/adaptations concise (max 3 each).\n"
            "- Keep each failure_mode/adaptation <= 10 words.\n"
            "- Keep outcome <= 3 words.\n"
            "- Omit the matrix field entirely (it will be computed).\n"
            "- Keep responses compact; avoid extra commentary.\n"
        )
        if len(chunks) > 1 and idx > 1:
            prompt += "- For this chunk, set break_conditions and triggers to empty arrays.\n"
        try:
            response = client.generate_json(prompt, schema)
        except Exception as exc:
            write_trace_artifact(
                run_id=run_id,
                artifact_name=f"wind_tunnel_failure_{idx}",
                payload={
                    "node": "wind_tunnel",
                    "chunk_index": idx,
                    "chunks": len(chunks),
                    "error": str(exc),
                    "strategy_count": len(strategy_chunk),
                    "scenario_count": len(scenario_chunk),
                },
                input_values={"chunk": idx, "chunks": len(chunks)},
                prompt_values={
                    "prompt_name": prompt_bundle.name,
                    "prompt_sha256": prompt_bundle.sha256,
                },
                tool_versions={"wind_tunnel_node": "0.1.0"},
                base_dir=base_dir,
            )
            raise

        response_raw = getattr(response, "raw", None)
        parsed = ensure_dict(response, node_name="wind_tunnel")
        raw_tests = parsed.get("tests", [])
        if isinstance(raw_tests, list):
            collected_tests.extend(raw_tests)
        if isinstance(parsed.get("break_conditions"), list):
            collected_break_conditions.extend(parsed.get("break_conditions") or [])
        if isinstance(parsed.get("triggers"), list):
            collected_triggers.extend(parsed.get("triggers") or [])

    parsed["tests"] = collected_tests
    if collected_break_conditions:
        parsed["break_conditions"] = collected_break_conditions
    if collected_triggers:
        parsed["triggers"] = collected_triggers

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

    if len(chunks) > 1:
        matrix = _matrix_from_tests()
        log_normalization(
            run_id=run_id,
            node_name="wind_tunnel",
            operation="matrix_rebuilt_from_tests",
            details={"entries": len(matrix)},
            base_dir=base_dir,
        )
    elif isinstance(matrix, dict):
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

    evaluation_payload = build_wind_tunnel_evaluations(
        EvaluationInputs(
            run_id=run_id,
            strategies=strategies_full,
            scenarios=scenarios_full,
            tests=normalized_tests,
            forces=(state.forces or {}).get("forces", []) if state.forces else [],
            break_conditions=payload.get("break_conditions", []),
            triggers=payload.get("triggers", []),
        )
    )
    validate_artifact("wind_tunnel_evaluations_v2", evaluation_payload)
    write_artifact(
        run_id=run_id,
        artifact_name="wind_tunnel_evaluations_v2",
        payload=evaluation_payload,
        ext="json",
        input_values={
            "strategy_count": len(strategies_full),
            "scenario_count": len(scenarios_full),
        },
        tool_versions={"wind_tunnel_evaluations_v2": "0.1.0"},
        base_dir=base_dir,
    )

    tests = [WindTunnelTest(**test) for test in normalized_tests]
    state.wind_tunnel = WindTunnel(
        id=payload["id"], title=payload["title"], tests=tests
    )
    return state
