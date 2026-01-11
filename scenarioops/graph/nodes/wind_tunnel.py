from __future__ import annotations

from pathlib import Path
from typing import Any

from app.config import LLMConfig
from scenarioops.graph.nodes.utils import get_client, load_prompt, render_prompt
from scenarioops.graph.state import ScenarioOpsState, WindTunnel, WindTunnelTest
from scenarioops.graph.tools.schema_validate import load_schema, validate_artifact
from scenarioops.graph.tools.scoring import score_with_rubric
from scenarioops.graph.tools.storage import write_artifact
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


def run_wind_tunnel_node(
    *,
    run_id: str,
    state: ScenarioOpsState,
    llm_client=None,
    base_dir: Path | None = None,
    config: LLMConfig | None = None,
) -> ScenarioOpsState:
    if state.strategies is None or state.logic is None:
        raise ValueError("Strategies and logic are required for the wind tunnel.")

    prompt_template = load_prompt("wind_tunnel")
    context = {
        "strategies": [strategy.__dict__ for strategy in state.strategies.strategies],
        "scenarios": [scenario.__dict__ for scenario in state.logic.scenarios],
    }
    prompt = render_prompt(prompt_template, context)

    client = get_client(llm_client, config)
    schema = load_schema("wind_tunnel")
    response = client.generate_json(prompt, schema)
    parsed = ensure_dict(response, node_name="wind_tunnel")

    raw_tests = parsed.get("tests", [])
    normalized_tests = _normalize_tests(raw_tests)
    payload = {
        "id": parsed.get("id", f"wind-tunnel-{run_id}"),
        "title": parsed.get("title", "Wind Tunnel"),
        "tests": normalized_tests,
    }

    validate_artifact("wind_tunnel", payload)

    write_artifact(
        run_id=run_id,
        artifact_name="wind_tunnel",
        payload=payload,
        ext="json",
        input_values={"strategy_ids": [strategy.id for strategy in state.strategies.strategies]},
        prompt_values={"prompt": prompt},
        tool_versions={"wind_tunnel_node": "0.1.0"},
        base_dir=base_dir,
    )

    tests = [WindTunnelTest(**test) for test in normalized_tests]
    state.wind_tunnel = WindTunnel(
        id=payload["id"], title=payload["title"], tests=tests
    )
    return state
