from __future__ import annotations

from pathlib import Path
from typing import Any

from scenarioops.graph.state import ScenarioOpsState
from scenarioops.graph.tools.schema_validate import validate_artifact
from scenarioops.graph.tools.storage import write_trace_artifact


def _collect_ids(items: list[dict[str, Any]], key: str) -> list[str]:
    collected = []
    for item in items:
        value = item.get(key)
        if value:
            collected.append(str(value))
    return sorted(set(collected))


def run_trace_map_node(
    *,
    run_id: str,
    state: ScenarioOpsState,
    base_dir: Path | None = None,
) -> ScenarioOpsState:
    if not isinstance(state.scenario_profiles, dict):
        raise ValueError("Scenario profiles required for trace map.")
    if not isinstance(state.epistemic_summary, dict):
        raise ValueError("Epistemic summary required for trace map.")

    scenarios = []
    for scenario in state.scenario_profiles.get("scenarios", []) or []:
        scenarios.append(
            {
                "scenario_id": scenario.get("id"),
                "links": [
                    {
                        "claim_type": "drivers",
                        "artifact": "artifacts/drivers.jsonl",
                        "ids": _collect_ids(scenario.get("drivers", []), "id"),
                    },
                    {
                        "claim_type": "assumptions",
                        "artifact": "artifacts/belief_sets.json",
                        "ids": _collect_ids(scenario.get("assumptions", []), "id"),
                    },
                    {
                        "claim_type": "unknowns",
                        "artifact": "artifacts/certainty_uncertainty.json",
                        "ids": _collect_ids(scenario.get("unknowns", []), "id"),
                    },
                    {
                        "claim_type": "causal_chain",
                        "artifact": "artifacts/effects.json",
                        "ids": _collect_ids(scenario.get("causal_chain", []), "id"),
                    },
                    {
                        "claim_type": "signals",
                        "artifact": "artifacts/ewi.json",
                        "ids": _collect_ids(scenario.get("signals", []), "id"),
                    },
                    {
                        "claim_type": "options",
                        "artifact": "artifacts/wind_tunnel.json",
                        "ids": _collect_ids(scenario.get("options", []), "strategy_id"),
                    },
                ],
            }
        )

    trace_map = {
        "run_id": run_id,
        "scenarios": scenarios,
        "epistemic_summary": {
            "artifact": "artifacts/epistemic_summary.json",
            "facts": _collect_ids(state.epistemic_summary.get("facts", []), "id"),
            "assumptions": _collect_ids(state.epistemic_summary.get("assumptions", []), "id"),
            "interpretations": _collect_ids(state.epistemic_summary.get("interpretations", []), "id"),
            "unknowns": _collect_ids(state.epistemic_summary.get("unknowns", []), "id"),
        },
    }

    validate_artifact("trace_map", trace_map)
    write_trace_artifact(
        run_id=run_id,
        artifact_name="trace_map",
        payload=trace_map,
        input_values={"scenario_count": len(scenarios)},
        prompt_values={"prompt": "trace_map"},
        tool_versions={"trace_map_node": "0.1.0"},
        base_dir=base_dir,
    )

    state.trace_map = trace_map
    return state
