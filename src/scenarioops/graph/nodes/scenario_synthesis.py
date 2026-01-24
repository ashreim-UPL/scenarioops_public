from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from scenarioops.app.config import LLMConfig
from scenarioops.graph.nodes.utils import build_prompt, get_client
from scenarioops.graph.state import ScenarioOpsState
from scenarioops.graph.tools.schema_validate import load_schema, validate_artifact
from scenarioops.graph.tools.storage import write_artifact
from scenarioops.graph.tools.traceability import build_run_metadata
from scenarioops.llm.guards import ensure_dict


def _selected_axes(payload: Mapping[str, Any]) -> list[str]:
    selected = payload.get("selected_axis_ids", [])
    if isinstance(selected, list) and selected:
        return [str(item) for item in selected if str(item)]
    axes = payload.get("axes", [])
    if not isinstance(axes, list):
        return []
    sorted_axes = sorted(
        axes,
        key=lambda axis: (
            -float(axis.get("impact_score", 0)) * float(axis.get("uncertainty_score", 0)),
            str(axis.get("axis_id")),
        ),
    )
    return [str(axis.get("axis_id")) for axis in sorted_axes[:2] if axis.get("axis_id")]


def run_scenario_synthesis_node(
    *,
    run_id: str,
    state: ScenarioOpsState,
    user_params: Mapping[str, Any],
    llm_client=None,
    base_dir: Path | None = None,
    config: LLMConfig | None = None,
) -> ScenarioOpsState:
    if state.uncertainty_axes is None:
        raise ValueError("Uncertainty axes are required before scenarios.")
    if state.clusters is None:
        raise ValueError("Clusters are required before scenarios.")
    if state.forces is None:
        raise ValueError("Forces are required before scenarios.")

    axis_ids = _selected_axes(state.uncertainty_axes)
    if len(axis_ids) < 2:
        raise ValueError("At least two axes are required for scenario synthesis.")
    axes = state.uncertainty_axes.get("axes", [])
    selected_axes = [
        axis for axis in axes if axis.get("axis_id") in set(axis_ids)
    ]
    prompt_bundle = build_prompt(
        "scenario_synthesis",
        {
            "selected_axes": selected_axes,
            "clusters": state.clusters.get("clusters", []),
            "forces": state.forces.get("forces", []),
        },
    )

    client = get_client(llm_client, config)
    schema = load_schema("scenarios_payload")
    response = client.generate_json(prompt_bundle.text, schema)
    parsed = ensure_dict(response, node_name="scenarios")

    scenarios = parsed.get("scenarios", [])
    if not isinstance(scenarios, list):
        raise TypeError("Scenarios payload must include scenarios list.")

    warnings: list[str] = []
    for idx, scenario in enumerate(scenarios, start=1):
        if not scenario.get("scenario_id"):
            scenario["scenario_id"] = f"S{idx}"
        axis_states = scenario.get("axis_states", {})
        if not isinstance(axis_states, Mapping):
            warnings.append(f"scenario_axis_states_invalid:{scenario.get('scenario_id')}")
            continue
        missing = [axis_id for axis_id in axis_ids if axis_id not in axis_states]
        if missing:
            warnings.append(f"scenario_missing_axes:{scenario.get('scenario_id')}:{missing}")
        touchpoints = scenario.get("evidence_touchpoints", {})
        cluster_ids = touchpoints.get("cluster_ids", []) if isinstance(touchpoints, Mapping) else []
        force_ids = touchpoints.get("force_ids", []) if isinstance(touchpoints, Mapping) else []
        if len(cluster_ids) < 2 or len(force_ids) < 2:
            warnings.append(f"scenario_insufficient_touchpoints:{scenario.get('scenario_id')}")

    metadata = build_run_metadata(run_id=run_id, user_params=user_params)
    payload = {
        **metadata,
        "needs_correction": bool(warnings),
        "warnings": warnings,
        "axes": axis_ids,
        "scenarios": scenarios,
    }
    validate_artifact("scenarios", payload)

    write_artifact(
        run_id=run_id,
        artifact_name="scenarios",
        payload=payload,
        ext="json",
        input_values={"scenario_count": len(scenarios)},
        prompt_values={
            "prompt_name": prompt_bundle.name,
            "prompt_sha256": prompt_bundle.sha256,
        },
        tool_versions={"scenario_synthesis_node": "0.1.0"},
        base_dir=base_dir,
    )
    state.scenarios = payload
    if warnings:
        raise RuntimeError("scenario_synthesis_needs_correction: " + "; ".join(warnings[:2]))
    return state
