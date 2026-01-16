from __future__ import annotations

from pathlib import Path
from typing import Any

from scenarioops.app.config import LLMConfig
from scenarioops.graph.nodes.utils import get_client, load_prompt, render_prompt
from scenarioops.graph.state import ScenarioOpsState
from scenarioops.graph.tools.schema_validate import load_schema, validate_artifact
from scenarioops.graph.tools.storage import write_artifact
from scenarioops.llm.guards import ensure_dict


def _scope_payload(state: ScenarioOpsState) -> dict[str, Any]:
    if isinstance(state.focal_issue, dict):
        scope = state.focal_issue.get("scope")
        if isinstance(scope, dict):
            return scope
    return {}


def _driving_forces_payload(state: ScenarioOpsState) -> dict[str, Any]:
    if isinstance(state.driving_forces, dict):
        return state.driving_forces
    return {"forces": []}


def run_washout_node(
    *,
    run_id: str,
    state: ScenarioOpsState,
    llm_client=None,
    base_dir: Path | None = None,
    config: LLMConfig | None = None,
) -> ScenarioOpsState:
    prompt_template = load_prompt("washout_audit")
    context = {
        "scope_json": _scope_payload(state),
        "driving_forces_json": _driving_forces_payload(state),
    }
    prompt = render_prompt(prompt_template, context)
    client = get_client(llm_client, config)
    schema = load_schema("washout_report.schema")
    response = client.generate_json(prompt, schema)
    parsed = ensure_dict(response, node_name="washout")
    validate_artifact("washout_report.schema", parsed)

    write_artifact(
        run_id=run_id,
        artifact_name="washout_report",
        payload=parsed,
        ext="json",
        input_values={"driving_force_count": len(context["driving_forces_json"].get("forces", []))},
        prompt_values={"prompt": prompt},
        tool_versions={"washout_node": "0.1.0"},
        base_dir=base_dir,
    )

    state.washout_report = parsed
    return state
