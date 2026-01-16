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


def _evidence_payload(state: ScenarioOpsState) -> dict[str, Any]:
    if isinstance(state.evidence_units, dict):
        return state.evidence_units
    return {"evidence_units": []}


def run_classify_node(
    *,
    run_id: str,
    state: ScenarioOpsState,
    llm_client=None,
    base_dir: Path | None = None,
    config: LLMConfig | None = None,
) -> ScenarioOpsState:
    prompt_template = load_prompt("certainty_uncertainty")
    context = {
        "scope_json": _scope_payload(state),
        "evidence_units_json": _evidence_payload(state),
    }
    prompt = render_prompt(prompt_template, context)
    client = get_client(llm_client, config)
    schema = load_schema("certainty_uncertainty.schema")
    response = client.generate_json(prompt, schema)
    parsed = ensure_dict(response, node_name="classify")
    validate_artifact("certainty_uncertainty.schema", parsed)

    write_artifact(
        run_id=run_id,
        artifact_name="certainty_uncertainty",
        payload=parsed,
        ext="json",
        input_values={"evidence_unit_count": len(context["evidence_units_json"].get("evidence_units", []))},
        prompt_values={"prompt": prompt},
        tool_versions={"classify_node": "0.1.0"},
        base_dir=base_dir,
    )

    state.certainty_uncertainty = parsed
    return state
