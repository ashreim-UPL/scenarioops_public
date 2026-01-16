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


def _beliefs_payload(state: ScenarioOpsState) -> dict[str, Any]:
    if isinstance(state.belief_sets, dict):
        return state.belief_sets
    return {"belief_sets": []}


def run_effects_node(
    *,
    run_id: str,
    state: ScenarioOpsState,
    llm_client=None,
    base_dir: Path | None = None,
    config: LLMConfig | None = None,
) -> ScenarioOpsState:
    prompt_template = load_prompt("effects")
    context = {
        "scope_json": _scope_payload(state),
        "belief_sets_json": _beliefs_payload(state),
    }
    prompt = render_prompt(prompt_template, context)
    client = get_client(llm_client, config)
    schema = load_schema("effects.schema")
    response = client.generate_json(prompt, schema)
    parsed = ensure_dict(response, node_name="effects")
    validate_artifact("effects.schema", parsed)

    write_artifact(
        run_id=run_id,
        artifact_name="effects",
        payload=parsed,
        ext="json",
        input_values={"belief_set_count": len(context["belief_sets_json"].get("belief_sets", []))},
        prompt_values={"prompt": prompt},
        tool_versions={"effects_node": "0.1.0"},
        base_dir=base_dir,
    )

    state.effects = parsed
    return state
