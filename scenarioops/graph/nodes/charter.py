from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from scenarioops.graph.state import Charter, ScenarioOpsState
from scenarioops.graph.tools.schema_validate import load_schema, validate_artifact
from scenarioops.graph.tools.storage import write_artifact
from scenarioops.graph.nodes.utils import get_client, load_prompt, render_prompt
from scenarioops.llm.guards import ensure_dict
from app.config import LLMConfig


def run_charter_node(
    user_params: Mapping[str, Any],
    *,
    run_id: str,
    state: ScenarioOpsState,
    llm_client=None,
    base_dir: Path | None = None,
    config: LLMConfig | None = None,
) -> ScenarioOpsState:
    prompt_template = load_prompt("charter")
    prompt = render_prompt(prompt_template, {"user_params": user_params})
    client = get_client(llm_client, config)
    schema = load_schema("charter")
    response = client.generate_json(prompt, schema)
    parsed = ensure_dict(response, node_name="charter")
    validate_artifact("charter", parsed)

    write_artifact(
        run_id=run_id,
        artifact_name="scenario_charter",
        payload=parsed,
        ext="json",
        input_values={"user_params": dict(user_params)},
        prompt_values={"prompt": prompt},
        tool_versions={"charter_node": "0.1.0"},
        base_dir=base_dir,
    )

    state.charter = Charter(**parsed)
    return state
