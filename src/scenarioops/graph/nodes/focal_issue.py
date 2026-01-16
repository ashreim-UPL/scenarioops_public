from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from scenarioops.app.config import LLMConfig
from scenarioops.graph.nodes.utils import get_client, load_prompt, render_prompt
from scenarioops.graph.state import ScenarioOpsState
from scenarioops.graph.tools.schema_validate import load_schema, validate_artifact
from scenarioops.graph.types import ArtifactData, NodeResult
from scenarioops.llm.guards import ensure_dict


def _user_intent(user_params: Mapping[str, Any]) -> str:
    intent = user_params.get("goal") or user_params.get("intent")
    if isinstance(intent, str) and intent.strip():
        return intent
    return json.dumps(user_params, sort_keys=True)


def run_focal_issue_node(
    user_params: Mapping[str, Any],
    *,
    state: ScenarioOpsState,
    llm_client=None,
    config: LLMConfig | None = None,
) -> NodeResult:
    prompt_template = load_prompt("focal_issue")
    context = {
        "user_intent": _user_intent(user_params),
        "org_context": user_params.get("org_context"),
    }
    prompt = render_prompt(prompt_template, context)
    client = get_client(llm_client, config)
    schema = load_schema("focal_issue.schema")
    response = client.generate_json(prompt, schema)
    parsed = ensure_dict(response, node_name="focal_issue")
    validate_artifact("focal_issue.schema", parsed)

    return NodeResult(
        state_updates={"focal_issue": parsed},
        artifacts=[
            ArtifactData(
                name="focal_issue",
                payload=parsed,
                ext="json",
                input_values={"user_params": dict(user_params)},
                prompt_values={"prompt": prompt},
                tool_versions={"focal_issue_node": "0.1.0"},
            )
        ],
    )
