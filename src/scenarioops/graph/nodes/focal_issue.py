from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from scenarioops.app.config import LLMConfig
from scenarioops.graph.nodes.utils import build_prompt, get_client
from scenarioops.graph.state import ScenarioOpsState
from scenarioops.graph.tools.schema_validate import load_schema, validate_artifact
from scenarioops.graph.types import ArtifactData, NodeResult
from scenarioops.llm.guards import ensure_dict
from scenarioops.graph.tools.traceability import build_run_metadata


def _user_intent(user_params: Mapping[str, Any]) -> str:
    intent = user_params.get("goal") or user_params.get("intent")
    if isinstance(intent, str) and intent.strip():
        return intent
    return json.dumps(user_params, sort_keys=True)


def run_focal_issue_node(
    user_params: Mapping[str, Any],
    *,
    state: ScenarioOpsState,
    run_id: str | None = None,
    llm_client=None,
    config: LLMConfig | None = None,
) -> NodeResult:
    context = {
        "user_intent": _user_intent(user_params),
        "org_context": user_params.get("org_context"),
    }
    prompt_bundle = build_prompt("focal_issue", context)
    prompt = prompt_bundle.text
    client = get_client(llm_client, config)
    schema = load_schema("focal_issue.schema")
    response = client.generate_json(prompt, schema)
    parsed = ensure_dict(response, node_name="focal_issue")
    scope = parsed.get("scope")
    if isinstance(scope, Mapping):
        horizon_years = scope.get("time_horizon_years")
        if isinstance(horizon_years, int):
            if horizon_years < 3:
                parsed["scope"] = {**scope, "time_horizon_years": 3}
            elif horizon_years > 10:
                parsed["scope"] = {**scope, "time_horizon_years": 10}
    validate_artifact("focal_issue.schema", parsed)

    metadata = build_run_metadata(
        run_id=run_id or "unknown",
        user_params=user_params,
        focal_issue=parsed,
    )
    metadata["prompt_name"] = prompt_bundle.name
    metadata["prompt_sha256"] = prompt_bundle.sha256
    parsed["metadata"] = metadata

    return NodeResult(
        state_updates={"focal_issue": parsed},
        artifacts=[
            ArtifactData(
                name="focal_issue",
                payload=parsed,
                ext="json",
                input_values={"user_params": dict(user_params)},
                prompt_values={
                    "prompt_name": prompt_bundle.name,
                    "prompt_sha256": prompt_bundle.sha256,
                },
                tool_versions={"focal_issue_node": "0.1.0"},
            )
        ],
    )
