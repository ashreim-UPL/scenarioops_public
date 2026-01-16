from __future__ import annotations

from typing import Any, Mapping

from scenarioops.graph.state import Charter, ScenarioOpsState
from scenarioops.graph.tools.normalization import stable_id
from scenarioops.graph.tools.schema_validate import load_schema, validate_artifact
from scenarioops.graph.tools.storage import log_normalization
from scenarioops.graph.nodes.utils import get_client, load_prompt, render_prompt
from scenarioops.graph.types import ArtifactData, NodeResult
from scenarioops.llm.guards import ensure_dict
from scenarioops.app.config import LLMConfig


def run_charter_node(
    user_params: Mapping[str, Any],
    *,
    run_id: str,
    state: ScenarioOpsState,
    llm_client=None,
    config: LLMConfig | None = None,
    base_dir=None,
) -> NodeResult:
    prompt_template = load_prompt("charter")
    prompt = render_prompt(prompt_template, {"user_params": user_params})
    client = get_client(llm_client, config)
    schema = load_schema("charter")
    response = client.generate_json(prompt, schema)
    parsed = ensure_dict(response, node_name="charter")

    # Deterministic normalization before schema validation
    if not parsed.get("id"):
        parsed["id"] = stable_id(
            "charter",
            parsed.get("title"),
            parsed.get("purpose"),
            parsed.get("scope"),
            parsed.get("time_horizon"),
        )
        log_normalization(
            run_id=run_id,
            node_name="charter",
            operation="stable_id_assigned",
            details={"field": "id"},
            base_dir=base_dir,
        )

    # Keep schema-required fields stable (avoid LLM omission)
    if not parsed.get("time_horizon"):
        horizon = user_params.get("horizon")
        if horizon is not None:
            parsed["time_horizon"] = f"{horizon} months"
            log_normalization(
                run_id=run_id,
                node_name="charter",
                operation="derived_time_horizon",
                details={"source": "user_params.horizon"},
                base_dir=base_dir,
            )

    if not parsed.get("decision_context"):
        decision_context = (
            user_params.get("decision_context")
            or user_params.get("goal")
            or user_params.get("intent")
        )
        if decision_context:
            parsed["decision_context"] = str(decision_context)
            log_normalization(
                run_id=run_id,
                node_name="charter",
                operation="derived_decision_context",
                details={"source": "user_params"},
                base_dir=base_dir,
            )

    artifacts = []
    artifacts.append(
        ArtifactData(
            name="scenario_charter_raw_prevalidate",
            payload=parsed,
            ext="json",
            input_values={"user_params": dict(user_params)},
            prompt_values={"prompt": prompt},
            tool_versions={"charter_node": "0.1.0"},
        )
    )

    validate_artifact("charter", parsed)

    artifacts.append(
        ArtifactData(
            name="scenario_charter",
            payload=parsed,
            ext="json",
            input_values={"user_params": dict(user_params)},
            prompt_values={"prompt": prompt},
            tool_versions={"charter_node": "0.1.0"},
        )
    )

    return NodeResult(
        state_updates={"charter": Charter(**parsed)}, artifacts=artifacts
    )
