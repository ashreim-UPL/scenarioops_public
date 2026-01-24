from __future__ import annotations

from typing import Any, Mapping, Sequence

from scenarioops.app.config import ScenarioOpsSettings
from scenarioops.graph.state import ScenarioOpsState
from scenarioops.graph.tools.schema_validate import validate_artifact
from scenarioops.graph.tools.traceability import build_run_metadata
from scenarioops.graph.types import ArtifactData, NodeResult


def _manual_input(user_params: Mapping[str, Any]) -> str:
    value = (
        user_params.get("company_description")
        or user_params.get("org_context")
        or user_params.get("value")
        or ""
    )
    return str(value)


def run_company_profile_node(
    user_params: Mapping[str, Any],
    sources: Sequence[str],
    *,
    run_id: str,
    state: ScenarioOpsState,
    settings: ScenarioOpsSettings | None = None,
) -> NodeResult:
    metadata = build_run_metadata(run_id=run_id, user_params=user_params)
    payload = {
        **metadata,
        "source_basis": {
            "urls": [str(item) for item in sources],
            "internal_docs": [],
            "manual_input": _manual_input(user_params),
        },
        "simulated": bool(getattr(settings, "simulate_evidence", False)),
        "metadata": metadata,
    }
    validate_artifact("company_profile", payload)
    return NodeResult(
        state_updates={"company_profile": payload},
        artifacts=[
            ArtifactData(
                name="company_profile",
                payload=payload,
                ext="json",
                input_values={"source_count": len(sources)},
                tool_versions={"company_profile_node": "0.1.0"},
            )
        ],
    )
