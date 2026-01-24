from __future__ import annotations

from pathlib import Path

from scenarioops.app.config import LLMConfig, ScenarioOpsSettings
from scenarioops.graph.nodes.utils import get_client, load_prompt, render_prompt
from scenarioops.graph.state import Ewi, EwiIndicator, ScenarioOpsState
from scenarioops.graph.tools.normalization import stable_id
from scenarioops.graph.tools.schema_validate import load_schema, validate_artifact
from scenarioops.graph.tools.storage import log_normalization, write_artifact
from scenarioops.llm.guards import ensure_dict


def _validate_ewi_counts(payload: dict, scenario_ids: list[str]) -> None:
    counts = {scenario_id: 0 for scenario_id in scenario_ids}
    for indicator in payload.get("indicators", []):
        linked = indicator.get("linked_scenarios", [])
        for scenario_id in linked:
            if scenario_id in counts and indicator.get("metric"):
                counts[scenario_id] += 1
    for scenario_id, count in counts.items():
        if count < 5:
            raise ValueError(f"Scenario {scenario_id} has {count} measurable EWIs.")


def _use_mock_payload(
    settings: ScenarioOpsSettings | None, config: LLMConfig | None
) -> bool:
    if settings is not None:
        if settings.sources_policy == "fixtures":
            return True
        if settings.mode == "demo":
            return True
        if settings.llm_provider == "mock":
            return True
    if config is not None and getattr(config, "mode", None) == "mock":
        return True
    return False


def run_ewis_node(
    *,
    run_id: str,
    state: ScenarioOpsState,
    llm_client=None,
    base_dir: Path | None = None,
    config: LLMConfig | None = None,
    settings: ScenarioOpsSettings | None = None,
) -> ScenarioOpsState:
    if state.logic is None:
        raise ValueError("Logic is required to generate EWIs.")

    scenario_ids = [scenario.id for scenario in state.logic.scenarios]
    schema = load_schema("ewi")
    if _use_mock_payload(settings, config):
        indicators = []
        for scenario_id in scenario_ids:
            indicators.append(
                {
                    "id": f"ewi-{scenario_id}-1",
                    "name": f"Signal for {scenario_id}",
                    "description": "Mock indicator",
                    "signal": "watch",
                    "metric": "index",
                    "linked_scenarios": [scenario_id],
                }
            )
        parsed = {
            "id": stable_id("ewi", scenario_ids),
            "title": "Early Warning Indicators",
            "indicators": indicators,
            "metadata": {"mocked": True},
        }
    else:
        prompt_template = load_prompt("ewis")
        prompt = render_prompt(prompt_template, {"scenario_ids": scenario_ids})
        client = get_client(llm_client, config)
        response = client.generate_json(prompt, schema)
        parsed = ensure_dict(response, node_name="ewis")

    validate_artifact("ewi", parsed)
    if not _use_mock_payload(settings, config):
        _validate_ewi_counts(parsed, scenario_ids)

    write_artifact(
        run_id=run_id,
        artifact_name="ewi",
        payload=parsed,
        ext="json",
        input_values={"scenario_ids": scenario_ids},
        prompt_values={"prompt": "ewis"},
        tool_versions={"ewis_node": "0.1.0"},
        base_dir=base_dir,
    )

    indicators = [EwiIndicator(**indicator) for indicator in parsed.get("indicators", [])]
    ewi_id = parsed.get("id")
    if not ewi_id:
        ewi_id = stable_id("ewi", scenario_ids)
        log_normalization(
            run_id=run_id,
            node_name="ewis",
            operation="stable_id_assigned",
            details={"field": "id"},
            base_dir=base_dir,
        )
    state.ewi = Ewi(
        id=ewi_id,
        title=parsed.get("title", "Early Warning Indicators"),
        indicators=indicators,
    )
    return state
