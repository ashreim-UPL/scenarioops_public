from __future__ import annotations

from pathlib import Path

from app.config import LLMConfig
from scenarioops.graph.nodes.utils import get_client, load_prompt, render_prompt
from scenarioops.graph.state import ScenarioEvent, ScenarioOpsState, ScenarioSkeleton, Skeleton
from scenarioops.graph.tools.schema_validate import load_schema, validate_artifact
from scenarioops.graph.tools.storage import write_artifact
from scenarioops.llm.guards import ensure_dict


REQUIRED_OPERATING_RULES = {"policy", "market", "operations"}


def _validate_operating_rules(payload: dict) -> None:
    for scenario in payload.get("scenarios", []):
        rules = scenario.get("operating_rules", {})
        missing = REQUIRED_OPERATING_RULES - set(rules.keys())
        if missing:
            raise ValueError(f"Skeleton {scenario.get('id')} missing operating_rules: {missing}")


def run_skeletons_node(
    *,
    run_id: str,
    state: ScenarioOpsState,
    llm_client=None,
    base_dir: Path | None = None,
    config: LLMConfig | None = None,
) -> ScenarioOpsState:
    if state.logic is None:
        raise ValueError("Logic is required to generate skeletons.")

    prompt_template = load_prompt("skeletons")
    scenarios = [scenario.__dict__ for scenario in state.logic.scenarios]
    prompt = render_prompt(prompt_template, {"scenarios": scenarios})

    client = get_client(llm_client, config)
    schema = load_schema("skeleton")
    response = client.generate_json(prompt, schema)
    parsed = ensure_dict(response, node_name="skeletons")

    validate_artifact("skeleton", parsed)
    _validate_operating_rules(parsed)

    write_artifact(
        run_id=run_id,
        artifact_name="skeletons",
        payload=parsed,
        ext="json",
        input_values={"scenario_ids": [item.id for item in state.logic.scenarios]},
        prompt_values={"prompt": prompt},
        tool_versions={"skeletons_node": "0.1.0"},
        base_dir=base_dir,
    )

    skeletons = []
    for scenario in parsed.get("scenarios", []):
        events = [ScenarioEvent(**event) for event in scenario.get("key_events", [])]
        skeletons.append(
            ScenarioSkeleton(
                id=scenario["id"],
                name=scenario["name"],
                narrative=scenario["narrative"],
                key_events=events,
                operating_rules=scenario.get("operating_rules", {}),
            )
        )
    state.skeleton = Skeleton(
        id=parsed.get("id", f"skeletons-{run_id}"),
        title=parsed.get("title", "Scenario Skeletons"),
        scenarios=skeletons,
    )
    return state
