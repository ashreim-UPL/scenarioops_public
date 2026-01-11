from __future__ import annotations

from difflib import SequenceMatcher
from pathlib import Path

from app.config import LLMConfig
from scenarioops.graph.nodes.utils import get_client, load_prompt, render_prompt
from scenarioops.graph.state import Logic, ScenarioAxis, ScenarioLogic, ScenarioOpsState
from scenarioops.graph.tools.schema_validate import load_schema, validate_artifact
from scenarioops.graph.tools.storage import write_artifact
from scenarioops.llm.guards import ensure_dict


AXIS_SIMILARITY_THRESHOLD = 0.6


def _axis_similarity(a_text: str, b_text: str) -> float:
    return SequenceMatcher(None, a_text.lower(), b_text.lower()).ratio()


def _validate_axes(payload: dict) -> None:
    axes = payload.get("axes", [])
    if len(axes) != 2:
        raise ValueError("Logic must include exactly 2 axes for a 2x2.")
    left = f"{axes[0].get('low', '')} {axes[0].get('high', '')}"
    right = f"{axes[1].get('low', '')} {axes[1].get('high', '')}"
    similarity = _axis_similarity(left, right)
    if similarity >= AXIS_SIMILARITY_THRESHOLD:
        raise ValueError(f"Axis similarity {similarity:.2f} exceeds threshold.")

    scenarios = payload.get("scenarios", [])
    if len(scenarios) != 4:
        raise ValueError("Logic must define exactly 4 scenarios.")


def run_logic_node(
    *,
    run_id: str,
    state: ScenarioOpsState,
    llm_client=None,
    base_dir: Path | None = None,
    config: LLMConfig | None = None,
) -> ScenarioOpsState:
    if state.uncertainties is None:
        raise ValueError("Uncertainties are required to generate logic.")

    prompt_template = load_prompt("logic")
    uncertainties = [entry.__dict__ for entry in state.uncertainties.uncertainties]
    prompt = render_prompt(prompt_template, {"uncertainties": uncertainties})

    client = get_client(llm_client, config)
    schema = load_schema("logic")
    response = client.generate_json(prompt, schema)
    parsed = ensure_dict(response, node_name="logic")

    validate_artifact("logic", parsed)
    _validate_axes(parsed)

    write_artifact(
        run_id=run_id,
        artifact_name="logic",
        payload=parsed,
        ext="json",
        input_values={"uncertainty_ids": [item.id for item in state.uncertainties.uncertainties]},
        prompt_values={"prompt": prompt},
        tool_versions={"logic_node": "0.1.0"},
        base_dir=base_dir,
    )

    axes = [ScenarioAxis(**axis) for axis in parsed.get("axes", [])]
    scenarios = [ScenarioLogic(**scenario) for scenario in parsed.get("scenarios", [])]
    state.logic = Logic(
        id=parsed.get("id", f"logic-{run_id}"),
        title=parsed.get("title", "Scenario Logic"),
        axes=axes,
        scenarios=scenarios,
    )
    return state
