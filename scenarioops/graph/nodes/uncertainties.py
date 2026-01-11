from __future__ import annotations

from pathlib import Path
from typing import Any

from app.config import LLMConfig
from scenarioops.graph.nodes.utils import get_client, load_prompt, render_prompt
from scenarioops.graph.state import ScenarioOpsState, UncertaintyEntry, Uncertainties
from scenarioops.graph.tools.schema_validate import load_schema, validate_artifact
from scenarioops.graph.tools.storage import write_artifact
from scenarioops.llm.guards import ensure_dict


def _validate_uncertainty_links(payload: dict[str, Any], driver_ids: set[str]) -> None:
    for entry in payload.get("uncertainties", []):
        linked = entry.get("driver_ids", [])
        if len(linked) < 2:
            raise ValueError(f"Uncertainty {entry.get('id')} must reference >=2 drivers.")
        missing = [item for item in linked if item not in driver_ids]
        if missing:
            raise ValueError(f"Uncertainty {entry.get('id')} references missing drivers: {missing}")


def run_uncertainties_node(
    *,
    run_id: str,
    state: ScenarioOpsState,
    llm_client=None,
    base_dir: Path | None = None,
    config: LLMConfig | None = None,
) -> ScenarioOpsState:
    if state.drivers is None:
        raise ValueError("Drivers are required to generate uncertainties.")

    prompt_template = load_prompt("uncertainties")
    driver_payload = [driver.__dict__ for driver in state.drivers.drivers]
    context = {"drivers": driver_payload}
    prompt = render_prompt(prompt_template, context)

    client = get_client(llm_client, config)
    schema = load_schema("uncertainties")
    response = client.generate_json(prompt, schema)
    parsed = ensure_dict(response, node_name="uncertainties")

    validate_artifact("uncertainties", parsed)
    driver_ids = {driver.id for driver in state.drivers.drivers}
    _validate_uncertainty_links(parsed, driver_ids)

    write_artifact(
        run_id=run_id,
        artifact_name="uncertainties",
        payload=parsed,
        ext="json",
        input_values={"driver_ids": sorted(driver_ids)},
        prompt_values={"prompt": prompt},
        tool_versions={"uncertainties_node": "0.1.0"},
        base_dir=base_dir,
    )

    uncertainties = [
        UncertaintyEntry(**entry) for entry in parsed.get("uncertainties", [])
    ]
    state.uncertainties = Uncertainties(
        id=parsed.get("id", f"uncertainties-{run_id}"),
        title=parsed.get("title", "Uncertainties"),
        uncertainties=uncertainties,
    )
    return state
