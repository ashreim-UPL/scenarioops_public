from __future__ import annotations

from pathlib import Path
from typing import Any

from scenarioops.app.config import LLMConfig
from scenarioops.graph.nodes.utils import get_client, load_prompt, render_prompt
from scenarioops.graph.state import ScenarioOpsState, UncertaintyEntry, Uncertainties
from scenarioops.graph.tools.normalization import stable_id
from scenarioops.graph.tools.schema_validate import load_schema, validate_artifact
from scenarioops.graph.tools.storage import log_normalization, write_artifact
from scenarioops.llm.guards import ensure_dict


def _forces_as_drivers(state: ScenarioOpsState) -> list[dict[str, Any]]:
    forces = []
    if isinstance(state.driving_forces, dict):
        forces = state.driving_forces.get("forces", [])
    if not isinstance(forces, list):
        return []
    drivers: list[dict[str, Any]] = []
    for force in forces:
        if not isinstance(force, dict):
            continue
        drivers.append(
            {
                "id": force.get("id"),
                "name": force.get("name"),
                "description": force.get("description"),
                "category": force.get("domain") or "other",
                "trend": "unspecified",
                "impact": "unspecified",
                "citations": force.get("citations", []),
            }
        )
    return drivers


def _validate_uncertainty_links(payload: dict[str, Any], driver_ids: set[str]) -> None:
    for entry in payload.get("uncertainties", []):
        linked = entry.get("driver_ids", [])
        if len(linked) < 2:
            raise ValueError(f"Uncertainty {entry.get('id')} must reference >=2 drivers.")
        missing = [item for item in linked if item not in driver_ids]
        if missing:
            raise ValueError(f"Uncertainty {entry.get('id')} references missing drivers: {missing}")


def _fallback_uncertainties(driver_payload: list[dict[str, Any]]) -> dict[str, Any]:
    drivers: list[dict[str, str]] = []
    for entry in driver_payload:
        if not isinstance(entry, dict):
            continue
        driver_id = entry.get("id")
        name = entry.get("name")
        if not isinstance(driver_id, str) or not driver_id.strip():
            driver_id = stable_id("driver", name, entry.get("description"))
        if not isinstance(name, str) or not name.strip():
            name = "Unknown driver"
        drivers.append({"id": str(driver_id), "name": str(name)})

    if len(drivers) < 2:
        raise ValueError("Need at least two drivers to build fallback uncertainties.")

    pairs: list[tuple[dict[str, str], dict[str, str]]] = []
    for idx in range(0, len(drivers) - 1, 2):
        pairs.append((drivers[idx], drivers[idx + 1]))
    if not pairs:
        pairs.append((drivers[0], drivers[1]))

    uncertainties: list[dict[str, Any]] = []
    for idx, (left, right) in enumerate(pairs, start=1):
        name = f"{left['name']} vs {right['name']}"
        uncertainties.append(
            {
                "id": stable_id("uncertainty", left["id"], right["id"], idx),
                "name": name,
                "description": f"Uncertainty in the balance between {left['name']} and {right['name']}.",
                "extremes": [f"{left['name']} dominates", f"{right['name']} dominates"],
                "driver_ids": [left["id"], right["id"]],
                "criticality": 3,
                "volatility": 3,
                "implications": [],
            }
        )

    return {
        "id": stable_id("uncertainties", [entry["id"] for entry in drivers]),
        "title": "Critical uncertainties (fallback)",
        "uncertainties": uncertainties,
        "metadata": {"fallback": True, "reason": "llm_output_not_json"},
    }


def run_uncertainties_node(
    *,
    run_id: str,
    state: ScenarioOpsState,
    llm_client=None,
    base_dir: Path | None = None,
    config: LLMConfig | None = None,
) -> ScenarioOpsState:
    if state.drivers is None and state.driving_forces is None:
        raise ValueError("Drivers or driving forces are required to generate uncertainties.")

    prompt_template = load_prompt("uncertainties")
    if state.drivers is not None:
        driver_payload = [driver.__dict__ for driver in state.drivers.drivers]
    else:
        driver_payload = _forces_as_drivers(state)
    context = {"drivers": driver_payload}
    prompt = render_prompt(prompt_template, context)

    client = get_client(llm_client, config)
    schema = load_schema("uncertainties")
    strict_prompt = (
        f"{prompt}\n\nReturn ONLY a JSON object that matches the schema. "
        "No prose, no markdown, no extra keys."
    )
    try:
        response = client.generate_json(prompt, schema)
        parsed = ensure_dict(response, node_name="uncertainties")
    except Exception as exc:
        log_normalization(
            run_id=run_id,
            node_name="uncertainties",
            operation="retry_strict_json",
            details={"error": str(exc)},
            base_dir=base_dir,
        )
        try:
            response = client.generate_json(strict_prompt, schema)
            parsed = ensure_dict(response, node_name="uncertainties")
        except Exception as retry_exc:
            log_normalization(
                run_id=run_id,
                node_name="uncertainties",
                operation="fallback_uncertainties",
                details={"error": str(retry_exc)},
                base_dir=base_dir,
            )
            parsed = _fallback_uncertainties(driver_payload)

    validate_artifact("uncertainties", parsed)
    driver_ids = {
        driver.get("id")
        for driver in driver_payload
        if isinstance(driver, dict) and driver.get("id")
    }
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
    uncertainty_id = parsed.get("id")
    if not uncertainty_id:
        uncertainty_id = stable_id("uncertainties", sorted(driver_ids))
        log_normalization(
            run_id=run_id,
            node_name="uncertainties",
            operation="stable_id_assigned",
            details={"field": "id"},
            base_dir=base_dir,
        )

    state.uncertainties = Uncertainties(
        id=uncertainty_id,
        title=parsed.get("title", "Uncertainties"),
        uncertainties=uncertainties,
    )
    return state
