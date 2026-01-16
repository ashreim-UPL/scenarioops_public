from __future__ import annotations

from pathlib import Path
from typing import Any

from scenarioops.graph.state import ScenarioOpsState
from scenarioops.graph.tools.schema_validate import validate_artifact
from scenarioops.graph.tools.storage import write_artifact


STEEP_DIMENSIONS = [
    "political",
    "economic",
    "social",
    "technological",
    "environmental",
    "legal",
]
STUDY_LENSES = ["macro", "law", "geopolitics", "ethics", "culture"]


def _counts(items: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        value = str(item.get(key, "")).strip().lower()
        if not value:
            continue
        counts[value] = counts.get(value, 0) + 1
    return counts


def _lens_counts(items: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        lenses = item.get("lenses", [])
        if not isinstance(lenses, list):
            continue
        for lens in lenses:
            label = str(lens).strip().lower()
            if not label:
                continue
            counts[label] = counts.get(label, 0) + 1
    return counts


def run_coverage_node(
    *,
    run_id: str,
    state: ScenarioOpsState,
    base_dir: Path | None = None,
) -> ScenarioOpsState:
    if not isinstance(state.driving_forces, dict):
        raise ValueError("Driving forces are required for coverage checks.")
    forces = state.driving_forces.get("forces", [])
    if not isinstance(forces, list) or not forces:
        raise ValueError("Coverage requires driving forces.")

    domain_counts = _counts(forces, "domain")
    lens_counts = _lens_counts(forces)

    steep = []
    missing_domains = []
    for dimension in STEEP_DIMENSIONS:
        count = domain_counts.get(dimension, 0)
        status = "covered" if count > 0 else "missing"
        if count == 0:
            missing_domains.append(dimension)
        steep.append({"dimension": dimension, "count": count, "status": status})

    lenses = []
    missing_lenses = []
    for lens in STUDY_LENSES:
        count = lens_counts.get(lens, 0)
        status = "covered" if count > 0 else "missing"
        if count == 0:
            missing_lenses.append(lens)
        lenses.append({"lens": lens, "count": count, "status": status})

    if missing_domains:
        raise ValueError(f"Coverage missing STEEP dimensions: {missing_domains}")
    if missing_lenses:
        raise ValueError(f"Coverage missing study lenses: {missing_lenses}")

    report = {"steep": steep, "lenses": lenses}
    validate_artifact("coverage_report", report)

    write_artifact(
        run_id=run_id,
        artifact_name="coverage_report",
        payload=report,
        ext="json",
        input_values={"force_count": len(forces)},
        prompt_values={"prompt": "coverage"},
        tool_versions={"coverage_node": "0.1.0"},
        base_dir=base_dir,
    )

    state.coverage_report = report
    return state
