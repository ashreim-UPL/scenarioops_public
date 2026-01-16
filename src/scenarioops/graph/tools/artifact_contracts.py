from __future__ import annotations

from typing import Final


_JSON_MAPPING: Final[dict[str, str]] = {
    "focal_issue": "focal_issue.schema",
    "scenario_charter": "charter",
    "scenario_charter_raw_prevalidate": "charter",
    "driving_forces": "driving_forces.schema",
    "washout_report": "washout_report.schema",
    "evidence_units": "evidence_units.schema",
    "coverage_report": "coverage_report",
    "certainty_uncertainty": "certainty_uncertainty.schema",
    "belief_sets": "belief_sets.schema",
    "effects": "effects.schema",
    "epistemic_summary": "epistemic_summary",
    "uncertainties": "uncertainties",
    "logic": "logic",
    "skeletons": "skeleton",
    "ewi": "ewi",
    "strategies": "strategies",
    "wind_tunnel": "wind_tunnel",
    "daily_brief": "daily_brief",
    "scenario_profiles": "scenario_profiles",
    "trace_map": "trace_map",
    "audit_report": "audit_report",
    "index": "artifact_index",
    "view_model": "view_model",
}


def schema_for_artifact(name: str, suffix: str) -> str | None:
    if suffix == ".md":
        return "markdown"
    if suffix == ".jsonl":
        if name == "drivers":
            return "driver_entry"
        return None
    if suffix == ".json":
        if name.startswith("narrative_"):
            return "markdown"
        return _JSON_MAPPING.get(name)
    return None
