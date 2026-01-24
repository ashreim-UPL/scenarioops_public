from __future__ import annotations

from typing import Final


_JSON_MAPPING: Final[dict[str, str]] = {
    "focal_issue": "focal_issue.schema",
    "scenario_charter": "charter",
    "scenario_charter_raw_prevalidate": "charter",
    "driving_forces": "driving_forces.schema",
    "forces": "forces",
    "forces_ranked": "forces_ranked",
    "clusters": "clusters",
    "washout_report": "washout_report.schema",
    "evidence_units": "evidence_units.schema",
    "evidence_units_uploads": "evidence_units.schema",
    "coverage_report": "coverage_report",
    "certainty_uncertainty": "certainty_uncertainty.schema",
    "belief_sets": "belief_sets.schema",
    "effects": "effects.schema",
    "retrieval_report": "retrieval_report",
    "epistemic_summary": "epistemic_summary",
    "uncertainties": "uncertainties",
    "uncertainty_axes": "uncertainty_axes",
    "logic": "logic",
    "skeletons": "skeleton",
    "ewi": "ewi",
    "strategies": "strategies",
    "wind_tunnel": "wind_tunnel",
    "daily_brief": "daily_brief",
    "scenario_profiles": "scenario_profiles",
    "scenarios": "scenarios",
    "trace_map": "trace_map",
    "audit_report": "audit_report",
    "index": "artifact_index",
    "view_model": "view_model",
    "prompt_manifest": "prompt_manifest",
    "company_profile": "company_profile",
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
