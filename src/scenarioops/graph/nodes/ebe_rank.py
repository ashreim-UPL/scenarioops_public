from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from scenarioops.graph.state import ScenarioOpsState
from scenarioops.graph.tools.schema_validate import validate_artifact
from scenarioops.graph.tools.storage import write_artifact
from scenarioops.graph.tools.traceability import build_run_metadata


_GRADE_SCORE = {"A": 5.0, "B": 4.0, "C": 3.0, "D": 2.0}
_DIMENSION_WEIGHT = {
    "cost": 1.0,
    "demand": 1.0,
    "regulation": 1.0,
    "tech_feasibility": 0.8,
    "capital": 0.8,
    "labor": 0.6,
    "risk": 0.6,
}


def _emergence_score(force: Mapping[str, Any]) -> float:
    text = f"{force.get('label', '')} {force.get('mechanism', '')}".lower()
    score = 3.0
    if any(token in text for token in ("emerging", "nascent", "accelerat", "disrupt")):
        score = 4.0
    if any(token in text for token in ("breakthrough", "paradigm", "revolution")):
        score = 5.0
    if any(token in text for token in ("mature", "stagnant", "declin")):
        score = 2.0
    layer = str(force.get("layer", "")).lower()
    if layer == "primary":
        score = min(5.0, score + 0.5)
    elif layer == "tertiary":
        score = max(1.0, score - 0.5)
    return max(0.0, min(5.0, score))


def _business_impact_score(force: Mapping[str, Any]) -> float:
    dims = [str(item).lower() for item in force.get("affected_dimensions", [])]
    if not dims:
        return 2.5
    total = sum(_DIMENSION_WEIGHT.get(dim, 0.5) for dim in dims)
    score = min(5.0, 2.0 + total)
    return max(0.0, score)


def _evidence_strength_score(
    force: Mapping[str, Any], evidence_scores: Mapping[str, float]
) -> float:
    ids = [str(item) for item in force.get("evidence_unit_ids", [])]
    if not ids:
        return 0.0
    scores = [evidence_scores.get(item, 0.0) for item in ids if item in evidence_scores]
    if not scores:
        return 0.0
    avg = sum(scores) / len(scores)
    return max(0.0, min(5.0, avg))


def _coverage_stats(forces: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(forces)
    linked = sum(1 for force in forces if not force.get("unlinked"))
    by_layer: dict[str, int] = {}
    for force in forces:
        layer = str(force.get("layer", "unknown"))
        by_layer[layer] = by_layer.get(layer, 0) + 1
    return {
        "total_forces": total,
        "linked_forces": linked,
        "linked_ratio": linked / max(1, total),
        "by_layer": by_layer,
    }


def run_ebe_rank_node(
    *,
    run_id: str,
    state: ScenarioOpsState,
    base_dir: Path | None = None,
) -> ScenarioOpsState:
    if state.forces is None:
        raise ValueError("Forces are required before EBE scoring.")
    if state.evidence_units is None:
        raise ValueError("Evidence units are required before EBE scoring.")

    forces_payload = state.forces
    forces = forces_payload.get("forces", [])
    if not isinstance(forces, list):
        raise TypeError("Forces payload must include a list of forces.")

    evidence_units = state.evidence_units.get("evidence_units", [])
    evidence_scores = {}
    if isinstance(evidence_units, list):
        for unit in evidence_units:
            if not isinstance(unit, Mapping):
                continue
            unit_id = unit.get("evidence_unit_id")
            grade = str(unit.get("reliability_grade", "")).upper()
            if unit_id and grade in _GRADE_SCORE:
                evidence_scores[str(unit_id)] = _GRADE_SCORE[grade]

    weights = {"emergence": 0.4, "business_impact": 0.4, "evidence_strength": 0.2}
    ranked = []
    for force in forces:
        if not isinstance(force, Mapping):
            continue
        emergence = _emergence_score(force)
        impact = _business_impact_score(force)
        evidence_strength = _evidence_strength_score(force, evidence_scores)
        ebe_score = (
            emergence * weights["emergence"]
            + impact * weights["business_impact"]
            + evidence_strength * weights["evidence_strength"]
        )
        ranked.append(
            {
                "force_id": str(force.get("force_id")),
                "E_emergence": emergence,
                "B_business_impact": impact,
                "E_evidence_strength": evidence_strength,
                "ebe_score": ebe_score,
                "rationale": f"{force.get('label', '')}: {force.get('mechanism', '')}",
            }
        )

    ranked.sort(key=lambda item: (-item["ebe_score"], item["force_id"]))
    force_map = {str(force.get("force_id")): force for force in forces if isinstance(force, Mapping)}
    top_by_layer: dict[str, list[str]] = {}
    top_by_domain: dict[str, list[str]] = {}
    for entry in ranked:
        force_id = entry["force_id"]
        force = force_map.get(force_id, {})
        layer = str(force.get("layer", "unknown"))
        domain = str(force.get("domain", "unknown"))
        top_by_layer.setdefault(layer, [])
        top_by_domain.setdefault(domain, [])
        if len(top_by_layer[layer]) < 5:
            top_by_layer[layer].append(force_id)
        if len(top_by_domain[domain]) < 5:
            top_by_domain[domain].append(force_id)
    user_params: dict[str, Any] = {}
    if isinstance(state.company_profile, Mapping):
        user_params = {
            "value": state.company_profile.get("company_name"),
            "scope": state.company_profile.get("geography"),
            "horizon": state.company_profile.get("horizon_months"),
        }
    metadata = build_run_metadata(
        run_id=run_id,
        user_params=user_params,
        focal_issue=state.focal_issue if isinstance(state.focal_issue, Mapping) else None,
    )
    payload = {
        **metadata,
        "weights": weights,
        "coverage_stats": _coverage_stats([dict(force) for force in forces if isinstance(force, Mapping)]),
        "top_by_layer": top_by_layer,
        "top_by_domain": top_by_domain,
        "forces": ranked,
    }
    validate_artifact("forces_ranked", payload)

    write_artifact(
        run_id=run_id,
        artifact_name="forces_ranked",
        payload=payload,
        ext="json",
        input_values={"force_count": len(ranked)},
        tool_versions={"ebe_rank_node": "0.1.0"},
        base_dir=base_dir,
    )
    state.forces_ranked = payload
    return state
