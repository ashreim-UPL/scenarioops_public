from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scenarioops.graph.tools.storage import read_latest_status


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    items: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            items.append(parsed)
    return items


def _load_text(path: Path) -> str | None:
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def _group_drivers_by_domain(drivers: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for entry in drivers:
        domain = str(entry.get("category") or entry.get("domain") or "Other")
        grouped.setdefault(domain, []).append(entry)
    return {key: grouped[key] for key in sorted(grouped.keys())}


def _load_narratives(artifacts_dir: Path) -> list[dict[str, str]]:
    narratives: list[dict[str, str]] = []
    for path in sorted(artifacts_dir.glob("narrative_*.md")):
        scenario_id = path.stem.replace("narrative_", "")
        narratives.append({"scenario_id": scenario_id, "markdown": path.read_text(encoding="utf-8")})
    return narratives


def build_view_model(run_dir: Path) -> dict[str, Any]:
    artifacts_dir = run_dir / "artifacts"

    charter = _load_json(artifacts_dir / "scenario_charter.json")
    focal_issue = _load_json(artifacts_dir / "focal_issue.json")
    company_profile = _load_json(artifacts_dir / "company_profile.json")
    prompt_manifest = _load_json(artifacts_dir / "prompt_manifest.json")

    driving_forces_payload = _load_json(artifacts_dir / "driving_forces.json") or {}
    driving_forces = driving_forces_payload.get("forces", [])
    if not isinstance(driving_forces, list):
        driving_forces = []
    forces_payload = _load_json(artifacts_dir / "forces.json") or {}
    forces = forces_payload.get("forces", [])
    if not isinstance(forces, list):
        forces = []
    forces_ranked = _load_json(artifacts_dir / "forces_ranked.json")
    clusters_payload = _load_json(artifacts_dir / "clusters.json") or {}
    clusters = clusters_payload.get("clusters", [])
    if not isinstance(clusters, list):
        clusters = []
    drivers = _load_jsonl(artifacts_dir / "drivers.jsonl")
    drivers_by_domain = _group_drivers_by_domain(drivers)

    washout_report = _load_json(artifacts_dir / "washout_report.json")
    retrieval_report = _load_json(artifacts_dir / "retrieval_report.json")
    evidence_units_payload = _load_json(artifacts_dir / "evidence_units.json") or {}
    evidence_units = evidence_units_payload.get("evidence_units", [])
    if not isinstance(evidence_units, list):
        evidence_units = []
    belief_sets_payload = _load_json(artifacts_dir / "belief_sets.json") or {}
    belief_sets = belief_sets_payload.get("belief_sets", [])
    if not isinstance(belief_sets, list):
        belief_sets = []
    effects_payload = _load_json(artifacts_dir / "effects.json") or {}
    effects = effects_payload.get("effects", [])
    if not isinstance(effects, list):
        effects = []

    uncertainties_payload = _load_json(artifacts_dir / "uncertainties.json") or {}
    uncertainties = uncertainties_payload.get("uncertainties", [])
    uncertainty_axes_payload = _load_json(artifacts_dir / "uncertainty_axes.json") or {}
    uncertainty_axes = uncertainty_axes_payload.get("axes", [])
    if not isinstance(uncertainty_axes, list):
        uncertainty_axes = []

    scenario_logic = _load_json(artifacts_dir / "logic.json") or {}
    skeletons_payload = _load_json(artifacts_dir / "skeletons.json") or {}
    scenarios = skeletons_payload.get("scenarios", [])
    if not scenarios:
        scenarios = scenario_logic.get("scenarios", [])
    scenarios_payload = _load_json(artifacts_dir / "scenarios.json") or {}
    scenarios_v2 = scenarios_payload.get("scenarios", [])
    if isinstance(scenarios_v2, list) and scenarios_v2:
        scenarios = scenarios_v2

    narratives = _load_narratives(artifacts_dir)

    ewi_payload = _load_json(artifacts_dir / "ewi.json") or {}
    ewis = ewi_payload.get("indicators", [])

    daily_brief_md = _load_text(artifacts_dir / "daily_brief.md")
    strategies_payload = _load_json(artifacts_dir / "strategies.json") or {}
    strategies = strategies_payload.get("strategies", [])
    if not isinstance(strategies, list):
        strategies = []
    wind_tunnel_payload = _load_json(artifacts_dir / "wind_tunnel.json")

    latest_status = read_latest_status(run_dir.parent) or {}
    run_config = _load_json(run_dir / "run_config.json")
    run_meta = {
        "run_id": run_dir.name,
        "status": latest_status.get("status", "unknown"),
    }
    if run_config:
        run_meta["run_config"] = run_config

    return {
        "charter": charter,
        "focal_issue": focal_issue,
        "company_profile": company_profile,
        "prompt_manifest": prompt_manifest,
        "driving_forces": driving_forces,
        "forces": forces,
        "forces_ranked": forces_ranked,
        "clusters": clusters,
        "washout_report": washout_report,
        "retrieval_report": retrieval_report,
        "evidence_units": evidence_units,
        "belief_sets": belief_sets,
        "effects": effects,
        "drivers": drivers,
        "drivers_by_domain": drivers_by_domain,
        "uncertainties": uncertainties,
        "uncertainty_axes": uncertainty_axes,
        "scenario_logic": scenario_logic,
        "scenarios": scenarios,
        "narratives": narratives,
        "ewis": ewis,
        "strategies": strategies,
        "wind_tunnel": wind_tunnel_payload,
        "daily_brief_md": daily_brief_md,
        "run_meta": run_meta,
    }
