from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from datetime import datetime

from scenarioops.graph.tools.storage import read_latest_status, ensure_local_file
from scenarioops.storage.run_store import get_run_store, run_store_mode, runs_root


def _load_json(path: Path) -> dict[str, Any] | None:
    ensure_local_file(path)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    ensure_local_file(path)
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
    ensure_local_file(path)
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _derive_run_status(run_dir: Path) -> dict[str, Any] | None:
    log_path = run_dir / "logs" / "node_events.jsonl"
    ensure_local_file(log_path)
    if not log_path.exists():
        return None
    events = _load_jsonl(log_path)
    if not events:
        return None
    latest_by_node: dict[str, dict[str, Any]] = {}
    for entry in events:
        node = str(entry.get("node") or entry.get("step") or entry.get("id") or "")
        if not node:
            continue
        ts = _parse_ts(entry.get("timestamp"))
        current = latest_by_node.get(node)
        if not current:
            latest_by_node[node] = {"entry": entry, "timestamp": ts}
            continue
        current_ts = current.get("timestamp")
        if current_ts is None or (ts and ts > current_ts):
            latest_by_node[node] = {"entry": entry, "timestamp": ts}

    latest_entries = [item["entry"] for item in latest_by_node.values()]
    has_fail = any("FAIL" in str(entry.get("status") or "").upper() for entry in latest_entries)
    has_running = any(
        any(flag in str(entry.get("status") or "").upper() for flag in ["RUN", "START", "IN_PROGRESS"])
        for entry in latest_entries
    )
    if has_fail:
        return {"status": "FAIL"}
    if has_running:
        return {"status": "RUNNING"}
    return {"status": "OK"}


def _group_drivers_by_domain(drivers: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for entry in drivers:
        domain = str(entry.get("category") or entry.get("domain") or "Other")
        grouped.setdefault(domain, []).append(entry)
    return {key: grouped[key] for key in sorted(grouped.keys())}


def _load_narratives(artifacts_dir: Path) -> list[dict[str, str]]:
    if run_store_mode() == "gcs":
        run_dir = artifacts_dir.parent
        base_dir = run_dir.parent
        root_dir = runs_root()
        try:
            relative_prefix = base_dir.resolve().relative_to(root_dir.resolve())
        except Exception:
            relative_prefix = Path("")
        prefix = str(relative_prefix).replace("\\", "/").strip("/")
        if prefix:
            prefix = f"{prefix}/{run_dir.name}/artifacts/"
        else:
            prefix = f"{run_dir.name}/artifacts/"
        store = get_run_store()
        for key in store.list(prefix):
            if not key.startswith(prefix):
                continue
            name = key[len(prefix):]
            if not (name.startswith("narrative_") and name.endswith(".md")):
                continue
            local_path = artifacts_dir / name
            if not local_path.exists():
                local_path.parent.mkdir(parents=True, exist_ok=True)
                local_path.write_bytes(store.get_bytes(key))

    narratives: list[dict[str, str]] = []
    for path in sorted(artifacts_dir.glob("narrative_*.md")):
        scenario_id = path.stem.replace("narrative_", "")
        narratives.append({"scenario_id": scenario_id, "markdown": path.read_text(encoding="utf-8")})
    return narratives


def build_view_model(run_dir: Path) -> dict[str, Any]:
    artifacts_dir = run_dir / "artifacts"
    run_meta_payload = (
        _load_json(run_dir / "run.json")
        or _load_json(run_dir / "run_meta.json")
        or {}
    )

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
    scenarios_enriched_payload = _load_json(artifacts_dir / "scenarios_enriched.json") or {}
    scenarios_enriched = scenarios_enriched_payload.get("scenarios", [])
    if isinstance(scenarios_enriched, list) and scenarios_enriched:
        scenarios = scenarios_enriched

    narratives = _load_narratives(artifacts_dir)

    ewi_payload = _load_json(artifacts_dir / "ewi.json") or {}
    ewis = ewi_payload.get("indicators", [])

    daily_brief_md = _load_text(artifacts_dir / "daily_brief.md")
    strategies_payload = _load_json(artifacts_dir / "strategies.json") or {}
    strategies = strategies_payload.get("strategies", [])
    if not isinstance(strategies, list):
        strategies = []
    wind_tunnel_payload = _load_json(artifacts_dir / "wind_tunnel.json")
    wind_tunnel_evaluations_v2 = _load_json(
        artifacts_dir / "wind_tunnel_evaluations_v2.json"
    )

    latest_status = read_latest_status(run_dir.parent) or {}
    run_config = _load_json(run_dir / "run_config.json")
    derived_status = _derive_run_status(run_dir) or {}
    status = derived_status.get("status")
    if not status and latest_status.get("run_id") == run_dir.name:
        status = latest_status.get("status")
    run_meta: dict[str, Any] = {}
    if isinstance(run_meta_payload, dict):
        run_meta.update(run_meta_payload)
    run_meta["run_id"] = run_dir.name
    run_meta["status"] = status or run_meta.get("status") or "unknown"
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
        "wind_tunnel_evaluations_v2": wind_tunnel_evaluations_v2,
        "daily_brief_md": daily_brief_md,
        "run_meta": run_meta,
    }
