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
    drivers = _load_jsonl(artifacts_dir / "drivers.jsonl")
    drivers_by_domain = _group_drivers_by_domain(drivers)

    uncertainties_payload = _load_json(artifacts_dir / "uncertainties.json") or {}
    uncertainties = uncertainties_payload.get("uncertainties", [])

    scenario_logic = _load_json(artifacts_dir / "logic.json") or {}
    skeletons_payload = _load_json(artifacts_dir / "skeletons.json") or {}
    scenarios = skeletons_payload.get("scenarios", [])
    if not scenarios:
        scenarios = scenario_logic.get("scenarios", [])

    narratives = _load_narratives(artifacts_dir)

    ewi_payload = _load_json(artifacts_dir / "ewi.json") or {}
    ewis = ewi_payload.get("indicators", [])

    daily_brief_md = _load_text(artifacts_dir / "daily_brief.md")

    latest_status = read_latest_status(run_dir.parent) or {}
    run_meta = {
        "run_id": run_dir.name,
        "status": latest_status.get("status", "unknown"),
    }

    return {
        "charter": charter,
        "drivers": drivers,
        "drivers_by_domain": drivers_by_domain,
        "uncertainties": uncertainties,
        "scenario_logic": scenario_logic,
        "scenarios": scenarios,
        "narratives": narratives,
        "ewis": ewis,
        "daily_brief_md": daily_brief_md,
        "run_meta": run_meta,
    }
