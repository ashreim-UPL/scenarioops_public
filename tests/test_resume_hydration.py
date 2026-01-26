from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scenarioops.app.config import ScenarioOpsSettings
from scenarioops.graph.build_graph import run_graph
from scenarioops.graph.types import GraphInputs


def _write_artifact(base_dir: Path, run_id: str, name: str, payload: dict[str, Any]) -> None:
    path = base_dir / run_id / "artifacts"
    path.mkdir(parents=True, exist_ok=True)
    (path / f"{name}.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _charter_payload() -> dict[str, Any]:
    return {
        "id": "charter-1",
        "title": "Test Charter",
        "purpose": "Test purpose",
        "decision_context": "Test context",
        "scope": "Global",
        "time_horizon": "12 months",
        "stakeholders": ["Exec"],
        "constraints": ["None"],
        "assumptions": ["Stable market"],
        "success_criteria": ["Actionable"],
    }


def _focal_issue_payload() -> dict[str, Any]:
    return {
        "focal_issue": "Test issue",
        "scope": {
            "geography": "Global",
            "sectors": ["Technology"],
            "time_horizon_months": 60,
        },
        "decision_type": "Strategy",
        "exclusions": ["Ops"],
        "success_criteria": "Clarity",
    }


def _company_profile_payload(run_id: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "company_name": "Acme",
        "geography": "Global",
        "horizon_months": 60,
        "source_basis": {"urls": [], "internal_docs": [], "manual_input": "Acme"},
        "simulated": False,
    }


def _evidence_units_payload(run_id: str) -> dict[str, Any]:
    timestamp = datetime.now(timezone.utc).isoformat()
    return {
        "run_id": run_id,
        "timestamp": timestamp,
        "company_name": "Acme",
        "geography": "Global",
        "horizon_months": 60,
        "simulated": False,
        "evidence_units": [
            {
                "evidence_unit_id": "ev-1",
                "source_type": "primary",
                "title": "Evidence",
                "publisher": "Example",
                "date_published": timestamp,
                "url": "https://example.com",
                "excerpt": "Evidence excerpt.",
                "claims": [],
                "metrics": [],
                "reliability_grade": "A",
                "reliability_reason": "test",
                "geography_tags": ["Global"],
                "domain_tags": [],
                "simulated": False,
            }
        ],
    }


def _read_node_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entries.append(json.loads(line))
    return entries


def _settings() -> ScenarioOpsSettings:
    return ScenarioOpsSettings(
        mode="demo",
        llm_provider="mock",
        sources_policy="fixtures",
        allow_web=False,
        min_forces=1,
        min_forces_per_domain=0,
    )


def test_resume_hydrates_existing_artifacts(tmp_path: Path) -> None:
    base_dir = tmp_path / "runs"
    run_id = "resume-artifacts"
    _write_artifact(base_dir, run_id, "scenario_charter", _charter_payload())
    _write_artifact(base_dir, run_id, "focal_issue", _focal_issue_payload())
    _write_artifact(base_dir, run_id, "company_profile", _company_profile_payload(run_id))
    _write_artifact(base_dir, run_id, "evidence_units", _evidence_units_payload(run_id))

    inputs = GraphInputs(
        user_params={"scope": "world", "value": "Acme", "horizon": 12},
        sources=[],
        signals=[],
    )
    run_graph(
        inputs,
        run_id=run_id,
        base_dir=base_dir,
        settings=_settings(),
        generate_strategies=False,
        resume_from="forces",
    )

    log_path = base_dir / run_id / "logs" / "node_events.jsonl"
    events = _read_node_events(log_path)
    retrieval_events = [item for item in events if item.get("node") == "retrieval_real"]
    assert retrieval_events
    assert all(item.get("status") == "HYDRATED" for item in retrieval_events)


def test_resume_uses_cache_when_missing_artifact(tmp_path: Path) -> None:
    base_dir = tmp_path / "runs"
    run_id = "resume-cache"
    _write_artifact(base_dir, run_id, "scenario_charter", _charter_payload())
    _write_artifact(base_dir, run_id, "focal_issue", _focal_issue_payload())
    _write_artifact(base_dir, run_id, "company_profile", _company_profile_payload(run_id))

    cache_dir = tmp_path / "cache" / "evidence_units"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached_payload = {
        "cache_key": "cache-1",
        "cached_at": datetime.now(timezone.utc).isoformat(),
        "metadata": {
            "company_name": "Acme",
            "geography": "Global",
            "horizon_months": 60,
        },
        "evidence_units": _evidence_units_payload(run_id)["evidence_units"],
    }
    (cache_dir / "cache-1.json").write_text(
        json.dumps(cached_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    inputs = GraphInputs(
        user_params={"scope": "world", "value": "Acme", "horizon": 12},
        sources=[],
        signals=[],
    )
    run_graph(
        inputs,
        run_id=run_id,
        base_dir=base_dir,
        settings=_settings(),
        generate_strategies=False,
        resume_from="forces",
    )

    evidence_path = base_dir / run_id / "artifacts" / "evidence_units.json"
    assert evidence_path.exists()
    payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert payload.get("evidence_units")

    log_path = base_dir / run_id / "logs" / "node_events.jsonl"
    events = _read_node_events(log_path)
    retrieval_events = [item for item in events if item.get("node") == "retrieval_real"]
    assert retrieval_events
    assert all(item.get("status") == "HYDRATED" for item in retrieval_events)
