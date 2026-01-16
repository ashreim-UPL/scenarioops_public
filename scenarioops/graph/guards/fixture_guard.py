from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable, Mapping

from scenarioops.app.config import ScenarioOpsSettings
from scenarioops.graph.state import Drivers, ScenarioOpsState
from scenarioops.graph.tools.storage import default_runs_dir, write_artifact, write_latest_status


_FIXTURE_NAME_RE = re.compile(r".*(signal|driver)\s+\d+", re.IGNORECASE)


def _iter_entries(items: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(items, list):
        return [entry for entry in items if isinstance(entry, Mapping)]
    if isinstance(items, Drivers):
        return [driver.__dict__ for driver in items.drivers]
    return []


def _collect_fixture_evidence(entries: Iterable[Mapping[str, Any]]) -> list[str]:
    evidence: list[str] = []
    for entry in entries:
        name = str(entry.get("name", ""))
        if _FIXTURE_NAME_RE.search(name):
            evidence.append(f"name:{name}")
        citations = entry.get("citations", [])
        if not isinstance(citations, Iterable):
            continue
        for citation in citations:
            if not isinstance(citation, Mapping):
                continue
            url = str(citation.get("url", "")).lower()
            if "example.com" in url:
                evidence.append(f"url:{url}")
            excerpt_hash = str(citation.get("excerpt_hash", ""))
            if excerpt_hash.startswith("hash-"):
                evidence.append(f"excerpt_hash:{excerpt_hash}")
    return evidence


def detect_fixture_evidence(state: ScenarioOpsState) -> list[str]:
    evidence: list[str] = []
    driving_forces = state.driving_forces or {}
    if isinstance(driving_forces, Mapping):
        evidence.extend(_collect_fixture_evidence(_iter_entries(driving_forces.get("forces", []))))
    evidence.extend(_collect_fixture_evidence(_iter_entries(state.drivers)))
    return evidence


def _write_fixture_washout_report(
    run_id: str, evidence: list[str], base_dir: Path | None
) -> None:
    payload = {
        "duplicate_ratio": 1.0,
        "duplicate_groups": [],
        "undercovered_domains": [],
        "missing_categories": [],
        "proposed_forces": [],
        "reason": "fixture_content_detected",
        "notes": "; ".join(evidence[:5]),
    }
    write_artifact(
        run_id=run_id,
        artifact_name="washout_report",
        payload=payload,
        ext="json",
        input_values={"fixture_evidence_count": len(evidence)},
        prompt_values={"prompt": "fixture_guard"},
        tool_versions={"fixture_guard": "0.1.0"},
        base_dir=base_dir,
    )


def validate_or_fail(
    *,
    run_id: str,
    state: ScenarioOpsState,
    settings: ScenarioOpsSettings,
    base_dir: Path | None = None,
    command: str | None = None,
) -> None:
    if settings.mode != "live" or not settings.forbid_fixture_citations:
        return

    evidence = detect_fixture_evidence(state)
    if not evidence:
        return

    if base_dir is None:
        base_dir = default_runs_dir()
    _write_fixture_washout_report(run_id, evidence, base_dir)
    write_latest_status(
        run_id=run_id,
        status="FAIL",
        command=command or "fixture_guard",
        error_summary="fixture_content_detected",
        base_dir=base_dir,
        run_config=settings.as_dict(),
    )
    raise RuntimeError(
        "LIVE run contains fixture content; check retriever and sources policy."
    )
