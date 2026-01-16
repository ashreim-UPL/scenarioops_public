from __future__ import annotations

import re
from typing import Any, Iterable, Mapping

_FIXTURE_NAME_RE = re.compile(r".*(signal|driver)\s+\d+", re.IGNORECASE)


def detect_fixture_markers(driving_forces: Mapping[str, Any]) -> list[str]:
    forces = driving_forces.get("forces", [])
    if not isinstance(forces, list):
        return []
    evidence: list[str] = []
    for force in forces:
        if not isinstance(force, Mapping):
            continue
        name = str(force.get("name", ""))
        if _FIXTURE_NAME_RE.search(name):
            evidence.append(f"name:{name}")
        citations = force.get("citations", [])
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


def assert_no_fixture_forces(
    driving_forces: Mapping[str, Any], *, mode: str
) -> None:
    if str(mode).strip().lower() == "demo":
        return
    evidence = detect_fixture_markers(driving_forces)
    if evidence:
        raise RuntimeError("Fixture data detected in live mode.")
