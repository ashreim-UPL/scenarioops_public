from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

from scenarioops.app.config import LLMConfig
from scenarioops.graph.nodes.utils import get_client, load_prompt, render_prompt
from scenarioops.graph.state import DriverEntry, Drivers, ScenarioOpsState
from scenarioops.graph.tools.normalization import stable_id
from scenarioops.graph.tools.schema_validate import load_schema, validate_jsonl
from scenarioops.graph.tools.storage import log_normalization, write_artifact
from scenarioops.llm.guards import ensure_dict, ensure_key, ensure_list, truncate_for_log


def _normalize_url(url: str) -> str:
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return url.strip().rstrip("/")
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    path = parsed.path or ""
    if path != "/" and path.endswith("/"):
        path = path[:-1]
    if parsed.query:
        path = f"{path}?{parsed.query}"
    return f"{scheme}://{netloc}{path}"


def _excerpt_hash(text: str, length: int = 1000) -> str:
    excerpt = text[:length]
    return hashlib.sha256(excerpt.encode("utf-8")).hexdigest()


def _evidence_payload(state: ScenarioOpsState) -> dict[str, Any]:
    if isinstance(state.evidence_units, dict):
        return state.evidence_units
    return {"evidence_units": []}


def _validate_citations(
    drivers: Sequence[Mapping[str, Any]],
    evidence_hashes: Mapping[str, str],
    min_citations: int,
) -> None:
    for entry in drivers:
        citations = entry.get("citations", [])
        if not citations:
            raise ValueError(f"Driver {entry.get('id')} missing citations.")
        if min_citations > 1 and len(citations) < min_citations:
            raise ValueError(
                f"Driver {entry.get('id')} requires at least {min_citations} citations."
            )
        for citation in citations:
            url = str(citation.get("url", ""))
            normalized = _normalize_url(url)
            if normalized not in evidence_hashes:
                raise ValueError(f"Citation url not in evidence units: {url}")


def run_drivers_node(
    *,
    run_id: str,
    state: ScenarioOpsState,
    llm_client=None,
    base_dir: Path | None = None,
    config: LLMConfig | None = None,
    min_citations: int = 1,
) -> ScenarioOpsState:
    prompt_template = load_prompt("drivers")
    evidence_payload = _evidence_payload(state)
    evidence_units = evidence_payload.get("evidence_units", [])
    if not isinstance(evidence_units, list) or not evidence_units:
        raise ValueError("Evidence units are required before running drivers.")

    evidence_with_hash: list[dict[str, Any]] = []
    evidence_hashes: dict[str, str] = {}
    evidence_publishers: dict[str, str] = {}
    evidence_ids: dict[str, str] = {}
    for entry in evidence_units:
        if not isinstance(entry, Mapping):
            continue
        url = str(entry.get("url", ""))
        excerpt = str(entry.get("excerpt", ""))
        publisher = str(entry.get("publisher", "")) or url
        evidence_id = str(entry.get("id", "")) or url
        excerpt_hash = _excerpt_hash(excerpt)
        normalized = _normalize_url(url)
        evidence_hashes[normalized] = excerpt_hash
        evidence_publishers[normalized] = publisher
        evidence_ids[normalized] = evidence_id
        enriched = dict(entry)
        enriched["excerpt_hash"] = excerpt_hash
        evidence_with_hash.append(enriched)

    context = {
        "charter": state.charter.__dict__ if state.charter else None,
        "evidence_units": evidence_with_hash,
    }
    prompt = render_prompt(prompt_template, context)
    client = get_client(llm_client, config)
    schema = load_schema("drivers_list")
    response = client.generate_json(prompt, schema)
    try:
        payload = ensure_dict(response, node_name="drivers")
    except TypeError as exc:
        if isinstance(response, list):
            raw = truncate_for_log(getattr(response, "raw", None) or repr(response))
            raise TypeError(
                "drivers: Expected payload['drivers'] list, "
                f"received {type(response)}. raw={raw}"
            ) from exc
        raise

    drivers_payload = ensure_key(payload, "drivers", node_name="drivers")
    if not isinstance(drivers_payload, list):
        raw = truncate_for_log(getattr(payload, "raw", None) or repr(payload))
        raise TypeError(
            "drivers: Expected payload['drivers'] list, "
            f"received {type(drivers_payload)}. raw={raw}"
        )
    drivers_payload = ensure_list(drivers_payload, node_name="drivers")

    for entry in drivers_payload:
        if not entry.get("id"):
            entry["id"] = stable_id(
                "driver",
                entry.get("name"),
                entry.get("description"),
                entry.get("category"),
                entry.get("trend"),
            )
            log_normalization(
                run_id=run_id,
                node_name="drivers",
                operation="stable_id_assigned",
                details={"field": "id", "name": entry.get("name", "")},
                base_dir=base_dir,
            )

    _validate_citations(drivers_payload, evidence_hashes, min_citations)
    for entry in drivers_payload:
        citations = entry.get("citations", [])
        for citation in citations:
            url = str(citation.get("url", ""))
            normalized = _normalize_url(url)
            if normalized in evidence_hashes:
                citation["excerpt_hash"] = evidence_hashes[normalized]
                citation["publisher"] = evidence_publishers.get(
                    normalized, citation.get("publisher", "")
                )
                citation["evidence_id"] = evidence_ids.get(
                    normalized, citation.get("evidence_id", "")
                )

    validate_jsonl("driver_entry", drivers_payload)

    write_artifact(
        run_id=run_id,
        artifact_name="drivers",
        payload=drivers_payload,
        ext="jsonl",
        input_values={"evidence_unit_count": len(evidence_units)},
        prompt_values={"prompt": prompt},
        tool_versions={"drivers_node": "0.1.0"},
        base_dir=base_dir,
    )

    driver_entries = [DriverEntry(**entry) for entry in drivers_payload]
    state.drivers = Drivers(id=f"drivers-{run_id}", title="Drivers", drivers=driver_entries)
    return state
