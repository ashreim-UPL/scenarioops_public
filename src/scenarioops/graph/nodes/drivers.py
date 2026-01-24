from __future__ import annotations

import hashlib
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

from scenarioops.app.config import LLMConfig, ScenarioOpsSettings
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


def _safe_url(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if not isinstance(value, Mapping):
        return ""
    for key in ("url", "file_name", "source_url"):
        candidate = value.get(key)
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return ""


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
    *,
    allow_fixture_citations: bool = False,
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
            url = _safe_url(citation)
            normalized = _normalize_url(url)
            if normalized not in evidence_hashes:
                if not allow_fixture_citations:
                    raise ValueError(f"Citation url not in evidence units: {url}")
                citation.setdefault("excerpt_hash", _excerpt_hash(url))
                citation.setdefault("publisher", url or "unknown")
                citation.setdefault("evidence_id", url or "unknown")
                continue


def _relax_drivers_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    relaxed = deepcopy(schema)
    driver_schema = load_schema("driver_entry")
    required = driver_schema.get("required")
    if isinstance(required, list):
        driver_schema["required"] = []
    driver_schema["additionalProperties"] = True
    citations = driver_schema.get("properties", {}).get("citations")
    if isinstance(citations, dict):
        items = citations.get("items")
        if isinstance(items, dict):
            items["additionalProperties"] = True
    try:
        relaxed["properties"]["drivers"]["items"] = driver_schema
    except Exception:
        pass
    return relaxed


def _drivers_from_forces(
    forces: Sequence[Mapping[str, Any]],
    *,
    evidence_hashes: Mapping[str, str],
    evidence_publishers: Mapping[str, str],
    evidence_ids: Mapping[str, str],
    run_id: str,
    base_dir: Path | None,
) -> list[dict[str, Any]]:
    drivers: list[dict[str, Any]] = []
    for force in forces:
        name = str(force.get("name") or "Unknown driver").strip()
        description = str(
            force.get("description")
            or force.get("why_it_matters")
            or "No description provided."
        ).strip()
        category = str(force.get("domain") or "unknown").strip()
        trend = str(force.get("trend") or "increasing").strip()
        impact = str(force.get("impact") or "medium").strip()
        citations = force.get("citations", [])
        if not isinstance(citations, list):
            citations = []
        normalized_citations: list[dict[str, Any]] = []
        for citation in citations:
            if not isinstance(citation, Mapping):
                continue
            url = _safe_url(citation)
            if not url:
                continue
            normalized = _normalize_url(url)
            excerpt_hash = citation.get("excerpt_hash") or evidence_hashes.get(normalized)
            if not excerpt_hash:
                excerpt_hash = _excerpt_hash(url)
            publisher = citation.get("publisher") or evidence_publishers.get(normalized) or "unknown"
            evidence_id = citation.get("evidence_id") or evidence_ids.get(normalized) or normalized
            normalized_citations.append(
                {
                    "url": url,
                    "excerpt_hash": str(excerpt_hash),
                    "publisher": str(publisher),
                    "evidence_id": str(evidence_id),
                }
            )

        entry = {
            "id": stable_id("driver", name, description, category, trend),
            "name": name,
            "description": description,
            "category": category,
            "trend": trend,
            "impact": impact,
            "citations": normalized_citations,
        }
        log_normalization(
            run_id=run_id,
            node_name="drivers",
            operation="fallback_from_forces",
            details={"name": name},
            base_dir=base_dir,
        )
        drivers.append(entry)
    return drivers


def _normalize_citation_value(
    value: Any,
    evidence_hashes: Mapping[str, str],
    evidence_publishers: Mapping[str, str],
    evidence_ids: Mapping[str, str],
) -> dict[str, str] | None:
    url = _safe_url(value)
    if not url and isinstance(value, Mapping):
        url = str(
            value.get("link")
            or value.get("source")
            or value.get("href")
            or ""
        ).strip()
    if not url:
        return None
    normalized = _normalize_url(url)
    excerpt_hash = ""
    if isinstance(value, Mapping):
        excerpt_hash = str(value.get("excerpt_hash") or "")
    if not excerpt_hash:
        excerpt_hash = evidence_hashes.get(normalized) or _excerpt_hash(url)
    publisher = ""
    evidence_id = ""
    if isinstance(value, Mapping):
        publisher = str(value.get("publisher") or "")
        evidence_id = str(value.get("evidence_id") or "")
    if not publisher:
        publisher = evidence_publishers.get(normalized) or "unknown"
    if not evidence_id:
        evidence_id = evidence_ids.get(normalized) or normalized
    return {
        "url": url,
        "excerpt_hash": str(excerpt_hash),
        "publisher": str(publisher),
        "evidence_id": str(evidence_id),
    }


def _normalize_driver_entry(
    entry: Mapping[str, Any],
    *,
    evidence_hashes: Mapping[str, str],
    evidence_publishers: Mapping[str, str],
    evidence_ids: Mapping[str, str],
) -> dict[str, Any]:
    data = dict(entry)
    driver_data = data.get("driver")
    if isinstance(driver_data, Mapping):
        if not data.get("name"):
            data["name"] = driver_data.get("name") or driver_data.get("title")
        if not data.get("description"):
            data["description"] = driver_data.get("description") or driver_data.get("summary")
        if not data.get("category"):
            data["category"] = driver_data.get("category") or driver_data.get("domain")
        if not data.get("trend"):
            data["trend"] = driver_data.get("trend")
        if not data.get("impact"):
            data["impact"] = driver_data.get("impact")
        if not data.get("citations") and driver_data.get("citations"):
            data["citations"] = driver_data.get("citations")

    if "citations" not in data and "citation" in data:
        data["citations"] = data.get("citation")

    citations = data.get("citations", [])
    if isinstance(citations, Mapping) or isinstance(citations, str):
        citations = [citations]
    normalized_citations: list[dict[str, str]] = []
    if isinstance(citations, list):
        for item in citations:
            normalized = _normalize_citation_value(
                item,
                evidence_hashes=evidence_hashes,
                evidence_publishers=evidence_publishers,
                evidence_ids=evidence_ids,
            )
            if normalized:
                normalized_citations.append(normalized)
    data["citations"] = normalized_citations

    allowed_keys = {
        "id",
        "name",
        "description",
        "category",
        "trend",
        "impact",
        "citations",
        "evidence",
        "signals",
        "confidence",
        "notes",
    }
    cleaned: dict[str, Any] = {}
    for key in allowed_keys:
        if key in data:
            cleaned[key] = data[key]
    return cleaned


def run_drivers_node(
    *,
    run_id: str,
    state: ScenarioOpsState,
    llm_client=None,
    base_dir: Path | None = None,
    config: LLMConfig | None = None,
    min_citations: int = 1,
    settings: ScenarioOpsSettings | None = None,
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
        url = _safe_url(entry)
        excerpt = str(entry.get("excerpt", ""))
        publisher = str(entry.get("publisher", "")) or url or str(entry.get("file_name", ""))
        evidence_id = (
            str(entry.get("evidence_unit_id") or entry.get("id") or "")
            or url
            or str(entry.get("file_name", ""))
        )
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
    response = client.generate_json(prompt, _relax_drivers_schema(schema))
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

    normalized_payload: list[dict[str, Any]] = []
    for entry in drivers_payload:
        if not isinstance(entry, Mapping):
            normalized_payload.append({})
            continue
        normalized = _normalize_driver_entry(
            entry,
            evidence_hashes=evidence_hashes,
            evidence_publishers=evidence_publishers,
            evidence_ids=evidence_ids,
        )
        if normalized != entry:
            log_normalization(
                run_id=run_id,
                node_name="drivers",
                operation="normalized_driver_payload",
                details={"keys": sorted(set(entry.keys()) - set(normalized.keys()))},
                base_dir=base_dir,
            )
        normalized_payload.append(normalized)
    drivers_payload = normalized_payload

    required_fields = ("id", "name", "description", "category", "trend", "impact", "citations")
    missing_required = False
    for entry in drivers_payload:
        if not isinstance(entry, Mapping):
            missing_required = True
            break
        for field in required_fields:
            value = entry.get(field)
            if field == "citations":
                if not isinstance(value, list) or not value:
                    missing_required = True
                    break
            else:
                if not isinstance(value, str) or not value.strip():
                    missing_required = True
                    break
        if missing_required:
            break

    if missing_required:
        forces = []
        if isinstance(state.driving_forces, dict):
            forces = state.driving_forces.get("forces", [])
        if not isinstance(forces, list) or not forces:
            raise ValueError("Drivers payload missing required fields and no driving forces to recover.")
        drivers_payload = _drivers_from_forces(
            forces,
            evidence_hashes=evidence_hashes,
            evidence_publishers=evidence_publishers,
            evidence_ids=evidence_ids,
            run_id=run_id,
            base_dir=base_dir,
        )

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

    allow_fixture_citations = (
        settings is not None and settings.sources_policy == "fixtures"
    )
    _validate_citations(
        drivers_payload,
        evidence_hashes,
        min_citations,
        allow_fixture_citations=allow_fixture_citations,
    )
    for entry in drivers_payload:
        citations = entry.get("citations", [])
        for citation in citations:
            url = _safe_url(citation)
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
