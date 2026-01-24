from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

from scenarioops.app.config import LLMConfig
from scenarioops.app.config import ScenarioOpsSettings
from scenarioops.graph.nodes.utils import get_client, load_prompt, render_prompt
from scenarioops.graph.state import ScenarioOpsState
from scenarioops.graph.tools.normalization import stable_id
from scenarioops.graph.tools.schema_validate import load_schema, validate_artifact
from scenarioops.graph.tools.storage import log_normalization, write_artifact
from scenarioops.llm.guards import ensure_dict
from scenarioops.graph.nodes.drivers import _normalize_url as _normalize_citation_url
from scenarioops.graph.nodes.drivers import _excerpt_hash as _hash_excerpt


def _scope_payload(state: ScenarioOpsState) -> dict[str, Any]:
    if isinstance(state.focal_issue, dict):
        scope = state.focal_issue.get("scope")
        if isinstance(scope, dict):
            return scope
    return {}


def _evidence_payload(state: ScenarioOpsState) -> dict[str, Any]:
    if isinstance(state.evidence_units, dict):
        return state.evidence_units
    return {"evidence_units": []}


def _evidence_maps(
    state: ScenarioOpsState,
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    evidence_units = _evidence_payload(state).get("evidence_units", [])
    hashes: dict[str, str] = {}
    publishers: dict[str, str] = {}
    evidence_ids: dict[str, str] = {}
    if not isinstance(evidence_units, list):
        return hashes, publishers, evidence_ids
    for entry in evidence_units:
        if not isinstance(entry, dict):
            continue
        url = _safe_citation_url(entry)
        excerpt = str(entry.get("excerpt", ""))
        publisher = str(entry.get("publisher", "")) or url or str(entry.get("file_name", ""))
        evidence_id = (
            str(entry.get("evidence_unit_id") or entry.get("id") or "")
            or url
            or str(entry.get("file_name", ""))
        )
        normalized = _normalize_citation_url(url)
        hashes[normalized] = _hash_excerpt(excerpt)
        publishers[normalized] = publisher
        evidence_ids[normalized] = evidence_id
    return hashes, publishers, evidence_ids


def _safe_citation_url(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if not isinstance(value, Mapping):
        return ""
    for key in ("url", "file_name", "source_url"):
        candidate = value.get(key)
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return ""


def _focus_payload(
    domains: Iterable[str] | None, categories: Iterable[str] | None
) -> dict[str, list[str]]:
    return {
        "domains": [item for item in (domains or []) if item],
        "categories": [item for item in (categories or []) if item],
    }


def run_scan_node(
    *,
    run_id: str,
    state: ScenarioOpsState,
    llm_client=None,
    base_dir: Path | None = None,
    config: LLMConfig | None = None,
    min_forces: int = 30,
    min_per_domain: int = 5,
    focus_domains: Iterable[str] | None = None,
    focus_categories: Iterable[str] | None = None,
    settings: ScenarioOpsSettings | None = None,
) -> ScenarioOpsState:
    prompt_template = load_prompt("scan_pestel")
    context = {
        "scope_json": _scope_payload(state),
        "evidence_units_json": _evidence_payload(state),
        "min_forces": min_forces,
        "min_per_domain": min_per_domain,
        "scan_focus": _focus_payload(focus_domains, focus_categories),
    }
    prompt = render_prompt(prompt_template, context)
    client = get_client(llm_client, config)
    schema = load_schema("driving_forces.schema")
    response = client.generate_json(prompt, schema)
    parsed = ensure_dict(response, node_name="scan_pestel")
    evidence_hashes, evidence_publishers, evidence_ids = _evidence_maps(state)
    if not evidence_hashes:
        raise ValueError("Evidence units are required before scan.")
    allow_fixture_citations = (
        settings is not None and settings.sources_policy == "fixtures"
    )
    
    raw_forces = parsed.get("forces", [])
    normalized_forces = []

    if isinstance(raw_forces, list):
        for force in raw_forces:
            if not isinstance(force, dict):
                raise ValueError("Driving forces must be objects.")
            if not force.get("id"):
                force["id"] = stable_id(
                    "force",
                    force.get("name"),
                    force.get("domain"),
                    force.get("description"),
                )
                log_normalization(
                    run_id=run_id,
                    node_name="scan_pestel",
                    operation="stable_id_assigned",
                    details={"field": "id", "name": force.get("name", "")},
                    base_dir=base_dir,
                )

            raw_domain = force.get("domain")
            if isinstance(raw_domain, str) and raw_domain.strip():
                normalized_domain = raw_domain.strip().lower()
                if normalized_domain != raw_domain:
                    log_normalization(
                        run_id=run_id,
                        node_name="scan_pestel",
                        operation="normalized_domain",
                        details={"from": raw_domain, "to": normalized_domain},
                        base_dir=base_dir,
                    )
                force["domain"] = normalized_domain

            raw_lenses = force.get("lenses", [])
            if isinstance(raw_lenses, list):
                normalized_lenses = []
                for lens in raw_lenses:
                    if isinstance(lens, str) and lens.strip():
                        normalized_lenses.append(lens.strip().lower())
                if normalized_lenses != raw_lenses:
                    log_normalization(
                        run_id=run_id,
                        node_name="scan_pestel",
                        operation="normalized_lenses",
                        details={"from": raw_lenses, "to": normalized_lenses},
                        base_dir=base_dir,
                    )
                force["lenses"] = list(dict.fromkeys(normalized_lenses))

            raw_citations = force.get("citations", [])
            if not isinstance(raw_citations, list) or not raw_citations:
                raise ValueError(
                    f"Driving force '{force.get('name')}' missing citations."
                )

            for citation in raw_citations:
                if not isinstance(citation, dict):
                    raise ValueError("Driving force citations must be objects.")
                url = _safe_citation_url(citation)
                normalized = _normalize_citation_url(url)
                if normalized not in evidence_hashes:
                    if not allow_fixture_citations:
                        raise ValueError(
                            f"Driving force citation url not in evidence units: {url}"
                        )
                    log_normalization(
                        run_id=run_id,
                        node_name="scan_pestel",
                        operation="fixture_citation_fallback",
                        details={"url": url},
                        base_dir=base_dir,
                    )
                    citation["excerpt_hash"] = _hash_excerpt(url)
                    citation["publisher"] = citation.get("publisher") or url
                    citation["evidence_id"] = citation.get("evidence_id") or url
                    continue
                citation["excerpt_hash"] = evidence_hashes[normalized]
                citation["publisher"] = evidence_publishers.get(
                    normalized, citation.get("publisher", "")
                )
                citation["evidence_id"] = evidence_ids.get(
                    normalized, citation.get("evidence_id", "")
                )
            normalized_forces.append(force)

    parsed["forces"] = normalized_forces

    validate_artifact("driving_forces.schema", parsed)

    write_artifact(
        run_id=run_id,
        artifact_name="driving_forces",
        payload=parsed,
        ext="json",
        input_values={
            "min_forces": min_forces,
            "min_per_domain": min_per_domain,
            "focus_domains": list(focus_domains or []),
            "focus_categories": list(focus_categories or []),
        },
        prompt_values={"prompt": prompt},
        tool_versions={"scan_node": "0.1.0"},
        base_dir=base_dir,
    )

    state.driving_forces = parsed
    return state
