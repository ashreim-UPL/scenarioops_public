from __future__ import annotations

import hashlib
import json
import os
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from scenarioops.app.config import LLMConfig
from scenarioops.graph.nodes.utils import build_prompt, get_client
from scenarioops.graph.state import ScenarioOpsState
from scenarioops.graph.tools.normalization import stable_id
from scenarioops.graph.tools.prompts import load_prompt_spec
from scenarioops.graph.tools.schema_validate import load_schema, validate_artifact
from scenarioops.graph.tools.storage import ensure_run_dirs, log_normalization, write_artifact
from scenarioops.graph.tools.traceability import build_run_metadata
from scenarioops.llm.guards import ensure_dict


_PESTEL_DOMAINS = (
    "political",
    "economic",
    "social",
    "technological",
    "environmental",
    "legal",
)

_REQUIRED_FORCE_FIELDS = (
    "layer",
    "domain",
    "label",
    "mechanism",
    "directionality",
    "affected_dimensions",
    "evidence_unit_ids",
    "confidence",
    "confidence_rationale",
)


def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(key)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(1, value)


def _chunk_size() -> int:
    return _env_int("FORCES_CHUNK_SIZE", 12)


def _chunk_min() -> int:
    return _env_int("FORCES_CHUNK_MIN", 4)


def _evidence_limit() -> int:
    return _env_int("FORCES_EVIDENCE_LIMIT", 40)


def _excerpt_limit() -> int:
    return _env_int("FORCES_EVIDENCE_EXCERPT_CHARS", 360)


def _max_chunk_attempts() -> int:
    return _env_int("FORCES_CHUNK_ATTEMPTS", 3)


def _split_depth_max() -> int:
    return _env_int("FORCES_SPLIT_DEPTH_MAX", 3)


def _max_force_fix_retries() -> int:
    return _env_int("MAX_FORCE_FIX_RETRIES", 3)


def _is_truncation_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "unable to locate json object" in message
        or "truncated" in message
        or "json" in message and "output" in message and "missing" in message
    )


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _hash_payload(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return _hash_text(encoded)


def _evidence_ids(state: ScenarioOpsState) -> set[str]:
    payload = state.evidence_units or {}
    units = payload.get("evidence_units", [])
    if not isinstance(units, list):
        return set()
    ids: set[str] = set()
    for unit in units:
        if isinstance(unit, Mapping):
            status = str(unit.get("status", "ok")).lower()
            if status != "ok":
                continue
            value = unit.get("id") or unit.get("evidence_unit_id")
            if isinstance(value, str) and value.strip():
                ids.add(value)
    return ids


def _forces_artifacts_dir(run_id: str, base_dir: Path | None) -> Path:
    dirs = ensure_run_dirs(run_id, base_dir=base_dir)
    forces_dir = dirs["artifacts_dir"] / "forces"
    forces_dir.mkdir(parents=True, exist_ok=True)
    return forces_dir


def _manifest_path(run_id: str, base_dir: Path | None) -> Path:
    return _forces_artifacts_dir(run_id, base_dir) / "forces_assembly_manifest.json"


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def _validate_force_item(
    force: Mapping[str, Any], evidence_ids: set[str]
) -> tuple[bool, list[str]]:
    errors: list[str] = []
    for field in _REQUIRED_FORCE_FIELDS:
        if not force.get(field):
            errors.append(f"missing_{field}")
    layer = str(force.get("layer", "")).lower()
    if layer and layer not in {"primary", "secondary", "tertiary"}:
        errors.append("invalid_layer")
    domain = str(force.get("domain", "")).lower()
    if domain and domain not in _PESTEL_DOMAINS:
        errors.append("invalid_domain")
    dims = force.get("affected_dimensions", [])
    if not isinstance(dims, list) or not dims:
        errors.append("invalid_affected_dimensions")
    evidence_list = force.get("evidence_unit_ids", [])
    if not isinstance(evidence_list, list) or not evidence_list:
        errors.append("missing_evidence_unit_ids")
    else:
        missing = [item for item in evidence_list if str(item) not in evidence_ids]
        if missing:
            errors.append("unknown_evidence_unit_ids")
    confidence = force.get("confidence")
    if confidence is None or not isinstance(confidence, (int, float)):
        errors.append("invalid_confidence")
    elif confidence < 0 or confidence > 1:
        errors.append("invalid_confidence_range")
    return (len(errors) == 0), errors


def _force_stub(force: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "force_id": force.get("force_id"),
        "label": force.get("label"),
        "domain": force.get("domain"),
        "mechanism": force.get("mechanism"),
        "directionality": force.get("directionality"),
        "evidence_unit_ids": force.get("evidence_unit_ids", []),
    }


def _force_dedupe_key(force: Mapping[str, Any]) -> str:
    label = _normalize_text(force.get("label"))
    domain = _normalize_text(force.get("domain"))
    mechanism = _normalize_text(force.get("mechanism"))
    return _hash_text(f"{label}|{domain}|{mechanism}")


def _dedupe_forces(forces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for force in forces:
        force_id = str(force.get("force_id", "")).strip()
        force_hash = _force_dedupe_key(force)
        if force_id and force_id in seen_ids:
            continue
        if force_hash in seen_hashes:
            continue
        if force_id:
            seen_ids.add(force_id)
        seen_hashes.add(force_hash)
        deduped.append(force)
    return deduped

def _trim_evidence_units(
    evidence_units: Iterable[Mapping[str, Any]],
    *,
    max_units: int,
    max_excerpt_chars: int,
    chunk_index: int,
) -> list[dict[str, Any]]:
    grade_rank = {"A": 0, "B": 1, "C": 2, "D": 3}
    filtered = [
        unit for unit in evidence_units if isinstance(unit, Mapping)
    ]
    filtered.sort(
        key=lambda unit: grade_rank.get(str(unit.get("reliability_grade", "D")).upper(), 4)
    )
    if not filtered:
        return []
    if len(filtered) > max_units:
        start = (chunk_index * max_units) % len(filtered)
        window = filtered[start : start + max_units]
        if len(window) < max_units:
            window.extend(filtered[: max_units - len(window)])
    else:
        window = filtered
    trimmed: list[dict[str, Any]] = []
    for unit in window:
        status = str(unit.get("status", "ok")).lower()
        if status != "ok":
            continue
        excerpt = str(unit.get("summary") or unit.get("excerpt") or "")
        if len(excerpt) > max_excerpt_chars:
            excerpt = excerpt[: max_excerpt_chars].rstrip() + "..."
        trimmed.append(
            {
                "evidence_unit_id": unit.get("id") or unit.get("evidence_unit_id"),
                "title": unit.get("title"),
                "publisher": unit.get("publisher"),
                "date_published": unit.get("date_published"),
                "url": unit.get("url"),
                "excerpt": excerpt,
                "summary": unit.get("summary") or excerpt,
                "claims": unit.get("claims", [])[:3],
                "metrics": unit.get("metrics", [])[:3],
                "reliability_grade": unit.get("reliability_grade"),
                "geography_tags": unit.get("geography_tags", []),
                "domain_tags": unit.get("domain_tags", []),
            }
        )
    return trimmed


def _domain_targets(min_forces: int, min_per_domain: int) -> dict[str, int]:
    base_total = max(min_forces, min_per_domain * len(_PESTEL_DOMAINS))
    remainder = base_total - min_per_domain * len(_PESTEL_DOMAINS)
    targets = {domain: min_per_domain for domain in _PESTEL_DOMAINS}
    domains = list(_PESTEL_DOMAINS)
    idx = 0
    while remainder > 0:
        targets[domains[idx % len(domains)]] += 1
        idx += 1
        remainder -= 1
    return targets


def _plan_chunks(
    domain_targets: Mapping[str, int],
    chunk_size: int,
) -> list[dict[str, int]]:
    remaining = {
        domain: int(domain_targets.get(domain, 0)) for domain in _PESTEL_DOMAINS
    }
    chunks: list[dict[str, int]] = []
    total_remaining = sum(remaining.values())
    if total_remaining <= 0:
        return chunks
    domains = list(_PESTEL_DOMAINS)
    while total_remaining > 0:
        chunk: dict[str, int] = {domain: 0 for domain in domains}
        capacity = min(chunk_size, total_remaining)
        idx = 0
        while capacity > 0:
            domain = domains[idx % len(domains)]
            if remaining[domain] > 0:
                chunk[domain] += 1
                remaining[domain] -= 1
                capacity -= 1
                total_remaining -= 1
            idx += 1
        chunks.append({domain: count for domain, count in chunk.items() if count > 0})
    return chunks


def _split_targets(targets: Mapping[str, int]) -> tuple[dict[str, int], dict[str, int]]:
    left: dict[str, int] = {}
    right: dict[str, int] = {}
    for domain in _PESTEL_DOMAINS:
        count = int(targets.get(domain, 0))
        half = count // 2
        left[domain] = half
        right[domain] = count - half
    left = {domain: count for domain, count in left.items() if count > 0}
    right = {domain: count for domain, count in right.items() if count > 0}
    return left, right


def _missing_fields(force: Mapping[str, Any]) -> list[str]:
    missing = [field for field in _REQUIRED_FORCE_FIELDS if not force.get(field)]
    if "affected_dimensions" not in missing:
        dims = force.get("affected_dimensions", [])
        if not isinstance(dims, list) or not dims:
            missing.append("affected_dimensions")
    return missing


def _filter_by_domain_targets(
    forces: list[dict[str, Any]],
    domain_targets: Mapping[str, int],
    *,
    max_total: int | None = None,
) -> list[dict[str, Any]]:
    target_total = sum(int(value) for value in domain_targets.values())
    if target_total <= 0:
        return forces
    if max_total is None:
        max_total = target_total
    counts = {domain: 0 for domain in _PESTEL_DOMAINS}
    selected: list[dict[str, Any]] = []
    overflow: list[dict[str, Any]] = []
    for force in forces:
        domain = str(force.get("domain", "")).lower()
        limit = domain_targets.get(domain)
        if limit is None or limit <= 0:
            overflow.append(force)
            continue
        if counts[domain] < int(limit):
            selected.append(force)
            counts[domain] += 1
        else:
            overflow.append(force)
    if not selected:
        return forces[:max_total]
    if len(selected) < max_total and overflow:
        for force in overflow:
            if len(selected) >= max_total:
                break
            selected.append(force)
    return selected


def _domain_counts(forces: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts = {domain: 0 for domain in _PESTEL_DOMAINS}
    for force in forces:
        domain = str(force.get("domain", "")).lower()
        if domain in counts:
            counts[domain] += 1
    return counts


def _deficits(
    targets: Mapping[str, int], counts: Mapping[str, int]
) -> dict[str, int]:
    deficits: dict[str, int] = {}
    for domain, target in targets.items():
        missing = int(target) - int(counts.get(domain, 0))
        if missing > 0:
            deficits[domain] = missing
    return deficits


def _normalize_force_ids(
    *,
    run_id: str,
    forces: list[dict[str, Any]],
    base_dir: Path | None,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for entry in forces:
        force = dict(entry)
        if not force.get("force_id"):
            force["force_id"] = stable_id(
                "force",
                force.get("label"),
                force.get("domain"),
                force.get("mechanism"),
            )
            log_normalization(
                run_id=run_id,
                node_name="forces",
                operation="stable_id_assigned",
                details={"field": "force_id", "label": force.get("label", "")},
                base_dir=base_dir,
            )
        normalized.append(force)
    return normalized


def _normalize_force_fields(forces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for entry in forces:
        force = dict(entry)
        domain_value = force.get("domain")
        if not domain_value:
            domain_value = (
                force.get("pillar")
                or force.get("pestel_category")
                or force.get("category")
            )
            if domain_value:
                force["domain"] = domain_value
        if not force.get("label"):
            label = force.get("force") or force.get("name")
            if label:
                force["label"] = label
        if not force.get("directionality"):
            direction = force.get("direction") or force.get("trend")
            if direction:
                force["directionality"] = direction
        evidence_ids = force.get("evidence_unit_ids")
        if not evidence_ids:
            single = force.get("evidence_unit_id") or force.get("evidence_id")
            if single:
                force["evidence_unit_ids"] = [single]
        elif isinstance(evidence_ids, str):
            force["evidence_unit_ids"] = [evidence_ids]
        if isinstance(force.get("affected_dimensions"), str):
            force["affected_dimensions"] = [force["affected_dimensions"]]
        if isinstance(force.get("domain"), str):
            force["domain"] = force["domain"].strip().lower()
        if isinstance(force.get("layer"), str):
            force["layer"] = force["layer"].strip().lower()
        allowed_keys = set(_REQUIRED_FORCE_FIELDS) | {"force_id"}
        force = {key: value for key, value in force.items() if key in allowed_keys}
        normalized.append(force)
    return normalized


def _impute_force_fields(
    force: Mapping[str, Any], evidence_units: list[dict[str, Any]]
) -> dict[str, Any]:
    filled = dict(force)
    if not filled.get("label"):
        filled["label"] = (
            filled.get("force") or filled.get("name") or filled.get("mechanism") or "Unlabeled force"
        )
    if not filled.get("mechanism"):
        filled["mechanism"] = f"{filled.get('label')} shifts market dynamics."
    if not filled.get("directionality"):
        filled["directionality"] = "Mixed impact with uneven effects across segments."
    if not filled.get("layer"):
        filled["layer"] = "secondary"
    if not filled.get("domain"):
        domain_tags = []
        if isinstance(filled.get("domain_tags"), list):
            domain_tags = [str(item).lower() for item in filled.get("domain_tags") if str(item)]
        tag_to_domain = {
            "political": "political",
            "economic": "economic",
            "social": "social",
            "technological": "technological",
            "environmental": "environmental",
            "legal": "legal",
        }
        inferred = next((tag_to_domain[tag] for tag in domain_tags if tag in tag_to_domain), None)
        filled["domain"] = inferred or "economic"
    if not filled.get("affected_dimensions"):
        domain_to_dimensions = {
            "political": ["regulatory", "geopolitical"],
            "economic": ["demand", "cost"],
            "social": ["workforce", "customer"],
            "technological": ["capability", "productivity"],
            "environmental": ["sustainability", "resilience"],
            "legal": ["compliance", "risk"],
        }
        filled["affected_dimensions"] = domain_to_dimensions.get(
            str(filled.get("domain", "")).lower(), ["demand"]
        )
    evidence_ids = filled.get("evidence_unit_ids")
    if not evidence_ids:
        for unit in evidence_units:
            unit_id = unit.get("evidence_unit_id") or unit.get("id")
            if isinstance(unit_id, str) and unit_id:
                evidence_ids = [unit_id]
                break
        if evidence_ids:
            filled["evidence_unit_ids"] = evidence_ids
    if filled.get("confidence") is None:
        filled["confidence"] = 0.55
    if not filled.get("confidence_rationale"):
        filled["confidence_rationale"] = "Imputed from partial evidence; confidence is provisional."
    return filled


def _relax_forces_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    relaxed = deepcopy(schema)
    if isinstance(relaxed, dict):
        relaxed["additionalProperties"] = True
    properties = relaxed.get("properties")
    if isinstance(properties, dict):
        forces_prop = properties.get("forces")
        if isinstance(forces_prop, dict):
            forces_prop["minItems"] = 0
            items = forces_prop.get("items")
            if isinstance(items, dict):
                items["additionalProperties"] = True
                items["required"] = []
    return relaxed


def _relax_force_item_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    relaxed = deepcopy(schema)
    relaxed["additionalProperties"] = True
    relaxed["required"] = []
    return relaxed


def _generate_forces_chunk(
    *,
    client,
    schema: dict[str, Any],
    prompt_name: str,
    correction_prompt_name: str,
    evidence_units: list[dict[str, Any]],
    domain_targets: Mapping[str, int],
    chunk_target_total: int,
    chunk_index: int,
    run_id: str,
    base_dir: Path | None,
) -> tuple[list[dict[str, Any]], str, int, int]:
    attempts = _max_chunk_attempts()
    for attempt in range(attempts):
        prompt_bundle = build_prompt(
            prompt_name,
            {
                "chunk_target_total": chunk_target_total,
                "domain_targets": domain_targets,
                "evidence_units": evidence_units,
            },
        )
        response = client.generate_json(prompt_bundle.text, schema)
        parsed = ensure_dict(response, node_name="forces_chunk")
        forces = parsed.get("forces", [])
        if not isinstance(forces, list):
            forces = []
        forces = [dict(force) for force in forces if isinstance(force, Mapping)]
        forces = _normalize_force_fields(forces)
        forces = [_impute_force_fields(force, evidence_units) for force in forces]
        forces = _filter_by_domain_targets(
            forces,
            domain_targets,
            max_total=chunk_target_total,
        )
        if not forces:
            continue
        if len(forces) > chunk_target_total:
            forces = forces[:chunk_target_total]
        invalid = [force for force in forces if _missing_fields(force)]
        if not invalid:
            raw = getattr(response, "raw", None) or ""
            return forces, raw, attempt + 1, 0
        correction_bundle = build_prompt(
            correction_prompt_name,
            {
                "chunk_target_total": chunk_target_total,
                "domain_targets": domain_targets,
                "forces": forces,
                "evidence_units": evidence_units,
            },
        )
        correction_response = client.generate_json(correction_bundle.text, schema)
        parsed = ensure_dict(correction_response, node_name="forces_chunk_correction")
        corrected = parsed.get("forces", [])
        if not isinstance(corrected, list):
            continue
        corrected = [dict(force) for force in corrected if isinstance(force, Mapping)]
        corrected = _normalize_force_fields(corrected)
        corrected = [_impute_force_fields(force, evidence_units) for force in corrected]
        corrected = _filter_by_domain_targets(
            corrected,
            domain_targets,
            max_total=chunk_target_total,
        )
        if len(corrected) > chunk_target_total:
            corrected = corrected[:chunk_target_total]
        invalid = [force for force in corrected if _missing_fields(force)]
        if not invalid and corrected:
            raw = getattr(correction_response, "raw", None) or ""
            return corrected, raw, attempt + 1, 1
    raise RuntimeError(
        f"force_chunk_invalid: chunk {chunk_index} missing required fields or counts"
    )


def _generate_chunk_with_recovery(
    *,
    client,
    schema: dict[str, Any],
    evidence_units: list[dict[str, Any]],
    domain_targets: Mapping[str, int],
    chunk_target_total: int,
    chunk_index: int,
    run_id: str,
    base_dir: Path | None,
    split_depth: int = 0,
) -> list[dict[str, Any]]:
    try:
        forces, raw, attempts, correction_retries = _generate_forces_chunk(
            client=client,
            schema=schema,
            prompt_name="forces_chunk",
            correction_prompt_name="forces_chunk_correction",
            evidence_units=evidence_units,
            domain_targets=domain_targets,
            chunk_target_total=chunk_target_total,
            chunk_index=chunk_index,
            run_id=run_id,
            base_dir=base_dir,
        )
        return [
            {
                "forces": forces,
                "raw": raw,
                "attempts": attempts,
                "correction_retries": correction_retries,
                "truncated_retries": 0,
                "split_depth": split_depth,
            }
        ]
    except Exception as exc:
        if not _is_truncation_error(exc):
            raise
        if chunk_target_total <= _chunk_min():
            raise
        if split_depth >= _split_depth_max():
            raise RuntimeError(
                f"force_chunk_split_depth_exceeded: depth={split_depth}"
            ) from exc
        left_targets, right_targets = _split_targets(domain_targets)
        results: list[dict[str, Any]] = []
        if left_targets:
            results.extend(
                _generate_chunk_with_recovery(
                    client=client,
                    schema=schema,
                    evidence_units=evidence_units,
                    domain_targets=left_targets,
                    chunk_target_total=sum(left_targets.values()),
                    chunk_index=chunk_index,
                    run_id=run_id,
                    base_dir=base_dir,
                    split_depth=split_depth + 1,
                )
            )
        if right_targets:
            results.extend(
                _generate_chunk_with_recovery(
                    client=client,
                    schema=schema,
                    evidence_units=evidence_units,
                    domain_targets=right_targets,
                    chunk_target_total=sum(right_targets.values()),
                    chunk_index=chunk_index,
                    run_id=run_id,
                    base_dir=base_dir,
                    split_depth=split_depth + 1,
                )
            )
        for entry in results:
            entry["truncated_retries"] = entry.get("truncated_retries", 0) + 1
        return results


def _correct_force_item(
    *,
    client,
    schema: dict[str, Any],
    force: Mapping[str, Any],
    errors: list[str],
    evidence_units: list[dict[str, Any]],
) -> dict[str, Any]:
    prompt_bundle = build_prompt(
        "force_item_correction",
        {
            "force": dict(force),
            "errors": errors,
            "evidence_units": evidence_units,
        },
    )
    response = client.generate_json(prompt_bundle.text, schema)
    parsed = ensure_dict(response, node_name="force_item_correction")
    return dict(parsed)


def _repair_invalid_forces(
    *,
    client,
    schema: dict[str, Any],
    forces: list[dict[str, Any]],
    evidence_units: list[dict[str, Any]],
    evidence_ids: set[str],
    batch_index: int,
    rejected_items: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    corrected: list[dict[str, Any]] = []
    correction_retries = 0
    for force in forces:
        force = _impute_force_fields(force, evidence_units)
        ok, errors = _validate_force_item(force, evidence_ids)
        if ok:
            corrected.append(force)
            continue
        retries = _max_force_fix_retries()
        fixed = None
        for attempt in range(retries):
            correction_retries += 1
            candidate = _correct_force_item(
                client=client,
                schema=schema,
                force=force,
                errors=errors,
                evidence_units=evidence_units,
            )
            normalized_candidates = _normalize_force_fields([dict(candidate)])
            if normalized_candidates:
                candidate = normalized_candidates[0]
            if not candidate.get("force_id") and force.get("force_id"):
                candidate["force_id"] = force.get("force_id")
            ok, errors = _validate_force_item(candidate, evidence_ids)
            if ok:
                fixed = candidate
                break
        if fixed:
            corrected.append(fixed)
        else:
            rejected_items.append(
                {
                    "reason": ",".join(errors) if errors else "validation_failed",
                    "force_stub": _force_stub(force),
                    "batch_index": batch_index,
                }
            )
    return corrected, correction_retries

def _validate_distribution(
    forces: list[dict[str, Any]], min_forces: int, min_per_domain: int
) -> list[str]:
    warnings: list[str] = []
    if len(forces) < min_forces:
        warnings.append(
            f"force_count_below_min: {len(forces)} < {min_forces}"
        )
    domain_counts = {domain: 0 for domain in _PESTEL_DOMAINS}
    for force in forces:
        domain = str(force.get("domain", "")).lower()
        if domain in domain_counts:
            domain_counts[domain] += 1
    for domain, count in domain_counts.items():
        if count < min_per_domain:
            warnings.append(
                f"domain_count_below_min: {domain}={count} < {min_per_domain}"
            )
    layer_counts: dict[str, int] = {"primary": 0, "secondary": 0, "tertiary": 0}
    for force in forces:
        layer = str(force.get("layer", "")).lower()
        if layer in layer_counts:
            layer_counts[layer] += 1
    for layer, count in layer_counts.items():
        if count == 0:
            warnings.append(f"layer_missing: {layer}")
    return warnings


def run_force_builder_node(
    *,
    run_id: str,
    state: ScenarioOpsState,
    user_params: Mapping[str, Any],
    llm_client=None,
    base_dir: Path | None = None,
    config: LLMConfig | None = None,
    min_forces: int = 60,
    min_per_domain: int = 10,
) -> ScenarioOpsState:
    if state.evidence_units is None:
        raise ValueError("Evidence units are required before force generation.")

    client = get_client(llm_client, config)
    strict_schema = load_schema("forces_payload")
    schema = _relax_forces_schema(strict_schema)
    force_item_schema = load_schema("force_item")
    relaxed_force_item_schema = _relax_force_item_schema(force_item_schema)

    evidence_units = state.evidence_units.get("evidence_units", [])
    if not isinstance(evidence_units, list):
        raise TypeError("Evidence units payload must include a list.")
    evidence_units = [
        unit
        for unit in evidence_units
        if isinstance(unit, Mapping)
        and str(unit.get("status", "ok")).lower() == "ok"
    ]

    forces_dir = _forces_artifacts_dir(run_id, base_dir)
    manifest_path = _manifest_path(run_id, base_dir)
    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "batch_size": _chunk_size(),
        "split_depth_max": _split_depth_max(),
        "batches": [],
        "rejected_items": [],
        "evidence_used_ids": [],
        "hashes": {},
    }
    _write_manifest(manifest_path, manifest)

    domain_targets = _domain_targets(min_forces, min_per_domain)
    chunk_plan = _plan_chunks(domain_targets, _chunk_size())
    raw_forces: list[dict[str, Any]] = []
    evidence_ids = _evidence_ids(state)
    part_index = 0
    evidence_used: set[str] = set()

    for idx, chunk_targets in enumerate(chunk_plan):
        chunk_total = sum(chunk_targets.values())
        evidence_subset = _trim_evidence_units(
            evidence_units,
            max_units=_evidence_limit(),
            max_excerpt_chars=_excerpt_limit(),
            chunk_index=idx,
        )
        chunk_outputs = _generate_chunk_with_recovery(
            client=client,
            schema=schema,
            evidence_units=evidence_subset,
            domain_targets=chunk_targets,
            chunk_target_total=chunk_total,
            chunk_index=idx + 1,
            run_id=run_id,
            base_dir=base_dir,
        )
        for chunk_output in chunk_outputs:
            part_index += 1
            batch_started = datetime.now(timezone.utc)
            batch_forces = chunk_output.get("forces", [])
            if not isinstance(batch_forces, list):
                batch_forces = []
            batch_forces = _normalize_force_fields(batch_forces)
            batch_forces = _normalize_force_ids(
                run_id=run_id, forces=batch_forces, base_dir=base_dir
            )
            batch_forces = _filter_by_domain_targets(
                batch_forces,
                chunk_targets,
                max_total=chunk_total,
            )
            batch_forces, correction_retries = _repair_invalid_forces(
                client=client,
                schema=relaxed_force_item_schema,
                forces=batch_forces,
                evidence_units=evidence_subset,
                evidence_ids=evidence_ids,
                batch_index=part_index,
                rejected_items=manifest["rejected_items"],
            )
            batch_forces = _dedupe_forces(batch_forces)
            batch_forces = _filter_by_domain_targets(
                batch_forces,
                chunk_targets,
                max_total=chunk_total,
            )
            raw_forces.extend(batch_forces)

            batch_evidence_ids: set[str] = set()
            for force in batch_forces:
                for item in force.get("evidence_unit_ids", []):
                    if item is None:
                        continue
                    value = str(item)
                    batch_evidence_ids.add(value)
                    evidence_used.add(value)

            part_payload = {
                "schema_version": "1.0",
                "batch_index": part_index,
                "forces": batch_forces,
            }
            part_path = forces_dir / f"forces.part_{part_index}.json"
            part_path.write_text(
                json.dumps(part_payload, indent=2, sort_keys=True), encoding="utf-8"
            )
            raw_path = forces_dir / f"forces.part_{part_index}.raw.txt"
            raw_path.write_text(str(chunk_output.get("raw", "")), encoding="utf-8")

            batch_completed = datetime.now(timezone.utc)
            batch_entry = {
                "batch_index": part_index,
                "part_path": str(part_path),
                "raw_path": str(raw_path),
                "forces_count": len(batch_forces),
                "retries": int(chunk_output.get("attempts", 1)),
                "truncated_retries": int(chunk_output.get("truncated_retries", 0)),
                "correction_retries": int(correction_retries),
                "split_depth": int(chunk_output.get("split_depth", 0)),
                "hash": _hash_payload(part_payload),
                "raw_hash": _hash_text(str(chunk_output.get("raw", ""))),
                "evidence_used_ids": sorted(batch_evidence_ids),
                "started_at": batch_started.isoformat(),
                "completed_at": batch_completed.isoformat(),
                "duration_seconds": (batch_completed - batch_started).total_seconds(),
            }
            manifest["batches"].append(batch_entry)
            manifest["evidence_used_ids"] = sorted(evidence_used)
            _write_manifest(manifest_path, manifest)

    normalized_forces = _dedupe_forces(raw_forces)

    deficit_attempts = 0
    deficits = _deficits(domain_targets, _domain_counts(normalized_forces))
    while deficits and deficit_attempts < 2:
        deficit_attempts += 1
        deficit_chunks = _plan_chunks(deficits, _chunk_size())
        for idx, deficit_targets in enumerate(deficit_chunks):
            deficit_total = sum(deficit_targets.values())
            evidence_subset = _trim_evidence_units(
                evidence_units,
                max_units=_evidence_limit(),
                max_excerpt_chars=_excerpt_limit(),
                chunk_index=idx + len(chunk_plan),
            )
            extra_outputs = _generate_chunk_with_recovery(
                client=client,
                schema=schema,
                evidence_units=evidence_subset,
                domain_targets=deficit_targets,
                chunk_target_total=deficit_total,
                chunk_index=idx + 1,
                run_id=run_id,
                base_dir=base_dir,
            )
            for chunk_output in extra_outputs:
                part_index += 1
                batch_started = datetime.now(timezone.utc)
                extra_forces = chunk_output.get("forces", [])
                if not isinstance(extra_forces, list):
                    extra_forces = []
                extra_forces = _normalize_force_fields(extra_forces)
                extra_forces = _normalize_force_ids(
                    run_id=run_id, forces=extra_forces, base_dir=base_dir
                )
                extra_forces = _filter_by_domain_targets(
                    extra_forces,
                    deficit_targets,
                    max_total=deficit_total,
                )
                extra_forces, correction_retries = _repair_invalid_forces(
                    client=client,
                    schema=relaxed_force_item_schema,
                    forces=extra_forces,
                    evidence_units=evidence_subset,
                    evidence_ids=evidence_ids,
                    batch_index=part_index,
                    rejected_items=manifest["rejected_items"],
                )
                extra_forces = _dedupe_forces(extra_forces)
                extra_forces = _filter_by_domain_targets(
                    extra_forces,
                    deficit_targets,
                    max_total=deficit_total,
                )
                raw_forces.extend(extra_forces)
                normalized_forces = _dedupe_forces(raw_forces)

                batch_evidence_ids: set[str] = set()
                for force in extra_forces:
                    for item in force.get("evidence_unit_ids", []):
                        if item is None:
                            continue
                        value = str(item)
                        batch_evidence_ids.add(value)
                        evidence_used.add(value)

                part_payload = {
                    "schema_version": "1.0",
                    "batch_index": part_index,
                    "forces": extra_forces,
                }
                part_path = forces_dir / f"forces.part_{part_index}.json"
                part_path.write_text(
                    json.dumps(part_payload, indent=2, sort_keys=True), encoding="utf-8"
                )
                raw_path = forces_dir / f"forces.part_{part_index}.raw.txt"
                raw_path.write_text(
                    str(chunk_output.get("raw", "")), encoding="utf-8"
                )
                batch_completed = datetime.now(timezone.utc)
                batch_entry = {
                    "batch_index": part_index,
                    "part_path": str(part_path),
                    "raw_path": str(raw_path),
                    "forces_count": len(extra_forces),
                    "retries": int(chunk_output.get("attempts", 1)),
                    "truncated_retries": int(chunk_output.get("truncated_retries", 0)),
                    "correction_retries": int(correction_retries),
                    "split_depth": int(chunk_output.get("split_depth", 0)),
                    "hash": _hash_payload(part_payload),
                    "raw_hash": _hash_text(str(chunk_output.get("raw", ""))),
                    "evidence_used_ids": sorted(batch_evidence_ids),
                    "started_at": batch_started.isoformat(),
                    "completed_at": batch_completed.isoformat(),
                    "duration_seconds": (batch_completed - batch_started).total_seconds(),
                }
                manifest["batches"].append(batch_entry)
                manifest["evidence_used_ids"] = sorted(evidence_used)
                _write_manifest(manifest_path, manifest)
        deficits = _deficits(domain_targets, _domain_counts(normalized_forces))

    warnings: list[str] = []
    linked = 0
    for force in normalized_forces:
        linked_ids = [
            str(item) for item in force.get("evidence_unit_ids", []) if str(item)
        ]
        missing = [item for item in linked_ids if item not in evidence_ids]
        if missing:
            warnings.append(
                f"force_unlinked:{force.get('force_id')} missing {missing[:2]}"
            )
        if linked_ids and not missing:
            linked += 1
            force["unlinked"] = False
        else:
            force["unlinked"] = True
        ok, errors = _validate_force_item(force, evidence_ids)
        if not ok:
            warnings.append(
                f"force_invalid:{force.get('force_id')}:{','.join(errors)}"
            )

    distribution_warnings = _validate_distribution(
        normalized_forces, min_forces, min_per_domain
    )
    warnings.extend(distribution_warnings)
    linked_ratio = linked / max(1, len(normalized_forces))
    if linked_ratio < 0.8:
        warnings.append(f"linked_ratio_below_threshold: {linked_ratio:.2f} < 0.80")

    metadata = build_run_metadata(run_id=run_id, user_params=user_params)
    payload = {
        **metadata,
        "needs_correction": bool(warnings),
        "warnings": warnings,
        "forces": normalized_forces,
    }
    validate_artifact("forces", payload)

    write_artifact(
        run_id=run_id,
        artifact_name="forces",
        payload=payload,
        ext="json",
        input_values={
            "min_forces": min_forces,
            "min_per_domain": min_per_domain,
            "chunk_count": len(chunk_plan),
        },
        prompt_values={
            "prompt_name": "forces_chunk",
            "prompt_sha256": load_prompt_spec("forces_chunk").sha256,
        },
        tool_versions={"force_builder_node": "0.1.0"},
        base_dir=base_dir,
    )
    forces_dir.joinpath("forces.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    manifest["hashes"]["forces_payload"] = _hash_payload(payload)
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    _write_manifest(manifest_path, manifest)

    state.forces = payload
    if not normalized_forces:
        raise RuntimeError("force_builder_empty: no valid forces generated.")
    return state
