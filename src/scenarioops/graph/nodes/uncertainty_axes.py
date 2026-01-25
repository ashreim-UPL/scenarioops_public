from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from scenarioops.app.config import LLMConfig, ScenarioOpsSettings
from scenarioops.graph.nodes.utils import build_prompt, get_client
from scenarioops.graph.state import ScenarioOpsState
from scenarioops.graph.tools.schema_validate import load_schema, validate_artifact
from scenarioops.graph.tools.storage import ensure_run_dirs, write_artifact
from scenarioops.graph.tools.traceability import build_run_metadata
from scenarioops.llm.guards import ensure_dict


_REQUIRED_AXIS_FIELDS = (
    "axis_name",
    "pole_a",
    "pole_b",
    "impact_score",
    "uncertainty_score",
    "tension_basis",
    "what_would_change_mind",
    "independence_notes",
)


def _axis_similarity(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
    def _ids(axis: Mapping[str, Any]) -> set[str]:
        tension = axis.get("tension_basis", {})
        cluster_ids = set(str(item) for item in tension.get("cluster_ids", []) if str(item))
        force_ids = set(str(item) for item in tension.get("force_ids", []) if str(item))
        return cluster_ids | force_ids

    set_a = _ids(a)
    set_b = _ids(b)
    if not set_a or not set_b:
        return 0.0
    overlap = len(set_a & set_b)
    union = len(set_a | set_b)
    return overlap / max(1, union)


def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(key)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(1, value)


def _max_axis_fix_retries() -> int:
    return _env_int("MAX_AXIS_FIX_RETRIES", 3)


def _allow_needs_correction(
    settings: ScenarioOpsSettings | None, config: LLMConfig | None
) -> bool:
    if settings is not None:
        if settings.sources_policy == "fixtures":
            return True
        if settings.mode == "demo":
            return True
        if settings.llm_provider == "mock":
            return True
    if config is not None and getattr(config, "mode", None) == "mock":
        return True
    return False


def _axis_item_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    properties = schema.get("properties")
    if isinstance(properties, Mapping):
        axes_prop = properties.get("axes")
        if isinstance(axes_prop, Mapping):
            items = axes_prop.get("items")
            if isinstance(items, Mapping):
                item_schema = dict(items)
                item_schema.setdefault("title", "Uncertainty Axis Item")
                return item_schema
    return {"title": "Uncertainty Axis Item", "type": "object"}


def _relax_uncertainty_axes_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    relaxed = deepcopy(schema)
    if isinstance(relaxed, dict):
        relaxed["additionalProperties"] = True
    properties = relaxed.get("properties")
    if isinstance(properties, dict):
        axes_prop = properties.get("axes")
        if isinstance(axes_prop, dict):
            axes_prop["minItems"] = 0
            items = axes_prop.get("items")
            if isinstance(items, dict):
                items["additionalProperties"] = True
                items["required"] = []
                what_change = items.get("properties", {}).get("what_would_change_mind")
                if isinstance(what_change, dict):
                    items["properties"]["what_would_change_mind"] = {
                        "anyOf": [what_change, {"type": "string"}]
                    }
    return relaxed


def _relax_axis_item_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    relaxed = deepcopy(schema)
    if isinstance(relaxed, dict):
        relaxed["additionalProperties"] = True
        relaxed["required"] = []
        what_change = relaxed.get("properties", {}).get("what_would_change_mind")
        if isinstance(what_change, dict):
            relaxed["properties"]["what_would_change_mind"] = {
                "anyOf": [what_change, {"type": "string"}]
            }
    return relaxed


def _ensure_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _normalize_axis_fields(axis: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(axis)
    if not normalized.get("axis_name"):
        axis_name = normalized.get("name") or normalized.get("title")
        if axis_name:
            normalized["axis_name"] = axis_name
    if not normalized.get("pole_a"):
        pole_a = normalized.get("low") or normalized.get("pole_low")
        if pole_a:
            normalized["pole_a"] = pole_a
    if not normalized.get("pole_b"):
        pole_b = normalized.get("high") or normalized.get("pole_high")
        if pole_b:
            normalized["pole_b"] = pole_b
    if normalized.get("impact_score") is None and normalized.get("impact") is not None:
        normalized["impact_score"] = normalized.get("impact")
    if (
        normalized.get("uncertainty_score") is None
        and normalized.get("uncertainty") is not None
    ):
        normalized["uncertainty_score"] = normalized.get("uncertainty")
    if normalized.get("uncertainty_score") is None and normalized.get("volatility") is not None:
        normalized["uncertainty_score"] = normalized.get("volatility")
    for key in ("impact_score", "uncertainty_score"):
        value = normalized.get(key)
        if isinstance(value, str):
            try:
                normalized[key] = float(value)
            except ValueError:
                pass
    if isinstance(normalized.get("what_would_change_mind"), str):
        normalized["what_would_change_mind"] = [normalized["what_would_change_mind"]]
    tension = normalized.get("tension_basis")
    if not isinstance(tension, Mapping):
        tension = {}
    cluster_ids = _ensure_list(tension.get("cluster_ids") or normalized.get("cluster_ids"))
    force_ids = _ensure_list(tension.get("force_ids") or normalized.get("force_ids"))
    normalized["tension_basis"] = {"cluster_ids": cluster_ids, "force_ids": force_ids}
    return normalized


def _validate_axis_item(axis: Mapping[str, Any]) -> tuple[bool, list[str]]:
    errors: list[str] = []
    for field in _REQUIRED_AXIS_FIELDS:
        if not axis.get(field):
            errors.append(f"missing_{field}")
    for key in ("impact_score", "uncertainty_score"):
        value = axis.get(key)
        if not isinstance(value, (int, float)):
            errors.append(f"invalid_{key}")
        elif value < 0 or value > 5:
            errors.append(f"{key}_out_of_range")
    tension = axis.get("tension_basis")
    if not isinstance(tension, Mapping):
        errors.append("invalid_tension_basis")
    else:
        cluster_ids = tension.get("cluster_ids")
        force_ids = tension.get("force_ids")
        if not isinstance(cluster_ids, list):
            errors.append("invalid_cluster_ids")
        if not isinstance(force_ids, list):
            errors.append("invalid_force_ids")
    what_change = axis.get("what_would_change_mind")
    if not isinstance(what_change, list) or not what_change:
        errors.append("invalid_what_would_change_mind")
    return (len(errors) == 0), errors


def _correct_axis_item(
    *,
    client,
    schema: Mapping[str, Any],
    axis: Mapping[str, Any],
    errors: list[str],
    clusters: list[dict[str, Any]],
    forces: list[dict[str, Any]],
) -> dict[str, Any]:
    prompt_bundle = build_prompt(
        "uncertainty_axis_correction",
        {
            "axis": dict(axis),
            "errors": errors,
            "clusters": clusters,
            "forces": forces,
        },
    )
    response = client.generate_json(prompt_bundle.text, schema)
    parsed = ensure_dict(response, node_name="uncertainty_axis_correction")
    return dict(parsed)


def _repair_invalid_axes(
    *,
    client,
    schema: Mapping[str, Any],
    axes: list[dict[str, Any]],
    clusters: list[dict[str, Any]],
    forces: list[dict[str, Any]],
    rejected_axes: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    corrected: list[dict[str, Any]] = []
    correction_retries = 0
    for axis in axes:
        ok, errors = _validate_axis_item(axis)
        if ok:
            corrected.append(axis)
            continue
        retries = _max_axis_fix_retries()
        fixed = None
        for attempt in range(retries):
            correction_retries += 1
            candidate = _correct_axis_item(
                client=client,
                schema=schema,
                axis=axis,
                errors=errors,
                clusters=clusters,
                forces=forces,
            )
            candidate = _normalize_axis_fields(candidate)
            if not candidate.get("axis_id") and axis.get("axis_id"):
                candidate["axis_id"] = axis.get("axis_id")
            ok, errors = _validate_axis_item(candidate)
            if ok:
                fixed = candidate
                break
        if fixed:
            corrected.append(fixed)
        else:
            rejected_axes.append(
                {
                    "axis_id": axis.get("axis_id"),
                    "axis_name": axis.get("axis_name"),
                    "reason": ",".join(errors) if errors else "validation_failed",
                }
            )
    return corrected, correction_retries


def run_uncertainty_axes_node(
    *,
    run_id: str,
    state: ScenarioOpsState,
    user_params: Mapping[str, Any],
    llm_client=None,
    base_dir: Path | None = None,
    config: LLMConfig | None = None,
    settings: ScenarioOpsSettings | None = None,
    max_selected: int = 2,
    allow_needs_correction: bool | None = None,
) -> ScenarioOpsState:
    if state.clusters is None:
        raise ValueError("Clusters are required before uncertainty axes.")
    if state.forces is None:
        raise ValueError("Forces are required before uncertainty axes.")

    clusters = state.clusters.get("clusters", [])
    forces = state.forces.get("forces", [])
    prompt_bundle = build_prompt(
        "uncertainty_axes",
        {"clusters": clusters, "forces": forces},
    )
    client = get_client(llm_client, config)
    strict_schema = load_schema("uncertainty_axes_payload")
    schema = _relax_uncertainty_axes_schema(strict_schema)
    axis_item_schema = _relax_axis_item_schema(_axis_item_schema(strict_schema))
    response = client.generate_json(prompt_bundle.text, schema)
    parsed = ensure_dict(response, node_name="uncertainty_axes")

    axes = parsed.get("axes", [])
    if not isinstance(axes, list):
        raise TypeError("Uncertainty axes payload must include axes list.")
    axes = [dict(axis) for axis in axes if isinstance(axis, Mapping)]
    axes = [_normalize_axis_fields(axis) for axis in axes]
    if os.environ.get("DEBUG_UNCERTAINTY_AXES") == "1":
        debug_axes = []
        for axis in axes:
            debug_axes.append(
                {
                    "axis_id": axis.get("axis_id"),
                    "axis_name": axis.get("axis_name"),
                    "what_would_change_mind": axis.get("what_would_change_mind"),
                    "what_would_change_mind_type": type(
                        axis.get("what_would_change_mind")
                    ).__name__,
                }
            )
        dirs = ensure_run_dirs(run_id, base_dir=base_dir)
        debug_path = dirs["logs_dir"] / "uncertainty_axes.debug.json"
        debug_path.write_text(
            json.dumps(debug_axes, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    warnings: list[str] = []
    if len(axes) < 6 or len(axes) > 10:
        warnings.append(f"axis_count_out_of_range:{len(axes)}")
    rejected_axes: list[dict[str, Any]] = []
    axes, correction_retries = _repair_invalid_axes(
        client=client,
        schema=axis_item_schema,
        axes=axes,
        clusters=clusters if isinstance(clusters, list) else [],
        forces=forces if isinstance(forces, list) else [],
        rejected_axes=rejected_axes,
    )
    if rejected_axes:
        warnings.append(f"axis_rejected:{len(rejected_axes)}")
    for idx, axis in enumerate(axes, start=1):
        if not axis.get("axis_id"):
            axis["axis_id"] = f"axis-{idx}"
        tension = axis.get("tension_basis", {})
        if not tension.get("cluster_ids") or not tension.get("force_ids"):
            warnings.append(f"axis_missing_links:{axis.get('axis_id')}")

    scored = []
    for axis in axes:
        impact = float(axis.get("impact_score", 0))
        uncertainty = float(axis.get("uncertainty_score", 0))
        scored.append((impact * uncertainty, axis))
    scored.sort(key=lambda item: (-item[0], str(item[1].get("axis_id"))))

    selected: list[Mapping[str, Any]] = []
    for _, axis in scored:
        if len(selected) >= max_selected:
            break
        if any(_axis_similarity(axis, other) > 0.6 for other in selected):
            continue
        selected.append(axis)

    allow = (
        allow_needs_correction
        if allow_needs_correction is not None
        else _allow_needs_correction(settings, config)
    )
    if len(selected) < max_selected:
        warnings.append("axis_selection_insufficient")
        if allow:
            for _, axis in scored:
                if axis in selected:
                    continue
                selected.append(axis)
                if len(selected) >= max_selected:
                    break

    metadata = build_run_metadata(run_id=run_id, user_params=user_params)
    payload = {
        **metadata,
        "needs_correction": bool(warnings),
        "warnings": warnings,
        "axes": selected,
        "selected_axis_ids": [str(axis.get("axis_id")) for axis in selected],
    }
    validate_artifact("uncertainty_axes", payload)

    write_artifact(
        run_id=run_id,
        artifact_name="uncertainty_axes",
        payload=payload,
        ext="json",
        input_values={"axis_count": len(selected), "selected_count": len(selected)},
        prompt_values={
            "prompt_name": prompt_bundle.name,
            "prompt_sha256": prompt_bundle.sha256,
        },
        tool_versions={"uncertainty_axes_node": "0.1.0"},
        base_dir=base_dir,
    )
    state.uncertainty_axes = payload
    if warnings and not allow:
        raise RuntimeError("uncertainty_axes_needs_correction")
    return state
