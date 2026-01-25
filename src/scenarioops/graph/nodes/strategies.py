from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from scenarioops.app.config import LLMConfig
from scenarioops.graph.nodes.utils import build_prompt, get_client
from scenarioops.graph.state import ScenarioOpsState, Strategies, Strategy
from scenarioops.graph.tools.normalization import stable_id
from scenarioops.graph.tools.schema_validate import load_schema, validate_artifact
from scenarioops.graph.tools.storage import log_normalization, write_artifact
from scenarioops.llm.guards import ensure_dict, ensure_list


_REQUIRED_STRATEGY_FIELDS = ("name", "objective", "kpis")


def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(key)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(1, value)


def _max_strategy_fix_retries() -> int:
    return _env_int("MAX_STRATEGY_FIX_RETRIES", 2)


def _strategies_item_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    defs = schema.get("$defs")
    if isinstance(defs, Mapping):
        item_schema = defs.get("strategy")
        if isinstance(item_schema, Mapping):
            return dict(item_schema)
    return {"title": "Strategy Item", "type": "object"}


def _relax_strategies_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    relaxed = deepcopy(schema)
    if isinstance(relaxed, dict):
        relaxed["additionalProperties"] = True
        relaxed["required"] = []
    properties = relaxed.get("properties")
    if isinstance(properties, dict):
        strategies_prop = properties.get("strategies")
        if isinstance(strategies_prop, dict):
            strategies_prop["minItems"] = 0
            items = strategies_prop.get("items")
            if isinstance(items, dict):
                items["additionalProperties"] = True
                items["required"] = []
    defs = relaxed.get("$defs")
    if isinstance(defs, dict):
        strategy_def = defs.get("strategy")
        if isinstance(strategy_def, dict):
            strategy_def["additionalProperties"] = True
            strategy_def["required"] = []
    return relaxed


def _relax_strategy_item_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    relaxed = deepcopy(schema)
    if isinstance(relaxed, dict):
        relaxed["additionalProperties"] = True
        relaxed["required"] = []
    return relaxed


def _ensure_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _normalize_strategy_fields(
    strategy: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    normalized = dict(strategy)
    changed: list[str] = []
    if not normalized.get("name"):
        name = (
            normalized.get("title")
            or normalized.get("label")
            or normalized.get("strategy_name")
            or normalized.get("strategy")
        )
        if name:
            normalized["name"] = name
            changed.append("name")
    if not normalized.get("objective"):
        objective = (
            normalized.get("goal")
            or normalized.get("intent")
            or normalized.get("outcome")
            or normalized.get("why")
        )
        if objective:
            normalized["objective"] = objective
            changed.append("objective")
    for key in ("actions", "owners", "dependencies", "risks", "fit", "kpis"):
        value = normalized.get(key)
        if isinstance(value, str) or isinstance(value, list):
            normalized[key] = _ensure_list(value)
            if normalized.get(key) != value:
                changed.append(key)
    if not normalized.get("kpis"):
        for alt in ("kpi", "metrics", "success_metrics", "success_criteria"):
            value = normalized.get(alt)
            if value:
                normalized["kpis"] = _ensure_list(value)
                changed.append("kpis")
                break
    return normalized, changed


def _impute_strategy_fields(
    strategy: Mapping[str, Any], index: int
) -> tuple[dict[str, Any], list[str]]:
    filled = dict(strategy)
    imputed: list[str] = []
    if not filled.get("name"):
        filled["name"] = f"Strategy {index}"
        imputed.append("name")
    if not filled.get("objective"):
        filled["objective"] = f"Advance {filled.get('name')} in the scenarios."
        imputed.append("objective")
    if not filled.get("actions"):
        filled["actions"] = ["Define execution plan"]
        imputed.append("actions")
    if not filled.get("kpis"):
        filled["kpis"] = ["Define KPI"]
        imputed.append("kpis")
    return filled, imputed


def _validate_strategy_item(strategy: Mapping[str, Any]) -> tuple[bool, list[str]]:
    errors: list[str] = []
    for field in _REQUIRED_STRATEGY_FIELDS:
        if not strategy.get(field):
            errors.append(f"missing_{field}")
    kpis = strategy.get("kpis")
    if not isinstance(kpis, list) or not kpis:
        errors.append("invalid_kpis")
    return (len(errors) == 0), errors


def _correct_strategy_item(
    *,
    client,
    schema: Mapping[str, Any],
    strategy: Mapping[str, Any],
    errors: list[str],
    scenarios: list[Mapping[str, Any]],
) -> dict[str, Any]:
    prompt_bundle = build_prompt(
        "strategy_item_correction",
        {
            "strategy": dict(strategy),
            "errors": errors,
            "scenarios": scenarios,
        },
    )
    response = client.generate_json(prompt_bundle.text, schema)
    parsed = ensure_dict(response, node_name="strategy_item_correction")
    return dict(parsed)


def _repair_invalid_strategies(
    *,
    client,
    schema: Mapping[str, Any],
    strategies: list[dict[str, Any]],
    scenarios: list[Mapping[str, Any]],
    rejected: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    corrected: list[dict[str, Any]] = []
    correction_retries = 0
    for idx, strategy in enumerate(strategies, start=1):
        strategy, _ = _impute_strategy_fields(strategy, idx)
        ok, errors = _validate_strategy_item(strategy)
        if ok:
            corrected.append(strategy)
            continue
        retries = _max_strategy_fix_retries()
        fixed = None
        for attempt in range(retries):
            correction_retries += 1
            candidate = _correct_strategy_item(
                client=client,
                schema=schema,
                strategy=strategy,
                errors=errors,
                scenarios=scenarios,
            )
            candidate = _normalize_strategy_fields(candidate)
            candidate = _impute_strategy_fields(candidate, idx)
            if not candidate.get("id") and strategy.get("id"):
                candidate["id"] = strategy.get("id")
            ok, errors = _validate_strategy_item(candidate)
            if ok:
                fixed = candidate
                break
        if fixed:
            corrected.append(fixed)
        else:
            rejected.append(
                {
                    "strategy_id": strategy.get("id"),
                    "strategy_name": strategy.get("name"),
                    "reason": ",".join(errors) if errors else "validation_failed",
                }
            )
            corrected.append(strategy)
    return corrected, correction_retries


def run_strategies_node(
    *,
    run_id: str,
    state: ScenarioOpsState,
    strategy_notes: str | None = None,
    llm_client=None,
    base_dir: Path | None = None,
    config: LLMConfig | None = None,
) -> ScenarioOpsState:
    scenario_payload = None
    if state.logic is None:
        if state.scenarios is None:
            raise ValueError("Logic or scenarios are required to generate strategies.")
        scenario_payload = state.scenarios.get("scenarios", [])

    scenarios = (
        [scenario.__dict__ for scenario in state.logic.scenarios]
        if state.logic is not None
        else (scenario_payload or [])
    )
    prompt_context = {"scenarios": scenarios}
    if strategy_notes:
        prompt_context["strategy_notes"] = strategy_notes
    prompt_bundle = build_prompt("strategies", prompt_context)
    prompt = prompt_bundle.text

    client = get_client(llm_client, config)
    strict_schema = load_schema("strategies")
    schema = _relax_strategies_schema(strict_schema)
    strategy_item_schema = _relax_strategy_item_schema(_strategies_item_schema(strict_schema))
    response = client.generate_json(prompt, schema)
    try:
        parsed = ensure_dict(response, node_name="strategies")
    except TypeError:
        strategies_list = ensure_list(response, node_name="strategies")
        parsed = {"strategies": strategies_list}
        log_normalization(
            run_id=run_id,
            node_name="strategies",
            operation="wrapped_list_payload",
            details={"count": len(strategies_list)},
            base_dir=base_dir,
        )

    strategies = parsed.get("strategies", [])
    if not isinstance(strategies, list):
        raise TypeError("Strategies payload must include strategies list.")
    strategies = [dict(strategy) for strategy in strategies if isinstance(strategy, Mapping)]
    normalized_strategies: list[dict[str, Any]] = []
    for idx, strategy in enumerate(strategies, start=1):
        normalized, changed = _normalize_strategy_fields(strategy)
        if changed:
            log_normalization(
                run_id=run_id,
                node_name="strategies",
                operation="normalized_strategy_fields",
                details={"fields": changed, "index": idx},
                base_dir=base_dir,
            )
        normalized, imputed = _impute_strategy_fields(normalized, idx)
        if imputed:
            log_normalization(
                run_id=run_id,
                node_name="strategies",
                operation="imputed_strategy_fields",
                details={"fields": imputed, "index": idx},
                base_dir=base_dir,
            )
        normalized_strategies.append(normalized)
    strategies = normalized_strategies

    if not parsed.get("id"):
        scenario_ids = (
            [scenario.id for scenario in state.logic.scenarios]
            if state.logic is not None
            else [str(entry.get("scenario_id") or entry.get("id")) for entry in scenarios if isinstance(entry, dict)]
        )
        parsed["id"] = stable_id(
            "strategies",
            parsed.get("title"),
            scenario_ids,
        )
        log_normalization(
            run_id=run_id,
            node_name="strategies",
            operation="stable_id_assigned",
            details={"field": "id"},
            base_dir=base_dir,
        )

    rejected: list[dict[str, Any]] = []
    strategies, correction_retries = _repair_invalid_strategies(
        client=client,
        schema=strategy_item_schema,
        strategies=strategies,
        scenarios=[scenario for scenario in scenarios if isinstance(scenario, Mapping)],
        rejected=rejected,
    )
    parsed["strategies"] = strategies
    if rejected:
        metadata = parsed.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        metadata["needs_correction"] = True
        metadata["rejected_strategies"] = rejected
        parsed["metadata"] = metadata

    validate_artifact("strategies", parsed)
    for strategy in parsed.get("strategies", []):
        if not strategy.get("id"):
            strategy["id"] = stable_id(
                "strategy",
                strategy.get("name"),
                strategy.get("objective"),
                strategy.get("actions", []),
            )
            log_normalization(
                run_id=run_id,
                node_name="strategies",
                operation="stable_id_assigned",
                details={"field": "strategy.id", "name": strategy.get("name", "")},
                base_dir=base_dir,
            )
        if not strategy.get("kpis"):
            raise ValueError(f"Strategy {strategy.get('id')} missing KPIs.")

    write_artifact(
        run_id=run_id,
        artifact_name="strategies",
        payload=parsed,
        ext="json",
        input_values={
            "scenario_ids": (
                [scenario.id for scenario in state.logic.scenarios]
                if state.logic is not None
                else [str(entry.get("scenario_id") or entry.get("id")) for entry in scenarios if isinstance(entry, dict)]
            ),
            "strategy_notes": strategy_notes or "",
        },
        prompt_values={
            "prompt_name": prompt_bundle.name,
            "prompt_sha256": prompt_bundle.sha256,
        },
        tool_versions={
            "strategies_node": "0.1.1",
            "strategy_correction_retries": str(correction_retries),
        },
        base_dir=base_dir,
    )

    strategies = [Strategy(**strategy) for strategy in parsed.get("strategies", [])]
    state.strategies = Strategies(
        id=parsed.get("id", f"strategies-{run_id}"),
        title=parsed.get("title", "Strategies"),
        strategies=strategies,
    )
    return state
