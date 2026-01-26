from __future__ import annotations

from typing import Any, Mapping


def build_matrix(
    evaluations: list[Mapping[str, Any]],
    *,
    strategies: list[Mapping[str, Any]] | None = None,
    scenarios: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    strategy_names = {
        str(item.get("id")): item.get("name") or str(item.get("id"))
        for item in (strategies or [])
        if isinstance(item, Mapping)
    }
    scenario_names = {
        str(item.get("id") or item.get("scenario_id")): item.get("name")
        or str(item.get("id") or item.get("scenario_id"))
        for item in (scenarios or [])
        if isinstance(item, Mapping)
    }
    for evaluation in evaluations:
        strategy_id = str(evaluation.get("strategy_id"))
        scenario_id = str(evaluation.get("scenario_id"))
        if strategy_id and strategy_id not in strategy_names:
            strategy_names[strategy_id] = (
                evaluation.get("strategy_name") or strategy_id
            )
        if scenario_id and scenario_id not in scenario_names:
            scenario_names[scenario_id] = (
                evaluation.get("scenario_name") or scenario_id
            )

    matrix: dict[str, dict[str, Mapping[str, Any]]] = {}
    for evaluation in evaluations:
        strategy_id = str(evaluation.get("strategy_id"))
        scenario_id = str(evaluation.get("scenario_id"))
        matrix.setdefault(strategy_id, {})[scenario_id] = evaluation

    return {
        "strategy_names": strategy_names,
        "scenario_names": scenario_names,
        "matrix": matrix,
    }


def get_cell_detail(
    evaluations: list[Mapping[str, Any]],
    *,
    strategy_id: str,
    scenario_id: str,
) -> Mapping[str, Any] | None:
    for evaluation in evaluations:
        if (
            str(evaluation.get("strategy_id")) == strategy_id
            and str(evaluation.get("scenario_id")) == scenario_id
        ):
            return evaluation
    return None


__all__ = ["build_matrix", "get_cell_detail"]
