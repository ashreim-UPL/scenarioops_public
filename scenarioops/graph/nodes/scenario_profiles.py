from __future__ import annotations

from pathlib import Path
from typing import Any

from scenarioops.graph.state import ScenarioOpsState
from scenarioops.graph.tools.normalization import stable_id
from scenarioops.graph.tools.schema_validate import validate_artifact
from scenarioops.graph.tools.storage import log_normalization, write_artifact


DEFAULT_DRIVER_CONFIDENCE = 0.5


def _driver_refs(state: ScenarioOpsState) -> list[dict[str, Any]]:
    drivers: list[dict[str, Any]] = []
    if state.drivers is not None:
        for entry in state.drivers.drivers:
            confidence = entry.confidence
            basis = "driver_confidence" if confidence is not None else "default"
            drivers.append(
                {
                    "id": entry.id,
                    "name": entry.name,
                    "category": entry.category,
                    "trend": entry.trend,
                    "impact": entry.impact,
                    "confidence": confidence if confidence is not None else DEFAULT_DRIVER_CONFIDENCE,
                    "confidence_basis": basis,
                    "citations": entry.citations,
                }
            )
    elif isinstance(state.driving_forces, dict):
        for force in state.driving_forces.get("forces", []) or []:
            drivers.append(
                {
                    "id": force.get("id"),
                    "name": force.get("name"),
                    "category": force.get("domain"),
                    "trend": "unspecified",
                    "impact": "unspecified",
                    "confidence": DEFAULT_DRIVER_CONFIDENCE,
                    "confidence_basis": "default",
                    "citations": force.get("citations", []),
                }
            )
    return sorted(drivers, key=lambda item: str(item.get("id", "")))


def _assumption_refs(summary: dict[str, Any]) -> list[dict[str, Any]]:
    assumptions = []
    for entry in summary.get("assumptions", []) or []:
        assumptions.append(
            {
                "id": entry.get("id"),
                "statement": entry.get("statement"),
                "uncertainty_id": entry.get("uncertainty_id"),
                "stance": entry.get("stance"),
                "evidence_ids": entry.get("evidence_ids", []),
                "confidence": entry.get("confidence"),
                "confidence_basis": entry.get("confidence_basis"),
            }
        )
    return sorted(assumptions, key=lambda item: str(item.get("id", "")))


def _unknown_refs(summary: dict[str, Any]) -> list[dict[str, Any]]:
    unknowns = []
    for entry in summary.get("unknowns", []) or []:
        unknowns.append(
            {
                "id": entry.get("id"),
                "statement": entry.get("statement"),
                "evidence_ids": entry.get("evidence_ids", []),
                "impact": entry.get("impact"),
                "uncertainty": entry.get("uncertainty"),
                "confidence": entry.get("confidence"),
            }
        )
    return sorted(unknowns, key=lambda item: str(item.get("id", "")))


def _causal_chain(state: ScenarioOpsState) -> list[dict[str, Any]]:
    chain = []
    if isinstance(state.effects, dict):
        for entry in state.effects.get("effects", []) or []:
            chain.append(
                {
                    "id": entry.get("id"),
                    "belief_id": entry.get("belief_id"),
                    "order": entry.get("order"),
                    "description": entry.get("description"),
                    "domains": entry.get("domains", []),
                }
            )
    return sorted(chain, key=lambda item: (item.get("order", 0), str(item.get("id", ""))))


def _signals(state: ScenarioOpsState, scenario_id: str) -> list[dict[str, Any]]:
    signals = []
    if state.ewi is None:
        return signals
    for indicator in state.ewi.indicators:
        if scenario_id in indicator.linked_scenarios:
            payload = {
                "id": indicator.id,
                "name": indicator.name,
                "description": indicator.description,
                "signal": indicator.signal,
                "metric": indicator.metric,
            }
            if indicator.threshold is not None:
                payload["threshold"] = indicator.threshold
            signals.append(payload)
    return sorted(signals, key=lambda item: str(item.get("id", "")))


def _options(state: ScenarioOpsState, scenario_id: str) -> list[dict[str, Any]]:
    if state.wind_tunnel is None:
        return []
    options = []
    for test in state.wind_tunnel.tests:
        if test.scenario_id != scenario_id:
            continue
        options.append(
            {
                "strategy_id": test.strategy_id,
                "action": test.action,
                "outcome": test.outcome,
                "feasibility_score": test.feasibility_score,
                "rubric_score": test.rubric_score,
            }
        )
    return sorted(options, key=lambda item: str(item.get("strategy_id", "")))


def _risks_opportunities(state: ScenarioOpsState, scenario_id: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if state.wind_tunnel is None:
        return [], []
    risks: list[dict[str, Any]] = []
    opportunities: list[dict[str, Any]] = []
    for test in state.wind_tunnel.tests:
        if test.scenario_id != scenario_id:
            continue
        for failure in test.failure_modes:
            risks.append({"description": failure, "source_test_id": test.id})
        for adaptation in test.adaptations:
            opportunities.append({"description": adaptation, "source_test_id": test.id})
    risks_sorted = sorted(risks, key=lambda item: item["description"])
    opp_sorted = sorted(opportunities, key=lambda item: item["description"])
    return risks_sorted, opp_sorted


def _implications(state: ScenarioOpsState, scenario_id: str) -> list[dict[str, Any]]:
    implications: list[dict[str, Any]] = []
    if state.skeleton is None:
        return implications
    for scenario in state.skeleton.scenarios:
        if scenario.id != scenario_id:
            continue
        for key, value in scenario.operating_rules.items():
            implications.append({"type": "operating_rule", "description": f"{key}: {value}"})
        for event in scenario.key_events:
            implications.append({"type": "event", "description": event.event})
    return sorted(implications, key=lambda item: item["description"])


def _scenario_confidence(drivers: list[dict[str, Any]]) -> dict[str, Any]:
    if not drivers:
        return {"overall": DEFAULT_DRIVER_CONFIDENCE, "basis": "default"}
    values = [float(entry.get("confidence", DEFAULT_DRIVER_CONFIDENCE)) for entry in drivers]
    overall = sum(values) / len(values)
    return {"overall": round(overall, 3), "basis": "driver_mean"}


def run_scenario_profiles_node(
    *,
    run_id: str,
    state: ScenarioOpsState,
    base_dir: Path | None = None,
) -> ScenarioOpsState:
    if state.logic is None:
        raise ValueError("Scenario logic required for scenario profiles.")
    if state.skeleton is None:
        raise ValueError("Skeletons required for scenario profiles.")
    if not isinstance(state.epistemic_summary, dict):
        raise ValueError("Epistemic summary required for scenario profiles.")

    drivers = _driver_refs(state)
    assumptions = _assumption_refs(state.epistemic_summary)
    unknowns = _unknown_refs(state.epistemic_summary)
    causal_chain = _causal_chain(state)

    scenarios = []
    narrative_map = state.narratives or {}
    for scenario in state.logic.scenarios:
        scenario_id = scenario.id
        narrative = narrative_map.get(scenario_id)
        if not narrative:
            for skel in state.skeleton.scenarios:
                if skel.id == scenario_id:
                    narrative = skel.narrative
                    break
        if not narrative:
            raise ValueError(f"Scenario {scenario_id} missing narrative text.")
        signals = _signals(state, scenario_id)
        options = _options(state, scenario_id)
        risks, opportunities = _risks_opportunities(state, scenario_id)
        implications = _implications(state, scenario_id)

        scenarios.append(
            {
                "id": scenario_id,
                "name": scenario.name,
                "logic": scenario.logic,
                "narrative": narrative or "",
                "drivers": drivers,
                "assumptions": assumptions,
                "unknowns": unknowns,
                "causal_chain": causal_chain,
                "signals": signals,
                "risks": risks,
                "opportunities": opportunities,
                "implications": implications,
                "options": options,
                "what_would_change_my_mind": [
                    {
                        "indicator_id": signal.get("id"),
                        "signal": signal.get("signal"),
                        "metric": signal.get("metric"),
                    }
                    for signal in signals
                ],
                "epistemic_refs": {
                    "facts": [item.get("id") for item in state.epistemic_summary.get("facts", [])],
                    "assumptions": [item.get("id") for item in assumptions],
                    "interpretations": [
                        item.get("id") for item in state.epistemic_summary.get("interpretations", [])
                    ],
                    "unknowns": [item.get("id") for item in unknowns],
                },
                "confidence": _scenario_confidence(drivers),
            }
        )

    profile_id = stable_id(
        "scenario-profiles",
        [scenario.id for scenario in state.logic.scenarios],
    )
    log_normalization(
        run_id=run_id,
        node_name="scenario_profiles",
        operation="stable_id_assigned",
        details={"field": "id"},
        base_dir=base_dir,
    )
    profiles = {
        "id": profile_id,
        "title": "Scenario Profiles",
        "scenarios": scenarios,
    }

    validate_artifact("scenario_profiles", profiles)

    write_artifact(
        run_id=run_id,
        artifact_name="scenario_profiles",
        payload=profiles,
        ext="json",
        input_values={"scenario_count": len(scenarios)},
        prompt_values={"prompt": "scenario_profiles"},
        tool_versions={"scenario_profiles_node": "0.1.0"},
        base_dir=base_dir,
    )

    state.scenario_profiles = profiles
    return state
