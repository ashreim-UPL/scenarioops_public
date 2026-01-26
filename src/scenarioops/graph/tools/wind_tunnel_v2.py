from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping


_OUTCOME_BANDS: dict[str, tuple[int, int]] = {
    "optimal success": (90, 100),
    "hegemonic success": (85, 95),
    "moderate success": (70, 84),
    "partial integration": (60, 74),
    "stalled growth": (45, 59),
    "inefficient dominance": (40, 55),
    "irrelevant strategy": (25, 45),
    "total strategic failure": (0, 20),
    "inconclusive": (35, 65),
}


def _normalize_label(label: str) -> str:
    return " ".join(str(label or "inconclusive").strip().lower().split())


def outcome_band(label: str) -> tuple[int, int]:
    return _OUTCOME_BANDS.get(_normalize_label(label), _OUTCOME_BANDS["inconclusive"])


def baseline_score(label: str, rubric_score: float | None = None) -> float:
    low, high = outcome_band(label)
    if isinstance(rubric_score, (int, float)) and 0.0 <= float(rubric_score) <= 1.0:
        return low + (high - low) * float(rubric_score)
    return (low + high) / 2.0


def grade_from_score(score: float) -> str:
    if score >= 85:
        return "A"
    if score >= 70:
        return "B"
    if score >= 55:
        return "C"
    if score >= 40:
        return "D"
    return "F"


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _confidence_from_test(test: Mapping[str, Any]) -> float:
    raw = test.get("confidence")
    if isinstance(raw, (int, float)):
        return _clamp(float(raw), 0.0, 1.0)
    raw = test.get("rubric_score")
    if isinstance(raw, (int, float)):
        return _clamp(float(raw), 0.0, 1.0)
    return 0.5


def _keywords(text: str) -> set[str]:
    tokens = [
        token.strip(".,:;()[]{}\"'").lower()
        for token in str(text or "").split()
    ]
    return {token for token in tokens if len(token) >= 4}


def _select_dominant_forces(
    forces: Iterable[Mapping[str, Any]],
    strategy_text: str,
    scenario_text: str,
    outcome_text: str,
    *,
    limit: int = 3,
) -> list[str]:
    strategy_kw = _keywords(strategy_text)
    scenario_kw = _keywords(scenario_text)
    outcome_kw = _keywords(outcome_text)
    combined = strategy_kw | scenario_kw | outcome_kw
    scored: list[tuple[int, str]] = []
    for force in forces:
        label = str(force.get("label") or force.get("force") or force.get("name") or "")
        mechanism = str(force.get("mechanism") or "")
        force_kw = _keywords(label + " " + mechanism)
        score = len(combined & force_kw)
        confidence = force.get("confidence")
        if isinstance(confidence, (int, float)):
            score += int(round(float(confidence) * 3))
        force_id = str(force.get("force_id") or label).strip()
        if force_id:
            scored.append((score, force_id))
    scored.sort(key=lambda item: item[0], reverse=True)
    top = [force_id for score, force_id in scored if score > 0]
    if not top:
        top = [force_id for _, force_id in scored][:limit]
    return top[:limit]


def _infer_break_conditions(
    outcome_label: str, score: float, break_conditions: list[str]
) -> list[str]:
    if not break_conditions:
        return []
    label = _normalize_label(outcome_label)
    if label == "inconclusive":
        return []
    if score < 40 or "failure" in label or "irrelevant" in label:
        return break_conditions[: min(2, len(break_conditions))]
    if score < 55:
        return break_conditions[:1]
    return []


def _kpi_projection(
    kpis: list[str], score: float, outcome_label: str
) -> dict[str, dict[str, Any]]:
    projections: dict[str, dict[str, Any]] = {}
    if not kpis:
        return projections
    direction = "flat"
    if score >= 70:
        direction = "up"
    elif score < 40:
        direction = "down"
    for kpi in kpis[:6]:
        projections[str(kpi)] = {
            "direction": direction,
            "rationale": f"Outcome '{outcome_label}' implies KPI trend {direction}.",
            "est_range_if_any": None,
        }
    return projections


def _rationale_bullets(
    *,
    outcome_label: str,
    score: float,
    confidence: float,
    failure_modes: list[str],
    adaptations: list[str],
    dominant_forces: list[str],
    break_conditions_triggered: list[str],
) -> list[str]:
    bullets: list[str] = [
        f"Outcome '{outcome_label}' with score {score:.0f}/100 and confidence {confidence:.2f}.",
    ]
    if failure_modes:
        bullets.append(f"Key failure risks: {', '.join(failure_modes[:3])}.")
    if adaptations:
        bullets.append(f"Adaptations available: {', '.join(adaptations[:3])}.")
    if dominant_forces:
        bullets.append(f"Dominant forces: {', '.join(dominant_forces[:3])}.")
    if break_conditions_triggered:
        bullets.append(
            f"Break conditions triggered: {', '.join(break_conditions_triggered[:2])}."
        )
    while len(bullets) < 3:
        bullets.append("Maintain monitoring to validate assumptions.")
    return bullets[:6]


def _hardening_actions(adaptations: list[str], failure_modes: list[str]) -> list[str]:
    actions: list[str] = []
    actions.extend([str(item) for item in adaptations if item])
    if not actions:
        actions.extend([f"Mitigate: {item}" for item in failure_modes[:3] if item])
    if not actions:
        actions = [
            "Add leading KPI monitoring for early warning.",
            "Stress-test critical dependencies quarterly.",
        ]
    return actions[:6]


def _trigger_points(
    *,
    strategy_id: str,
    dominant_forces: list[str],
    break_conditions_triggered: list[str],
    kpis: list[str],
) -> list[dict[str, Any]]:
    triggers: list[dict[str, Any]] = []
    for idx, force in enumerate(dominant_forces[:2], start=1):
        triggers.append(
            {
                "trigger_id": f"{strategy_id}-force-{idx}",
                "description": f"Force '{force}' shifts materially.",
                "signal_source": "external",
                "threshold": "sustained 2+ sigma move in indicators",
                "recommended_action": "hedge",
            }
        )
    for idx, cond in enumerate(break_conditions_triggered[:2], start=1):
        triggers.append(
            {
                "trigger_id": f"{strategy_id}-break-{idx}",
                "description": cond,
                "signal_source": "external",
                "threshold": "break condition observed",
                "recommended_action": "switch",
            }
        )
    for idx, kpi in enumerate(kpis[:2], start=1):
        triggers.append(
            {
                "trigger_id": f"{strategy_id}-kpi-{idx}",
                "description": f"KPI '{kpi}' underperforms.",
                "signal_source": "internal",
                "threshold": "2 consecutive periods below target",
                "recommended_action": "adapt",
            }
        )
    return triggers


@dataclass(frozen=True)
class EvaluationInputs:
    run_id: str
    strategies: list[Mapping[str, Any]]
    scenarios: list[Mapping[str, Any]]
    tests: list[Mapping[str, Any]]
    forces: list[Mapping[str, Any]]
    break_conditions: list[str]
    triggers: list[str]


def build_wind_tunnel_evaluations(
    inputs: EvaluationInputs,
) -> dict[str, Any]:
    strategies = inputs.strategies
    scenarios = inputs.scenarios
    tests = inputs.tests
    forces = inputs.forces
    break_conditions = [str(item) for item in inputs.break_conditions if str(item)]

    strategy_map = {
        str(item.get("id")): item for item in strategies if isinstance(item, Mapping)
    }
    scenario_map = {
        str(item.get("id") or item.get("scenario_id")): item
        for item in scenarios
        if isinstance(item, Mapping)
    }

    tests_by_pair: dict[tuple[str, str], Mapping[str, Any]] = {}
    for test in tests:
        strategy_id = str(test.get("strategy_id"))
        scenario_id = str(test.get("scenario_id"))
        if strategy_id and scenario_id:
            tests_by_pair[(strategy_id, scenario_id)] = test

    evaluations: list[dict[str, Any]] = []
    for strategy_id, strategy in strategy_map.items():
        for scenario_id, scenario in scenario_map.items():
            test = tests_by_pair.get((strategy_id, scenario_id), {})
            outcome_label = str(test.get("outcome") or "INCONCLUSIVE")
            rubric_score = test.get("rubric_score")
            base = baseline_score(outcome_label, rubric_score)
            confidence = _confidence_from_test(test)
            failure_modes = list(test.get("failure_modes") or [])
            adaptations = list(test.get("adaptations") or [])
            strategy_text = " ".join(
                [
                    str(strategy.get("name") or ""),
                    str(strategy.get("objective") or ""),
                    " ".join(strategy.get("actions") or []),
                ]
            )
            scenario_text = " ".join(
                [
                    str(scenario.get("name") or ""),
                    str(scenario.get("summary") or scenario.get("narrative") or ""),
                    str(scenario.get("logic") or ""),
                ]
            )
            dominant_forces = _select_dominant_forces(
                forces,
                strategy_text,
                scenario_text,
                outcome_label,
            )
            break_conditions_triggered = _infer_break_conditions(
                outcome_label, base, break_conditions
            )
            failed_assumptions = [str(item) for item in failure_modes[:4] if str(item)]
            kpis = list(strategy.get("kpis") or [])
            kpi_projection = _kpi_projection(kpis, base, outcome_label)
            kpi_support = bool(kpis) and base >= 70

            confidence_modifier = _clamp((confidence - 0.5) * 30.0, -15.0, 15.0)
            break_penalty = max(-40.0, -20.0 * len(break_conditions_triggered))
            assumption_penalty = max(-20.0, -5.0 * len(failed_assumptions))
            kpi_bonus = 10.0 if kpi_support else 0.0
            score = base + confidence_modifier + break_penalty + assumption_penalty + kpi_bonus
            score = _clamp(score, 0.0, 100.0)

            evaluation = {
                "run_id": inputs.run_id,
                "strategy_id": strategy_id,
                "strategy_name": strategy.get("name") or strategy_id,
                "scenario_id": scenario_id,
                "scenario_name": scenario.get("name") or scenario_id,
                "outcome_label": outcome_label,
                "grade_letter": grade_from_score(score),
                "score_0_100": round(score, 1),
                "confidence_0_1": round(confidence, 2),
                "kpi_projection": kpi_projection,
                "dominant_forces": dominant_forces,
                "failed_assumptions": failed_assumptions,
                "break_conditions_triggered": break_conditions_triggered,
                "rationale_bullets": _rationale_bullets(
                    outcome_label=outcome_label,
                    score=score,
                    confidence=confidence,
                    failure_modes=failure_modes,
                    adaptations=adaptations,
                    dominant_forces=dominant_forces,
                    break_conditions_triggered=break_conditions_triggered,
                ),
                "recommended_hardening_actions": _hardening_actions(
                    adaptations, failure_modes
                ),
                "trigger_points": _trigger_points(
                    strategy_id=strategy_id,
                    dominant_forces=dominant_forces,
                    break_conditions_triggered=break_conditions_triggered,
                    kpis=kpis,
                ),
            }
            evaluations.append(evaluation)

    summary = _build_summary(
        evaluations=evaluations,
        strategy_map=strategy_map,
        scenario_map=scenario_map,
    )
    return {
        "run_id": inputs.run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "evaluations": evaluations,
        **summary,
    }


def _build_summary(
    *,
    evaluations: list[dict[str, Any]],
    strategy_map: Mapping[str, Mapping[str, Any]],
    scenario_map: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    by_scenario: dict[str, list[dict[str, Any]]] = {}
    by_strategy: dict[str, list[dict[str, Any]]] = {}
    for evaluation in evaluations:
        by_scenario.setdefault(evaluation["scenario_id"], []).append(evaluation)
        by_strategy.setdefault(evaluation["strategy_id"], []).append(evaluation)

    per_scenario_rankings: list[dict[str, Any]] = []
    for scenario_id, items in by_scenario.items():
        ranked = sorted(items, key=lambda item: item["score_0_100"], reverse=True)
        scenario_name = scenario_map.get(scenario_id, {}).get("name", scenario_id)
        margin = None
        if len(ranked) > 1:
            margin = round(ranked[0]["score_0_100"] - ranked[1]["score_0_100"], 1)
        per_scenario_rankings.append(
            {
                "scenario_id": scenario_id,
                "scenario_name": scenario_name,
                "rankings": [
                    {
                        "strategy_id": item["strategy_id"],
                        "strategy_name": strategy_map.get(item["strategy_id"], {}).get("name", item["strategy_id"]),
                        "score_0_100": item["score_0_100"],
                        "grade_letter": item["grade_letter"],
                    }
                    for item in ranked
                ],
                "top_strategy_id": ranked[0]["strategy_id"] if ranked else None,
                "margin_vs_second": margin,
            }
        )

    overall_rankings: list[dict[str, Any]] = []
    scenario_fit: list[dict[str, Any]] = []
    for strategy_id, items in by_strategy.items():
        scores = [float(item["score_0_100"]) for item in items]
        overall = sum(scores) / max(1, len(scores))
        min_score = min(scores) if scores else 0.0
        variance = 0.0
        if scores:
            mean = overall
            variance = sum((score - mean) ** 2 for score in scores) / len(scores)
        robustness = 0.6 * overall + 0.4 * min_score - 0.1 * variance
        overall_rankings.append(
            {
                "strategy_id": strategy_id,
                "strategy_name": strategy_map.get(strategy_id, {}).get("name", strategy_id),
                "overall_score": round(overall, 1),
                "min_score": round(min_score, 1),
                "variance": round(variance, 1),
                "robustness_index": round(robustness, 1),
            }
        )
        best_fit = [
            scenario_map.get(item["scenario_id"], {}).get("name", item["scenario_id"])
            for item in items
            if item["score_0_100"] >= 75
        ]
        fail = [
            scenario_map.get(item["scenario_id"], {}).get("name", item["scenario_id"])
            for item in items
            if item["score_0_100"] < 40
        ]
        sensitive = []
        if variance >= 150:
            sensitive = [
                scenario_map.get(item["scenario_id"], {}).get("name", item["scenario_id"])
                for item in items
            ]
        scenario_fit.append(
            {
                "strategy_id": strategy_id,
                "strategy_name": strategy_map.get(strategy_id, {}).get("name", strategy_id),
                "best_fit_scenarios": best_fit,
                "fail_scenarios": fail,
                "sensitive_scenarios": sensitive,
            }
        )

    overall_rankings.sort(key=lambda item: item["robustness_index"], reverse=True)
    for idx, entry in enumerate(overall_rankings, start=1):
        entry["rank"] = idx

    recommendations = _build_recommendations(
        overall_rankings=overall_rankings,
        per_scenario_rankings=per_scenario_rankings,
        evaluations=evaluations,
    )

    return {
        "rankings": {
            "per_scenario": per_scenario_rankings,
            "overall": overall_rankings,
            "scenario_fit": scenario_fit,
        },
        "recommendations": recommendations,
    }


def _build_recommendations(
    *,
    overall_rankings: list[dict[str, Any]],
    per_scenario_rankings: list[dict[str, Any]],
    evaluations: list[dict[str, Any]],
) -> dict[str, Any]:
    primary = overall_rankings[0] if overall_rankings else {}
    primary_id = primary.get("strategy_id")
    conditional: list[dict[str, Any]] = []
    for entry in per_scenario_rankings:
        top_id = entry.get("top_strategy_id")
        if top_id and top_id != primary_id:
            conditional.append(
                {
                    "condition": f"If scenario '{entry.get('scenario_name')}' signals dominate",
                    "strategy_id": top_id,
                    "strategy_name": entry.get("rankings", [{}])[0].get("strategy_name", top_id),
                    "rationale": f"Top score {entry.get('rankings', [{}])[0].get('score_0_100')} for this scenario.",
                }
            )
    hardening_actions: list[str] = []
    triggers_to_watch: list[dict[str, Any]] = []
    hedge_actions: list[str] = []
    if primary_id:
        for evaluation in evaluations:
            if evaluation.get("strategy_id") == primary_id:
                hardening_actions.extend(evaluation.get("recommended_hardening_actions", []))
                triggers_to_watch.extend(evaluation.get("trigger_points", []))
    hardening_actions = list(dict.fromkeys(hardening_actions))[:5]
    triggers_to_watch = triggers_to_watch[:5]
    for entry in overall_rankings[1:3]:
        hedge_actions.append(f"Keep option on {entry.get('strategy_name')} for downside coverage.")

    rationale = ""
    if primary:
        rationale = (
            f"Best robustness index {primary.get('robustness_index')} with "
            f"overall score {primary.get('overall_score')} and min score {primary.get('min_score')}."
        )

    return {
        "primary_recommended_strategy": {
            "strategy_id": primary_id,
            "strategy_name": primary.get("strategy_name"),
            "rationale": rationale,
        },
        "conditional_recommendations": conditional,
        "hardening_actions": hardening_actions,
        "hedge_actions": hedge_actions,
        "triggers_to_watch": triggers_to_watch,
    }


__all__ = [
    "EvaluationInputs",
    "baseline_score",
    "build_wind_tunnel_evaluations",
    "grade_from_score",
    "outcome_band",
]
