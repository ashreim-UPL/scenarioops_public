from __future__ import annotations

from pathlib import Path
from typing import Any

from scenarioops.graph.state import ScenarioOpsState
from scenarioops.graph.tools.schema_validate import validate_artifact
from scenarioops.graph.tools.storage import write_artifact


FACT_CONFIDENCE = 0.8
ASSUMPTION_CONFIDENCE = 0.5
INTERPRETATION_CONFIDENCE = 0.4


def _sorted(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(items, key=lambda item: str(item.get("id", "")))


def _facts(certainty: dict[str, Any]) -> list[dict[str, Any]]:
    facts = []
    for entry in certainty.get("predetermined_elements", []) or []:
        facts.append(
            {
                "id": entry.get("id"),
                "statement": entry.get("description") or entry.get("name"),
                "evidence_ids": entry.get("evidence_ids", []),
                "confidence": FACT_CONFIDENCE,
                "confidence_basis": "predetermined_element",
            }
        )
    return _sorted(facts)


def _unknowns(certainty: dict[str, Any]) -> list[dict[str, Any]]:
    unknowns = []
    for entry in certainty.get("uncertainties", []) or []:
        uncertainty = float(entry.get("uncertainty", 0.0))
        confidence = max(0.0, min(1.0, 1.0 - uncertainty))
        unknowns.append(
            {
                "id": entry.get("id"),
                "statement": entry.get("description") or entry.get("name"),
                "evidence_ids": entry.get("evidence_ids", []),
                "impact": entry.get("impact"),
                "uncertainty": uncertainty,
                "confidence": confidence,
                "confidence_basis": "1_minus_uncertainty",
            }
        )
    return _sorted(unknowns)


def _assumptions(belief_sets: dict[str, Any]) -> list[dict[str, Any]]:
    assumptions = []
    for belief_set in belief_sets.get("belief_sets", []) or []:
        uncertainty_id = belief_set.get("uncertainty_id")
        for stance_key in ("dominant_belief", "counter_belief"):
            belief = belief_set.get(stance_key, {}) or {}
            assumptions.append(
                {
                    "id": belief.get("id"),
                    "statement": belief.get("statement"),
                    "assumptions": belief.get("assumptions", []),
                    "evidence_ids": belief.get("evidence_ids", []),
                    "uncertainty_id": uncertainty_id,
                    "stance": "dominant" if stance_key == "dominant_belief" else "counter",
                    "confidence": ASSUMPTION_CONFIDENCE,
                    "confidence_basis": "belief_statement",
                }
            )
    return _sorted(assumptions)


def _interpretations(effects: dict[str, Any]) -> list[dict[str, Any]]:
    interpretations = []
    for entry in effects.get("effects", []) or []:
        interpretations.append(
            {
                "id": entry.get("id"),
                "statement": entry.get("description"),
                "belief_id": entry.get("belief_id"),
                "domains": entry.get("domains", []),
                "order": entry.get("order"),
                "confidence": INTERPRETATION_CONFIDENCE,
                "confidence_basis": "effect_chain",
            }
        )
    return _sorted(interpretations)


def run_epistemic_summary_node(
    *,
    run_id: str,
    state: ScenarioOpsState,
    base_dir: Path | None = None,
) -> ScenarioOpsState:
    if not isinstance(state.certainty_uncertainty, dict):
        raise ValueError("Certainty/uncertainty output required for epistemic summary.")
    if not isinstance(state.belief_sets, dict):
        raise ValueError("Belief sets required for epistemic summary.")
    if not isinstance(state.effects, dict):
        raise ValueError("Effects required for epistemic summary.")

    summary = {
        "facts": _facts(state.certainty_uncertainty),
        "assumptions": _assumptions(state.belief_sets),
        "interpretations": _interpretations(state.effects),
        "unknowns": _unknowns(state.certainty_uncertainty),
        "confidence_rules": [
            "facts: 0.8 (predetermined_element)",
            "assumptions: 0.5 (belief_statement)",
            "interpretations: 0.4 (effect_chain)",
            "unknowns: 1 - uncertainty",
        ],
    }

    validate_artifact("epistemic_summary", summary)

    write_artifact(
        run_id=run_id,
        artifact_name="epistemic_summary",
        payload=summary,
        ext="json",
        input_values={"belief_set_count": len(state.belief_sets.get("belief_sets", []))},
        prompt_values={"prompt": "epistemic_summary"},
        tool_versions={"epistemic_summary_node": "0.1.0"},
        base_dir=base_dir,
    )

    state.epistemic_summary = summary
    return state
