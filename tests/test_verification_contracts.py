import pytest

from scenarioops.graph.nodes.coverage import run_coverage_node
from scenarioops.graph.nodes.epistemic_summary import run_epistemic_summary_node
from scenarioops.graph.nodes.retrieval import run_retrieval_node
from scenarioops.graph.nodes.scan import run_scan_node
from scenarioops.graph.state import ScenarioOpsState
from scenarioops.graph.tools.normalization import stable_id
from scenarioops.graph.tools.schema_validate import SchemaValidationError, validate_artifact
from scenarioops.llm.client import MockLLMClient


def _charter_payload() -> dict:
    return {
        "id": "charter-001",
        "title": "ScenarioOps Pilot",
        "purpose": "Assess operational resilience.",
        "decision_context": "Capital allocation.",
        "scope": "Global supply chain",
        "time_horizon": "12 months",
        "stakeholders": ["Operations"],
        "constraints": ["No headcount increase"],
        "assumptions": ["Stable demand"],
        "success_criteria": ["Decision-ready scenarios"],
    }


def test_charter_rejects_extra_keys() -> None:
    payload = dict(_charter_payload())
    payload["extra_key"] = "not allowed"
    with pytest.raises(SchemaValidationError):
        validate_artifact("charter", payload)


def test_stable_id_is_deterministic() -> None:
    first = stable_id("driver", "Regulatory shift", "policy")
    second = stable_id("driver", "Regulatory shift", "policy")
    third = stable_id("driver", "Supplier consolidation", "market")
    assert first == second
    assert first != third


def test_retrieval_fails_closed(tmp_path) -> None:
    def failing_retriever(url: str, **_: object) -> object:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="retrieval_failed"):
        run_retrieval_node(
            ["https://example.com/a"],
            run_id="retrieval-fail",
            state=ScenarioOpsState(),
            retriever=failing_retriever,
            base_dir=tmp_path / "runs",
        )


def test_scan_fails_on_unknown_citation(tmp_path) -> None:
    evidence_units = {
        "evidence_units": [
            {
                "id": "ev-1",
                "title": "Example",
                "url": "https://example.com/a",
                "publisher": "example.com",
                "retrieved_at": "2026-01-01T00:00:00Z",
                "excerpt": "Source text",
            }
        ]
    }
    payload = {
        "forces": [
            {
                "id": "force-1",
                "name": "Political signal",
                "domain": "political",
                "lenses": ["macro"],
                "description": "Driver description.",
                "why_it_matters": "It matters.",
                "citations": [{"url": "https://bad.example.com"}],
            }
        ]
    }
    state = ScenarioOpsState(evidence_units=evidence_units)
    with pytest.raises(ValueError, match="citation url not in evidence units"):
        run_scan_node(
            run_id="scan-fail",
            state=state,
            llm_client=MockLLMClient(json_payload=payload),
            base_dir=tmp_path / "runs",
        )


def test_coverage_requires_all_lenses(tmp_path) -> None:
    forces = [
        {
            "id": "force-1",
            "name": "Political signal",
            "domain": "political",
            "lenses": ["geopolitics"],
            "description": "Driver",
            "why_it_matters": "Reason",
            "citations": [{"url": "https://example.com/a"}],
        },
        {
            "id": "force-2",
            "name": "Economic signal",
            "domain": "economic",
            "lenses": ["macro"],
            "description": "Driver",
            "why_it_matters": "Reason",
            "citations": [{"url": "https://example.com/a"}],
        },
        {
            "id": "force-3",
            "name": "Social signal",
            "domain": "social",
            "lenses": ["culture"],
            "description": "Driver",
            "why_it_matters": "Reason",
            "citations": [{"url": "https://example.com/a"}],
        },
        {
            "id": "force-4",
            "name": "Technological signal",
            "domain": "technological",
            "lenses": ["macro"],
            "description": "Driver",
            "why_it_matters": "Reason",
            "citations": [{"url": "https://example.com/a"}],
        },
        {
            "id": "force-5",
            "name": "Environmental signal",
            "domain": "environmental",
            "lenses": ["ethics"],
            "description": "Driver",
            "why_it_matters": "Reason",
            "citations": [{"url": "https://example.com/a"}],
        },
        {
            "id": "force-6",
            "name": "Legal signal",
            "domain": "legal",
            "lenses": ["law"],
            "description": "Driver",
            "why_it_matters": "Reason",
            "citations": [{"url": "https://example.com/a"}],
        },
    ]
    state = ScenarioOpsState(driving_forces={"forces": forces})
    run_coverage_node(run_id="coverage-ok", state=state, base_dir=tmp_path / "runs")


def test_epistemic_summary_labels(tmp_path) -> None:
    certainty = {
        "predetermined_elements": [
            {
                "id": "pre-1",
                "name": "Baseline policy",
                "description": "Existing regulations apply.",
                "evidence_ids": ["ev-1"],
                "reasoning": "Law already enacted.",
            }
        ],
        "uncertainties": [
            {
                "id": "unc-1",
                "name": "Policy tempo",
                "description": "Pace of regulatory change.",
                "evidence_ids": ["ev-1"],
                "reasoning": "Signals diverge.",
                "impact": 0.8,
                "uncertainty": 0.6,
            }
        ],
    }
    belief_sets = {
        "belief_sets": [
            {
                "uncertainty_id": "unc-1",
                "dominant_belief": {
                    "id": "bel-1",
                    "statement": "Policy accelerates.",
                    "assumptions": ["Regulators prioritize rapid updates."],
                    "evidence_ids": ["ev-1"],
                },
                "counter_belief": {
                    "id": "bel-2",
                    "statement": "Policy slows.",
                    "assumptions": ["Regulators prefer stability."],
                    "evidence_ids": ["ev-1"],
                },
            }
        ]
    }
    effects = {
        "effects": [
            {
                "id": "eff-1",
                "belief_id": "bel-1",
                "order": 1,
                "description": "Compliance costs rise.",
                "domains": ["cost"],
            }
        ]
    }
    state = ScenarioOpsState(
        certainty_uncertainty=certainty, belief_sets=belief_sets, effects=effects
    )
    run_epistemic_summary_node(
        run_id="epistemic-ok", state=state, base_dir=tmp_path / "runs"
    )
    assert state.epistemic_summary is not None
    assert state.epistemic_summary["facts"]
    assert state.epistemic_summary["assumptions"]
    assert state.epistemic_summary["interpretations"]
    assert state.epistemic_summary["unknowns"]
