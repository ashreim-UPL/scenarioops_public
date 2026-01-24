from __future__ import annotations

import numpy as np
import pytest

from scenarioops.app.config import ScenarioOpsSettings
from scenarioops.graph.nodes.force_builder import run_force_builder_node
from scenarioops.graph.nodes.retrieval_real import run_retrieval_real_node
from scenarioops.graph.nodes.scenario_synthesis import run_scenario_synthesis_node
from scenarioops.graph.state import ScenarioOpsState
from scenarioops.graph.tools.clustering import cluster_vectors
from scenarioops.graph.tools.embeddings import embed_texts
from scenarioops.llm.client import MockLLMClient


def test_retrieval_real_requires_sources_without_simulation(tmp_path):
    settings = ScenarioOpsSettings(allow_web=False, simulate_evidence=False)
    state = ScenarioOpsState()
    llm = MockLLMClient(
        json_payload={"primary": [], "secondary": [], "tertiary": [], "counter": []}
    )
    with pytest.raises(RuntimeError):
        run_retrieval_real_node(
            [],
            run_id="test-run",
            state=state,
            user_params={"value": "Acme", "scope": "world", "horizon": 12},
            base_dir=tmp_path,
            llm_client=llm,
            settings=settings,
        )


def test_retrieval_real_simulation_flag(tmp_path):
    settings = ScenarioOpsSettings(allow_web=False, simulate_evidence=True)
    state = ScenarioOpsState()
    llm = MockLLMClient(
        json_payload={"primary": ["test"], "secondary": [], "tertiary": [], "counter": []}
    )
    state = run_retrieval_real_node(
        [],
        run_id="test-run",
        state=state,
        user_params={"value": "Acme", "scope": "world", "horizon": 12},
        base_dir=tmp_path,
        llm_client=llm,
        settings=settings,
        simulate_evidence=True,
    )
    assert state.evidence_units is not None
    assert state.evidence_units.get("simulated") is True


def test_force_traceability_links(tmp_path):
    state = ScenarioOpsState()
    state.evidence_units = {
        "run_id": "test-run",
        "timestamp": "2026-01-01T00:00:00+00:00",
        "company_name": "Acme",
        "geography": "Global",
        "horizon_months": 12,
        "simulated": False,
        "evidence_units": [
            {
                "evidence_unit_id": "ev-1",
                "source_type": "primary",
                "title": "Test",
                "publisher": "Example",
                "date_published": "2026-01-01T00:00:00+00:00",
                "url": "https://example.com",
                "excerpt": "Test excerpt",
                "claims": [],
                "metrics": [],
                "reliability_grade": "A",
                "reliability_reason": "test",
                "geography_tags": ["Global"],
                "domain_tags": [],
                "simulated": False,
            }
        ],
    }
    llm = MockLLMClient()
    state = run_force_builder_node(
        run_id="test-run",
        state=state,
        user_params={"value": "Acme", "scope": "world", "horizon": 12},
        llm_client=llm,
        base_dir=tmp_path,
        min_forces=1,
        min_per_domain=0,
    )
    forces = state.forces.get("forces", []) if state.forces else []
    assert forces
    assert forces[0]["evidence_unit_ids"]


def test_clustering_stability(tmp_path):
    items = [
        ("force-1", "capital cost volatility finance rates"),
        ("force-2", "supply chain disruption logistics risks"),
        ("force-3", "capital cost volatility investment"),
    ]
    embeddings = embed_texts(items, seed=42, cache_root=tmp_path)
    vectors = [embeddings[item[0]]["embedding"] for item in items]
    labels_a, _ = cluster_vectors(
        np.array(vectors), distance_threshold=0.4, min_cluster_size=2
    )
    labels_b, _ = cluster_vectors(
        np.array(vectors), distance_threshold=0.4, min_cluster_size=2
    )
    assert labels_a == labels_b


def test_scenario_constraints(tmp_path):
    state = ScenarioOpsState()
    state.uncertainty_axes = {
        "axes": [
            {"axis_id": "axis-1", "impact_score": 4, "uncertainty_score": 4},
            {"axis_id": "axis-2", "impact_score": 3, "uncertainty_score": 3},
        ],
        "selected_axis_ids": ["axis-1", "axis-2"],
    }
    state.clusters = {"clusters": [{"cluster_id": "cluster-1"}]}
    state.forces = {"forces": [{"force_id": "force-1"}]}
    llm = MockLLMClient()
    state = run_scenario_synthesis_node(
        run_id="test-run",
        state=state,
        user_params={"value": "Acme", "scope": "world", "horizon": 12},
        llm_client=llm,
        base_dir=tmp_path,
    )
    scenarios = state.scenarios.get("scenarios", []) if state.scenarios else []
    assert len(scenarios) == 4
