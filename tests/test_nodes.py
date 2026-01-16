import json
import pytest

from scenarioops.__main__ import main as cli_main
from scenarioops.graph.nodes.auditor import run_auditor_node
from scenarioops.graph.nodes.daily_runner import run_daily_runner_node
from scenarioops.graph.nodes.drivers import run_drivers_node
from scenarioops.graph.nodes.ewis import run_ewis_node
from scenarioops.graph.nodes.logic import run_logic_node
from scenarioops.graph.nodes.narratives import (
    extract_numeric_claims_without_citations,
    run_narratives_node,
)
from scenarioops.graph.nodes.skeletons import run_skeletons_node
from scenarioops.graph.nodes.strategies import run_strategies_node
from scenarioops.graph.nodes.uncertainties import run_uncertainties_node
from scenarioops.graph.nodes.wind_tunnel import run_wind_tunnel_node
from scenarioops.graph.state import (
    DriverEntry,
    Drivers,
    Logic,
    ScenarioAxis,
    ScenarioLogic,
    ScenarioOpsState,
    ScenarioSkeleton,
    Skeleton,
    Strategies,
    Strategy,
    UncertaintyEntry,
    Uncertainties,
)
from scenarioops.graph.tools.storage import write_artifact
from scenarioops.llm.client import MockLLMClient


def _make_charter_payload() -> dict:
    return {
        "id": "charter-001",
        "title": "Resilience Charter",
        "purpose": "Assess operational resilience.",
        "decision_context": "Budget allocation for resilience.",
        "scope": "Supply chain",
        "time_horizon": "12 months",
        "stakeholders": ["Operations", "Finance"],
        "constraints": ["No headcount increase"],
        "assumptions": ["Stable demand"],
        "success_criteria": ["Decision-ready scenarios"],
    }


def _make_drivers_payload(sources: list[str]) -> dict:
    return {
        "drivers": [
            {
                "id": "drv-1",
                "name": "Regulatory shift",
                "description": "New reporting requirements.",
                "category": "policy",
                "trend": "tightening",
                "impact": "medium",
                "citations": [
                    {"url": sources[0], "excerpt_hash": "hash-a"},
                ],
            },
            {
                "id": "drv-2",
                "name": "Supplier consolidation",
                "description": "Fewer tier-2 suppliers.",
                "category": "market",
                "trend": "consolidating",
                "impact": "high",
                "citations": [
                    {"url": sources[1], "excerpt_hash": "hash-b"},
                ],
            },
            {
                "id": "drv-3",
                "name": "Logistics volatility",
                "description": "Transit variability rising.",
                "category": "operations",
                "trend": "volatile",
                "impact": "high",
                "citations": [
                    {"url": sources[2], "excerpt_hash": "hash-c"},
                ],
            },
        ]
    }


def _make_evidence_units_payload(sources: list[str]) -> dict:
    units = []
    for idx, url in enumerate(sources, start=1):
        units.append(
            {
                "id": f"ev-{idx}",
                "title": "Example",
                "url": url,
                "publisher": "example.com",
                "retrieved_at": "2026-01-01T00:00:00Z",
                "excerpt": f"Source content {url}",
            }
        )
    return {"evidence_units": units}


def _make_uncertainties_payload() -> dict:
    return {
        "id": "unc-1",
        "title": "Critical uncertainties",
        "uncertainties": [
            {
                "id": "u1",
                "name": "Regulatory tempo",
                "description": "Pace of compliance changes",
                "extremes": ["slow", "rapid"],
                "driver_ids": ["drv-1", "drv-2"],
                "criticality": 4,
                "volatility": 3,
                "implications": ["Planning cycles shrink"],
            },
            {
                "id": "u2",
                "name": "Logistics stability",
                "description": "Predictability of transit",
                "extremes": ["stable", "turbulent"],
                "driver_ids": ["drv-2", "drv-3"],
                "criticality": 5,
                "volatility": 4,
                "implications": ["Buffer stock decisions"],
            },
        ],
    }


def _make_logic_payload() -> dict:
    return {
        "id": "logic-1",
        "title": "2x2 logic",
        "axes": [
            {
                "uncertainty_id": "u1",
                "low": "Slow compliance",
                "high": "Rapid compliance",
            },
            {
                "uncertainty_id": "u2",
                "low": "Stable logistics",
                "high": "Turbulent logistics",
            },
        ],
        "scenarios": [
            {"id": "S1", "name": "Steady climb", "logic": "Slow + Stable"},
            {"id": "S2", "name": "Policy shock", "logic": "Rapid + Stable"},
            {"id": "S3", "name": "Turbulent road", "logic": "Slow + Turbulent"},
            {"id": "S4", "name": "Full volatility", "logic": "Rapid + Turbulent"},
        ],
    }


def _make_skeleton_payload() -> dict:
    return {
        "id": "sk-1",
        "title": "Skeletons",
        "scenarios": [
            {
                "id": "S1",
                "name": "Steady climb",
                "narrative": "Calm reforms and reliable transport.",
                "operating_rules": {
                    "policy": "Incremental updates",
                    "market": "Stable demand",
                    "operations": "Predictable lanes",
                },
                "key_events": [
                    {"date": "2026-01-01", "event": "Baseline review"},
                ],
            },
            {
                "id": "S2",
                "name": "Policy shock",
                "narrative": "Sudden rule changes with steady logistics.",
                "operating_rules": {
                    "policy": "Rapid compliance shifts",
                    "market": "Demand steady",
                    "operations": "Stable lanes",
                },
                "key_events": [],
            },
            {
                "id": "S3",
                "name": "Turbulent road",
                "narrative": "Slow policy but volatile transport.",
                "operating_rules": {
                    "policy": "Slow compliance",
                    "market": "Demand steady",
                    "operations": "Volatile lanes",
                },
                "key_events": [],
            },
            {
                "id": "S4",
                "name": "Full volatility",
                "narrative": "Rapid policy shifts and volatile transport.",
                "operating_rules": {
                    "policy": "Rapid changes",
                    "market": "Demand stress",
                    "operations": "Volatile lanes",
                },
                "key_events": [],
            },
        ],
    }


def _make_ewi_payload() -> dict:
    indicators = []
    for scenario_id in ["S1", "S2", "S3", "S4"]:
        for idx in range(5):
            indicators.append(
                {
                    "id": f"{scenario_id}-ewi-{idx}",
                    "name": f"EWI {idx}",
                    "description": "Track metric",
                    "signal": "Rising",
                    "metric": "days",
                    "linked_scenarios": [scenario_id],
                }
            )
    return {"id": "ewi-1", "title": "EWIs", "indicators": indicators}


def _make_strategies_payload() -> dict:
    return {
        "id": "strat-1",
        "title": "Strategies",
        "strategies": [
            {
                "id": "st-1",
                "name": "Buffer inventory",
                "objective": "Reduce disruptions",
                "actions": ["Increase safety stock"],
                "kpis": ["Fill rate"],
            },
            {
                "id": "st-2",
                "name": "Dual sourcing",
                "objective": "Reduce single points of failure",
                "actions": ["Qualify secondary supplier"],
                "kpis": ["Qualified suppliers"],
            },
        ],
    }


def _make_wind_tunnel_payload() -> dict:
    return {
        "id": "wt-1",
        "title": "Wind Tunnel",
        "tests": [
            {
                "id": "wt-1",
                "strategy_id": "st-1",
                "scenario_id": "S1",
                "outcome": "Resilient",
                "failure_modes": ["Demand spike"],
                "adaptations": ["Scale buffers"],
                "feasibility_score": 0.8,
                "rubric_inputs": {
                    "relevance": 0.9,
                    "credibility": 0.9,
                    "recency": 0.8,
                    "specificity": 0.8,
                },
            }
        ],
    }


def test_charter_cli(tmp_path, monkeypatch) -> None:
    run_id = "run-cli"
    base_dir = tmp_path / "runs"
    charter_payload = _make_charter_payload()
    payload_file = tmp_path / "mock.json"
    payload_file.write_text(json.dumps(charter_payload), encoding="utf-8")
    params = json.dumps({"goal": "test"})

    monkeypatch.setattr(
        "sys.argv",
        [
            "scenarioops",
            "charter",
            "--params",
            params,
            "--run-id",
            run_id,
            "--base-dir",
            str(base_dir),
            "--mock-payload",
            str(payload_file),
        ],
    )
    cli_main()

    artifact_path = base_dir / run_id / "artifacts" / "scenario_charter.json"
    assert artifact_path.exists()


def test_drivers_node_citations(tmp_path) -> None:
    run_id = "run-drivers"
    base_dir = tmp_path / "runs"
    sources = ["https://example.com/a", "https://example.com/b", "https://example.com/c"]

    payload = _make_drivers_payload(sources)
    state = ScenarioOpsState(evidence_units=_make_evidence_units_payload(sources))
    run_drivers_node(
        run_id=run_id,
        state=state,
        llm_client=MockLLMClient(json_payload=payload),
        base_dir=base_dir,
    )

    artifact_path = base_dir / run_id / "artifacts" / "drivers.jsonl"
    assert artifact_path.exists()
    for entry in payload["drivers"]:
        assert entry["citations"]


def test_uncertainties_node_driver_links(tmp_path) -> None:
    run_id = "run-unc"
    base_dir = tmp_path / "runs"
    drivers = Drivers(
        id="drivers-1",
        title="Drivers",
        drivers=[
            DriverEntry(
                id="drv-1",
                name="A",
                description="A",
                category="cat",
                trend="up",
                impact="high",
                citations=[{"url": "https://example.com/a", "excerpt_hash": "hash"}],
            ),
            DriverEntry(
                id="drv-2",
                name="B",
                description="B",
                category="cat",
                trend="up",
                impact="high",
                citations=[{"url": "https://example.com/b", "excerpt_hash": "hash"}],
            ),
            DriverEntry(
                id="drv-3",
                name="C",
                description="C",
                category="cat",
                trend="up",
                impact="high",
                citations=[{"url": "https://example.com/c", "excerpt_hash": "hash"}],
            ),
        ],
    )
    state = ScenarioOpsState(drivers=drivers)
    run_uncertainties_node(
        run_id=run_id,
        state=state,
        llm_client=MockLLMClient(json_payload=_make_uncertainties_payload()),
        base_dir=base_dir,
    )
    assert state.uncertainties is not None
    for entry in state.uncertainties.uncertainties:
        assert len(entry.driver_ids) >= 2


def test_logic_node_axis_similarity(tmp_path) -> None:
    run_id = "run-logic"
    base_dir = tmp_path / "runs"
    uncertainties = Uncertainties(
        id="unc-1",
        title="Uncertainties",
        uncertainties=[
            UncertaintyEntry(
                id="u1",
                name="A",
                description="A",
                extremes=["low", "high"],
                driver_ids=["drv-1", "drv-2"],
            ),
            UncertaintyEntry(
                id="u2",
                name="B",
                description="B",
                extremes=["low", "high"],
                driver_ids=["drv-2", "drv-3"],
            ),
        ],
    )
    state = ScenarioOpsState(uncertainties=uncertainties)
    run_logic_node(
        run_id=run_id,
        state=state,
        llm_client=MockLLMClient(json_payload=_make_logic_payload()),
        base_dir=base_dir,
    )
    assert state.logic is not None


def test_skeletons_node_operating_rules(tmp_path) -> None:
    run_id = "run-skel"
    base_dir = tmp_path / "runs"
    logic = Logic(
        id="logic-1",
        title="Logic",
        axes=[
            ScenarioAxis(uncertainty_id="u1", low="low", high="high"),
            ScenarioAxis(uncertainty_id="u2", low="low", high="high"),
        ],
        scenarios=[
            ScenarioLogic(id="S1", name="S1", logic="L1"),
            ScenarioLogic(id="S2", name="S2", logic="L2"),
            ScenarioLogic(id="S3", name="S3", logic="L3"),
            ScenarioLogic(id="S4", name="S4", logic="L4"),
        ],
    )
    state = ScenarioOpsState(logic=logic)
    run_skeletons_node(
        run_id=run_id,
        state=state,
        llm_client=MockLLMClient(json_payload=_make_skeleton_payload()),
        base_dir=base_dir,
    )
    assert state.skeleton is not None
    for scenario in state.skeleton.scenarios:
        assert set(scenario.operating_rules.keys()) == {"policy", "market", "operations"}


def test_narratives_node_and_claim_extractor(tmp_path) -> None:
    run_id = "run-narr"
    base_dir = tmp_path / "runs"
    skeleton = Skeleton(
        id="sk-1",
        title="Skeletons",
        scenarios=[
            ScenarioSkeleton(id="S1", name="S1", narrative="n1"),
            ScenarioSkeleton(id="S2", name="S2", narrative="n2"),
            ScenarioSkeleton(id="S3", name="S3", narrative="n3"),
            ScenarioSkeleton(id="S4", name="S4", narrative="n4"),
        ],
    )
    state = ScenarioOpsState(skeleton=skeleton)
    markdown = (
        "## Scenario: S1\nNarrative text.\n"
        "## Scenario: S2\nNarrative text.\n"
        "## Scenario: S3\nNarrative text.\n"
        "## Scenario: S4\nNarrative text.\n"
    )
    run_narratives_node(
        run_id=run_id,
        state=state,
        llm_client=MockLLMClient(markdown_payload=markdown),
        base_dir=base_dir,
    )
    artifact = base_dir / run_id / "artifacts" / "narrative_S1.md"
    assert artifact.exists()

    flagged = extract_numeric_claims_without_citations("Revenue grew 12% year over year.")
    assert flagged


def test_ewi_node_minimum_per_scenario(tmp_path) -> None:
    run_id = "run-ewi"
    base_dir = tmp_path / "runs"
    logic = Logic(
        id="logic-1",
        title="Logic",
        axes=[],
        scenarios=[
            ScenarioLogic(id="S1", name="S1", logic="L1"),
            ScenarioLogic(id="S2", name="S2", logic="L2"),
            ScenarioLogic(id="S3", name="S3", logic="L3"),
            ScenarioLogic(id="S4", name="S4", logic="L4"),
        ],
    )
    state = ScenarioOpsState(logic=logic)
    run_ewis_node(
        run_id=run_id,
        state=state,
        llm_client=MockLLMClient(json_payload=_make_ewi_payload()),
        base_dir=base_dir,
    )
    assert state.ewi is not None


def test_strategies_node_kpis(tmp_path) -> None:
    run_id = "run-strat"
    base_dir = tmp_path / "runs"
    logic = Logic(
        id="logic-1",
        title="Logic",
        axes=[],
        scenarios=[ScenarioLogic(id="S1", name="S1", logic="L1")],
    )
    state = ScenarioOpsState(logic=logic)
    run_strategies_node(
        run_id=run_id,
        state=state,
        llm_client=MockLLMClient(json_payload=_make_strategies_payload()),
        base_dir=base_dir,
    )
    assert state.strategies is not None
    for strategy in state.strategies.strategies:
        assert strategy.kpis


def test_wind_tunnel_keep_threshold(tmp_path) -> None:
    run_id = "run-wt"
    base_dir = tmp_path / "runs"
    strategies = Strategies(
        id="strat-1",
        title="Strategies",
        strategies=[
            Strategy(
                id="st-1",
                name="Buffer",
                objective="Resilience",
                kpis=["Fill rate"],
            )
        ],
    )
    logic = Logic(
        id="logic-1",
        title="Logic",
        axes=[],
        scenarios=[ScenarioLogic(id="S1", name="S1", logic="L1")],
    )
    state = ScenarioOpsState(strategies=strategies, logic=logic)
    run_wind_tunnel_node(
        run_id=run_id,
        state=state,
        llm_client=MockLLMClient(json_payload=_make_wind_tunnel_payload()),
        base_dir=base_dir,
    )
    assert state.wind_tunnel is not None
    assert state.wind_tunnel.tests[0].action == "KEEP"


def test_daily_runner_missing_data(tmp_path) -> None:
    run_id = "run-daily"
    base_dir = tmp_path / "runs"
    state = ScenarioOpsState()
    run_daily_runner_node(
        [],
        run_id=run_id,
        state=state,
        llm_client=MockLLMClient(markdown_payload="# Daily Brief"),
        base_dir=base_dir,
        report_date="2026-01-01",
    )
    artifact_path = base_dir / run_id / "artifacts" / "daily_brief.json"
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert "data unavailable" in payload["markdown"].lower()
    assert "data unavailable" in payload["notes"].lower()
    assert payload["implications"] == []
    markdown_path = base_dir / run_id / "artifacts" / "daily_brief.md"
    assert markdown_path.exists()


def test_auditor_fails_on_missing_citations(tmp_path) -> None:
    run_id = "run-audit"
    base_dir = tmp_path / "runs"
    artifact_payload = [
        {
            "id": "drv-1",
            "name": "Driver",
            "description": "Desc",
            "category": "cat",
            "trend": "up",
            "impact": "high",
            "citations": [],
        }
    ]
    write_artifact(
        run_id=run_id,
        artifact_name="drivers",
        payload=artifact_payload,
        ext="jsonl",
        base_dir=base_dir,
    )

    with pytest.raises(RuntimeError):
        run_auditor_node(run_id=run_id, state=ScenarioOpsState(), base_dir=base_dir)

    report_path = base_dir / run_id / "artifacts" / "audit_report.json"
    assert report_path.exists()
