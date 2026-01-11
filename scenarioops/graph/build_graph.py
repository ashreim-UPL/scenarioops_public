from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.config import LLMConfig
from scenarioops.graph.nodes import (
    run_auditor_node,
    run_charter_node,
    run_daily_runner_node,
    run_drivers_node,
    run_ewis_node,
    run_logic_node,
    run_narratives_node,
    run_skeletons_node,
    run_strategies_node,
    run_uncertainties_node,
    run_wind_tunnel_node,
)
from scenarioops.graph.state import ScenarioOpsState
from scenarioops.graph.tools.storage import ensure_run_dirs, write_latest_status
from scenarioops.graph.tools.web_retriever import RetrievedContent, retrieve_url
from scenarioops.llm.client import LLMClient, MockLLMClient


@dataclass(frozen=True)
class GraphInputs:
    user_params: Mapping[str, Any]
    sources: Sequence[str]
    signals: Sequence[Mapping[str, Any]]


def _default_sources() -> list[str]:
    return [
        "https://example.com/a",
        "https://example.com/b",
        "https://example.com/c",
    ]


def _mock_retriever(url: str, **_: Any) -> RetrievedContent:
    text = f"Mock source content for {url}."
    excerpt_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return RetrievedContent(
        url=url,
        title="Mock Source",
        date="2026-01-01T00:00:00+00:00",
        text=text,
        excerpt_hash=excerpt_hash,
    )


def _mock_payloads(sources: Sequence[str]) -> dict[str, dict[str, Any]]:
    safe_sources = list(sources)
    while len(safe_sources) < 3:
        safe_sources.append(_default_sources()[len(safe_sources)])
    drivers = [
        {
            "id": "drv-1",
            "name": "Regulatory shift",
            "description": "New compliance requirements.",
            "category": "policy",
            "trend": "tightening",
            "impact": "medium",
            "citations": [{"url": safe_sources[0], "excerpt_hash": "hash-a"}],
        },
        {
            "id": "drv-2",
            "name": "Supplier consolidation",
            "description": "Fewer tier-2 suppliers.",
            "category": "market",
            "trend": "consolidating",
            "impact": "high",
            "citations": [{"url": safe_sources[1], "excerpt_hash": "hash-b"}],
        },
        {
            "id": "drv-3",
            "name": "Logistics volatility",
            "description": "Transit variability rising.",
            "category": "operations",
            "trend": "volatile",
            "impact": "high",
            "citations": [{"url": safe_sources[2], "excerpt_hash": "hash-c"}],
        },
    ]
    uncertainties = {
        "id": "unc-1",
        "title": "Critical uncertainties",
        "uncertainties": [
            {
                "id": "u1",
                "name": "Regulatory tempo",
                "description": "Pace of compliance changes.",
                "extremes": ["slow", "rapid"],
                "driver_ids": ["drv-1", "drv-2"],
                "criticality": 4,
                "volatility": 3,
                "implications": ["Planning cycles shrink"],
            },
            {
                "id": "u2",
                "name": "Logistics stability",
                "description": "Predictability of transit.",
                "extremes": ["stable", "turbulent"],
                "driver_ids": ["drv-2", "drv-3"],
                "criticality": 5,
                "volatility": 4,
                "implications": ["Buffer stock decisions"],
            },
        ],
    }
    logic = {
        "id": "logic-1",
        "title": "Scenario Logic",
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
    skeletons = {
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
                "key_events": [],
            },
            {
                "id": "S2",
                "name": "Policy shock",
                "narrative": "Sudden rule changes with steady logistics.",
                "operating_rules": {
                    "policy": "Rapid compliance shifts",
                    "market": "Steady demand",
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
                    "market": "Steady demand",
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
    narrative_md = (
        "## Scenario: S1\nSteady demand with incremental policy shifts.\n\n"
        "## Scenario: S2\nRapid compliance changes with steady logistics.\n\n"
        "## Scenario: S3\nVolatile transport with slow policy movement.\n\n"
        "## Scenario: S4\nRapid policy changes and volatile transport.\n"
    )
    indicators = []
    for scenario_id in ["S1", "S2", "S3", "S4"]:
        for idx in range(5):
            indicators.append(
                {
                    "id": f"{scenario_id}-ewi-{idx}",
                    "name": f"EWI {idx}",
                    "description": "Track metric movement.",
                    "signal": "Rising",
                    "metric": "days",
                    "linked_scenarios": [scenario_id],
                }
            )
    ewis = {"id": "ewi-1", "title": "EWIs", "indicators": indicators}
    strategies = {
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
    wind_tunnel = {
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
    daily_runner = "# Daily Brief\n\nSignals monitored without anomalies.\n"
    auditor = "- No remediation needed."
    return {
        "charter": {"json": _mock_charter_payload()},
        "drivers": {"json": {"drivers": drivers}},
        "uncertainties": {"json": uncertainties},
        "logic": {"json": logic},
        "skeletons": {"json": skeletons},
        "narratives": {"markdown": narrative_md},
        "ewis": {"json": ewis},
        "strategies": {"json": strategies},
        "wind_tunnel": {"json": wind_tunnel},
        "daily_runner": {"markdown": daily_runner},
        "auditor": {"markdown": auditor},
    }


def _mock_charter_payload() -> dict[str, Any]:
    return {
        "id": "charter-001",
        "title": "ScenarioOps Charter",
        "purpose": "Assess operational resilience.",
        "scope": "Supply chain",
        "time_horizon": "12 months",
        "stakeholders": ["Operations", "Finance"],
    }


def _client_for(
    node_name: str,
    *,
    default_client: LLMClient | None = None,
    mock_payloads: Mapping[str, Mapping[str, Any]] | None = None,
) -> LLMClient | None:
    if not mock_payloads:
        return default_client
    payload = mock_payloads.get(node_name, {})
    json_payload = payload.get("json")
    markdown_payload = payload.get("markdown")
    return MockLLMClient(json_payload=json_payload, markdown_payload=markdown_payload)


def default_sources() -> list[str]:
    return _default_sources()


def mock_payloads_for_sources(sources: Sequence[str]) -> dict[str, dict[str, Any]]:
    return _mock_payloads(sources)


def client_for_node(
    node_name: str,
    *,
    default_client: LLMClient | None = None,
    mock_payloads: Mapping[str, Mapping[str, Any]] | None = None,
) -> LLMClient | None:
    return _client_for(
        node_name, default_client=default_client, mock_payloads=mock_payloads
    )


def mock_retriever(url: str, **kwargs: Any) -> RetrievedContent:
    return _mock_retriever(url, **kwargs)


def _signals_from_ewi(indicators: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"indicator_id": indicator.get("id"), "score": 0.6}
        for indicator in indicators
        if indicator.get("id")
    ]


def run_graph(
    inputs: GraphInputs,
    *,
    run_id: str | None = None,
    base_dir: Path | None = None,
    state: ScenarioOpsState | None = None,
    llm_client: LLMClient | None = None,
    retriever=retrieve_url,
    config: LLMConfig | None = None,
    mock_mode: bool = False,
    generate_strategies: bool = True,
    report_date: str | None = None,
) -> ScenarioOpsState:
    if run_id is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    if state is None:
        state = ScenarioOpsState()

    ensure_run_dirs(run_id, base_dir=base_dir)

    try:
        sources = list(inputs.sources) if inputs.sources else []
        if not sources:
            if mock_mode:
                sources = _default_sources()
            else:
                raise ValueError("Sources are required for drivers node.")
        if mock_mode and len(sources) < 3:
            defaults = _default_sources()
            while len(sources) < 3:
                sources.append(defaults[len(sources)])

        mock_payloads = _mock_payloads(sources) if mock_mode else None
        if mock_mode:
            retriever = _mock_retriever
            if not inputs.user_params:
                inputs = GraphInputs(
                    user_params={"goal": "mock run"},
                    sources=sources,
                    signals=inputs.signals,
                )

        state = run_charter_node(
            inputs.user_params,
            run_id=run_id,
            state=state,
            llm_client=_client_for(
                "charter", default_client=llm_client, mock_payloads=mock_payloads
            ),
            base_dir=base_dir,
            config=config,
        )
        state = run_drivers_node(
            sources,
            run_id=run_id,
            state=state,
            llm_client=_client_for(
                "drivers", default_client=llm_client, mock_payloads=mock_payloads
            ),
            retriever=retriever,
            base_dir=base_dir,
            config=config,
        )
        state = run_uncertainties_node(
            run_id=run_id,
            state=state,
            llm_client=_client_for(
                "uncertainties", default_client=llm_client, mock_payloads=mock_payloads
            ),
            base_dir=base_dir,
            config=config,
        )
        state = run_logic_node(
            run_id=run_id,
            state=state,
            llm_client=_client_for(
                "logic", default_client=llm_client, mock_payloads=mock_payloads
            ),
            base_dir=base_dir,
            config=config,
        )
        state = run_skeletons_node(
            run_id=run_id,
            state=state,
            llm_client=_client_for(
                "skeletons", default_client=llm_client, mock_payloads=mock_payloads
            ),
            base_dir=base_dir,
            config=config,
        )
        state = run_narratives_node(
            run_id=run_id,
            state=state,
            llm_client=_client_for(
                "narratives", default_client=llm_client, mock_payloads=mock_payloads
            ),
            base_dir=base_dir,
            config=config,
        )
        state = run_ewis_node(
            run_id=run_id,
            state=state,
            llm_client=_client_for(
                "ewis", default_client=llm_client, mock_payloads=mock_payloads
            ),
            base_dir=base_dir,
            config=config,
        )

        if generate_strategies and state.strategies is None:
            state = run_strategies_node(
                run_id=run_id,
                state=state,
                llm_client=_client_for(
                    "strategies", default_client=llm_client, mock_payloads=mock_payloads
                ),
                base_dir=base_dir,
                config=config,
            )

        if state.strategies is not None:
            state = run_wind_tunnel_node(
                run_id=run_id,
                state=state,
                llm_client=_client_for(
                    "wind_tunnel", default_client=llm_client, mock_payloads=mock_payloads
                ),
                base_dir=base_dir,
                config=config,
            )
            signals = list(inputs.signals) if inputs.signals else []
            if not signals and mock_mode and state.ewi is not None:
                signals = _signals_from_ewi(
                    [indicator.__dict__ for indicator in state.ewi.indicators]
                )
            state = run_daily_runner_node(
                signals,
                run_id=run_id,
                state=state,
                llm_client=_client_for(
                    "daily_runner", default_client=llm_client, mock_payloads=mock_payloads
                ),
                base_dir=base_dir,
                config=config,
                report_date=report_date,
            )

        state = run_auditor_node(
            run_id=run_id,
            state=state,
            base_dir=base_dir,
            llm_client=_client_for(
                "auditor", default_client=llm_client, mock_payloads=mock_payloads
            ),
            config=config,
        )
    except Exception as exc:
        write_latest_status(
            run_id=run_id,
            status="FAIL",
            error_summary=str(exc),
            base_dir=base_dir,
        )
        raise

    write_latest_status(run_id=run_id, status="OK", base_dir=base_dir)
    return state
