from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import time
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

from scenarioops.app.config import LLMConfig
from scenarioops.app.config import (
    ScenarioOpsSettings,
    llm_config_from_settings,
    load_settings,
)
from scenarioops.graph.gates.washout_gate import (
    WashoutGateConfig,
    assert_washout_pass,
    washout_deficits,
)
from scenarioops.graph.guards.fixture_guard import validate_or_fail as validate_fixture_or_fail
from scenarioops.graph.nodes import (
    run_auditor_node,
    run_beliefs_node,
    run_charter_node,
    run_classify_node,
    run_coverage_node,
    run_daily_runner_node,
    run_drivers_node,
    run_effects_node,
    run_epistemic_summary_node,
    run_ewis_node,
    run_focal_issue_node,
    run_logic_node,
    run_narratives_node,
    run_retrieval_node,
    run_scenario_profiles_node,
    run_scan_node,
    run_skeletons_node,
    run_strategies_node,
    run_trace_map_node,
    run_uncertainties_node,
    run_washout_node,
    run_wind_tunnel_node,
)
from scenarioops.graph.state import ScenarioOpsState
from scenarioops.graph.tools.provenance import utc_now_iso
from scenarioops.graph.tools.run_manifest import write_artifact_index, write_run_manifest
from scenarioops.graph.tools.storage import (
    default_runs_dir,
    ensure_run_dirs,
    log_node_event,
    register_run_timestamp,
    write_artifact,
    write_latest_status,
    write_run_config,
)
from scenarioops.graph.tools.web_retriever import RetrievedContent, retrieve_url
from scenarioops.graph.types import NodeResult
from scenarioops.llm.client import LLMClient, MockLLMClient
from scenarioops.sources.policy import default_sources_for_policy, policy_for_name


@dataclass(frozen=True)
class GraphInputs:
    user_params: Mapping[str, Any]
    sources: Sequence[str]
    signals: Sequence[Mapping[str, Any]]


def _default_sources() -> list[str]:
    return default_sources_for_policy("fixtures")


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
    domains = [
        "political",
        "economic",
        "social",
        "technological",
        "environmental",
        "legal",
    ]
    lenses_by_domain = {
        "political": ["geopolitics", "macro"],
        "economic": ["macro"],
        "social": ["culture"],
        "technological": ["macro"],
        "environmental": ["ethics"],
        "legal": ["law"],
    }
    forces = []
    force_id = 1
    for domain in domains:
        for idx in range(5):
            forces.append(
                {
                    "id": f"force-{force_id}",
                    "name": f"{domain.title()} signal {idx + 1}",
                    "domain": domain,
                    "lenses": lenses_by_domain.get(domain, ["macro"]),
                    "description": f"{domain.title()} driver {idx + 1} in the scan.",
                    "why_it_matters": "It shapes the operating context.",
                    "citations": [
                        {
                            "url": safe_sources[force_id % len(safe_sources)],
                            "excerpt_hash": f"hash-{force_id}",
                        }
                    ],
                }
            )
            force_id += 1
    driving_forces = {"forces": forces}
    focal_issue = {
        "focal_issue": "How should the organization prioritize resilience investments over the next 5 years?",
        "scope": {
            "geography": "UAE",
            "sectors": ["supply chain"],
            "time_horizon_years": 5,
        },
        "decision_type": "strategic planning",
        "exclusions": ["Product portfolio redesign", "M&A decisions"],
        "success_criteria": "Scenarios enable resilient investment sequencing and risk-aware budgeting.",
    }
    washout_report = {
        "duplicate_ratio": 0.1,
        "duplicate_groups": [],
        "undercovered_domains": [],
        "missing_categories": [],
        "proposed_forces": [],
    }
    evidence_units_list = []
    for idx, url in enumerate(safe_sources, start=1):
        evidence_units_list.append(
            {
                "id": f"ev-{idx}",
                "title": "Mock Source",
                "url": url,
                "publisher": urlparse(url).hostname or "mock",
                "retrieved_at": "2026-01-01T00:00:00+00:00",
                "excerpt": f"Mock source content for {url}.",
            }
        )
    evidence_units = {"evidence_units": evidence_units_list}
    certainty_uncertainty = {
        "predetermined_elements": [
            {
                "id": "pre-1",
                "name": "Baseline policy obligations",
                "description": "Compliance baselines remain in effect.",
                "evidence_ids": ["ev-1"],
                "reasoning": "Regulatory baselines are already enacted.",
            }
        ],
        "uncertainties": [
            {
                "id": "unc-1",
                "name": "Policy tempo",
                "description": "Pace of regulatory changes.",
                "evidence_ids": ["ev-1", "ev-2"],
                "reasoning": "Signals point to potential acceleration or slowdown.",
                "impact": 0.8,
                "uncertainty": 0.7,
            },
            {
                "id": "unc-2",
                "name": "Technology adoption",
                "description": "Speed of platform and automation uptake.",
                "evidence_ids": ["ev-2", "ev-3"],
                "reasoning": "Adoption timing remains contested.",
                "impact": 0.7,
                "uncertainty": 0.6,
            },
        ],
    }
    belief_sets = {
        "belief_sets": [
            {
                "uncertainty_id": "unc-1",
                "dominant_belief": {
                    "id": "bel-1",
                    "statement": "Policy changes accelerate and increase compliance load.",
                    "assumptions": ["Regulators prioritize rapid updates."],
                    "evidence_ids": ["ev-1", "ev-2"],
                },
                "counter_belief": {
                    "id": "bel-2",
                    "statement": "Policy updates slow as institutions focus on stability.",
                    "assumptions": ["Regulators prefer phased rollouts."],
                    "evidence_ids": ["ev-1", "ev-2"],
                },
            },
            {
                "uncertainty_id": "unc-2",
                "dominant_belief": {
                    "id": "bel-3",
                    "statement": "Automation adoption accelerates across core processes.",
                    "assumptions": ["Capital is available for modernization."],
                    "evidence_ids": ["ev-2", "ev-3"],
                },
                "counter_belief": {
                    "id": "bel-4",
                    "statement": "Adoption slows due to cost and talent constraints.",
                    "assumptions": ["Budget pressures delay upgrades."],
                    "evidence_ids": ["ev-2", "ev-3"],
                },
            },
        ]
    }
    effects = {
        "effects": [
            {
                "id": "eff-1",
                "belief_id": "bel-1",
                "order": 1,
                "description": "Compliance costs rise and policy teams expand.",
                "domains": ["policy", "cost"],
            },
            {
                "id": "eff-2",
                "belief_id": "bel-2",
                "order": 1,
                "description": "Planning cycles stabilize with fewer regulatory surprises.",
                "domains": ["policy", "operations"],
            },
            {
                "id": "eff-3",
                "belief_id": "bel-3",
                "order": 2,
                "description": "Automation improves throughput and demand response.",
                "domains": ["technology", "operations", "demand"],
            },
            {
                "id": "eff-4",
                "belief_id": "bel-4",
                "order": 2,
                "description": "Talent gaps slow transformation and limit savings.",
                "domains": ["talent", "cost"],
            },
        ]
    }
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
    wind_tunnel_tests = []
    for idx, scenario_id in enumerate(["S1", "S2", "S3", "S4"], start=1):
        wind_tunnel_tests.append(
            {
                "id": f"wt-{idx}",
                "strategy_id": "st-1" if idx % 2 == 1 else "st-2",
                "scenario_id": scenario_id,
                "outcome": "Resilient" if idx % 2 == 1 else "Stressed",
                "failure_modes": ["Demand spike"] if idx % 2 == 1 else ["Port delays"],
                "adaptations": ["Scale buffers"] if idx % 2 == 1 else ["Shift routing"],
                "feasibility_score": 0.8 if idx % 2 == 1 else 0.7,
                "rubric_inputs": {
                    "relevance": 0.9,
                    "credibility": 0.9,
                    "recency": 0.8,
                    "specificity": 0.8,
                },
            }
        )
    wind_tunnel = {
        "id": "wt-1",
        "title": "Wind Tunnel",
        "tests": wind_tunnel_tests,
    }
    daily_runner = "# Daily Brief\n\nSignals monitored without anomalies.\n"
    auditor = "- No remediation needed."
    return {
        "charter": {"json": _mock_charter_payload()},
        "focal_issue": {"json": focal_issue},
        "scan_pestel": {"json": driving_forces},
        "washout_audit": {"json": washout_report},
        "certainty_uncertainty": {"json": certainty_uncertainty},
        "beliefs": {"json": belief_sets},
        "effects": {"json": effects},
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
        "decision_context": "Resilience investment prioritization.",
        "scope": "Supply chain",
        "time_horizon": "12 months",
        "stakeholders": ["Operations", "Finance"],
        "constraints": ["No headcount increase"],
        "assumptions": ["Stable demand"],
        "success_criteria": ["Decision-ready scenario set"],
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


def apply_node_result(
    run_id: str,
    base_dir: Path | None,
    state: ScenarioOpsState,
    result: NodeResult | ScenarioOpsState,
) -> ScenarioOpsState:
    if isinstance(result, ScenarioOpsState):
        return result

    for key, value in result.state_updates.items():
        if hasattr(state, key):
            setattr(state, key, value)

    for artifact in result.artifacts:
        write_artifact(
            run_id=run_id,
            artifact_name=artifact.name,
            payload=artifact.payload,
            ext=artifact.ext,
            input_values=artifact.input_values,
            prompt_values=artifact.prompt_values,
            tool_versions=artifact.tool_versions,
            base_dir=base_dir,
        )
    return state


def run_graph(
    inputs: GraphInputs,
    *,
    run_id: str | None = None,
    run_timestamp: str | None = None,
    base_dir: Path | None = None,
    state: ScenarioOpsState | None = None,
    llm_client: LLMClient | None = None,
    retriever=retrieve_url,
    config: LLMConfig | None = None,
    mock_mode: bool = False,
    settings: ScenarioOpsSettings | None = None,
    settings_overrides: Mapping[str, Any] | None = None,
    generate_strategies: bool = True,
    report_date: str | None = None,
    command: str | None = None,
) -> ScenarioOpsState:
    if run_id is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    if state is None:
        state = ScenarioOpsState()

    if run_timestamp is None:
        run_timestamp = utc_now_iso()
    register_run_timestamp(run_id, run_timestamp)

    ensure_run_dirs(run_id, base_dir=base_dir)
    if settings is None:
        overrides = dict(settings_overrides or {})
        if mock_mode:
            overrides["mode"] = "demo"
            overrides["sources_policy"] = "fixtures"
            overrides["llm_provider"] = "mock"
        settings = load_settings(overrides)
    if config is None:
        config = llm_config_from_settings(settings)
    use_fixtures = settings.sources_policy == "fixtures"
    policy = policy_for_name(settings.sources_policy)
    write_run_config(
        run_id=run_id, run_config=settings.as_dict(), base_dir=base_dir
    )

    node_events: list[dict[str, Any]] = []

    def _record_event(
        node_name: str,
        inputs: list[str],
        outputs: list[str],
        duration_seconds: float,
        schema_validated: bool,
        error: str | None = None,
    ) -> None:
        event: dict[str, Any] = {
            "node": node_name,
            "inputs": inputs,
            "outputs": outputs,
            "duration_seconds": round(duration_seconds, 6),
            "schema_validated": schema_validated,
        }
        if error:
            event["error"] = error
        node_events.append(event)
        log_node_event(
            run_id=run_id,
            node_name=node_name,
            inputs=inputs,
            outputs=outputs,
            schema_validated=schema_validated,
            duration_seconds=duration_seconds,
            base_dir=base_dir,
            error=error,
        )

    def _run_node(
        node_name: str,
        func,
        *,
        inputs: list[str],
        outputs,
        **kwargs: Any,
    ):
        started = time.perf_counter()
        try:
            result = func(**kwargs)
        except Exception as exc:
            duration = time.perf_counter() - started
            _record_event(
                node_name,
                inputs,
                [],
                duration,
                False,
                error=str(exc),
            )
            raise
        duration = time.perf_counter() - started
        resolved_outputs = outputs(result) if callable(outputs) else outputs
        _record_event(
            node_name,
            inputs,
            resolved_outputs,
            duration,
            True,
        )
        return result

    try:
        sources = list(inputs.sources) if inputs.sources else []
        if not sources:
            sources = policy.default_sources()
        if not sources:
            raise ValueError("Sources are required for exploration context.")
        if use_fixtures and len(sources) < 3:
            defaults = _default_sources()
            while len(sources) < 3:
                sources.append(defaults[len(sources)])

        mock_payloads = _mock_payloads(sources) if use_fixtures else None
        if use_fixtures:
            retriever = _mock_retriever
            if not inputs.user_params:
                inputs = GraphInputs(
                    user_params={"goal": "mock run"},
                    sources=sources,
                    signals=inputs.signals,
                )

        charter_result = _run_node(
            "charter",
            run_charter_node,
            inputs=["params:user_params"],
            outputs=lambda result: [
                f"artifacts/{artifact.name}.{artifact.ext}"
                for artifact in result.artifacts
            ],
            user_params=inputs.user_params,
            run_id=run_id,
            state=state,
            llm_client=_client_for(
                "charter", default_client=llm_client, mock_payloads=mock_payloads
            ),
            config=config,
            base_dir=base_dir,
        )
        state = apply_node_result(
            run_id=run_id, base_dir=base_dir, state=state, result=charter_result
        )
        focal_result = _run_node(
            "focal_issue",
            run_focal_issue_node,
            inputs=["params:user_params"],
            outputs=lambda result: [
                f"artifacts/{artifact.name}.{artifact.ext}"
                for artifact in result.artifacts
            ],
            user_params=inputs.user_params,
            state=state,
            llm_client=_client_for(
                "focal_issue",
                default_client=llm_client,
                mock_payloads=mock_payloads,
            ),
            config=config,
        )
        state = apply_node_result(
            run_id=run_id, base_dir=base_dir, state=state, result=focal_result
        )

        state = _run_node(
            "retrieval",
            run_retrieval_node,
            inputs=["sources"],
            outputs=["artifacts/evidence_units.json"],
            sources=sources,
            run_id=run_id,
            state=state,
            retriever=retriever,
            base_dir=base_dir,
            settings=settings,
        )

        washout_config = WashoutGateConfig()
        focus_domains: list[str] | None = None
        focus_categories: list[str] | None = None
        for iteration in range(washout_config.max_iterations + 1):
            state = _run_node(
                f"scan_pestel[{iteration}]",
                run_scan_node,
                inputs=[
                    "artifacts/evidence_units.json",
                    "artifacts/focal_issue.json",
                ],
                outputs=["artifacts/driving_forces.json"],
                run_id=run_id,
                state=state,
                llm_client=_client_for(
                    "scan_pestel", default_client=llm_client, mock_payloads=mock_payloads
                ),
                base_dir=base_dir,
                config=config,
                min_forces=washout_config.min_total_forces,
                min_per_domain=washout_config.min_per_domain,
                focus_domains=focus_domains,
                focus_categories=focus_categories,
                settings=settings,
            )
            state = _run_node(
                f"washout_audit[{iteration}]",
                run_washout_node,
                inputs=["artifacts/driving_forces.json"],
                outputs=["artifacts/washout_report.json"],
                run_id=run_id,
                state=state,
                llm_client=_client_for(
                    "washout_audit", default_client=llm_client, mock_payloads=mock_payloads
                ),
                base_dir=base_dir,
                config=config,
            )
            try:
                assert_washout_pass(
                    state.driving_forces, state.washout_report, washout_config
                )
                break
            except Exception as exc:
                if iteration >= washout_config.max_iterations:
                    raise RuntimeError("exploration_washout_failed") from exc
                deficits = washout_deficits(
                    state.driving_forces, state.washout_report, washout_config
                )
                focus_domains = deficits.get("missing_domains") or None
                focus_categories = deficits.get("missing_categories") or None

        state = _run_node(
            "coverage",
            run_coverage_node,
            inputs=["artifacts/driving_forces.json"],
            outputs=["artifacts/coverage_report.json"],
            run_id=run_id,
            state=state,
            base_dir=base_dir,
        )

        state = _run_node(
            "classify",
            run_classify_node,
            inputs=["artifacts/evidence_units.json"],
            outputs=["artifacts/certainty_uncertainty.json"],
            run_id=run_id,
            state=state,
            llm_client=_client_for(
                "certainty_uncertainty",
                default_client=llm_client,
                mock_payloads=mock_payloads,
            ),
            base_dir=base_dir,
            config=config,
        )
        state = _run_node(
            "beliefs",
            run_beliefs_node,
            inputs=["artifacts/certainty_uncertainty.json", "artifacts/evidence_units.json"],
            outputs=["artifacts/belief_sets.json"],
            run_id=run_id,
            state=state,
            llm_client=_client_for(
                "beliefs", default_client=llm_client, mock_payloads=mock_payloads
            ),
            base_dir=base_dir,
            config=config,
        )
        state = _run_node(
            "effects",
            run_effects_node,
            inputs=["artifacts/belief_sets.json"],
            outputs=["artifacts/effects.json"],
            run_id=run_id,
            state=state,
            llm_client=_client_for(
                "effects", default_client=llm_client, mock_payloads=mock_payloads
            ),
            base_dir=base_dir,
            config=config,
        )

        state = _run_node(
            "epistemic_summary",
            run_epistemic_summary_node,
            inputs=[
                "artifacts/certainty_uncertainty.json",
                "artifacts/belief_sets.json",
                "artifacts/effects.json",
            ],
            outputs=["artifacts/epistemic_summary.json"],
            run_id=run_id,
            state=state,
            base_dir=base_dir,
        )

        min_citations = (
            settings.min_citations_per_driver if settings.mode == "live" else 1
        )
        state = _run_node(
            "drivers",
            run_drivers_node,
            inputs=["artifacts/evidence_units.json"],
            outputs=["artifacts/drivers.jsonl"],
            run_id=run_id,
            state=state,
            llm_client=_client_for(
                "drivers", default_client=llm_client, mock_payloads=mock_payloads
            ),
            base_dir=base_dir,
            config=config,
            min_citations=min_citations,
        )
        validate_fixture_or_fail(
            run_id=run_id,
            state=state,
            settings=settings,
            base_dir=base_dir,
            command=command,
        )

        state = _run_node(
            "uncertainties",
            run_uncertainties_node,
            inputs=["artifacts/drivers.jsonl"],
            outputs=["artifacts/uncertainties.json"],
            run_id=run_id,
            state=state,
            llm_client=_client_for(
                "uncertainties", default_client=llm_client, mock_payloads=mock_payloads
            ),
            base_dir=base_dir,
            config=config,
        )
        state = _run_node(
            "logic",
            run_logic_node,
            inputs=["artifacts/uncertainties.json"],
            outputs=["artifacts/logic.json"],
            run_id=run_id,
            state=state,
            llm_client=_client_for(
                "logic", default_client=llm_client, mock_payloads=mock_payloads
            ),
            base_dir=base_dir,
            config=config,
        )
        state = _run_node(
            "skeletons",
            run_skeletons_node,
            inputs=["artifacts/logic.json"],
            outputs=["artifacts/skeletons.json"],
            run_id=run_id,
            state=state,
            llm_client=_client_for(
                "skeletons", default_client=llm_client, mock_payloads=mock_payloads
            ),
            base_dir=base_dir,
            config=config,
        )
        state = _run_node(
            "narratives",
            run_narratives_node,
            inputs=["artifacts/skeletons.json"],
            outputs=lambda result: [
                f"artifacts/narrative_{scenario_id}.md"
                for scenario_id in sorted((result.narratives or {}).keys())
            ],
            run_id=run_id,
            state=state,
            llm_client=_client_for(
                "narratives", default_client=llm_client, mock_payloads=mock_payloads
            ),
            base_dir=base_dir,
            config=config,
        )
        state = _run_node(
            "ewis",
            run_ewis_node,
            inputs=["artifacts/logic.json"],
            outputs=["artifacts/ewi.json"],
            run_id=run_id,
            state=state,
            llm_client=_client_for(
                "ewis", default_client=llm_client, mock_payloads=mock_payloads
            ),
            base_dir=base_dir,
            config=config,
        )

        if generate_strategies and state.strategies is None:
            state = _run_node(
                "strategies",
                run_strategies_node,
                inputs=["artifacts/logic.json"],
                outputs=["artifacts/strategies.json"],
                run_id=run_id,
                state=state,
                llm_client=_client_for(
                    "strategies", default_client=llm_client, mock_payloads=mock_payloads
                ),
                base_dir=base_dir,
                config=config,
            )

        if state.strategies is not None:
            state = _run_node(
                "wind_tunnel",
                run_wind_tunnel_node,
                inputs=["artifacts/strategies.json", "artifacts/logic.json"],
                outputs=["artifacts/wind_tunnel.json"],
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
            state = _run_node(
                "daily_runner",
                run_daily_runner_node,
                inputs=["artifacts/ewi.json", "artifacts/wind_tunnel.json"],
                outputs=["artifacts/daily_brief.md", "artifacts/daily_brief.json"],
                signals=signals,
                run_id=run_id,
                state=state,
                llm_client=_client_for(
                    "daily_runner", default_client=llm_client, mock_payloads=mock_payloads
                ),
                base_dir=base_dir,
                config=config,
                report_date=report_date,
            )

        state = _run_node(
            "scenario_profiles",
            run_scenario_profiles_node,
            inputs=[
                "artifacts/logic.json",
                "artifacts/skeletons.json",
                "artifacts/ewi.json",
                "artifacts/wind_tunnel.json",
                "artifacts/epistemic_summary.json",
                "artifacts/drivers.jsonl",
            ],
            outputs=["artifacts/scenario_profiles.json"],
            run_id=run_id,
            state=state,
            base_dir=base_dir,
        )

        state = _run_node(
            "trace_map",
            run_trace_map_node,
            inputs=[
                "artifacts/scenario_profiles.json",
                "artifacts/epistemic_summary.json",
            ],
            outputs=["trace/trace_map.json"],
            run_id=run_id,
            state=state,
            base_dir=base_dir,
        )

        state = _run_node(
            "auditor",
            run_auditor_node,
            inputs=["artifacts"],
            outputs=["artifacts/audit_report.json"],
            run_id=run_id,
            state=state,
            base_dir=base_dir,
            llm_client=_client_for(
                "auditor", default_client=llm_client, mock_payloads=mock_payloads
            ),
            config=config,
            settings=settings,
        )

        index_path = _run_node(
            "artifact_index",
            write_artifact_index,
            inputs=["artifacts"],
            outputs=["artifacts/index.json"],
            run_id=run_id,
            base_dir=base_dir,
            strict=True,
        )
        runs_dir = base_dir if base_dir is not None else default_runs_dir()
        trace_path = runs_dir / run_id / "trace" / "trace_map.json"
        input_parameters = {
            "user_params": dict(inputs.user_params),
            "sources": sources,
            "signals": list(inputs.signals),
            "settings": settings.as_dict(),
            "run_timestamp": run_timestamp,
        }
        _run_node(
            "run_manifest",
            write_run_manifest,
            inputs=["artifacts/index.json", "trace/trace_map.json"],
            outputs=["manifest.json"],
            run_id=run_id,
            run_timestamp=run_timestamp,
            status="OK",
            input_parameters=input_parameters,
            node_sequence=node_events,
            artifact_index_path=index_path,
            trace_map_path=trace_path if trace_path.exists() else None,
            run_config=settings.as_dict(),
            base_dir=base_dir,
            errors=None,
        )
    except Exception as exc:
        write_latest_status(
            run_id=run_id,
            status="FAIL",
            command=command or "run-graph",
            error_summary=str(exc),
            base_dir=base_dir,
            run_config=settings.as_dict(),
        )
        errors = [str(exc)]
        runs_dir = base_dir if base_dir is not None else default_runs_dir()
        index_path: Path | None = None
        try:
            started = time.perf_counter()
            index_path = write_artifact_index(
                run_id=run_id, base_dir=base_dir, strict=False
            )
            duration = time.perf_counter() - started
            _record_event(
                "artifact_index",
                ["artifacts"],
                ["artifacts/index.json"],
                duration,
                False,
            )
        except Exception as index_exc:
            errors.append(f"artifact_index_failed: {index_exc}")

        trace_path = runs_dir / run_id / "trace" / "trace_map.json"
        input_parameters = {
            "user_params": dict(inputs.user_params),
            "sources": list(inputs.sources or []),
            "signals": list(inputs.signals),
            "settings": settings.as_dict(),
            "run_timestamp": run_timestamp,
        }
        try:
            started = time.perf_counter()
            write_run_manifest(
                run_id=run_id,
                run_timestamp=run_timestamp,
                status="FAIL",
                input_parameters=input_parameters,
                node_sequence=node_events,
                artifact_index_path=index_path
                if index_path is not None
                else (runs_dir / run_id / "artifacts" / "index.json"),
                trace_map_path=trace_path if trace_path.exists() else None,
                run_config=settings.as_dict(),
                base_dir=base_dir,
                errors=errors,
            )
            duration = time.perf_counter() - started
            _record_event(
                "run_manifest",
                ["artifacts/index.json", "trace/trace_map.json"],
                ["manifest.json"],
                duration,
                False,
            )
        except Exception as manifest_exc:
            errors.append(f"run_manifest_failed: {manifest_exc}")
        raise

    write_latest_status(
        run_id=run_id,
        status="OK",
        command=command or "run-graph",
        base_dir=base_dir,
        run_config=settings.as_dict(),
    )
    return state
