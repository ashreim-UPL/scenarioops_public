from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from scenarioops.app.config import load_settings, llm_config_from_settings
from scenarioops.graph.state import (
    AuditFinding,
    AuditReport,
    Charter,
    ScenarioOpsState,
    Strategies,
    Strategy,
    WindTunnel,
    WindTunnelTest,
)
from scenarioops.graph.types import GraphInputs
from scenarioops.graph.setup import (
    apply_node_result,
    default_sources,
    mock_payloads_for_sources,
    mock_retriever,
)
from scenarioops.graph.nodes.charter import run_charter_node
from scenarioops.graph.nodes.company_profile import run_company_profile_node
from scenarioops.graph.nodes.focal_issue import run_focal_issue_node
from scenarioops.graph.nodes.retrieval_real import run_retrieval_real_node
from scenarioops.graph.nodes.force_builder import run_force_builder_node
from scenarioops.graph.nodes.ingest_docs import run_ingest_docs_node
from scenarioops.graph.nodes.ebe_rank import run_ebe_rank_node
from scenarioops.graph.nodes.cluster import run_cluster_node
from scenarioops.graph.nodes.uncertainty_axes import run_uncertainty_axes_node
from scenarioops.graph.nodes.scenario_synthesis import run_scenario_synthesis_node
from scenarioops.graph.nodes.strategies import run_strategies_node
from scenarioops.graph.nodes.wind_tunnel import run_wind_tunnel_node
from scenarioops.graph.nodes.auditor import run_auditor_node
from scenarioops.graph.nodes.ewis import run_ewis_node
from scenarioops.graph.nodes.scenario_profiles import run_scenario_profiles_node
from scenarioops.graph.nodes.trace_map import run_trace_map_node
from scenarioops.graph.nodes.daily_runner import run_daily_runner_node
from scenarioops.graph.nodes.utils import get_client
from scenarioops.graph.tools.web_retriever import retrieve_url
from scenarioops.graph.tools.storage import default_runs_dir, log_node_event, write_artifact
from scenarioops.graph.tools.schema_validate import validate_artifact
from scenarioops.graph.tools.artifact_contracts import schema_for_artifact
from scenarioops.graph.tools.traceability import build_run_metadata

from .sentinel import Sentinel
from .analyst import Analyst
from .strategist import Strategist
from .critic import Critic
from .client_wrapper import SquadClient
from .telemetry import record_node_event

def run_squad_orchestrator(
    inputs: GraphInputs,
    *,
    run_id: str,
    base_dir: Path | None = None,
    mock_mode: bool = False,
    generate_strategies: bool = True,
    legacy_mode: bool = False,
    settings=None,
    resume_from: str | None = None,
    report_date: str | None = None,
) -> ScenarioOpsState:
    """Orchestrates the Dynamic Strategy Team (Squad)."""

    overrides: dict[str, Any] = {}
    if mock_mode:
        overrides["mode"] = "demo"
        overrides["sources_policy"] = "fixtures"
        overrides["llm_provider"] = "mock"

    settings = settings or load_settings(overrides)
    config = llm_config_from_settings(settings)
    
    if resume_from and legacy_mode:
        raise ValueError("resume_from is not supported in legacy mode.")
    if resume_from and not generate_strategies:
        blocked = {"strategies", "wind_tunnel"}
        if resume_from.strip().lower() in blocked:
            raise ValueError(
                "resume_from requires strategies enabled when resuming at strategies or wind_tunnel."
            )

    # Initialize State
    state = ScenarioOpsState()
    if resume_from:
        state = _load_resume_state(
            run_id=run_id,
            base_dir=base_dir,
            resume_from=resume_from,
        )
    
    # Initialize Base Client (Shared History Holder)
    base_client = get_client(None, config)
    mock_payloads = mock_payloads_for_sources(inputs.sources or default_sources()) if mock_mode else None
    
    # In a real impl, we'd wrap the specific node clients. 
    # Here we create a single shared SquadClient to hold the "Thought Signatures" (history).
    # This fulfills: "Update the orchestrator to pass the Gemini3Client.history... between agents"
    squad_client = SquadClient(base_client, thinking_level="low")
    
    # Pre-Squad Setup (Charter/Focal Issue/Company Profile) - typically part of setup
    # We run these using the squad client to maintain context.
    # Note: Using run_charter_node directly as it wasn't assigned to a specific agent in instructions,
    # or arguably belongs to Analyst or Sentinel. We'll run it as prep.
    llm_label = f"llm:{getattr(config, 'mode', 'unknown')}"
    if _should_run("charter", resume_from):
        charter_result = record_node_event(
            run_id=run_id,
            node_name="charter",
            inputs=["user_params"],
            outputs=["charter.json"],
            tools=[llm_label],
            base_dir=base_dir,
            action=lambda: run_charter_node(
                user_params=inputs.user_params,
                run_id=run_id,
                state=state,
                llm_client=squad_client,
                base_dir=base_dir,
                config=config,
            ),
        )
        state = apply_node_result(run_id, base_dir, state, charter_result)

    if _should_run("focal_issue", resume_from):
        focal_result = record_node_event(
            run_id=run_id,
            node_name="focal_issue",
            inputs=["charter.json"],
            outputs=["focal_issue.json"],
            tools=[llm_label],
            base_dir=base_dir,
            action=lambda: run_focal_issue_node(
                user_params=inputs.user_params,
                run_id=run_id,
                state=state,
                llm_client=squad_client,
                config=config,
            ),
        )
        state = apply_node_result(run_id, base_dir, state, focal_result)

    if _should_run("company_profile", resume_from):
        company_profile_result = record_node_event(
            run_id=run_id,
            node_name="company_profile",
            inputs=["user_params", "sources"],
            outputs=["company_profile.json"],
            tools=[llm_label],
            base_dir=base_dir,
            action=lambda: run_company_profile_node(
                user_params=inputs.user_params,
                sources=list(inputs.sources or []),
                input_docs=list(inputs.input_docs or []),
                run_id=run_id,
                state=state,
                base_dir=base_dir,
                settings=settings,
                config=config,
            ),
        )
        state = apply_node_result(run_id, base_dir, state, company_profile_result)

    if _should_run("ingest_docs", resume_from):
        has_input_docs = bool(inputs.input_docs)
        has_preloaded = bool(
            state.evidence_units
            and state.evidence_units.get("evidence_units")
        )
        if has_input_docs or has_preloaded:
            ingest_result = record_node_event(
                run_id=run_id,
                node_name="ingest_docs",
                inputs=["input_docs"],
                outputs=["evidence_units_uploads.json"],
                tools=["system"],
                base_dir=base_dir,
                action=lambda: run_ingest_docs_node(
                    doc_paths=list(inputs.input_docs),
                    run_id=run_id,
                    state=state,
                    user_params=inputs.user_params,
                    base_dir=base_dir,
                    settings=settings,
                    config=config,
                ),
            )
            state = apply_node_result(run_id, base_dir, state, ingest_result)
        else:
            log_node_event(
                run_id=run_id,
                node_name="ingest_docs",
                inputs=["input_docs"],
                outputs=["evidence_units_uploads.json"],
                schema_validated=True,
                duration_seconds=0.0,
                base_dir=base_dir,
                tools=["system"],
                status="SKIPPED",
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
    
    if legacy_mode:
        # 1. Sentinel (Exploration)
        # "Configure sentinel.py to accept company and country. Use enable_search=True and thinking_level='low'"
        # We extract company/country from user_params or default.
        company = str(
            inputs.user_params.get("company")
            or inputs.user_params.get("value")
            or "Unknown Company"
        )
        geography = None
        if isinstance(state.focal_issue, Mapping):
            scope = state.focal_issue.get("scope")
            if isinstance(scope, Mapping):
                geography = scope.get("geography")
        country = str(
            geography
            or inputs.user_params.get("country")
            or inputs.user_params.get("scope")
            or "Unknown Region"
        )
        
        sentinel = Sentinel(
            company=company, 
            country=country, 
            enable_search=True, 
            thinking_level="low"
        )
        # Sentinel uses the squad_client, potentially updating its thinking_level locally if needed,
        # but since we pass the client instance, we should ensure the client respects the agent's pref.
        # Our SquadClient wrapper allows init with level, but here we share the instance. 
        # We might need to update the client's thinking level on the fly or create a child context.
        squad_client.thinking_level = sentinel.thinking_level
        
        retriever = mock_retriever if mock_mode else retrieve_url

        state = sentinel.explore(
            state=state,
            sources=list(inputs.sources) if inputs.sources else [],
            run_id=run_id,
            user_params=dict(inputs.user_params),
            base_dir=base_dir,
            config=config,
            settings=settings,
            llm_client=squad_client,
            retriever=retriever
        )
        
        # 2. Analyst (Logic/Narratives)
        analyst = Analyst()
        squad_client.thinking_level = "low" # Reset/Default for Analyst
        state = analyst.process(
            state=state,
            run_id=run_id,
            base_dir=base_dir,
            config=config,
            settings=settings,
            llm_client=squad_client
        )
        
        # 3. Strategist (Strategies + ROI)
        if generate_strategies:
            strategist = Strategist()
            squad_client.thinking_level = "low"
            state = strategist.generate(
                state=state,
                run_id=run_id,
                base_dir=base_dir,
                config=config,
                llm_client=squad_client
            )
        
        # 4. Critic (Wind Tunnel / Validation)
        critic = Critic(thinking_level="high")
        squad_client.thinking_level = critic.thinking_level
        state = critic.validate(
            state=state,
            run_id=run_id,
            base_dir=base_dir,
            config=config,
            settings=settings,
            llm_client=squad_client
        )

        state = record_node_event(
            run_id=run_id,
            node_name="ewi",
            inputs=["logic.json"],
            outputs=["ewi.json"],
            tools=[llm_label],
            base_dir=base_dir,
            action=lambda: run_ewis_node(
                run_id=run_id,
                state=state,
                llm_client=squad_client,
                base_dir=base_dir,
                config=config,
                settings=settings,
            ),
        )

        state = record_node_event(
            run_id=run_id,
            node_name="scenario_profiles",
            inputs=["logic.json", "skeletons.json", "epistemic_summary.json"],
            outputs=["scenario_profiles.json"],
            tools=["system"],
            base_dir=base_dir,
            action=lambda: run_scenario_profiles_node(
                run_id=run_id,
                state=state,
                base_dir=base_dir,
            ),
        )

        state = record_node_event(
            run_id=run_id,
            node_name="trace_map",
            inputs=["scenario_profiles.json", "epistemic_summary.json"],
            outputs=["trace_map.json"],
            tools=["system"],
            base_dir=base_dir,
            action=lambda: run_trace_map_node(
                run_id=run_id,
                state=state,
                base_dir=base_dir,
            ),
        )

        state = record_node_event(
            run_id=run_id,
            node_name="daily_brief",
            inputs=["ewi.json", "wind_tunnel.json"],
            outputs=["daily_brief.json", "daily_brief.md"],
            tools=[llm_label],
            base_dir=base_dir,
            action=lambda: run_daily_runner_node(
                list(inputs.signals or []),
                run_id=run_id,
                state=state,
                llm_client=squad_client,
                base_dir=base_dir,
                config=config,
                report_date=report_date,
            ),
        )

        return state

    # Professionalized pipeline (non-legacy)
    if _should_run("retrieval_real", resume_from):
        retriever_fn = mock_retriever if mock_mode else retrieve_url
        state = record_node_event(
            run_id=run_id,
            node_name="retrieval_real",
            inputs=["sources", "focal_issue.json"],
            outputs=["evidence_units.json", "retrieval_report.json"],
            tools=[llm_label, "search:gemini"],
            base_dir=base_dir,
            action=lambda: run_retrieval_real_node(
                list(inputs.sources) if inputs.sources else [],
                run_id=run_id,
                state=state,
                user_params=dict(inputs.user_params),
                focal_issue=state.focal_issue if isinstance(state.focal_issue, Mapping) else None,
                base_dir=base_dir,
                config=config,
                settings=settings,
                llm_client=squad_client,
                retriever=retriever_fn,
            ),
        )

    if _should_run("forces", resume_from):
        state = record_node_event(
            run_id=run_id,
            node_name="forces",
            inputs=["evidence_units.json"],
            outputs=["forces.json"],
            tools=[llm_label],
            base_dir=base_dir,
            action=lambda: run_force_builder_node(
                run_id=run_id,
                state=state,
                user_params=dict(inputs.user_params),
                llm_client=squad_client,
                base_dir=base_dir,
                config=config,
                min_forces=int(getattr(settings, "min_forces", 60)),
                min_per_domain=int(getattr(settings, "min_forces_per_domain", 10)),
            ),
        )

    if _should_run("ebe_rank", resume_from):
        state = record_node_event(
            run_id=run_id,
            node_name="ebe_rank",
            inputs=["forces.json", "evidence_units.json"],
            outputs=["forces_ranked.json"],
            tools=["system"],
            base_dir=base_dir,
            action=lambda: run_ebe_rank_node(
                run_id=run_id,
                state=state,
                base_dir=base_dir,
            ),
        )

    if _should_run("clusters", resume_from):
        state = record_node_event(
            run_id=run_id,
            node_name="clusters",
            inputs=["forces.json"],
            outputs=["clusters.json"],
            tools=[llm_label],
            base_dir=base_dir,
            action=lambda: run_cluster_node(
                run_id=run_id,
                state=state,
                user_params=dict(inputs.user_params),
                llm_client=squad_client,
                base_dir=base_dir,
                config=config,
                settings=settings,
                seed=int(getattr(settings, "seed", 0) or 0),
            ),
        )

    if _should_run("uncertainty_axes", resume_from):
        state = record_node_event(
            run_id=run_id,
            node_name="uncertainty_axes",
            inputs=["clusters.json", "forces.json"],
            outputs=["uncertainty_axes.json"],
            tools=[llm_label],
            base_dir=base_dir,
            action=lambda: run_uncertainty_axes_node(
                run_id=run_id,
                state=state,
                user_params=dict(inputs.user_params),
                llm_client=squad_client,
                base_dir=base_dir,
                config=config,
                settings=settings,
            ),
        )

    if _should_run("scenarios", resume_from):
        state = record_node_event(
            run_id=run_id,
            node_name="scenarios",
            inputs=["uncertainty_axes.json", "clusters.json"],
            outputs=["scenarios.json"],
            tools=[llm_label],
            base_dir=base_dir,
            action=lambda: run_scenario_synthesis_node(
                run_id=run_id,
                state=state,
                user_params=dict(inputs.user_params),
                llm_client=squad_client,
                base_dir=base_dir,
                config=config,
            ),
        )

    if generate_strategies:
        if _should_run("strategies", resume_from):
            state = record_node_event(
                run_id=run_id,
                node_name="strategies",
                inputs=["scenarios.json"],
                outputs=["strategies.json"],
                tools=[llm_label],
                base_dir=base_dir,
                action=lambda: run_strategies_node(
                    run_id=run_id,
                    state=state,
                    llm_client=squad_client,
                    base_dir=base_dir,
                    config=config,
                ),
            )
        if _should_run("wind_tunnel", resume_from):
            state = record_node_event(
                run_id=run_id,
                node_name="wind_tunnel",
                inputs=["strategies.json", "scenarios.json"],
                outputs=["wind_tunnel.json"],
                tools=[llm_label],
                base_dir=base_dir,
                action=lambda: run_wind_tunnel_node(
                    run_id=run_id,
                    state=state,
                    llm_client=squad_client,
                    base_dir=base_dir,
                    config=config,
                    settings=settings,
                ),
            )

    if _should_run("auditor", resume_from):
        state = record_node_event(
            run_id=run_id,
            node_name="auditor",
            inputs=["artifacts/*"],
            outputs=["audit_report.json"],
            tools=[llm_label],
            base_dir=base_dir,
            action=lambda: run_auditor_node(
                run_id=run_id,
                state=state,
                base_dir=base_dir,
                settings=settings,
                config=config,
                llm_client=squad_client,
            ),
        )

    return state


_PRO_STEPS = [
    "charter",
    "focal_issue",
    "company_profile",
    "ingest_docs",
    "retrieval_real",
    "forces",
    "ebe_rank",
    "clusters",
    "uncertainty_axes",
    "scenarios",
    "strategies",
    "wind_tunnel",
    "auditor",
]

_NODE_EVENT_META: dict[str, dict[str, list[str]]] = {
    "charter": {"inputs": ["user_params"], "outputs": ["charter.json"]},
    "focal_issue": {"inputs": ["charter.json"], "outputs": ["focal_issue.json"]},
    "company_profile": {"inputs": ["user_params", "sources", "input_docs"], "outputs": ["company_profile.json"]},
    "ingest_docs": {"inputs": ["input_docs"], "outputs": ["evidence_units_uploads.json"]},
    "retrieval_real": {"inputs": ["sources", "focal_issue.json"], "outputs": ["evidence_units.json", "retrieval_report.json"]},
    "forces": {"inputs": ["evidence_units.json"], "outputs": ["forces.json"]},
    "ebe_rank": {"inputs": ["forces.json", "evidence_units.json"], "outputs": ["forces_ranked.json"]},
    "clusters": {"inputs": ["forces.json"], "outputs": ["clusters.json"]},
    "uncertainty_axes": {"inputs": ["clusters.json", "forces.json"], "outputs": ["uncertainty_axes.json"]},
    "scenarios": {"inputs": ["uncertainty_axes.json", "clusters.json"], "outputs": ["scenarios.json"]},
    "strategies": {"inputs": ["scenarios.json"], "outputs": ["strategies.json"]},
    "wind_tunnel": {"inputs": ["strategies.json", "scenarios.json"], "outputs": ["wind_tunnel.json"]},
    "auditor": {"inputs": ["artifacts/*"], "outputs": ["audit_report.json"]},
}


def _should_run(node_name: str, resume_from: str | None) -> bool:
    if not resume_from:
        return True
    resume_from = resume_from.strip().lower()
    node_name = node_name.strip().lower()
    if resume_from not in _PRO_STEPS:
        raise ValueError(f"Unknown resume_from node: {resume_from}")
    return _PRO_STEPS.index(node_name) >= _PRO_STEPS.index(resume_from)


def _record_hydrated_event(
    *,
    run_id: str,
    node_name: str,
    base_dir: Path | None,
    tools: list[str] | None = None,
) -> None:
    meta = _NODE_EVENT_META.get(node_name, {"inputs": [], "outputs": []})
    log_node_event(
        run_id=run_id,
        node_name=node_name,
        inputs=meta.get("inputs", []),
        outputs=meta.get("outputs", []),
        schema_validated=True,
        duration_seconds=0.0,
        base_dir=base_dir,
        tools=tools or ["hydrate"],
        status="HYDRATED",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def _cache_dir(base_dir: Path | None) -> Path:
    if base_dir is None:
        repo_root = Path(__file__).resolve().parents[4]
        return repo_root / "cache" / "evidence_units"
    return base_dir.parent / "cache" / "evidence_units"


def _cache_ttl_days() -> int:
    raw = os.environ.get("EVIDENCE_CACHE_DAYS")
    if raw is None:
        return 7
    try:
        return max(0, int(raw))
    except ValueError:
        return 7


def _cache_is_fresh(cached_at: str, ttl_days: int) -> bool:
    if ttl_days <= 0:
        return False
    try:
        cached_ts = datetime.fromisoformat(cached_at.replace("Z", "+00:00"))
    except ValueError:
        return False
    age_days = (datetime.now(timezone.utc) - cached_ts).days
    return age_days <= ttl_days


def _normalize_match(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _load_cached_evidence(
    *,
    base_dir: Path | None,
    company_name: str,
    geography: str,
    horizon_months: int,
) -> dict[str, Any] | None:
    cache_root = _cache_dir(base_dir)
    if not cache_root.exists():
        return None
    ttl_days = _cache_ttl_days()
    target_company = _normalize_match(company_name)
    target_geo = _normalize_match(geography)
    target_horizon = int(horizon_months)
    best: dict[str, Any] | None = None
    best_ts: datetime | None = None
    for path in cache_root.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        if payload.get("schema_version") != "2.0":
            continue
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, Mapping):
            continue
        cached_at = payload.get("cached_at")
        if not isinstance(cached_at, str) or not _cache_is_fresh(cached_at, ttl_days):
            continue
        if _normalize_match(metadata.get("company_name")) != target_company:
            continue
        if _normalize_match(metadata.get("geography")) != target_geo:
            continue
        try:
            cached_horizon = int(metadata.get("horizon_months", 0))
        except (TypeError, ValueError):
            continue
        if cached_horizon != target_horizon:
            continue
        evidence_units = payload.get("evidence_units")
        if not isinstance(evidence_units, list) or not evidence_units:
            continue
        try:
            cached_ts = datetime.fromisoformat(cached_at.replace("Z", "+00:00"))
        except ValueError:
            cached_ts = None
        if cached_ts and (best_ts is None or cached_ts > best_ts):
            best = {
                "cache_key": path.stem,
                "cached_at": cached_at,
                "evidence_units": evidence_units,
            }
            best_ts = cached_ts
    return best


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}.")
    return payload


def _load_and_validate_json(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    schema_name = schema_for_artifact(path.stem, path.suffix)
    if schema_name and schema_name != "markdown":
        validate_artifact(schema_name, payload)
    return payload


def _load_resume_state(
    *,
    run_id: str,
    base_dir: Path | None,
    resume_from: str,
) -> ScenarioOpsState:
    run_root = base_dir if base_dir is not None else default_runs_dir()
    artifacts_dir = run_root / run_id / "artifacts"
    if not artifacts_dir.exists():
        raise FileNotFoundError(f"Run artifacts not found for resume: {run_id}")

    resume_from = resume_from.strip().lower()
    if resume_from not in _PRO_STEPS:
        raise ValueError(f"Unknown resume_from node: {resume_from}")

    state = ScenarioOpsState()
    profile_hint: dict[str, Any] = {}
    for node_name in _PRO_STEPS:
        if node_name == resume_from:
            break
        if node_name == "charter":
            payload = _load_and_validate_json(artifacts_dir / "scenario_charter.json")
            state.charter = Charter(**payload)
            _record_hydrated_event(
                run_id=run_id, node_name="charter", base_dir=base_dir
            )
        elif node_name == "focal_issue":
            state.focal_issue = _load_and_validate_json(
                artifacts_dir / "focal_issue.json"
            )
            _record_hydrated_event(
                run_id=run_id, node_name="focal_issue", base_dir=base_dir
            )
        elif node_name == "company_profile":
            state.company_profile = _load_and_validate_json(
                artifacts_dir / "company_profile.json"
            )
            profile_hint = {
                "company_name": str(state.company_profile.get("company_name", "")),
                "geography": str(state.company_profile.get("geography", "")),
                "horizon_months": int(state.company_profile.get("horizon_months", 0)),
            }
            _record_hydrated_event(
                run_id=run_id, node_name="company_profile", base_dir=base_dir
            )
        elif node_name == "ingest_docs":
            uploads_path = artifacts_dir / "evidence_units_uploads.json"
            if uploads_path.exists():
                state.evidence_units = _load_and_validate_json(uploads_path)
                _record_hydrated_event(
                    run_id=run_id, node_name="ingest_docs", base_dir=base_dir
                )
            else:
                log_node_event(
                    run_id=run_id,
                    node_name="ingest_docs",
                    inputs=_NODE_EVENT_META.get("ingest_docs", {}).get("inputs", []),
                    outputs=_NODE_EVENT_META.get("ingest_docs", {}).get("outputs", []),
                    schema_validated=True,
                    duration_seconds=0.0,
                    base_dir=base_dir,
                    tools=["hydrate"],
                    status="SKIPPED",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
        elif node_name == "retrieval_real":
            evidence_path = artifacts_dir / "evidence_units.json"
            if evidence_path.exists():
                state.evidence_units = _load_and_validate_json(evidence_path)
                _record_hydrated_event(
                    run_id=run_id, node_name="retrieval_real", base_dir=base_dir
                )
            else:
                company_name = profile_hint.get("company_name") or ""
                geography = profile_hint.get("geography") or ""
                horizon_months = profile_hint.get("horizon_months", 0)
                if not company_name or not geography:
                    raise FileNotFoundError(
                        "evidence_units.json missing and company_profile incomplete for cache lookup."
                    )
                cached = _load_cached_evidence(
                    base_dir=base_dir,
                    company_name=company_name,
                    geography=geography,
                    horizon_months=int(horizon_months),
                )
                if not cached:
                    raise FileNotFoundError(
                        "evidence_units.json missing and no matching cache found."
                    )
                user_params = {
                    "company": company_name,
                    "value": company_name,
                    "geography": geography,
                    "scope": geography,
                    "horizon": horizon_months,
                }
                metadata = build_run_metadata(
                    run_id=run_id,
                    user_params=user_params,
                    focal_issue=state.focal_issue
                    if isinstance(state.focal_issue, Mapping)
                    else None,
                )
                payload = {
                    **metadata,
                    "schema_version": "2.0",
                    "simulated": False,
                    "evidence_units": cached.get("evidence_units", []),
                }
                validate_artifact("evidence_units.schema", payload)
                write_artifact(
                    run_id=run_id,
                    artifact_name="evidence_units",
                    payload=payload,
                    ext="json",
                    input_values={
                        "cache_hit": True,
                        "cache_key": cached.get("cache_key"),
                    },
                    tool_versions={"resume_hydration": "0.1.0"},
                    base_dir=base_dir,
                )
                state.evidence_units = payload
                _record_hydrated_event(
                    run_id=run_id,
                    node_name="retrieval_real",
                    base_dir=base_dir,
                    tools=["cache"],
                )
        elif node_name == "forces":
            state.forces = _load_and_validate_json(artifacts_dir / "forces.json")
            _record_hydrated_event(
                run_id=run_id, node_name="forces", base_dir=base_dir
            )
        elif node_name == "ebe_rank":
            state.forces_ranked = _load_and_validate_json(
                artifacts_dir / "forces_ranked.json"
            )
            _record_hydrated_event(
                run_id=run_id, node_name="ebe_rank", base_dir=base_dir
            )
        elif node_name == "clusters":
            state.clusters = _load_and_validate_json(artifacts_dir / "clusters.json")
            _record_hydrated_event(
                run_id=run_id, node_name="clusters", base_dir=base_dir
            )
        elif node_name == "uncertainty_axes":
            state.uncertainty_axes = _load_and_validate_json(
                artifacts_dir / "uncertainty_axes.json"
            )
            _record_hydrated_event(
                run_id=run_id, node_name="uncertainty_axes", base_dir=base_dir
            )
        elif node_name == "scenarios":
            state.scenarios = _load_and_validate_json(artifacts_dir / "scenarios.json")
            _record_hydrated_event(
                run_id=run_id, node_name="scenarios", base_dir=base_dir
            )
        elif node_name == "strategies":
            payload = _load_and_validate_json(artifacts_dir / "strategies.json")
            strategies = [Strategy(**item) for item in payload.get("strategies", [])]
            state.strategies = Strategies(
                id=payload.get("id", f"strategies-{run_id}"),
                title=payload.get("title", "Strategies"),
                strategies=strategies,
            )
            _record_hydrated_event(
                run_id=run_id, node_name="strategies", base_dir=base_dir
            )
        elif node_name == "wind_tunnel":
            payload = _load_and_validate_json(artifacts_dir / "wind_tunnel.json")
            tests = [WindTunnelTest(**test) for test in payload.get("tests", [])]
            state.wind_tunnel = WindTunnel(
                id=payload.get("id", f"wind-tunnel-{run_id}"),
                title=payload.get("title", "Wind Tunnel"),
                tests=tests,
            )
            _record_hydrated_event(
                run_id=run_id, node_name="wind_tunnel", base_dir=base_dir
            )
        elif node_name == "auditor":
            payload = _load_and_validate_json(artifacts_dir / "audit_report.json")
            findings = [
                AuditFinding(**finding) for finding in payload.get("findings", [])
            ]
            state.audit_report = AuditReport(
                id=payload.get("id", f"audit-{run_id}"),
                period_start=payload.get("period_start", run_id),
                period_end=payload.get("period_end", run_id),
                summary=payload.get("summary", ""),
                findings=findings,
                lessons=payload.get("lessons", []),
                actions=payload.get("actions", []),
            )
            _record_hydrated_event(
                run_id=run_id, node_name="auditor", base_dir=base_dir
            )
    return state
