from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from scenarioops.app.config import (
    ScenarioOpsSettings,
    apply_overrides,
    llm_config_from_settings,
    load_settings,
    settings_from_dict,
)
from scenarioops.app.workflow import (
    ensure_signals,
    latest_run_id,
    list_artifacts,
    state_for_daily,
    state_for_strategies,
)
from scenarioops.graph.types import GraphInputs
from scenarioops.graph.setup import (
    apply_node_result,
    client_for_node,
    default_sources,
    mock_payloads_for_sources,
)
from scenarioops.graph.build_graph import run_graph
from scenarioops.graph.guards.fixture_guard import detect_fixture_evidence
from scenarioops.graph.nodes import (
    run_auditor_node,
    run_daily_runner_node,
    run_strategies_node,
    run_wind_tunnel_node,
)
from scenarioops.graph.nodes.charter import run_charter_node
from scenarioops.graph.state import ScenarioOpsState
from scenarioops.graph.tools.scoring import hash_scoring_result, score_with_rubric
from scenarioops.graph.tools.storage import (
    default_runs_dir,
    read_latest_status,
    write_artifact,
    write_latest_status,
)
from scenarioops.graph.tools.view_model import build_view_model
from scenarioops.graph.tools.web_retriever import retrieve_url
from scenarioops.llm.client import MockLLMClient, get_gemini_api_key


def _load_json_value(value: str) -> Any:
    path = Path(value)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return json.loads(value)


def _parse_sources(value: str | None) -> list[str]:
    if not value:
        return []
    path = Path(value)
    if path.exists():
        parsed = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    if value.strip().startswith("["):
        parsed = _load_json_value(value)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    return [item.strip() for item in value.split(",") if item.strip()]


import logging
import sys

# Configure structured JSON logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("scenarioops")

def _log_json(event: str, data: dict[str, Any] | None = None) -> None:
    payload = {"event": event, "timestamp": datetime.now(timezone.utc).isoformat()}
    if data:
        payload.update(data)
    logger.info(json.dumps(payload))


def _run_charter(args: argparse.Namespace) -> None:
    if not args.params:
        raise ValueError("--params is required for charter.")
    user_params = _load_json_value(args.params)
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    base_dir = Path(args.base_dir) if args.base_dir else None
    llm_client = None
    if args.mock_payload:
        mock_payload = _load_json_value(args.mock_payload)
        llm_client = MockLLMClient(json_payload=mock_payload)

    result = run_charter_node(
        user_params,
        run_id=run_id,
        state=ScenarioOpsState(),
        llm_client=llm_client,
        base_dir=base_dir,
    )
    apply_node_result(run_id=run_id, base_dir=base_dir, state=ScenarioOpsState(), result=result)

def _runs_dir(base_dir: Path | None) -> Path:
    return base_dir if base_dir is not None else default_runs_dir()


def _load_run_config(run_id: str, base_dir: Path | None) -> dict[str, Any] | None:
    path = _runs_dir(base_dir) / run_id / "run_config.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _settings_for_run(
    run_id: str, base_dir: Path | None, overrides: dict[str, Any]
) -> ScenarioOpsSettings:
    run_config = _load_run_config(run_id, base_dir)
    if run_config:
        settings = settings_from_dict(run_config)
        if overrides:
            settings = apply_overrides(settings, overrides)
        return settings
    return load_settings(overrides)


def _settings_overrides_from_args(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    mode = getattr(args, "mode", None)
    if mode:
        overrides["mode"] = mode
    llm_provider = getattr(args, "llm_provider", None)
    if llm_provider:
        overrides["llm_provider"] = llm_provider
    gemini_model = getattr(args, "gemini_model", None)
    if gemini_model:
        overrides["gemini_model"] = gemini_model
    sources_policy = getattr(args, "sources_policy", None)
    if sources_policy:
        overrides["sources_policy"] = sources_policy
    allow_web = getattr(args, "allow_web", None)
    if allow_web is not None:
        overrides["allow_web"] = allow_web
    min_sources_per_domain = getattr(args, "min_sources_per_domain", None)
    if min_sources_per_domain is not None:
        overrides["min_sources_per_domain"] = min_sources_per_domain
    min_citations_per_driver = getattr(args, "min_citations_per_driver", None)
    if min_citations_per_driver is not None:
        overrides["min_citations_per_driver"] = min_citations_per_driver
    min_forces = getattr(args, "min_forces", None)
    if min_forces is not None:
        overrides["min_forces"] = min_forces
    min_forces_per_domain = getattr(args, "min_forces_per_domain", None)
    if min_forces_per_domain is not None:
        overrides["min_forces_per_domain"] = min_forces_per_domain
    forbid_fixture_citations = getattr(args, "forbid_fixture_citations", None)
    if forbid_fixture_citations is not None:
        overrides["forbid_fixture_citations"] = forbid_fixture_citations
    simulate_evidence = getattr(args, "simulate_evidence", None)
    if simulate_evidence is not None:
        overrides["simulate_evidence"] = simulate_evidence
    seed = getattr(args, "seed", None)
    if seed is not None:
        overrides["seed"] = seed
    return overrides


def _use_fixtures(settings: ScenarioOpsSettings) -> bool:
    return settings.sources_policy == "fixtures"
def _print_result(run_id: str, base_dir: Path | None) -> None:
    artifacts = list_artifacts(run_id, base_dir)
    _log_json("result", {"run_id": run_id, "artifacts": artifacts})

def _verify_fail(message: str) -> None:
    _log_json("verify_failed", {"error": message})
    sys.exit(1)

def _format_audit_findings(report_path: Path) -> list[str]:
    if not report_path.exists():
        return [f"audit report missing: {report_path}"]
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"audit report invalid JSON: {exc}"]

    findings = report.get("findings", []) if isinstance(report, dict) else []
    if not isinstance(findings, list):
        return [f"audit report findings invalid: {type(findings)}"]

    lines = []
    for finding in findings:
        if not isinstance(finding, dict):
            lines.append(str(finding))
            continue
        finding_id = str(finding.get("id", "unknown"))
        finding_text = str(finding.get("finding", "")).strip()
        line = f"{finding_id}: {finding_text}".strip()
        evidence = finding.get("evidence")
        if isinstance(evidence, list) and evidence:
            line = f"{line} (evidence: {evidence[0]})".strip()
        elif evidence:
            line = f"{line} (evidence: {evidence})".strip()
        lines.append(line)
    return lines

def _record_audit_findings(run_id: str, base_dir: Path | None) -> None:
    report_path = _runs_dir(base_dir) / run_id / "artifacts" / "audit_report.json"
    lines = _format_audit_findings(report_path)
    if lines:
        _log_json("audit_findings", {"findings": lines})

    logs_dir = _runs_dir(base_dir) / run_id / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "verify_audit_findings.txt"
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_latest(
    *,
    run_id: str,
    status: str,
    command: str,
    base_dir: Path | None = None,
    error_summary: str | None = None,
    settings: ScenarioOpsSettings | None = None,
) -> None:
    write_latest_status(
        run_id=run_id,
        status=status,
        command=command,
        error_summary=error_summary,
        base_dir=base_dir,
        run_config=settings.as_dict() if settings else None,
    )


def _export_view_model(run_id: str, base_dir: Path | None) -> Path:
    runs_dir = base_dir if base_dir is not None else default_runs_dir()
    run_dir = runs_dir / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    view_model = build_view_model(run_dir)
    artifact_path, _, _ = write_artifact(
        run_id=run_id,
        artifact_name="view_model",
        payload=view_model,
        ext="json",
        tool_versions={"view_model_export": "0.1.0"},
        base_dir=base_dir,
    )
    return artifact_path


def _run_verify(args: argparse.Namespace) -> None:
    if args.demo and args.live:
        _verify_fail("verify requires exactly one of --demo or --live.")
    if not args.demo and not args.live:
        _verify_fail("verify requires --demo or --live.")

    base_dir = Path(args.base_dir) if args.base_dir else None
    run_id = args.run_id or f"verify-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    overrides = _settings_overrides_from_args(args)

    if args.demo:
        overrides["mode"] = "demo"
        overrides["sources_policy"] = "fixtures"
        overrides["llm_provider"] = "mock"
        settings = load_settings(overrides)
        config = llm_config_from_settings(settings)
        sources = default_sources()
        inputs = GraphInputs(
            user_params={"scope": "country", "value": "UAE", "horizon": 24},
            sources=sources,
            signals=[],
        )

        try:
                run_graph(
                    inputs,
                    run_id=run_id,
                    base_dir=base_dir,
                    settings=settings,
                    mock_mode=True,
                    generate_strategies=True,
                    report_date="2026-01-01",
                    legacy_mode=True,
                    command="verify",
                )
        except Exception as exc:
            _verify_fail(f"mock build failed: {exc}")

        latest = read_latest_status(base_dir)
        if not latest or latest.get("run_id") != run_id:
            _verify_fail("latest.json missing or does not match run_id.")
        if latest.get("status") != "OK":
            _verify_fail(f"latest.json status is {latest.get('status')}.")

        try:
            view_path = _export_view_model(run_id, base_dir)
        except Exception as exc:
            _verify_fail(f"view model export failed: {exc}")
        if not view_path.exists():
            _verify_fail("view_model.json missing after export.")

        try:
            run_auditor_node(
                run_id=run_id,
                state=ScenarioOpsState(),
                base_dir=base_dir,
                settings=settings,
                config=config,
            )
        except Exception as exc:
            _record_audit_findings(run_id, base_dir)
            _verify_fail(f"schema audit failed: {exc}")

        scores = {"relevance": 0.9, "credibility": 0.9, "recency": 0.8, "specificity": 0.8}
        hash_one = hash_scoring_result(score_with_rubric(scores))
        hash_two = hash_scoring_result(score_with_rubric(scores))
        if hash_one != hash_two:
            _verify_fail("scoring hash mismatch across runs.")

        daily_state = state_for_daily(run_id, base_dir, allow_missing=True)
        signals = []
        if daily_state.ewi is not None:
            signals = ensure_signals(daily_state.ewi.indicators)
        mock_payloads = mock_payloads_for_sources(sources)
        run_daily_runner_node(
            signals,
            run_id=run_id,
            state=daily_state,
            llm_client=client_for_node("daily_runner", mock_payloads=mock_payloads),
            base_dir=base_dir,
            report_date="2026-01-01",
            config=config,
        )
        brief_path = _runs_dir(base_dir) / run_id / "artifacts" / "daily_brief.md"
        if not brief_path.exists():
            _verify_fail("daily_brief.md missing after run-daily.")

        broken_run_id = f"{run_id}-broken"
        invalid_logic = {"id": "logic-1", "title": "Logic", "axes": "bad", "scenarios": []}
        write_artifact(
            run_id=broken_run_id,
            artifact_name="logic",
            payload=invalid_logic,
            ext="json",
            base_dir=base_dir,
        )
        try:
            run_auditor_node(
                run_id=broken_run_id, state=ScenarioOpsState(), base_dir=base_dir
            )
            _verify_fail("auditor should fail on broken artifact.")
        except RuntimeError:
            pass

        _log_json("verify_ok", {"mode": "demo"})
        return

    overrides["mode"] = "live"
    overrides.setdefault("allow_web", True)
    settings = load_settings(overrides)
    config = llm_config_from_settings(settings)
    try:
        get_gemini_api_key()
    except RuntimeError:
        _log_json("verify_skipped", {"reason": "GEMINI_API_KEY not set"})
        return

    sources = _parse_sources(args.sources)
    if not sources:
        _verify_fail("verify --live requires --sources.")

    inputs = GraphInputs(
        user_params={"scope": "country", "value": "UAE", "horizon": 24},
        sources=sources,
        signals=[],
    )

    try:
        state = run_graph(
            inputs,
            run_id=run_id,
            base_dir=base_dir,
            settings=settings,
            generate_strategies=False,
            retriever=retrieve_url,
            command="verify-live",
        )
    except Exception as exc:
        _verify_fail(f"live build failed: {exc}")

    evidence = detect_fixture_evidence(state)
    if evidence:
        _verify_fail("fixture markers detected in live verification.")

    try:
        run_auditor_node(
            run_id=run_id,
            state=ScenarioOpsState(),
            base_dir=base_dir,
            settings=settings,
            config=config,
        )
    except Exception as exc:
        _record_audit_findings(run_id, base_dir)
        _verify_fail(f"live audit failed: {exc}")

    _log_json("verify_ok", {"mode": "live"})

def _run_build_scenarios(args: argparse.Namespace) -> None:
    base_dir = Path(args.base_dir) if args.base_dir else None
    resume_from = getattr(args, "resume_from", None)
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    if resume_from and not args.run_id:
        latest = latest_run_id(base_dir)
        if not latest:
            raise ValueError("No run_id found to resume from. Run build-scenarios first.")
        run_id = latest
    user_params = {"scope": args.scope, "value": args.value, "horizon": args.horizon}
    if getattr(args, "company", None):
        user_params["company"] = args.company
    if getattr(args, "geography", None):
        user_params["geography"] = args.geography
    sources = _parse_sources(args.sources)
    overrides = _settings_overrides_from_args(args)
    
    # Enforce fixtures for demo if not specified
    if overrides.get("mode") == "demo" and not overrides.get("sources_policy"):
        overrides["sources_policy"] = "fixtures"

    settings = load_settings(overrides)
    use_fixtures = _use_fixtures(settings)
    if not sources and use_fixtures:
        sources = default_sources()
        _log_json("debug_sources", {"sources": sources, "origin": "default_fixtures"})
    else:
        _log_json("debug_sources", {"sources": sources, "origin": "args_or_empty"})

    inputs = GraphInputs(user_params=user_params, sources=sources, signals=[])
    command = "build-scenarios"
    try:
        if (
            getattr(args, "mode", None) == "live"
            and getattr(args, "allow_web", None) is False
        ):
            raise PermissionError("Network disabled (allow_web is False).")
        if settings.mode == "live" and not settings.allow_web:
            raise PermissionError("Network disabled (allow_web is False).")
        run_graph(
            inputs,
            run_id=run_id,
            base_dir=base_dir,
            mock_mode=use_fixtures,
            settings=settings,
            generate_strategies=not getattr(args, "no_strategies", False),
            retriever=retrieve_url,
            command=command,
            legacy_mode=getattr(args, "legacy_mode", False),
            resume_from=resume_from,
        )
        _write_latest(
            run_id=run_id,
            status="OK",
            command=command,
            base_dir=base_dir,
            settings=settings,
        )
    except Exception as exc:
        _write_latest(
            run_id=run_id,
            status="FAIL",
            command=command,
            error_summary=str(exc),
            base_dir=base_dir,
            settings=settings,
        )
        raise
    _print_result(run_id, base_dir)


def _run_add_strategies(args: argparse.Namespace) -> None:
    base_dir = Path(args.base_dir) if args.base_dir else None
    run_id = args.run_id or latest_run_id(base_dir)
    if not run_id:
        raise ValueError("No run_id found. Run build-scenarios first.")

    settings = _settings_for_run(
        run_id, base_dir, _settings_overrides_from_args(args)
    )
    config = llm_config_from_settings(settings)
    strategy_notes = Path(args.strategies_file).read_text(encoding="utf-8")
    state = state_for_strategies(run_id, base_dir)
    mock_payloads = (
        mock_payloads_for_sources(default_sources()) if _use_fixtures(settings) else None
    )

    state = run_strategies_node(
        run_id=run_id,
        state=state,
        strategy_notes=strategy_notes,
        llm_client=client_for_node(
            "strategies", mock_payloads=mock_payloads
        ),
        base_dir=base_dir,
        config=config,
    )
    state = run_wind_tunnel_node(
        run_id=run_id,
        state=state,
        llm_client=client_for_node(
            "wind_tunnel", mock_payloads=mock_payloads
        ),
        base_dir=base_dir,
        config=config,
    )
    run_auditor_node(
        run_id=run_id,
        state=state,
        base_dir=base_dir,
        settings=settings,
        config=config,
    )
    _print_result(run_id, base_dir)


def _run_daily(args: argparse.Namespace) -> None:
    base_dir = Path(args.base_dir) if args.base_dir else None
    run_id = args.run_id or latest_run_id(base_dir)
    if not run_id:
        raise ValueError("No run_id found. Run build-scenarios first.")

    settings = _settings_for_run(
        run_id, base_dir, _settings_overrides_from_args(args)
    )
    config = llm_config_from_settings(settings)
    state = state_for_daily(run_id, base_dir, allow_missing=True)
    signals = []
    if args.signals:
        signals = _load_json_value(args.signals)
    if not signals and _use_fixtures(settings):
        if state.ewi is not None:
            signals = ensure_signals(state.ewi.indicators)

    mock_payloads = (
        mock_payloads_for_sources(default_sources()) if _use_fixtures(settings) else None
    )
    command = "run-daily"
    try:
        state = run_daily_runner_node(
            signals,
            run_id=run_id,
            state=state,
            llm_client=client_for_node(
                "daily_runner", mock_payloads=mock_payloads
            ),
            base_dir=base_dir,
            config=config,
        )
        run_auditor_node(
            run_id=run_id,
            state=state,
            base_dir=base_dir,
            settings=settings,
            config=config,
        )
        _write_latest(
            run_id=run_id,
            status="OK",
            command=command,
            base_dir=base_dir,
            settings=settings,
        )
    except Exception as exc:
        _write_latest(
            run_id=run_id,
            status="FAIL",
            command=command,
            error_summary=str(exc),
            base_dir=base_dir,
            settings=settings,
        )
        raise
    _print_result(run_id, base_dir)


def _run_export_view(args: argparse.Namespace) -> None:
    base_dir = Path(args.base_dir) if args.base_dir else None
    run_id = args.run_id or latest_run_id(base_dir)
    if not run_id:
        raise ValueError("No run_id found. Run build-scenarios first.")

    output_path = _export_view_model(run_id, base_dir)
    _log_json("export_view", {"run_id": run_id, "view_model": str(output_path)})


def _add_settings_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--mode",
        choices=["demo", "live"],
        default=None,
        help="Override mode from config file.",
    )
    parser.add_argument(
        "--sources-policy",
        choices=["fixtures", "academic_only", "mixed_reputable"],
        default=None,
        help="Override sources policy from config file.",
    )
    parser.add_argument(
        "--allow-web",
        dest="allow_web",
        action="store_true",
        help="Allow network retrieval (overrides config).",
    )
    parser.add_argument(
        "--no-allow-web",
        dest="allow_web",
        action="store_false",
        help="Disable network retrieval (overrides config).",
    )
    parser.set_defaults(allow_web=None)
    parser.add_argument("--min-sources-per-domain", type=int, default=None)
    parser.add_argument("--min-citations-per-driver", type=int, default=None)
    parser.add_argument("--min-forces", type=int, default=None)
    parser.add_argument("--min-forces-per-domain", type=int, default=None)
    parser.add_argument(
        "--simulate-evidence",
        dest="simulate_evidence",
        action="store_true",
        help="Allow simulated evidence units (explicit opt-in).",
    )
    parser.add_argument(
        "--no-simulate-evidence",
        dest="simulate_evidence",
        action="store_false",
        help="Disallow simulated evidence units.",
    )
    parser.set_defaults(simulate_evidence=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--forbid-fixture-citations",
        dest="forbid_fixture_citations",
        action="store_true",
        help="Fail runs when fixture citations are detected.",
    )
    parser.add_argument(
        "--allow-fixture-citations",
        dest="forbid_fixture_citations",
        action="store_false",
        help="Allow fixture citations even in live mode.",
    )
    parser.set_defaults(forbid_fixture_citations=None)
    parser.add_argument("--gemini-model", default=None)
    parser.add_argument("--llm-provider", default=None)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="scenarioops")
    subparsers = parser.add_subparsers(dest="command", required=True)

    charter_parser = subparsers.add_parser("charter", help="Generate a scenario charter.")
    charter_parser.add_argument("--params", required=True, help="JSON string or path.")
    charter_parser.add_argument("--run-id", default=None)
    charter_parser.add_argument("--base-dir", default=None)
    charter_parser.add_argument("--mock-payload", default=None)
    charter_parser.set_defaults(handler=_run_charter)

    build_parser = subparsers.add_parser("build-scenarios")
    build_parser.add_argument("--scope", required=True, choices=["world", "region", "country"])
    build_parser.add_argument("--value", required=True)
    build_parser.add_argument("--company", default=None)
    build_parser.add_argument("--geography", default=None)
    build_parser.add_argument("--horizon", required=True, type=int)
    build_parser.add_argument("--run-id", default=None)
    build_parser.add_argument("--base-dir", default=None)
    build_parser.add_argument("--sources", default=None)
    build_parser.add_argument(
        "--resume-from",
        default=None,
        choices=[
            "charter",
            "focal_issue",
            "company_profile",
            "retrieval_real",
            "forces",
            "ebe_rank",
            "clusters",
            "uncertainty_axes",
            "scenarios",
            "strategies",
            "wind_tunnel",
            "auditor",
        ],
        help="Resume execution from a specific node using existing artifacts.",
    )
    build_parser.add_argument(
        "--no-strategies",
        action="store_true",
        help="Skip strategy and wind-tunnel generation.",
    )
    build_parser.add_argument(
        "--legacy-mode",
        action="store_true",
        help="Use legacy pipeline (pre-professionalization).",
    )
    build_parser.set_defaults(legacy_mode=False)
    _add_settings_args(build_parser)
    build_parser.set_defaults(handler=_run_build_scenarios)

    strategies_parser = subparsers.add_parser("add-strategies")
    strategies_parser.add_argument("strategies_file")
    strategies_parser.add_argument("--run-id", default=None)
    strategies_parser.add_argument("--base-dir", default=None)
    _add_settings_args(strategies_parser)
    strategies_parser.set_defaults(handler=_run_add_strategies)

    daily_parser = subparsers.add_parser("run-daily")
    daily_parser.add_argument("--run-id", default=None)
    daily_parser.add_argument("--base-dir", default=None)
    daily_parser.add_argument("--signals", default=None)
    _add_settings_args(daily_parser)
    daily_parser.set_defaults(handler=_run_daily)

    export_parser = subparsers.add_parser("export-view")
    export_parser.add_argument("--run-id", default=None)
    export_parser.add_argument("--base-dir", default=None)
    export_parser.set_defaults(handler=_run_export_view)

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--demo", action="store_true", help="Run offline demo checks.")
    verify_parser.add_argument("--live", action="store_true", help="Run live verification checks.")
    verify_parser.add_argument("--run-id", default=None)
    verify_parser.add_argument("--base-dir", default=None)
    verify_parser.add_argument("--sources", default=None, help="Comma-separated URLs for live verify.")
    verify_parser.set_defaults(handler=_run_verify)

    args = parser.parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
