from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from scenarioops.app.workflow import (
    ensure_signals,
    latest_run_id,
    list_artifacts,
    state_for_daily,
    state_for_strategies,
)
from scenarioops.graph.build_graph import (
    GraphInputs,
    client_for_node,
    default_sources,
    mock_payloads_for_sources,
    run_graph,
)
from scenarioops.graph.nodes import (
    run_auditor_node,
    run_daily_runner_node,
    run_strategies_node,
    run_wind_tunnel_node,
)
from scenarioops.graph.state import ScenarioOpsState
from scenarioops.graph.tools.scoring import hash_scoring_result, score_with_rubric
from scenarioops.graph.tools.storage import default_runs_dir, read_latest_status, write_artifact
from scenarioops.graph.tools.web_retriever import retrieve_url


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


def _print_result(run_id: str, base_dir: Path | None) -> None:
    artifacts = list_artifacts(run_id, base_dir)
    payload = {"run_id": run_id, "artifacts": artifacts}
    print(json.dumps(payload, indent=2))


def _runs_dir(base_dir: Path | None) -> Path:
    return base_dir if base_dir is not None else default_runs_dir()


def _verify_fail(message: str) -> None:
    print(f"verify failed: {message}")
    raise SystemExit(1)


def _run_verify(args: argparse.Namespace) -> None:
    if not args.demo:
        _verify_fail("verify requires --demo to run in mock mode.")

    base_dir = Path(args.base_dir) if args.base_dir else None
    run_id = args.run_id or f"verify-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
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
            mock_mode=True,
            generate_strategies=True,
            report_date="2026-01-01",
        )
    except Exception as exc:
        _verify_fail(f"mock build failed: {exc}")

    latest = read_latest_status(base_dir)
    if not latest or latest.get("run_id") != run_id:
        _verify_fail("latest.json missing or does not match run_id.")
    if latest.get("status") != "OK":
        _verify_fail(f"latest.json status is {latest.get('status')}.")

    try:
        run_auditor_node(run_id=run_id, state=ScenarioOpsState(), base_dir=base_dir)
    except Exception as exc:
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

    print("verify --demo ok")

def _run_build_scenarios(args: argparse.Namespace) -> None:
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    base_dir = Path(args.base_dir) if args.base_dir else None
    user_params = {"scope": args.scope, "value": args.value, "horizon": args.horizon}
    sources = _parse_sources(args.sources)
    if not sources and not args.live:
        sources = default_sources()

    inputs = GraphInputs(user_params=user_params, sources=sources, signals=[])
    if args.live:
        retriever = lambda url, **kwargs: retrieve_url(
            url, allow_network=True, public_demo_mode=False, **kwargs
        )
        run_graph(
            inputs,
            run_id=run_id,
            base_dir=base_dir,
            mock_mode=False,
            generate_strategies=False,
            retriever=retriever,
        )
    else:
        run_graph(
            inputs,
            run_id=run_id,
            base_dir=base_dir,
            mock_mode=True,
            generate_strategies=False,
        )
    _print_result(run_id, base_dir)


def _run_add_strategies(args: argparse.Namespace) -> None:
    base_dir = Path(args.base_dir) if args.base_dir else None
    run_id = args.run_id or latest_run_id(base_dir)
    if not run_id:
        raise ValueError("No run_id found. Run build-scenarios first.")

    strategy_notes = Path(args.strategies_file).read_text(encoding="utf-8")
    state = state_for_strategies(run_id, base_dir)
    mock_payloads = (
        mock_payloads_for_sources(default_sources()) if not args.live else None
    )

    state = run_strategies_node(
        run_id=run_id,
        state=state,
        strategy_notes=strategy_notes,
        llm_client=client_for_node(
            "strategies", mock_payloads=mock_payloads
        ),
        base_dir=base_dir,
    )
    state = run_wind_tunnel_node(
        run_id=run_id,
        state=state,
        llm_client=client_for_node(
            "wind_tunnel", mock_payloads=mock_payloads
        ),
        base_dir=base_dir,
    )
    run_auditor_node(run_id=run_id, state=state, base_dir=base_dir)
    _print_result(run_id, base_dir)


def _run_daily(args: argparse.Namespace) -> None:
    base_dir = Path(args.base_dir) if args.base_dir else None
    run_id = args.run_id or latest_run_id(base_dir)
    if not run_id:
        raise ValueError("No run_id found. Run build-scenarios first.")

    state = state_for_daily(run_id, base_dir, allow_missing=True)
    signals = []
    if args.signals:
        signals = _load_json_value(args.signals)
    if not signals and not args.live:
        if state.ewi is not None:
            signals = ensure_signals(state.ewi.indicators)

    mock_payloads = (
        mock_payloads_for_sources(default_sources()) if not args.live else None
    )
    state = run_daily_runner_node(
        signals,
        run_id=run_id,
        state=state,
        llm_client=client_for_node(
            "daily_runner", mock_payloads=mock_payloads
        ),
        base_dir=base_dir,
    )
    run_auditor_node(run_id=run_id, state=state, base_dir=base_dir)
    _print_result(run_id, base_dir)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="scenarioops-app")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build-scenarios")
    build_parser.add_argument("--scope", required=True, choices=["world", "region", "country"])
    build_parser.add_argument("--value", required=True)
    build_parser.add_argument("--horizon", required=True, type=int)
    build_parser.add_argument("--run-id", default=None)
    build_parser.add_argument("--base-dir", default=None)
    build_parser.add_argument("--sources", default=None)
    build_parser.add_argument(
        "--live",
        action="store_true",
        help="Use live LLM/retriever instead of mock payloads.",
    )
    build_parser.set_defaults(handler=_run_build_scenarios)

    strategies_parser = subparsers.add_parser("add-strategies")
    strategies_parser.add_argument("strategies_file")
    strategies_parser.add_argument("--run-id", default=None)
    strategies_parser.add_argument("--base-dir", default=None)
    strategies_parser.add_argument(
        "--live",
        action="store_true",
        help="Use live LLM instead of mock payloads.",
    )
    strategies_parser.set_defaults(handler=_run_add_strategies)

    daily_parser = subparsers.add_parser("run-daily")
    daily_parser.add_argument("--run-id", default=None)
    daily_parser.add_argument("--base-dir", default=None)
    daily_parser.add_argument("--signals", default=None)
    daily_parser.add_argument(
        "--live",
        action="store_true",
        help="Use live LLM instead of mock payloads.",
    )
    daily_parser.set_defaults(handler=_run_daily)

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--demo", action="store_true", help="Run offline demo checks.")
    verify_parser.add_argument("--run-id", default=None)
    verify_parser.add_argument("--base-dir", default=None)
    verify_parser.set_defaults(handler=_run_verify)

    args = parser.parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
