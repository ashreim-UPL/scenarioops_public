"""ScenarioOps module entrypoint."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from scenarioops.graph.build_graph import GraphInputs, run_graph
from scenarioops.graph.nodes.charter import run_charter_node
from scenarioops.graph.state import ScenarioOpsState
from scenarioops.llm.client import MockLLMClient


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


def _run_charter(args: argparse.Namespace) -> None:
    user_params = _load_json_value(args.params)
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    base_dir = Path(args.base_dir) if args.base_dir else None
    llm_client = None
    if args.mock_payload:
        mock_payload = _load_json_value(args.mock_payload)
        llm_client = MockLLMClient(json_payload=mock_payload)

    run_charter_node(
        user_params,
        run_id=run_id,
        state=ScenarioOpsState(),
        llm_client=llm_client,
        base_dir=base_dir,
    )


def _run_workflow(args: argparse.Namespace) -> None:
    if not args.params and not args.mock:
        raise ValueError("--params is required unless --mock is set.")
    user_params = _load_json_value(args.params) if args.params else {}
    sources = _parse_sources(args.sources)
    signals = _load_json_value(args.signals) if args.signals else []
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    base_dir = Path(args.base_dir) if args.base_dir else None

    inputs = GraphInputs(
        user_params=user_params,
        sources=sources,
        signals=signals if isinstance(signals, list) else [],
    )
    run_graph(
        inputs,
        run_id=run_id,
        base_dir=base_dir,
        mock_mode=args.mock,
        generate_strategies=not args.no_strategies,
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="scenarioops")
    subparsers = parser.add_subparsers(dest="command", required=True)

    charter_parser = subparsers.add_parser("charter", help="Generate a scenario charter.")
    charter_parser.add_argument("--params", required=True, help="JSON string or path.")
    charter_parser.add_argument("--run-id", default=None, help="Run identifier.")
    charter_parser.add_argument(
        "--base-dir",
        default=None,
        help="Base directory for runs (defaults to storage/runs).",
    )
    charter_parser.add_argument(
        "--mock-payload",
        default=None,
        help="JSON string or path for mock LLM output.",
    )
    charter_parser.set_defaults(handler=_run_charter)

    run_parser = subparsers.add_parser("run", help="Run the full ScenarioOps graph.")
    run_parser.add_argument("--params", default=None, help="JSON string or path.")
    run_parser.add_argument(
        "--sources",
        default=None,
        help="Comma-separated list of URLs or JSON array string.",
    )
    run_parser.add_argument(
        "--signals",
        default=None,
        help="JSON string or path for indicator signals.",
    )
    run_parser.add_argument("--run-id", default=None, help="Run identifier.")
    run_parser.add_argument(
        "--base-dir",
        default=None,
        help="Base directory for runs (defaults to storage/runs).",
    )
    run_parser.add_argument(
        "--mock",
        action="store_true",
        help="Run with mock payloads and offline retriever.",
    )
    run_parser.add_argument(
        "--no-strategies",
        action="store_true",
        help="Skip strategies and downstream wind-tunnel/daily runner.",
    )
    run_parser.set_defaults(handler=_run_workflow)

    args = parser.parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
