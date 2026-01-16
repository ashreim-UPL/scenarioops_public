from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scenarioops.app.config import (
    ScenarioOpsSettings,
    load_settings,
    llm_config_from_settings
)
from scenarioops.graph.types import GraphInputs
from scenarioops.graph.setup import default_sources, mock_payloads_for_sources, apply_node_result, client_for_node
from scenarioops.squad.orchestrator import run_squad_orchestrator
from scenarioops.graph.tools.storage import (
    default_runs_dir,
    write_latest_status,
    ensure_run_dirs,
)
from scenarioops.graph.tools.view_model import build_view_model
from scenarioops.graph.tools.storage import write_artifact

# Mapping old function signature to new Squad Orchestrator
def run_graph(
    inputs: GraphInputs,
    *,
    run_id: str | None = None,
    base_dir: Path | None = None,
    state: Any | None = None, 
    llm_client: Any | None = None, 
    retriever: Any = None, 
    config: Any | None = None,
    mock_mode: bool = False,
    settings: ScenarioOpsSettings | None = None,
    settings_overrides: dict[str, Any] | None = None,
    generate_strategies: bool = True,
    report_date: str | None = None,
    command: str | None = None,
) -> Any:
    """
    Legacy entry point mapped to Dynamic Strategy Squad.
    """
    if run_id is None:
        from datetime import datetime, timezone
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    
    # Ensure settings if not passed
    if settings is None:
        overrides = dict(settings_overrides or {})
        if mock_mode:
            overrides["mode"] = "demo"
            overrides["sources_policy"] = "fixtures"
            overrides["llm_provider"] = "mock"
        settings = load_settings(overrides)

    ensure_run_dirs(run_id, base_dir=base_dir)

    try:
        # Execute Squad Orchestrator
        final_state = run_squad_orchestrator(
            inputs,
            run_id=run_id,
            base_dir=base_dir,
            mock_mode=mock_mode,
            generate_strategies=generate_strategies
        )

        runs_dir = base_dir if base_dir is not None else default_runs_dir()
        run_dir = runs_dir / run_id
        if run_dir.exists():
            view_model = build_view_model(run_dir)
            write_artifact(
                run_id=run_id,
                artifact_name="view_model",
                payload=view_model,
                ext="json",
                base_dir=base_dir,
            )

        write_latest_status(
            run_id=run_id,
            status="OK",
            command=command or "run-graph",
            base_dir=base_dir,
            run_config=settings.as_dict() if settings else None,
        )

        return final_state

    except Exception as exc:
        write_latest_status(
            run_id=run_id,
            status="FAIL",
            command=command or "run-graph",
            error_summary=str(exc),
            base_dir=base_dir,
            run_config=settings.as_dict() if settings else None,
        )
        raise
