from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scenarioops.app.config import (
    ScenarioOpsSettings,
    apply_overrides,
    load_settings,
    llm_config_from_settings,
)
from scenarioops.graph.types import GraphInputs
from scenarioops.graph.setup import default_sources, mock_payloads_for_sources, apply_node_result, client_for_node
from scenarioops.squad.orchestrator import run_squad_orchestrator
from scenarioops.graph.tools.storage import (
    default_runs_dir,
    write_latest_status,
    ensure_run_dirs,
    register_run_timestamp,
    write_run_config,
    update_run_json,
)
from scenarioops.graph.tools.view_model import build_view_model
from scenarioops.graph.tools.storage import write_artifact
from scenarioops.graph.tools.prompts import build_prompt_manifest
from scenarioops.graph.tools.traceability import build_run_metadata
from scenarioops.graph.tools.run_manifest import write_artifact_index

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
    run_timestamp: str | None = None,
    command: str | None = None,
    legacy_mode: bool = False,
    resume_from: str | None = None,
) -> Any:
    """
    Legacy entry point mapped to Dynamic Strategy Squad.
    """
    if run_id is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    
    # Ensure settings if not passed
    if settings is None:
        overrides = dict(settings_overrides or {})
        if mock_mode:
            overrides["mode"] = "demo"
            overrides["sources_policy"] = "fixtures"
            overrides["llm_provider"] = "mock"
        settings = load_settings(overrides)
    elif mock_mode:
        if (
            settings.mode != "demo"
            or settings.sources_policy != "fixtures"
            or settings.llm_provider != "mock"
        ):
            settings = apply_overrides(
                settings,
                {
                    "mode": "demo",
                    "sources_policy": "fixtures",
                    "llm_provider": "mock",
                },
            )

    dirs = ensure_run_dirs(run_id, base_dir=base_dir)
    created_at = run_timestamp or datetime.now(timezone.utc).isoformat()
    existing_config = (dirs["run_dir"] / "run_config.json")
    if existing_config.exists():
        try:
            payload = json.loads(existing_config.read_text(encoding="utf-8"))
        except Exception:
            payload = None
        if isinstance(payload, dict):
            existing_created = payload.get("created_at")
            if isinstance(existing_created, str) and existing_created:
                created_at = existing_created
    register_run_timestamp(run_id, created_at)
    update_run_json(
        run_id=run_id,
        updates={
            "run_id": run_id,
            "status": "RUNNING",
            "is_final": False,
            "created_at": created_at,
            "updated_at": created_at,
        },
        base_dir=base_dir,
    )
    prompt_manifest = build_prompt_manifest()
    prompt_manifest_hash = hashlib.sha256(
        json.dumps(prompt_manifest, sort_keys=True).encode("utf-8")
    ).hexdigest()
    run_config_payload = {
        "run_id": run_id,
        "created_at": created_at,
        "legacy_mode": legacy_mode,
        "resume_from": resume_from,
        "user_params": dict(inputs.user_params or {}),
        "generate_strategies": bool(generate_strategies),
        "settings": settings.as_dict() if settings else {},
        "models": {
            "llm_model": settings.llm_model if settings else None,
            "search_model": settings.search_model if settings else None,
            "summarizer_model": settings.summarizer_model if settings else None,
            "embed_model": settings.embed_model if settings else None,
            "image_model": settings.image_model if settings else None,
        },
        "prompt_manifest_sha256": prompt_manifest_hash,
        "retriever": {
            "allow_web": bool(settings.allow_web) if settings else False,
            "sources_policy": settings.sources_policy if settings else None,
            "source_count": len(inputs.sources or []),
        },
    }
    write_run_config(run_id=run_id, run_config=run_config_payload, base_dir=base_dir)
    prompt_manifest.update(
        build_run_metadata(
            run_id=run_id,
            user_params=inputs.user_params,
            timestamp=created_at,
        )
    )
    write_artifact(
        run_id=run_id,
        artifact_name="prompt_manifest",
        payload=prompt_manifest,
        ext="json",
        input_values={"prompt_count": len(prompt_manifest.get("prompts", []))},
        tool_versions={"prompt_manifest": "0.1.0"},
        base_dir=base_dir,
    )

    try:
        # Execute Squad Orchestrator
        final_state = run_squad_orchestrator(
            inputs,
            run_id=run_id,
            base_dir=base_dir,
            mock_mode=mock_mode,
            generate_strategies=generate_strategies,
            legacy_mode=legacy_mode,
            settings=settings,
            resume_from=resume_from,
            report_date=report_date,
        )

        write_latest_status(
            run_id=run_id,
            status="OK",
            command=command or "run-graph",
            base_dir=base_dir,
            run_config=settings.as_dict() if settings else None,
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
            write_artifact_index(run_id=run_id, base_dir=base_dir, strict=True)

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
