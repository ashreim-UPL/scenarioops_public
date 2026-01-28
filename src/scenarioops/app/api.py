from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

from scenarioops.app.config import llm_config_from_settings, load_settings
from scenarioops.app.auth import TenantContext, ensure_default_user, resolve_tenant
from scenarioops.app.workflow import (
    ensure_signals,
    latest_run_id,
    list_artifacts,
    state_for_daily,
    state_for_strategies,
)
from scenarioops.graph.types import GraphInputs
from scenarioops.graph.setup import (
    client_for_node,
    default_sources,
    mock_payloads_for_sources,
)
from scenarioops.graph.build_graph import run_graph
from scenarioops.graph.tools.web_retriever import retrieve_url
from scenarioops.graph.nodes import (
    run_auditor_node,
    run_daily_runner_node,
    run_strategies_node,
    run_wind_tunnel_node,
)


ensure_default_user()
app = FastAPI(title="ScenarioOps API")


class BuildRequest(BaseModel):
    scope: Literal["world", "region", "country"]
    value: str
    company: str | None = None
    geography: str | None = None
    horizon: int = Field(..., ge=36, le=120)
    sources: list[str] | None = None
    run_id: str | None = None
    mock: bool = True


class StrategiesRequest(BaseModel):
    run_id: str | None = None
    strategies_text: str | None = None
    mock: bool = True


class DailyRequest(BaseModel):
    run_id: str | None = None
    signals: list[dict[str, Any]] | None = None
    mock: bool = True


class RunResponse(BaseModel):
    run_id: str
    artifacts: list[str]


class LatestResponse(BaseModel):
    run_id: str
    daily_brief: dict[str, Any]
    links: list[str]


def _load_daily_brief(run_id: str, base_dir: Path) -> dict[str, Any]:
    artifacts_dir = base_dir / run_id / "artifacts"
    path = artifacts_dir / "daily_brief.json"
    if not path.exists():
        raise FileNotFoundError("daily_brief.json not found.")
    return json.loads(path.read_text(encoding="utf-8"))


def _latest_run_with_daily(base_dir: Path) -> str | None:
    runs_dir = base_dir
    if not runs_dir.exists():
        return None
    runs = [path for path in runs_dir.iterdir() if path.is_dir()]
    runs.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    for run in runs:
        if (run / "artifacts" / "daily_brief.json").exists():
            return run.name
    return None


def _artifact_path(run_id: str, artifact_name: str, base_dir: Path) -> Path:
    if "/" in artifact_name or "\\" in artifact_name:
        raise ValueError("Invalid artifact name.")
    return base_dir / run_id / "artifacts" / artifact_name


def _tenant_context(
    x_api_key: str | None = Header(default=None, alias="X-Api-Key"),
    authorization: str | None = Header(default=None),
) -> TenantContext:
    api_key = x_api_key
    if not api_key and authorization:
        parts = authorization.split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            api_key = parts[1]
    return resolve_tenant(api_key)


@app.post("/build", response_model=RunResponse)
def build(
    payload: BuildRequest, tenant: TenantContext = Depends(_tenant_context)
) -> RunResponse:
    run_id = payload.run_id or datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    user_params = {"scope": payload.scope, "value": payload.value, "horizon": payload.horizon}
    if payload.company:
        user_params["company"] = payload.company
    if payload.geography:
        user_params["geography"] = payload.geography
    sources = payload.sources or []
    overrides = {"mode": "demo" if payload.mock else "live"}
    if payload.mock:
        overrides["sources_policy"] = "fixtures"
        overrides["llm_provider"] = "mock"
    settings = load_settings(overrides)
    use_fixtures = settings.sources_policy == "fixtures"
    if not sources and use_fixtures:
        sources = default_sources()

    inputs = GraphInputs(
        user_params=user_params,
        sources=sources,
        signals=[],
        input_docs=[],
    )
    try:
        run_graph(
            inputs,
            run_id=run_id,
            mock_mode=use_fixtures,
            settings=settings,
            generate_strategies=False,
            retriever=retrieve_url,
            base_dir=tenant.base_dir,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return RunResponse(
        run_id=run_id, artifacts=list_artifacts(run_id, base_dir=tenant.base_dir)
    )


@app.get("/")
def index() -> FileResponse:
    ui_path = Path(__file__).with_name("ui.html")
    return FileResponse(ui_path)


@app.get("/artifact/{run_id}/{artifact_name}")
def artifact(
    run_id: str,
    artifact_name: str,
    tenant: TenantContext = Depends(_tenant_context),
):
    try:
        path = _artifact_path(run_id, artifact_name, tenant.base_dir)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not path.exists():
        raise HTTPException(status_code=404, detail="Artifact not found.")

    if path.suffix == ".json":
        return JSONResponse(json.loads(path.read_text(encoding="utf-8")))
    if path.suffix == ".jsonl":
        return PlainTextResponse(path.read_text(encoding="utf-8"))
    return PlainTextResponse(path.read_text(encoding="utf-8"))


@app.post("/strategies", response_model=RunResponse)
def strategies(
    payload: StrategiesRequest, tenant: TenantContext = Depends(_tenant_context)
) -> RunResponse:
    run_id = payload.run_id or latest_run_id(base_dir=tenant.base_dir)
    if not run_id:
        raise HTTPException(status_code=404, detail="No runs available.")

    overrides = {"mode": "demo" if payload.mock else "live"}
    if payload.mock:
        overrides["sources_policy"] = "fixtures"
        overrides["llm_provider"] = "mock"
    settings = load_settings(overrides)
    config = llm_config_from_settings(settings)
    mock_payloads = (
        mock_payloads_for_sources(default_sources()) if payload.mock else None
    )
    try:
        state = state_for_strategies(run_id, base_dir=tenant.base_dir)
        state = run_strategies_node(
            run_id=run_id,
            state=state,
            strategy_notes=payload.strategies_text or "",
            llm_client=client_for_node("strategies", mock_payloads=mock_payloads),
            config=config,
            base_dir=tenant.base_dir,
        )
        state = run_wind_tunnel_node(
            run_id=run_id,
            state=state,
            llm_client=client_for_node("wind_tunnel", mock_payloads=mock_payloads),
            config=config,
            base_dir=tenant.base_dir,
        )
        run_auditor_node(
            run_id=run_id,
            state=state,
            settings=settings,
            config=config,
            base_dir=tenant.base_dir,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return RunResponse(
        run_id=run_id, artifacts=list_artifacts(run_id, base_dir=tenant.base_dir)
    )


@app.post("/daily", response_model=RunResponse)
def daily(
    payload: DailyRequest, tenant: TenantContext = Depends(_tenant_context)
) -> RunResponse:
    run_id = payload.run_id or latest_run_id(base_dir=tenant.base_dir)
    if not run_id:
        raise HTTPException(status_code=404, detail="No runs available.")

    overrides = {"mode": "demo" if payload.mock else "live"}
    if payload.mock:
        overrides["sources_policy"] = "fixtures"
        overrides["llm_provider"] = "mock"
    settings = load_settings(overrides)
    config = llm_config_from_settings(settings)
    mock_payloads = (
        mock_payloads_for_sources(default_sources()) if payload.mock else None
    )
    try:
        state = state_for_daily(run_id, base_dir=tenant.base_dir)
        signals = payload.signals or []
        if not signals and payload.mock:
            signals = ensure_signals(state.ewi.indicators)
        state = run_daily_runner_node(
            signals,
            run_id=run_id,
            state=state,
            llm_client=client_for_node("daily_runner", mock_payloads=mock_payloads),
            config=config,
            base_dir=tenant.base_dir,
        )
        run_auditor_node(
            run_id=run_id,
            state=state,
            settings=settings,
            config=config,
            base_dir=tenant.base_dir,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return RunResponse(
        run_id=run_id, artifacts=list_artifacts(run_id, base_dir=tenant.base_dir)
    )


@app.get("/latest", response_model=LatestResponse)
def latest(tenant: TenantContext = Depends(_tenant_context)) -> LatestResponse:
    run_id = _latest_run_with_daily(tenant.base_dir) or latest_run_id(
        base_dir=tenant.base_dir
    )
    if not run_id:
        raise HTTPException(status_code=404, detail="No runs available.")
    try:
        daily_brief = _load_daily_brief(run_id, tenant.base_dir)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return LatestResponse(
        run_id=run_id,
        daily_brief=daily_brief,
        links=list_artifacts(run_id, base_dir=tenant.base_dir),
    )
