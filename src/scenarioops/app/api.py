from __future__ import annotations

import json
import mimetypes
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

from scenarioops.app.config import llm_config_from_settings, load_settings
from scenarioops.app.auth import TenantContext, ensure_default_user, resolve_tenant
from scenarioops.app.tenant_config import get_tenant_config, update_tenant_config
from scenarioops.app.workflow import (
    ensure_signals,
    latest_run_id,
    list_artifacts,
    list_run_ids,
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
from scenarioops.graph.tools.injection_defense import strip_instruction_patterns
from scenarioops.graph.tools.vectordb import open_run_vector_store
from scenarioops.llm.client import get_llm_client
from scenarioops.graph.tools.view_model import build_view_model
from scenarioops.graph.nodes import (
    run_auditor_node,
    run_daily_runner_node,
    run_strategies_node,
    run_wind_tunnel_node,
)


ensure_default_user()
try:
    import multipart  # type: ignore
    _MULTIPART_AVAILABLE = True
except Exception:
    _MULTIPART_AVAILABLE = False

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
    resume_from: str | None = None
    input_docs: list[str] | None = None
    generate_strategies: bool = True
    settings_overrides: dict[str, Any] | None = None


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
    run_id: str | None = None
    daily_brief: dict[str, Any] = Field(default_factory=dict)
    links: list[str] = Field(default_factory=list)


class RunsResponse(BaseModel):
    runs: list[str]
    labels: dict[str, str] = Field(default_factory=dict)


class ChatRequest(BaseModel):
    run_id: str | None = None
    question: str = Field(..., min_length=3)
    top_k: int = Field(default=6, ge=1, le=15)


class ChatResponse(BaseModel):
    run_id: str
    label: str
    answer: str
    sources: list[dict[str, Any]]


class LogResponse(BaseModel):
    entries: list[dict[str, Any]]


def _load_daily_brief(run_id: str, base_dir: Path) -> dict[str, Any]:
    artifacts_dir = base_dir / run_id / "artifacts"
    path = artifacts_dir / "daily_brief.json"
    if not path.exists():
        return {}
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


def _safe_image_path(run_id: str, image_name: str, base_dir: Path) -> Path:
    if "/" in image_name or "\\" in image_name or ".." in image_name:
        raise ValueError("Invalid image name.")
    return base_dir / run_id / "artifacts" / "images" / image_name


def _safe_load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _run_label(run_id: str, base_dir: Path) -> str:
    artifacts_dir = base_dir / run_id / "artifacts"
    prompt = _safe_load_json(artifacts_dir / "prompt_manifest.json") or {}
    company_profile = _safe_load_json(artifacts_dir / "company_profile.json") or {}
    focal_issue = _safe_load_json(artifacts_dir / "focal_issue.json") or {}
    scope = focal_issue.get("scope") if isinstance(focal_issue.get("scope"), dict) else {}

    company = (
        company_profile.get("company_name")
        or prompt.get("company_name")
        or focal_issue.get("company_name")
        or prompt.get("value")
        or run_id
    )
    geography = (
        scope.get("geography")
        or focal_issue.get("geography")
        or prompt.get("geography")
        or company_profile.get("geography")
        or "Global"
    )
    horizon = (
        scope.get("time_horizon_months")
        or focal_issue.get("time_horizon_months")
        or prompt.get("horizon_months")
        or 60
    )
    try:
        horizon_value = int(horizon)
    except (TypeError, ValueError):
        horizon_value = 60
    return f"{company} | SP{horizon_value} | {geography}"


def _truncate_text(value: str, limit: int) -> str:
    if not value:
        return ""
    return value if len(value) <= limit else f"{value[: max(0, limit - 1)].strip()}…"


def _compact_forces(forces: list[dict[str, Any]], limit: int = 10) -> list[dict[str, Any]]:
    compacted: list[dict[str, Any]] = []
    for force in forces[:limit]:
        if not isinstance(force, dict):
            continue
        compacted.append(
            {
                "label": force.get("label") or force.get("name") or force.get("force"),
                "domain": force.get("domain"),
                "layer": force.get("layer"),
                "mechanism": _truncate_text(str(force.get("mechanism") or force.get("description") or ""), 220),
                "confidence": force.get("confidence"),
            }
        )
    return compacted


def _compact_scenarios(scenarios: list[dict[str, Any]], limit: int = 6) -> list[dict[str, Any]]:
    compacted: list[dict[str, Any]] = []
    for scenario in scenarios[:limit]:
        if not isinstance(scenario, dict):
            continue
        compacted.append(
            {
                "name": scenario.get("name") or scenario.get("scenario_name") or scenario.get("scenario_id"),
                "summary": _truncate_text(
                    str(scenario.get("summary") or scenario.get("narrative") or scenario.get("story_text") or ""),
                    280,
                ),
                "axis_states": scenario.get("axis_states") or {},
            }
        )
    return compacted


def _build_chat_context(run_id: str, base_dir: Path) -> dict[str, Any]:
    artifacts_dir = base_dir / run_id / "artifacts"
    view_model = _safe_load_json(artifacts_dir / "view_model.json") or {}
    prompt_manifest = view_model.get("prompt_manifest") or _safe_load_json(artifacts_dir / "prompt_manifest.json") or {}
    focal_issue = view_model.get("focal_issue") or _safe_load_json(artifacts_dir / "focal_issue.json") or {}
    company_profile = view_model.get("company_profile") or _safe_load_json(artifacts_dir / "company_profile.json") or {}
    forces = view_model.get("forces") or []
    if not isinstance(forces, list):
        forces = []
    driving_forces = view_model.get("driving_forces") or []
    if not isinstance(driving_forces, list):
        driving_forces = []
    scenarios = view_model.get("scenarios") or []
    if not isinstance(scenarios, list):
        scenarios = []
    strategies_payload = view_model.get("strategies") or []
    if not isinstance(strategies_payload, list):
        strategies_payload = []
    wind_tunnel = view_model.get("wind_tunnel_evaluations_v2") or {}
    if not isinstance(wind_tunnel, dict):
        wind_tunnel = {}
    scope = focal_issue.get("scope") if isinstance(focal_issue.get("scope"), dict) else {}

    context = {
        "run_id": run_id,
        "run_label": _run_label(run_id, base_dir),
        "company": {
            "name": company_profile.get("company_name") or prompt_manifest.get("company_name"),
            "industry": company_profile.get("industry"),
            "summary": _truncate_text(
                str(company_profile.get("summary") or company_profile.get("description") or ""), 360
            ),
            "geography": company_profile.get("geography") or prompt_manifest.get("geography"),
        },
        "focal_issue": {
            "focal_issue": focal_issue.get("focal_issue"),
            "purpose": focal_issue.get("purpose"),
            "scope": {
                "geography": scope.get("geography"),
                "time_horizon_months": scope.get("time_horizon_months"),
            },
        },
        "forces": _compact_forces(forces),
        "driving_forces": _compact_forces(driving_forces),
        "scenarios": _compact_scenarios(scenarios),
        "strategies": [
            {
                "name": item.get("strategy_name") or item.get("name") or item.get("strategy_id"),
                "summary": _truncate_text(str(item.get("summary") or item.get("description") or ""), 220),
            }
            for item in strategies_payload[:6]
            if isinstance(item, dict)
        ],
        "wind_tunnel": {
            "recommendation": (wind_tunnel.get("recommendations") or {}).get("primary_recommended_strategy"),
            "rankings": (wind_tunnel.get("rankings") or {}).get("overall", [])[:3],
        },
    }
    return context


def _safe_log_path(run_id: str, log_name: str, base_dir: Path) -> Path:
    if "/" in log_name or "\\" in log_name:
        raise ValueError("Invalid log name.")
    logs_dir = base_dir / run_id / "logs"
    return logs_dir / log_name


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


def _require_admin(tenant: TenantContext = Depends(_tenant_context)) -> TenantContext:
    if not tenant.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required.")
    return tenant


class ConfigResponse(BaseModel):
    settings: dict[str, Any]
    is_admin: bool


class ConfigUpdateRequest(BaseModel):
    settings: dict[str, Any]


@app.get("/config", response_model=ConfigResponse)
def config_get(tenant: TenantContext = Depends(_tenant_context)) -> ConfigResponse:
    settings = get_tenant_config(tenant.tenant_id)
    return ConfigResponse(settings=settings, is_admin=tenant.is_admin)


@app.post("/config", response_model=ConfigResponse)
def config_update(
    payload: ConfigUpdateRequest, tenant: TenantContext = Depends(_require_admin)
) -> ConfigResponse:
    update_tenant_config(tenant.tenant_id, payload.settings)
    return ConfigResponse(settings=get_tenant_config(tenant.tenant_id), is_admin=True)


@app.post("/build", response_model=RunResponse)
def build(
    payload: BuildRequest, tenant: TenantContext = Depends(_tenant_context)
) -> RunResponse:
    run_id = payload.run_id or datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    if payload.resume_from and not payload.run_id:
        raise HTTPException(status_code=400, detail="run_id required when resuming.")
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
    tenant_overrides = get_tenant_config(tenant.tenant_id)
    settings = load_settings(
        {**tenant_overrides, **overrides, **(payload.settings_overrides or {})}
    )
    use_fixtures = settings.sources_policy == "fixtures"
    if not sources:
        sources = default_sources()

    inputs = GraphInputs(
        user_params=user_params,
        sources=sources,
        signals=[],
        input_docs=payload.input_docs or [],
    )
    try:
        run_graph(
            inputs,
            run_id=run_id,
            mock_mode=use_fixtures,
            settings=settings,
            generate_strategies=payload.generate_strategies,
            retriever=retrieve_url,
            base_dir=tenant.base_dir,
            resume_from=payload.resume_from,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return RunResponse(
        run_id=run_id, artifacts=list_artifacts(run_id, base_dir=tenant.base_dir)
    )


def _commercial_ui() -> FileResponse:
    ui_path = Path(__file__).with_name("commercial_ui.html")
    return FileResponse(ui_path)


@app.get("/")
def index() -> FileResponse:
    return _commercial_ui()


@app.get("/ops")
def ops_ui() -> FileResponse:
    ui_path = Path(__file__).with_name("ui.html")
    return FileResponse(ui_path)


@app.get("/intelligence")
def intelligence_ui() -> FileResponse:
    return _commercial_ui()


@app.get("/scenarios")
def scenarios_ui() -> FileResponse:
    return _commercial_ui()


@app.get("/wind-tunnel")
def wind_tunnel_ui() -> FileResponse:
    return _commercial_ui()


@app.get("/actions")
def actions_ui() -> FileResponse:
    return _commercial_ui()


@app.get("/chat")
def chat_ui() -> FileResponse:
    return _commercial_ui()


@app.get("/runs/{run_id}")
def run_ui(run_id: str) -> FileResponse:
    return _commercial_ui()


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
        if artifact_name == "view_model.json":
            run_dir = tenant.base_dir / run_id
            if run_dir.exists():
                try:
                    view_model = build_view_model(run_dir)
                except Exception as exc:
                    raise HTTPException(status_code=500, detail=str(exc)) from exc
                return JSONResponse(view_model)
        raise HTTPException(status_code=404, detail="Artifact not found.")

    if path.suffix == ".json":
        return JSONResponse(json.loads(path.read_text(encoding="utf-8")))
    if path.suffix == ".jsonl":
        return PlainTextResponse(path.read_text(encoding="utf-8"))
    return PlainTextResponse(path.read_text(encoding="utf-8"))


@app.get("/artifact/{run_id}/image/{image_name}")
def artifact_image(
    run_id: str,
    image_name: str,
    tenant: TenantContext = Depends(_tenant_context),
) -> FileResponse:
    try:
        path = _safe_image_path(run_id, image_name, tenant.base_dir)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not path.exists():
        raise HTTPException(status_code=404, detail="Image not found.")
    media_type, _ = mimetypes.guess_type(str(path))
    return FileResponse(path, media_type=media_type or "application/octet-stream")


@app.get("/runs", response_model=RunsResponse)
def runs(tenant: TenantContext = Depends(_tenant_context)) -> RunsResponse:
    run_ids = list_run_ids(base_dir=tenant.base_dir)
    labels: dict[str, str] = {}
    for run_id in run_ids:
        try:
            labels[run_id] = _run_label(run_id, tenant.base_dir)
        except Exception:
            labels[run_id] = run_id
    return RunsResponse(runs=run_ids, labels=labels)


@app.post("/chat", response_model=ChatResponse)
def chat(
    payload: ChatRequest, tenant: TenantContext = Depends(_tenant_context)
) -> ChatResponse:
    run_id = payload.run_id or latest_run_id(base_dir=tenant.base_dir)
    if not run_id:
        raise HTTPException(status_code=404, detail="No runs available.")
    question = strip_instruction_patterns(payload.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is required.")

    tenant_overrides = get_tenant_config(tenant.tenant_id)
    settings = load_settings(tenant_overrides)
    config = llm_config_from_settings(settings)
    try:
        client = get_llm_client(config)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    context = _build_chat_context(run_id, tenant.base_dir)
    sources: list[dict[str, Any]] = []

    vector_text = ""
    try:
        vector_store = open_run_vector_store(
            run_id, base_dir=tenant.base_dir, embed_model=settings.embed_model, seed=int(settings.seed or 0)
        )
        matches = vector_store.query(question, top_k=payload.top_k)
        filtered: list[dict[str, Any]] = []
        for match in matches:
            metadata = match.metadata if isinstance(match.metadata, dict) else {}
            run_match = metadata.get("run_id")
            if run_match and str(run_match) != str(run_id):
                continue
            filtered.append(
                {
                    "doc_id": match.doc_id,
                    "score": round(float(match.score), 4),
                    "text": _truncate_text(str(match.text), 600),
                    "metadata": metadata,
                }
            )
        if filtered:
            vector_text = "\n".join(
                f"[vectordb:{item['doc_id']}] score={item['score']} text={item['text']} meta={item['metadata']}"
                for item in filtered
            )
            sources.extend(
                {
                    "type": "vectordb",
                    "id": item["doc_id"],
                    "score": item["score"],
                    "metadata": item["metadata"],
                }
                for item in filtered
            )
    except Exception as exc:
        sources.append({"type": "vectordb_error", "detail": str(exc)})

    prompt = (
        "You are ScenarioOps Chat. Use ONLY the provided context (Artifacts + Vector Evidence). "
        "If the answer is not available in the context, say so explicitly. "
        "Cite sources using tags like [artifact:company_profile] or [vectordb:doc_id] after claims.\n\n"
        f"Context (JSON):\n{json.dumps(context, indent=2)}\n\n"
        f"Vector Evidence:\n{vector_text or 'None'}\n\n"
        f"Question: {question}\n"
    )

    answer = client.generate_markdown(prompt)
    return ChatResponse(run_id=run_id, label=context.get("run_label", run_id), answer=answer, sources=sources)


@app.get("/run/{run_id}/node_events", response_model=LogResponse)
def node_events(
    run_id: str, tenant: TenantContext = Depends(_tenant_context)
) -> LogResponse:
    try:
        path = _safe_log_path(run_id, "node_events.jsonl", tenant.base_dir)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not path.exists():
        return LogResponse(entries=[])
    lines = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return LogResponse(entries=[item for item in lines if isinstance(item, dict)])


@app.get("/run/{run_id}/normalization", response_model=LogResponse)
def normalization_logs(
    run_id: str, tenant: TenantContext = Depends(_tenant_context)
) -> LogResponse:
    try:
        path = _safe_log_path(run_id, "normalization.jsonl", tenant.base_dir)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not path.exists():
        return LogResponse(entries=[])
    lines = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return LogResponse(entries=[item for item in lines if isinstance(item, dict)])


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
    tenant_overrides = get_tenant_config(tenant.tenant_id)
    settings = load_settings({**tenant_overrides, **overrides})
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
    tenant_overrides = get_tenant_config(tenant.tenant_id)
    settings = load_settings({**tenant_overrides, **overrides})
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
        return LatestResponse()
    try:
        daily_brief = _load_daily_brief(run_id, tenant.base_dir)
    except FileNotFoundError:
        daily_brief = {}
    return LatestResponse(
        run_id=run_id,
        daily_brief=daily_brief,
        links=list_artifacts(run_id, base_dir=tenant.base_dir),
    )


if _MULTIPART_AVAILABLE:
    from fastapi import File, UploadFile

    @app.post("/upload")
    async def upload_files(
        files: list[UploadFile] = File(...),
        tenant: TenantContext = Depends(_tenant_context),
    ):
        uploaded_paths: list[str] = []
        inputs_dir = tenant.base_dir / "uploads"
        inputs_dir.mkdir(parents=True, exist_ok=True)
        for file in files:
            if not file.filename:
                continue
            safe_name = Path(file.filename).name
            target = inputs_dir / safe_name
            counter = 1
            while target.exists():
                target = inputs_dir / f"{target.stem}-{counter}{target.suffix}"
                counter += 1
            content = await file.read()
            target.write_bytes(content)
            uploaded_paths.append(str(target))
        return {"files": uploaded_paths}
else:

    @app.post("/upload")
    async def upload_files_unavailable(
        tenant: TenantContext = Depends(_tenant_context),
    ):
        raise HTTPException(
            status_code=503,
            detail="Upload requires python-multipart. Ask your admin to install it.",
        )
