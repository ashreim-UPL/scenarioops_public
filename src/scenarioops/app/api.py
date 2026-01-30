from __future__ import annotations

import json
import mimetypes
import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Mapping

from fastapi import Depends, FastAPI, Header, HTTPException, Request
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
from scenarioops.graph.tools.prompts import build_prompt_manifest
from scenarioops.graph.tools.storage import (
    ensure_run_dirs,
    write_run_config,
    ensure_local_file,
    read_run_json,
    update_run_json,
    write_run_json,
    write_run_inputs,
    write_artifact,
)
from scenarioops.security.api_keys import resolve_api_key
from scenarioops.storage.run_store import run_store_mode, runs_prefix, runs_root
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
    pack: str | None = None
    sources: list[str] | None = None
    run_id: str | None = None
    mock: bool = True
    resume_from: str | None = None
    input_docs: list[str] | None = None
    generate_strategies: bool = True
    settings_overrides: dict[str, Any] | None = None
    force: bool = False
    rerun: bool = False


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
    reused: bool = False
    reuse_reason: str | None = None
    existing_run_id: str | None = None
    signature: str | None = None


class RunDeleteResponse(BaseModel):
    run_id: str
    deleted: bool


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


class ActionConsolePayload(BaseModel):
    run_id: str | None = None
    updated_at: str | None = None
    items: list[dict[str, Any]] = Field(default_factory=list)


class LogResponse(BaseModel):
    entries: list[dict[str, Any]]


def _load_daily_brief(run_id: str, base_dir: Path) -> dict[str, Any]:
    artifacts_dir = base_dir / run_id / "artifacts"
    path = artifacts_dir / "daily_brief.json"
    ensure_local_file(path, base_dir=base_dir)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _latest_run_with_daily(base_dir: Path) -> str | None:
    if run_store_mode() == "gcs":
        for run_id in list_run_ids(base_dir=base_dir, include_deleted=False):
            path = base_dir / run_id / "artifacts" / "daily_brief.json"
            ensure_local_file(path, base_dir=base_dir)
            if path.exists():
                return run_id
        return None
    runs_dir = base_dir
    if not runs_dir.exists():
        return None
    runs = [path for path in runs_dir.iterdir() if path.is_dir()]
    runs.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    for run in runs:
        run_meta = _safe_load_json(run / "run.json") or _safe_load_json(run / "run_meta.json") or {}
        if isinstance(run_meta, dict) and run_meta.get("is_deleted"):
            continue
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
    ensure_local_file(path)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _normalize_text(value: str | None) -> str:
    if not value:
        return ""
    return " ".join(value.strip().lower().split())


def _settings_dict(settings: Any) -> dict[str, Any]:
    if hasattr(settings, "as_dict"):
        try:
            return settings.as_dict()
        except Exception:
            return {}
    if isinstance(settings, Mapping):
        return dict(settings)
    return {}


def _prompt_manifest_sha256() -> str:
    manifest = build_prompt_manifest()
    return hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _build_signature_payload(
    *,
    company: str | None,
    geography: str | None,
    scope: str | None,
    value: str | None,
    horizon: int | None,
    pack: str | None,
    settings: Any,
    models: Mapping[str, Any] | None = None,
    generate_strategies: bool | None = None,
    prompt_manifest_sha256: str | None = None,
    source_count: int | None = None,
) -> dict[str, Any]:
    settings_payload = _settings_dict(settings)
    models_payload = dict(models or {})
    if not models_payload:
        models_payload = {
            "llm_model": settings_payload.get("llm_model") or settings_payload.get("gemini_model"),
            "search_model": settings_payload.get("search_model"),
            "summarizer_model": settings_payload.get("summarizer_model"),
            "embed_model": settings_payload.get("embed_model"),
            "image_model": settings_payload.get("image_model"),
        }
    normalized_models = {
        key: _normalize_text(str(value)) if value is not None else ""
        for key, value in models_payload.items()
    }
    return {
        "company": _normalize_text(company),
        "geography": _normalize_text(geography),
        "scope": _normalize_text(scope),
        "value": _normalize_text(value),
        "horizon_months": int(horizon or 0),
        "pack": _normalize_text(pack),
        "pipeline_version": _normalize_text(os.environ.get("SCENARIOOPS_PIPELINE_VERSION", "v1")),
        "app_version": _normalize_text(os.environ.get("SCENARIOOPS_VERSION", "")),
        "scenario_config_version": _normalize_text(os.environ.get("SCENARIOOPS_SCENARIO_VERSION", "")),
        "pack_version": _normalize_text(os.environ.get("SCENARIOOPS_PACK_VERSION", "")),
        "generate_strategies": bool(generate_strategies)
        if generate_strategies is not None
        else True,
        "mode": _normalize_text(settings_payload.get("mode")),
        "llm_provider": _normalize_text(settings_payload.get("llm_provider")),
        "models": normalized_models,
        "retriever": {
            "allow_web": bool(settings_payload.get("allow_web")),
            "sources_policy": _normalize_text(settings_payload.get("sources_policy")),
            "source_count": int(
                source_count
                or settings_payload.get("source_count")
                or 0
            ),
            "simulate_evidence": bool(settings_payload.get("simulate_evidence")),
        },
        "controls": {
            "temperature": float(settings_payload.get("temperature") or 0.0),
            "seed": settings_payload.get("seed"),
            "min_sources_per_domain": int(settings_payload.get("min_sources_per_domain") or 0),
            "min_citations_per_driver": int(settings_payload.get("min_citations_per_driver") or 0),
            "min_forces": int(settings_payload.get("min_forces") or 0),
            "min_forces_per_domain": int(settings_payload.get("min_forces_per_domain") or 0),
            "min_evidence_ok": int(settings_payload.get("min_evidence_ok") or 0),
            "min_evidence_total": int(settings_payload.get("min_evidence_total") or 0),
            "max_failed_ratio": settings_payload.get("max_failed_ratio"),
            "forbid_fixture_citations": bool(settings_payload.get("forbid_fixture_citations")),
        },
        "prompt_manifest_sha256": prompt_manifest_sha256 or "",
    }


def _hash_signature(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _load_run_meta(run_id: str, base_dir: Path) -> dict[str, Any] | None:
    return read_run_json(run_id, base_dir)


def _write_run_meta(run_id: str, base_dir: Path, updates: Mapping[str, Any]) -> dict[str, Any]:
    payload = update_run_json(run_id=run_id, updates=updates, base_dir=base_dir)
    run_dir = ensure_run_dirs(run_id, base_dir=base_dir)["run_dir"]
    run_meta_path = run_dir / "run_meta.json"
    run_meta_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    run_config_path = run_dir / "run_config.json"
    if run_config_path.exists():
        run_config = _safe_load_json(run_config_path) or {}
        run_config_meta = run_config.get("run_meta")
        if not isinstance(run_config_meta, dict):
            run_config_meta = {}
        for key in (
            "signature",
            "signature_payload",
            "parent_run_id",
            "is_deleted",
            "deleted_at",
            "deleted_by",
            "status",
            "updated_at",
            "created_at",
            "is_final",
            "completed_at",
        ):
            if key in payload:
                run_config_meta[key] = payload.get(key)
        run_config["run_meta"] = run_config_meta
        write_run_config(run_id=run_id, run_config=run_config, base_dir=base_dir)
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    ensure_local_file(path)
    if not path.exists():
        return []
    items: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            items.append(parsed)
    return items


def _derive_run_status(run_dir: Path) -> str | None:
    log_path = run_dir / "logs" / "node_events.jsonl"
    ensure_local_file(log_path, base_dir=run_dir.parent)
    if not log_path.exists():
        return None
    events = _load_jsonl(log_path)
    if not events:
        return None
    latest_by_node: dict[str, dict[str, Any]] = {}
    for entry in events:
        node = str(entry.get("node") or entry.get("step") or entry.get("id") or "")
        if not node:
            continue
        ts = entry.get("timestamp")
        current = latest_by_node.get(node)
        if not current:
            latest_by_node[node] = {"entry": entry, "timestamp": ts}
            continue
        if ts and (current.get("timestamp") is None or ts > current.get("timestamp")):
            latest_by_node[node] = {"entry": entry, "timestamp": ts}
    latest_entries = [item["entry"] for item in latest_by_node.values()]
    has_fail = any("FAIL" in str(entry.get("status") or "").upper() for entry in latest_entries)
    has_running = any(
        any(flag in str(entry.get("status") or "").upper() for flag in ["RUN", "START", "IN_PROGRESS"])
        for entry in latest_entries
    )
    if has_fail:
        return "FAIL"
    if has_running:
        return "RUNNING"
    return "OK"


def _run_status_for_dir(run_dir: Path, run_meta: Mapping[str, Any] | None) -> str:
    status = ""
    if isinstance(run_meta, Mapping):
        status = str(run_meta.get("status") or "").upper()
    if status:
        return status
    derived = _derive_run_status(run_dir)
    if derived:
        return derived
    if (run_dir / "artifacts" / "view_model.json").exists():
        return "OK"
    artifacts_dir = run_dir / "artifacts"
    if artifacts_dir.exists() and any(path.is_file() for path in artifacts_dir.iterdir()):
        return "OK"
    if (run_dir / "run_config.json").exists():
        return "PENDING"
    return "UNKNOWN"


def _signature_payload_for_run(run_dir: Path) -> dict[str, Any] | None:
    run_config = _safe_load_json(run_dir / "run_config.json") or {}
    settings = run_config.get("settings") or run_config
    models = run_config.get("models") if isinstance(run_config.get("models"), Mapping) else {}
    user_params = run_config.get("user_params") if isinstance(run_config.get("user_params"), Mapping) else {}
    prompt_manifest = None
    if not user_params:
        prompt_manifest = _safe_load_json(run_dir / "artifacts" / "prompt_manifest.json") or {}
        user_params = {
            "company": prompt_manifest.get("company_name"),
            "geography": prompt_manifest.get("geography"),
            "horizon": prompt_manifest.get("horizon_months"),
            "value": prompt_manifest.get("value"),
            "scope": prompt_manifest.get("scope"),
        }
    if not run_config and not prompt_manifest:
        return None
    retriever = run_config.get("retriever") if isinstance(run_config.get("retriever"), Mapping) else {}
    prompt_sha = run_config.get("prompt_manifest_sha256") or _prompt_manifest_sha256()
    return _build_signature_payload(
        company=user_params.get("company"),
        geography=user_params.get("geography"),
        scope=user_params.get("scope"),
        value=user_params.get("value"),
        horizon=user_params.get("horizon"),
        pack=user_params.get("pack") or run_config.get("pack"),
        settings=settings,
        models=models,
        generate_strategies=run_config.get("generate_strategies", True),
        prompt_manifest_sha256=prompt_sha,
        source_count=retriever.get("source_count"),
    )


def _find_existing_run(signature: str, base_dir: Path) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    for run_id in list_run_ids(base_dir=base_dir, include_deleted=False):
        run_dir = base_dir / run_id
        run_meta = _load_run_meta(run_id, base_dir) or {}
        if run_meta.get("is_deleted"):
            continue
        run_signature = run_meta.get("signature")
        signature_payload = run_meta.get("signature_payload")
        if not run_signature:
            signature_payload = _signature_payload_for_run(run_dir)
            if signature_payload:
                run_signature = _hash_signature(signature_payload)
                _write_run_meta(
                    run_id,
                    base_dir,
                    {"signature": run_signature, "signature_payload": signature_payload},
                )
        if run_signature != signature:
            continue
        status = _run_status_for_dir(run_dir, run_meta)
        candidates.append(
            {
                "run_id": run_id,
                "status": status,
                "mtime": run_dir.stat().st_mtime,
            }
        )
    if not candidates:
        return None

    def status_rank(value: str) -> int:
        if value in {"RUNNING", "PENDING", "IN_PROGRESS"}:
            return 0
        if value in {"OK", "COMPLETED", "COMPLETED_WITH_WARNINGS"}:
            return 1
        if value in {"FAIL", "FAILED"}:
            return 2
        return 3

    candidates.sort(key=lambda item: (status_rank(item["status"]), -item["mtime"]))
    return candidates[0]


def _action_console_path(run_id: str, base_dir: Path) -> Path:
    return _artifact_path(run_id, "action_console.json", base_dir)


def _build_action_console(view_model: dict[str, Any]) -> dict[str, Any]:
    wind_tunnel = view_model.get("wind_tunnel_evaluations_v2") or {}
    if not isinstance(wind_tunnel, dict):
        wind_tunnel = {}
    recommendations = wind_tunnel.get("recommendations") or {}
    if not isinstance(recommendations, dict):
        recommendations = {}
    primary = recommendations.get("primary_recommended_strategy") or {}
    primary_name = ""
    if isinstance(primary, dict):
        primary_name = primary.get("strategy_name") or ""
    items: list[dict[str, Any]] = []

    def add_item(
        *,
        title: str,
        priority: str,
        action_type: str,
        strategy: str | None = None,
        scenario: str | None = None,
        trigger: str | None = None,
        source: str = "wind_tunnel",
    ) -> None:
        if not title:
            return
        items.append(
            {
                "id": f"action-{len(items) + 1}",
                "priority": priority,
                "type": action_type,
                "title": title,
                "strategy": strategy,
                "scenario": scenario,
                "trigger": trigger,
                "owner": "",
                "due": "",
                "status": "open",
                "source": source,
            }
        )

    for action in recommendations.get("hardening_actions") or []:
        add_item(
            title=str(action),
            priority="P0",
            action_type="hardening",
            strategy=primary_name,
        )
    for action in recommendations.get("hedge_actions") or []:
        add_item(
            title=str(action),
            priority="P1",
            action_type="hedge",
            strategy="",
        )
    for trigger in recommendations.get("triggers_to_watch") or []:
        if isinstance(trigger, dict):
            title = trigger.get("recommended_action") or trigger.get("description") or ""
            add_item(
                title=str(title),
                priority="P2",
                action_type="trigger",
                trigger=str(trigger.get("description") or ""),
                strategy=primary_name,
            )
        else:
            add_item(
                title=str(trigger),
                priority="P2",
                action_type="trigger",
                trigger=str(trigger),
                strategy=primary_name,
            )

    return {
        "run_id": (view_model.get("run_meta") or {}).get("run_id"),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "items": items,
    }


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
    trace_dir = base_dir / run_id / "trace"
    logs_dir = base_dir / run_id / "logs"
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

    latest_failure = {}
    node_events_path = logs_dir / "node_events.jsonl"
    ensure_local_file(node_events_path, base_dir=base_dir)
    if node_events_path.exists():
        for line in reversed(node_events_path.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(entry, dict):
                continue
            status = str(entry.get("status") or "").upper()
            error = entry.get("error")
            if "FAIL" in status or error:
                latest_failure = {
                    "node": entry.get("node"),
                    "status": entry.get("status"),
                    "error": _truncate_text(str(error or ""), 360),
                    "timestamp": entry.get("timestamp"),
                }
                break

    wind_tunnel_failures: list[dict[str, Any]] = []
    if trace_dir.exists():
        for path in sorted(trace_dir.glob("wind_tunnel_failure_*.json"))[-3:]:
            payload = _safe_load_json(path) or {}
            if isinstance(payload, dict):
                wind_tunnel_failures.append(
                    {
                        "error": _truncate_text(str(payload.get("error") or ""), 360),
                        "chunk_index": payload.get("chunk_index"),
                        "chunks": payload.get("chunks"),
                        "strategy_count": payload.get("strategy_count"),
                        "scenario_count": payload.get("scenario_count"),
                    }
                )
    debug_path = trace_dir / "wind_tunnel_debug.json"
    ensure_local_file(debug_path, base_dir=base_dir)
    wind_tunnel_debug = _safe_load_json(debug_path) or {}
    if not isinstance(wind_tunnel_debug, dict):
        wind_tunnel_debug = {}

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
        "run_failures": {
            "latest": latest_failure,
            "wind_tunnel": {
                "failures": wind_tunnel_failures,
                "debug": {
                    "missing_outcomes": wind_tunnel_debug.get("missing_outcomes"),
                    "parsed_keys": wind_tunnel_debug.get("parsed_keys"),
                    "response_raw_excerpt": _truncate_text(
                        str(wind_tunnel_debug.get("response_raw_excerpt") or ""), 600
                    ),
                }
                if wind_tunnel_debug
                else {},
            },
        },
    }
    return context


def _safe_log_path(run_id: str, log_name: str, base_dir: Path) -> Path:
    if "/" in log_name or "\\" in log_name:
        raise ValueError("Invalid log name.")
    logs_dir = base_dir / run_id / "logs"
    return logs_dir / log_name


def _run_meta_for(run_id: str, base_dir: Path) -> dict[str, Any]:
    meta = read_run_json(run_id, base_dir) or {}
    if not meta:
        fallback = _safe_load_json(base_dir / run_id / "run_meta.json")
        if isinstance(fallback, dict):
            meta = fallback
    return meta if isinstance(meta, dict) else {}


def _ensure_mutable(run_id: str, base_dir: Path) -> None:
    meta = _run_meta_for(run_id, base_dir)
    if meta.get("is_final"):
        raise HTTPException(
            status_code=409,
            detail={"code": "RUN_READ_ONLY", "message": "Run is completed and read-only."},
        )


def _require_llm_key(request: Request) -> str:
    key, source = resolve_api_key(request)
    if not key:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "MISSING_API_KEY",
                "message": "No API key found. Provide in UI or set GEMINI_API_KEY in Cloud Run.",
            },
        )
    return key


def _tenant_context(
    request: Request,
    x_api_key: str | None = Header(default=None, alias="X-Api-Key"),
    authorization: str | None = Header(default=None),
) -> TenantContext:
    api_key, _ = resolve_api_key(request)
    return resolve_tenant(api_key)


def _tenant_context_optional(
    request: Request,
    x_api_key: str | None = Header(default=None, alias="X-Api-Key"),
    authorization: str | None = Header(default=None),
) -> TenantContext:
    api_key, _ = resolve_api_key(request)
    try:
        return resolve_tenant(api_key)
    except Exception:
        return TenantContext(
            tenant_id=os.environ.get("SCENARIOOPS_DEFAULT_TENANT", "public"),
            is_admin=False,
        )


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
    payload: BuildRequest,
    request: Request,
    tenant: TenantContext = Depends(_tenant_context),
) -> RunResponse:
    if not payload.mock:
        _require_llm_key(request)
    if payload.resume_from and not payload.run_id:
        raise HTTPException(status_code=400, detail="run_id required when resuming.")
    user_params = {"scope": payload.scope, "value": payload.value, "horizon": payload.horizon}
    if payload.company:
        user_params["company"] = payload.company
    if payload.geography:
        user_params["geography"] = payload.geography
    if payload.pack:
        user_params["pack"] = payload.pack
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

    signature_payload = _build_signature_payload(
        company=payload.company or user_params.get("company"),
        geography=payload.geography or user_params.get("geography"),
        scope=payload.scope,
        value=payload.value,
        horizon=payload.horizon,
        pack=payload.pack,
        settings=settings,
        generate_strategies=payload.generate_strategies,
        prompt_manifest_sha256=_prompt_manifest_sha256(),
        source_count=len(sources),
    )
    signature = _hash_signature(signature_payload)
    force_run = bool(payload.force or payload.rerun)
    resume_mode = bool(payload.resume_from)
    existing = _find_existing_run(signature, tenant.base_dir)
    if existing and not resume_mode and not force_run:
        status = existing.get("status", "UNKNOWN")
        if status in {"RUNNING", "PENDING", "IN_PROGRESS"}:
            existing_id = existing["run_id"]
            return RunResponse(
                run_id=existing_id,
                artifacts=list_artifacts(existing_id, base_dir=tenant.base_dir),
                reused=True,
                reuse_reason="in_progress",
                existing_run_id=existing_id,
                signature=signature,
            )
        if status in {"OK", "COMPLETED", "COMPLETED_WITH_WARNINGS"}:
            existing_id = existing["run_id"]
            return RunResponse(
                run_id=existing_id,
                artifacts=list_artifacts(existing_id, base_dir=tenant.base_dir),
                reused=True,
                reuse_reason="completed",
                existing_run_id=existing_id,
                signature=signature,
            )

    run_id = payload.run_id or datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    inputs = GraphInputs(
        user_params=user_params,
        sources=sources,
        signals=[],
        input_docs=payload.input_docs or [],
    )
    parent_run_id = existing["run_id"] if force_run and existing else None
    now = datetime.now(timezone.utc).isoformat()
    meta_updates: dict[str, Any] = {
        "run_id": run_id,
        "signature": signature,
        "signature_payload": signature_payload,
        "status": "RUNNING",
        "updated_at": now,
        "is_final": False,
    }
    if parent_run_id:
        meta_updates["parent_run_id"] = parent_run_id
    if not _load_run_meta(run_id, tenant.base_dir):
        meta_updates["created_at"] = now
    _write_run_meta(run_id, tenant.base_dir, meta_updates)
    write_run_inputs(run_id=run_id, payload=signature_payload, base_dir=tenant.base_dir)
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
        _write_run_meta(
            run_id,
            tenant.base_dir,
            {
                "status": "FAILED",
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "is_final": True,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return RunResponse(
        run_id=run_id,
        artifacts=list_artifacts(run_id, base_dir=tenant.base_dir),
        signature=signature,
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
def run_ui(
    run_id: str,
    request: Request,
    tenant: TenantContext = Depends(_tenant_context_optional),
):
    accept = request.headers.get("accept", "")
    if "application/json" in accept:
        meta = _run_meta_for(run_id, tenant.base_dir)
        artifacts = list_artifacts(run_id, base_dir=tenant.base_dir)
        return JSONResponse({"run_id": run_id, "meta": meta, "artifacts": artifacts})
    return _commercial_ui()


@app.get("/artifact/{run_id}/{artifact_name}")
def artifact(
    run_id: str,
    artifact_name: str,
    tenant: TenantContext = Depends(_tenant_context_optional),
):
    try:
        path = _artifact_path(run_id, artifact_name, tenant.base_dir)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    ensure_local_file(path)
    if not path.exists():
        if artifact_name == "view_model.json":
            run_dir = tenant.base_dir / run_id
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
    tenant: TenantContext = Depends(_tenant_context_optional),
) -> FileResponse:
    try:
        path = _safe_image_path(run_id, image_name, tenant.base_dir)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    ensure_local_file(path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Image not found.")
    media_type, _ = mimetypes.guess_type(str(path))
    return FileResponse(path, media_type=media_type or "application/octet-stream")


@app.get("/action-console/{run_id}")
def action_console_get(
    run_id: str, tenant: TenantContext = Depends(_tenant_context_optional)
) -> JSONResponse:
    path = _action_console_path(run_id, tenant.base_dir)
    ensure_local_file(path)
    if path.exists():
        return JSONResponse(json.loads(path.read_text(encoding="utf-8")))
    view_model = _safe_load_json(
        tenant.base_dir / run_id / "artifacts" / "view_model.json"
    )
    if not view_model:
        run_dir = tenant.base_dir / run_id
        try:
            view_model = build_view_model(run_dir)
        except Exception:
            view_model = {}
    payload = _build_action_console(view_model or {})
    return JSONResponse(payload)


@app.post("/action-console/{run_id}")
def action_console_update(
    run_id: str,
    payload: ActionConsolePayload,
    tenant: TenantContext = Depends(_tenant_context),
) -> JSONResponse:
    _ensure_mutable(run_id, tenant.base_dir)
    action_payload = payload.model_dump()
    action_payload["run_id"] = run_id
    action_payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    write_artifact(
        run_id=run_id,
        artifact_name="action_console",
        payload=action_payload,
        ext="json",
        base_dir=tenant.base_dir,
    )
    return JSONResponse(action_payload)


@app.post("/action-console/{run_id}/generate")
def action_console_generate(
    run_id: str, tenant: TenantContext = Depends(_tenant_context)
) -> JSONResponse:
    _ensure_mutable(run_id, tenant.base_dir)
    view_model = _safe_load_json(
        tenant.base_dir / run_id / "artifacts" / "view_model.json"
    )
    if not view_model:
        run_dir = tenant.base_dir / run_id
        try:
            view_model = build_view_model(run_dir)
        except Exception:
            view_model = {}
    payload = _build_action_console(view_model or {})
    payload["run_id"] = run_id
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    write_artifact(
        run_id=run_id,
        artifact_name="action_console",
        payload=payload,
        ext="json",
        base_dir=tenant.base_dir,
    )
    return JSONResponse(payload)


@app.get("/runs", response_model=RunsResponse)
def runs(tenant: TenantContext = Depends(_tenant_context_optional)) -> RunsResponse:
    run_ids = list_run_ids(base_dir=tenant.base_dir, include_deleted=False)
    labels: dict[str, str] = {}
    for run_id in run_ids:
        try:
            labels[run_id] = _run_label(run_id, tenant.base_dir)
        except Exception:
            labels[run_id] = run_id
    return RunsResponse(runs=run_ids, labels=labels)


@app.delete("/runs/{run_id}", response_model=RunDeleteResponse)
def delete_run(
    run_id: str, tenant: TenantContext = Depends(_tenant_context)
) -> RunDeleteResponse:
    run_dir = tenant.base_dir / run_id
    meta = _load_run_meta(run_id, tenant.base_dir) or {}
    if not meta and not run_dir.exists():
        raise HTTPException(status_code=404, detail="Run not found.")
    deleted_at = datetime.now(timezone.utc).isoformat()
    meta_updates = {
        "run_id": run_id,
        "is_deleted": True,
        "deleted_at": deleted_at,
        "updated_at": deleted_at,
        "deleted_by": tenant.tenant_id,
    }
    _write_run_meta(run_id, tenant.base_dir, {**meta, **meta_updates})
    return RunDeleteResponse(run_id=run_id, deleted=True)


@app.post("/runs/{run_id}/clone", response_model=RunResponse)
def clone_run(
    run_id: str,
    request: Request,
    tenant: TenantContext = Depends(_tenant_context),
) -> RunResponse:
    run_config = _safe_load_json(tenant.base_dir / run_id / "run_config.json") or {}
    if not isinstance(run_config, dict) or not run_config:
        raise HTTPException(status_code=404, detail="Run config not found.")
    user_params = run_config.get("user_params") if isinstance(run_config.get("user_params"), dict) else {}
    settings_override = run_config.get("settings") if isinstance(run_config.get("settings"), dict) else {}
    payload = BuildRequest(
        scope=user_params.get("scope") or "country",
        value=user_params.get("value") or user_params.get("geography") or user_params.get("company") or "Global",
        company=user_params.get("company"),
        geography=user_params.get("geography"),
        horizon=int(user_params.get("horizon") or 60),
        pack=user_params.get("pack"),
        sources=run_config.get("sources"),
        mock=settings_override.get("mode") == "demo",
        resume_from=None,
        input_docs=None,
        generate_strategies=bool(run_config.get("generate_strategies", True)),
        settings_overrides=settings_override or None,
        force=True,
        rerun=True,
    )
    if not payload.mock:
        _require_llm_key(request)
    return build(payload, tenant=tenant, request=request)


@app.post("/chat", response_model=ChatResponse)
def chat(
    payload: ChatRequest,
    request: Request,
    tenant: TenantContext = Depends(_tenant_context),
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
    if getattr(config, "mode", "mock") != "mock":
        _require_llm_key(request)
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
    run_id: str, tenant: TenantContext = Depends(_tenant_context_optional)
) -> LogResponse:
    try:
        path = _safe_log_path(run_id, "node_events.jsonl", tenant.base_dir)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    ensure_local_file(path)
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
    run_id: str, tenant: TenantContext = Depends(_tenant_context_optional)
) -> LogResponse:
    try:
        path = _safe_log_path(run_id, "normalization.jsonl", tenant.base_dir)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    ensure_local_file(path)
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
    payload: StrategiesRequest,
    request: Request,
    tenant: TenantContext = Depends(_tenant_context),
) -> RunResponse:
    run_id = payload.run_id or latest_run_id(base_dir=tenant.base_dir)
    if not run_id:
        raise HTTPException(status_code=404, detail="No runs available.")
    _ensure_mutable(run_id, tenant.base_dir)

    overrides = {"mode": "demo" if payload.mock else "live"}
    if payload.mock:
        overrides["sources_policy"] = "fixtures"
        overrides["llm_provider"] = "mock"
    else:
        _require_llm_key(request)
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
    payload: DailyRequest,
    request: Request,
    tenant: TenantContext = Depends(_tenant_context),
) -> RunResponse:
    run_id = payload.run_id or latest_run_id(base_dir=tenant.base_dir)
    if not run_id:
        raise HTTPException(status_code=404, detail="No runs available.")
    _ensure_mutable(run_id, tenant.base_dir)

    overrides = {"mode": "demo" if payload.mock else "live"}
    if payload.mock:
        overrides["sources_policy"] = "fixtures"
        overrides["llm_provider"] = "mock"
    else:
        _require_llm_key(request)
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
def latest(tenant: TenantContext = Depends(_tenant_context_optional)) -> LatestResponse:
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


@app.get("/health")
def health(request: Request):
    _, source = resolve_api_key(request)
    if source in {"header", "bearer", "query"}:
        source_label = "user"
    elif source == "env":
        source_label = "env"
    else:
        source_label = "none"
    storage_mode = run_store_mode()
    runs_dir = runs_root()
    gcs_bucket = os.environ.get("GCS_BUCKET", "").strip()
    gcs_prefix = runs_prefix()
    if storage_mode == "gcs" and gcs_bucket:
        runs_location = f"gs://{gcs_bucket}/{gcs_prefix}".rstrip("/")
    else:
        runs_location = str(runs_dir)
    vector_mode = os.environ.get("VECTOR_STORE", "local").strip().lower()
    vector_scope = os.environ.get("SCENARIOOPS_VECTORDB_SCOPE", "global").strip().lower()
    if vector_mode == "off":
        vector_location = "disabled"
    elif vector_scope == "run":
        vector_location = str(runs_dir / "<run_id>" / "vectordb")
    else:
        vector_location = str(runs_dir.parent / "vectordb")
    return {
        "storage_backend": storage_mode,
        "runs_dir": str(runs_dir),
        "runs_location": runs_location,
        "runs_prefix": gcs_prefix,
        "api_key_source": source_label,
        "vector_store": vector_mode,
        "vector_scope": vector_scope,
        "vector_location": vector_location,
        "version": os.environ.get("SCENARIOOPS_VERSION", "unknown"),
    }


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
