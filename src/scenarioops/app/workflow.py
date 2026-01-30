from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from scenarioops.graph.state import (
    Ewi,
    EwiIndicator,
    Logic,
    ScenarioAxis,
    ScenarioLogic,
    ScenarioOpsState,
    WindTunnel,
    WindTunnelTest,
)
from scenarioops.graph.tools.storage import (
    default_runs_dir,
    ensure_local_file,
    read_latest_status,
    read_run_json,
)
from scenarioops.storage.run_store import get_run_store, run_store_mode


def _runs_dir(base_dir: Path | None = None) -> Path:
    if base_dir is not None:
        return base_dir
    return default_runs_dir()


def _load_run_meta(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "run.json"
    ensure_local_file(path)
    if not path.exists():
        path = run_dir / "run_meta.json"
        ensure_local_file(path)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def list_run_ids(base_dir: Path | None = None, *, include_deleted: bool = False) -> list[str]:
    if run_store_mode() == "gcs":
        runs_dir = _runs_dir(base_dir)
        root_dir = _runs_dir(None)
        try:
            relative_prefix = runs_dir.resolve().relative_to(root_dir.resolve())
        except Exception:
            relative_prefix = Path("")
        prefix = str(relative_prefix).replace("\\", "/").strip("/")
        if prefix:
            prefix = f"{prefix}/"
        store = get_run_store()
        keys = store.list(prefix)
        run_ids: set[str] = set()
        for key in keys:
            if not key.endswith("/run.json"):
                continue
            rel = key[len(prefix):] if prefix and key.startswith(prefix) else key
            parts = rel.split("/")
            if len(parts) >= 2:
                run_ids.add(parts[0])
        runs = sorted(run_ids)
        if include_deleted:
            return runs
        filtered: list[str] = []
        for run_id in runs:
            meta = read_run_json(run_id, base_dir=base_dir) or {}
            if isinstance(meta, dict) and meta.get("is_deleted"):
                continue
            filtered.append(run_id)
        return filtered
    runs_dir = _runs_dir(base_dir)
    if not runs_dir.exists():
        return []
    excluded = {"vectordb", "vector_db", "cache", "embed_cache"}
    runs: list[str] = []
    for path in runs_dir.iterdir():
        if not path.is_dir():
            continue
        if path.name in excluded:
            continue
        if not include_deleted:
            meta = _load_run_meta(path)
            if isinstance(meta, dict) and meta.get("is_deleted"):
                continue
        if (path / "artifacts").exists() or (path / "logs").exists() or (path / "run_meta.json").exists():
            runs.append(path.name)
    return sorted(runs)


def latest_run_id(base_dir: Path | None = None) -> str | None:
    latest_status = read_latest_status(base_dir)
    if latest_status:
        run_id = latest_status.get("run_id")
        if isinstance(run_id, str) and run_id:
            runs_dir = _runs_dir(base_dir)
            run_dir = runs_dir / run_id
            meta = _load_run_meta(run_dir)
            if not (isinstance(meta, dict) and meta.get("is_deleted")):
                return run_id
    if run_store_mode() == "gcs":
        runs = list_run_ids(base_dir=base_dir, include_deleted=False)
        if not runs:
            return None
        best_id = None
        best_ts = None
        for run_id in runs:
            meta = read_run_json(run_id, base_dir=base_dir) or {}
            updated = meta.get("updated_at") or meta.get("created_at")
            if not isinstance(updated, str):
                continue
            try:
                ts = updated.replace("Z", "+00:00")
            except Exception:
                ts = updated
            if best_ts is None or ts > best_ts:
                best_ts = ts
                best_id = run_id
        return best_id or runs[-1]
    runs_dir = _runs_dir(base_dir)
    if not runs_dir.exists():
        return None
    runs = []
    for path in runs_dir.iterdir():
        if not path.is_dir():
            continue
        meta = _load_run_meta(path)
        if isinstance(meta, dict) and meta.get("is_deleted"):
            continue
        runs.append(path)
    if not runs:
        return None
    latest = max(runs, key=lambda path: path.stat().st_mtime)
    return latest.name


def _artifacts_dir(run_id: str, base_dir: Path | None = None) -> Path:
    return _runs_dir(base_dir) / run_id / "artifacts"


def list_artifacts(run_id: str, base_dir: Path | None = None) -> list[str]:
    if run_store_mode() == "gcs":
        runs_dir = _runs_dir(base_dir)
        root_dir = _runs_dir(None)
        try:
            relative_prefix = runs_dir.resolve().relative_to(root_dir.resolve())
        except Exception:
            relative_prefix = Path("")
        prefix = str(relative_prefix).replace("\\", "/").strip("/")
        if prefix:
            prefix = f"{prefix}/{run_id}/artifacts/"
        else:
            prefix = f"{run_id}/artifacts/"
        store = get_run_store()
        keys = store.list(prefix)
        artifacts: list[str] = []
        for key in keys:
            if not key.startswith(prefix):
                continue
            artifacts.append(key)
        return sorted(artifacts)
    artifacts_dir = _artifacts_dir(run_id, base_dir)
    if not artifacts_dir.exists():
        return []
    paths = sorted([path for path in artifacts_dir.iterdir() if path.is_file()])
    cwd = Path.cwd()
    listed = []
    for path in paths:
        try:
            listed.append(str(path.relative_to(cwd)))
        except ValueError:
            listed.append(str(path))
    return listed


def _load_json(path: Path) -> Any:
    ensure_local_file(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _load_required_json(path: Path) -> Any:
    ensure_local_file(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing artifact: {path}")
    return _load_json(path)


def load_logic(run_id: str, base_dir: Path | None = None) -> Logic:
    payload = _load_required_json(_artifacts_dir(run_id, base_dir) / "logic.json")
    axes = [ScenarioAxis(**axis) for axis in payload.get("axes", [])]
    scenarios = [ScenarioLogic(**scenario) for scenario in payload.get("scenarios", [])]
    return Logic(
        id=payload.get("id", f"logic-{run_id}"),
        title=payload.get("title", "Scenario Logic"),
        axes=axes,
        scenarios=scenarios,
    )


def load_ewi(run_id: str, base_dir: Path | None = None) -> Ewi:
    payload = _load_required_json(_artifacts_dir(run_id, base_dir) / "ewi.json")
    indicators = [EwiIndicator(**indicator) for indicator in payload.get("indicators", [])]
    return Ewi(
        id=payload.get("id", f"ewi-{run_id}"),
        title=payload.get("title", "Early Warning Indicators"),
        indicators=indicators,
    )


def load_wind_tunnel(run_id: str, base_dir: Path | None = None) -> WindTunnel:
    payload = _load_required_json(_artifacts_dir(run_id, base_dir) / "wind_tunnel.json")
    tests = [WindTunnelTest(**test) for test in payload.get("tests", [])]
    return WindTunnel(
        id=payload.get("id", f"wind-tunnel-{run_id}"),
        title=payload.get("title", "Wind Tunnel"),
        tests=tests,
    )


def state_for_strategies(run_id: str, base_dir: Path | None = None) -> ScenarioOpsState:
    artifacts_dir = _artifacts_dir(run_id, base_dir)
    try:
        return ScenarioOpsState(logic=load_logic(run_id, base_dir))
    except FileNotFoundError:
        scenarios_path = artifacts_dir / "scenarios.json"
        scenarios = _load_required_json(scenarios_path)
        return ScenarioOpsState(scenarios=scenarios)


def state_for_daily(
    run_id: str, base_dir: Path | None = None, *, allow_missing: bool = False
) -> ScenarioOpsState:
    ewi = None
    wind_tunnel = None
    if allow_missing:
        try:
            ewi = load_ewi(run_id, base_dir)
        except FileNotFoundError:
            ewi = None
        try:
            wind_tunnel = load_wind_tunnel(run_id, base_dir)
        except FileNotFoundError:
            wind_tunnel = None
    else:
        ewi = load_ewi(run_id, base_dir)
        wind_tunnel = load_wind_tunnel(run_id, base_dir)
    return ScenarioOpsState(ewi=ewi, wind_tunnel=wind_tunnel)


def ensure_signals(indicators: Iterable[EwiIndicator]) -> list[dict[str, Any]]:
    return [{"indicator_id": indicator.id, "score": 0.6} for indicator in indicators]
