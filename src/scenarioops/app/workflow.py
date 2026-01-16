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
from scenarioops.graph.tools.storage import default_runs_dir, read_latest_status


def _runs_dir(base_dir: Path | None = None) -> Path:
    if base_dir is not None:
        return base_dir
    return default_runs_dir()


def list_run_ids(base_dir: Path | None = None) -> list[str]:
    runs_dir = _runs_dir(base_dir)
    if not runs_dir.exists():
        return []
    return sorted([path.name for path in runs_dir.iterdir() if path.is_dir()])


def latest_run_id(base_dir: Path | None = None) -> str | None:
    latest_status = read_latest_status(base_dir)
    if latest_status:
        run_id = latest_status.get("run_id")
        if isinstance(run_id, str) and run_id:
            return run_id
    runs_dir = _runs_dir(base_dir)
    if not runs_dir.exists():
        return None
    runs = [path for path in runs_dir.iterdir() if path.is_dir()]
    if not runs:
        return None
    latest = max(runs, key=lambda path: path.stat().st_mtime)
    return latest.name


def _artifacts_dir(run_id: str, base_dir: Path | None = None) -> Path:
    return _runs_dir(base_dir) / run_id / "artifacts"


def list_artifacts(run_id: str, base_dir: Path | None = None) -> list[str]:
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
    return json.loads(path.read_text(encoding="utf-8"))


def _load_required_json(path: Path) -> Any:
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
    return ScenarioOpsState(logic=load_logic(run_id, base_dir))


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
