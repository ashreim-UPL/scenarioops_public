from argparse import Namespace
import json
from pathlib import Path

import pytest

from scenarioops.app.main import _run_build_scenarios, _run_daily


def _read_latest(base_dir: Path) -> dict:
    latest_path = base_dir / "latest.json"
    return json.loads(latest_path.read_text(encoding="utf-8"))


def test_build_scenarios_writes_latest_ok(tmp_path: Path) -> None:
    base_dir = tmp_path / "runs"
    args = Namespace(
        scope="world",
        value="UAE",
        horizon=12,
        run_id="latest-ok",
        base_dir=str(base_dir),
        sources=None,
        live=False,
    )

    _run_build_scenarios(args)

    latest = _read_latest(base_dir)
    assert latest["run_id"] == "latest-ok"
    assert latest["status"] == "OK"
    assert latest["command"] == "build-scenarios"
    assert latest["updated_at"]


def test_build_scenarios_writes_latest_fail(tmp_path: Path) -> None:
    base_dir = tmp_path / "runs"
    args = Namespace(
        scope="world",
        value="UAE",
        horizon=12,
        run_id="latest-fail",
        base_dir=str(base_dir),
        sources=None,
        live=True,
    )

    with pytest.raises(ValueError):
        _run_build_scenarios(args)

    latest = _read_latest(base_dir)
    assert latest["run_id"] == "latest-fail"
    assert latest["status"] == "FAIL"
    assert latest["command"] == "build-scenarios"
    assert "Sources are required" in latest.get("error_summary", "")


def test_run_daily_writes_latest_ok(tmp_path: Path) -> None:
    base_dir = tmp_path / "runs"
    args = Namespace(
        run_id="daily-ok",
        base_dir=str(base_dir),
        signals=None,
        live=False,
    )

    _run_daily(args)

    latest = _read_latest(base_dir)
    assert latest["run_id"] == "daily-ok"
    assert latest["status"] == "OK"
    assert latest["command"] == "run-daily"
    assert latest["updated_at"]
    assert (base_dir / "daily-ok" / "artifacts" / "daily_brief.md").exists()
