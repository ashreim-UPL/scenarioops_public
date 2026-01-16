import json
from pathlib import Path

import pytest

from scenarioops.app.config import load_settings
from scenarioops.graph.build_graph import GraphInputs, run_graph


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _compare_expected(run_dir: Path, expected_root: Path) -> None:
    for expected_path in sorted(expected_root.rglob("*")):
        if expected_path.is_dir():
            continue
        relative = expected_path.relative_to(expected_root)
        actual_path = run_dir / relative
        assert actual_path.exists(), f"Missing artifact: {actual_path}"
        if expected_path.suffix == ".json":
            assert _load_json(actual_path) == _load_json(expected_path)
        elif expected_path.suffix == ".jsonl":
            expected_lines = expected_path.read_text(encoding="utf-8").splitlines()
            actual_lines = actual_path.read_text(encoding="utf-8").splitlines()
            assert actual_lines == expected_lines
        else:
            assert actual_path.read_text(encoding="utf-8") == expected_path.read_text(
                encoding="utf-8"
            )


@pytest.mark.parametrize("fixture_name", ["small", "medium"])
def test_fixture_outputs_match_expected(tmp_path: Path, fixture_name: str) -> None:
    fixture_dir = Path("tests/fixtures") / fixture_name
    inputs = json.loads((fixture_dir / "inputs.json").read_text(encoding="utf-8-sig"))
    expected_root = fixture_dir / "expected"
    settings = load_settings(inputs.get("settings_overrides"))
    run_id = fixture_name
    graph_inputs = GraphInputs(
        user_params=inputs.get("user_params", {}),
        sources=inputs.get("sources", []),
        signals=inputs.get("signals", []),
    )
    run_graph(
        graph_inputs,
        run_id=run_id,
        run_timestamp=inputs.get("run_timestamp"),
        base_dir=tmp_path / "runs",
        mock_mode=True,
        settings=settings,
        generate_strategies=True,
        report_date=inputs.get("report_date"),
        command="fixture-test",
    )
    run_dir = tmp_path / "runs" / run_id
    _compare_expected(run_dir, expected_root)
