import hashlib
from pathlib import Path

import pytest

from scenarioops.graph.build_graph import GraphInputs, default_sources, run_graph
from scenarioops.graph.nodes.auditor import run_auditor_node
from scenarioops.graph.state import ScenarioOpsState
from scenarioops.graph.tools.storage import write_artifact


def _hash_artifacts(base_dir: Path, run_id: str) -> dict[str, str]:
    artifacts_dir = base_dir / run_id / "artifacts"
    hashes: dict[str, str] = {}
    for path in artifacts_dir.iterdir():
        if path.name.endswith(".meta.json"):
            continue
        payload = path.read_bytes()
        hashes[path.name] = hashlib.sha256(payload).hexdigest()
    return hashes


def test_mock_run_deterministic_hashes(tmp_path: Path) -> None:
    run_id = "deterministic-run"
    base_dir = tmp_path / "runs"
    inputs = GraphInputs(
        user_params={"scope": "country", "value": "UAE", "horizon": 24},
        sources=default_sources(),
        signals=[],
    )

    run_graph(
        inputs,
        run_id=run_id,
        base_dir=base_dir,
        mock_mode=True,
        generate_strategies=True,
        report_date="2026-01-01",
    )
    first = _hash_artifacts(base_dir, run_id)

    run_graph(
        inputs,
        run_id=run_id,
        base_dir=base_dir,
        mock_mode=True,
        generate_strategies=True,
        report_date="2026-01-01",
    )
    second = _hash_artifacts(base_dir, run_id)

    run_graph(
        inputs,
        run_id=run_id,
        base_dir=base_dir,
        mock_mode=True,
        generate_strategies=True,
        report_date="2026-01-01",
    )
    third = _hash_artifacts(base_dir, run_id)

    assert first == second == third


def test_schema_violation_blocks_publish(tmp_path: Path) -> None:
    run_id = "schema-fail"
    base_dir = tmp_path / "runs"

    invalid_logic = {"id": "logic-1", "title": "Logic", "axes": "bad", "scenarios": []}
    write_artifact(
        run_id=run_id,
        artifact_name="logic",
        payload=invalid_logic,
        ext="json",
        base_dir=base_dir,
    )

    with pytest.raises(RuntimeError):
        run_auditor_node(run_id=run_id, state=ScenarioOpsState(), base_dir=base_dir)
