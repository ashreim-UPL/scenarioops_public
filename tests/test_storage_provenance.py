import json

from scenarioops.graph.tools.provenance import hash_value
from scenarioops.graph.tools.storage import run_dummy_hello


def test_dummy_node_writes_artifact_and_meta(tmp_path) -> None:
    run_id = "run-001"
    base_dir = tmp_path / "runs"

    artifact_path, meta_path = run_dummy_hello(run_id, base_dir=base_dir)

    assert artifact_path.exists()
    assert meta_path.exists()
    assert (base_dir / run_id / "logs").exists()

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["artifact_name"] == "hello"
    assert meta["artifact_path"] == "hello.json"
    assert meta["run_id"] == run_id
    assert "created_at" in meta["timestamps"]
    assert meta["input_hashes"]["payload"] == hash_value({"message": "hello"})
    assert "prompt" in meta["prompt_hashes"]
    assert "scenarioops" in meta["tool_versions"]
