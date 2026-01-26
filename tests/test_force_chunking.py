from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pytest

from scenarioops.graph.nodes.force_builder import (
    _is_truncation_error,
    _plan_chunks,
    run_force_builder_node,
)
from scenarioops.graph.state import ScenarioOpsState


def _evidence_units_payload(run_id: str, count: int) -> dict[str, Any]:
    timestamp = datetime.now(timezone.utc).isoformat()
    units = []
    for idx in range(1, count + 1):
        units.append(
            {
                "evidence_unit_id": f"ev-{idx}",
                "source_type": "primary",
                "title": f"Evidence {idx}",
                "publisher": "Example",
                "date_published": timestamp,
                "url": f"https://example.com/{idx}",
                "excerpt": "Evidence excerpt.",
                "claims": [],
                "metrics": [],
                "reliability_grade": "A",
                "reliability_reason": "test",
                "geography_tags": ["Global"],
                "domain_tags": [],
                "simulated": False,
            }
        )
    return {
        "run_id": run_id,
        "timestamp": timestamp,
        "company_name": "Acme",
        "geography": "Global",
        "horizon_months": 60,
        "simulated": False,
        "evidence_units": units,
    }


def _force_payload(force_id: str, domain: str, evidence_id: str) -> dict[str, Any]:
    return {
        "force_id": force_id,
        "layer": "primary",
        "domain": domain,
        "label": f"{domain} force",
        "mechanism": f"{domain} mechanism",
        "directionality": "increase",
        "affected_dimensions": ["cost"],
        "evidence_unit_ids": [evidence_id],
        "confidence": 0.7,
        "confidence_rationale": "test",
    }


class ForceCorrectionClient:
    def generate_json(self, prompt: str, schema: Mapping[str, Any]) -> dict[str, Any]:
        title = schema.get("title") if isinstance(schema, Mapping) else ""
        if title == "Forces Payload":
            return {"forces": [_force_payload("force-1", "economic", "ev-missing")]}
        if title == "Force Item":
            return _force_payload("force-1", "economic", "ev-1")
        return {}


class ChunkingClient:
    def __init__(self) -> None:
        self.calls = 0

    def generate_json(self, prompt: str, schema: Mapping[str, Any]) -> dict[str, Any]:
        title = schema.get("title") if isinstance(schema, Mapping) else ""
        if title != "Forces Payload":
            return {}
        self.calls += 1
        if self.calls == 1:
            forces = [
                _force_payload("force-1", "political", "ev-1"),
                _force_payload("force-2", "economic", "ev-1"),
            ]
        elif self.calls == 2:
            forces = [
                _force_payload("force-3", "social", "ev-1"),
                _force_payload("force-4", "technological", "ev-1"),
            ]
        else:
            forces = [_force_payload(f"force-{self.calls}", "environmental", "ev-1")]
        return {"forces": forces}


class DuplicateClient:
    def __init__(self) -> None:
        self.calls = 0

    def generate_json(self, prompt: str, schema: Mapping[str, Any]) -> dict[str, Any]:
        title = schema.get("title") if isinstance(schema, Mapping) else ""
        if title == "Forces Payload":
            self.calls += 1
            return {"forces": [_force_payload("force-dup", "economic", "ev-1")]}
        return {}


class PillarClient:
    def generate_json(self, prompt: str, schema: Mapping[str, Any]) -> dict[str, Any]:
        title = schema.get("title") if isinstance(schema, Mapping) else ""
        if title == "Forces Payload":
            return {
                "forces": [
                    {
                        "force_id": "force-1",
                        "pillar": "Political",
                        "layer": "primary",
                        "label": "Data sovereignty pressure",
                        "mechanism": "New data rules increase compliance scope.",
                        "direction": "increasing",
                        "affected_dimensions": "risk",
                        "evidence_unit_ids": ["ev-1"],
                        "confidence": 0.7,
                        "confidence_rationale": "Test input.",
                    }
                ]
            }
        return {}


def test_truncation_detector() -> None:
    assert _is_truncation_error(ValueError("Unable to locate JSON object in output."))
    assert _is_truncation_error(RuntimeError("response was truncated mid-stream"))
    assert not _is_truncation_error(RuntimeError("other failure"))


def test_plan_chunks_handles_sparse_targets() -> None:
    chunks = _plan_chunks({"political": 2}, chunk_size=2)
    assert chunks
    assert sum(chunk.get("political", 0) for chunk in chunks) == 2


def test_force_item_correction(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("FORCES_CHUNK_SIZE", "1")
    monkeypatch.setenv("FORCES_CHUNK_MIN", "1")
    state = ScenarioOpsState()
    state.evidence_units = _evidence_units_payload("test-run", 1)

    state = run_force_builder_node(
        run_id="test-run",
        state=state,
        user_params={"value": "Acme", "scope": "world", "horizon": 12},
        llm_client=ForceCorrectionClient(),
        base_dir=tmp_path,
        min_forces=1,
        min_per_domain=0,
    )

    forces = state.forces.get("forces", []) if state.forces else []
    assert forces
    assert forces[0]["evidence_unit_ids"] == ["ev-1"]


def test_chunking_writes_manifest_parts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("FORCES_CHUNK_SIZE", "2")
    monkeypatch.setenv("FORCES_CHUNK_MIN", "1")
    state = ScenarioOpsState()
    state.evidence_units = _evidence_units_payload("chunk-run", 120)

    state = run_force_builder_node(
        run_id="chunk-run",
        state=state,
        user_params={"value": "Acme", "scope": "world", "horizon": 12},
        llm_client=ChunkingClient(),
        base_dir=tmp_path,
        min_forces=4,
        min_per_domain=0,
    )

    forces_dir = tmp_path / "chunk-run" / "artifacts" / "forces"
    manifest_path = forces_dir / "forces_assembly_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest.get("batches")
    assert manifest.get("hashes", {}).get("forces_payload")
    for batch in manifest["batches"]:
        part_path = Path(batch["part_path"])
        assert part_path.exists()
    forces = state.forces.get("forces", []) if state.forces else []
    assert forces
    force_ids = {force["force_id"] for force in forces}
    assert len(force_ids) == len(forces)


def test_dedupe_across_chunks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("FORCES_CHUNK_SIZE", "1")
    monkeypatch.setenv("FORCES_CHUNK_MIN", "1")
    state = ScenarioOpsState()
    state.evidence_units = _evidence_units_payload("dedupe-run", 3)

    state = run_force_builder_node(
        run_id="dedupe-run",
        state=state,
        user_params={"value": "Acme", "scope": "world", "horizon": 12},
        llm_client=DuplicateClient(),
        base_dir=tmp_path,
        min_forces=2,
        min_per_domain=0,
    )

    forces = state.forces.get("forces", []) if state.forces else []
    assert len(forces) == 1


def test_force_normalization_maps_pillar(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("FORCES_CHUNK_SIZE", "1")
    monkeypatch.setenv("FORCES_CHUNK_MIN", "1")
    state = ScenarioOpsState()
    state.evidence_units = _evidence_units_payload("pillar-run", 1)

    state = run_force_builder_node(
        run_id="pillar-run",
        state=state,
        user_params={"value": "Acme", "scope": "world", "horizon": 12},
        llm_client=PillarClient(),
        base_dir=tmp_path,
        min_forces=1,
        min_per_domain=0,
    )

    forces = state.forces.get("forces", []) if state.forces else []
    assert forces
    assert forces[0]["domain"] == "political"
    assert forces[0]["directionality"] == "increasing"
