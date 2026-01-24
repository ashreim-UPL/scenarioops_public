from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Mapping

import pytest

from scenarioops.app.config import ScenarioOpsSettings
from scenarioops.graph.nodes.force_builder import run_force_builder_node
from scenarioops.graph.nodes.retrieval_real import run_retrieval_real_node
from scenarioops.graph.state import ScenarioOpsState
from scenarioops.graph.tools.web_retriever import RetrievedContent
from scenarioops.llm.client import MockLLMClient


def _content(url: str, text: str) -> RetrievedContent:
    return RetrievedContent(
        url=url,
        title=url,
        date=datetime.now(timezone.utc).isoformat(),
        text=text,
        excerpt_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        content_type="text/html",
        http_status=200,
    )


def _make_retriever(text_map: Mapping[str, str]):
    def _retriever(url: str, **_: object) -> RetrievedContent:
        return _content(url, text_map.get(url, ""))

    return _retriever


def _force_payload(evidence_id: str) -> dict[str, object]:
    return {
        "force_id": "force-1",
        "layer": "primary",
        "domain": "economic",
        "label": "Capital cost volatility",
        "mechanism": "Rates push up financing costs and delay projects.",
        "directionality": "increase",
        "affected_dimensions": ["cost"],
        "evidence_unit_ids": [evidence_id],
        "confidence": 0.7,
        "confidence_rationale": "Test harness evidence link.",
    }


class ForceClient:
    def __init__(self, evidence_id: str) -> None:
        self._evidence_id = evidence_id

    def generate_json(self, prompt: str, schema: Mapping[str, object]) -> dict[str, object]:
        title = schema.get("title") if isinstance(schema, Mapping) else ""
        if title in {"Forces Payload", "Forces"}:
            return {"forces": [_force_payload(self._evidence_id)]}
        if title == "Force Item":
            return _force_payload(self._evidence_id)
        return {}


def test_retrieval_real_stops_below_thresholds(tmp_path) -> None:
    run_id = "retrieval-threshold"
    settings = ScenarioOpsSettings(
        allow_web=False,
        llm_provider="mock",
        min_evidence_ok=1,
        min_evidence_total=1,
    )
    sources = ["https://bad.example.com/a"]
    retriever = _make_retriever({sources[0]: ""})

    with pytest.raises(RuntimeError, match="retrieval_failed"):
        run_retrieval_real_node(
            sources,
            run_id=run_id,
            state=ScenarioOpsState(),
            user_params={"value": "Acme", "scope": "global", "horizon": 12},
            base_dir=tmp_path,
            llm_client=MockLLMClient(),
            settings=settings,
            retriever=retriever,
        )

    report_path = tmp_path / run_id / "artifacts" / "retrieval_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert report["counts"]["ok"] == 0


def test_retrieval_real_recovery_continues_to_forces(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    run_id = "retrieval-recovery"
    settings = ScenarioOpsSettings(
        allow_web=True,
        llm_provider="mock",
        min_evidence_ok=2,
        min_evidence_total=3,
        max_failed_ratio=0.4,
    )
    sources = [
        "https://bad.example.com/a",
        "https://bad.example.com/b",
        "https://bad.example.com/c",
    ]
    good_text = "Evidence content " * 20
    replacements = ["https://good.example.com/1", "https://good.example.com/2"]
    retriever = _make_retriever(
        {
            sources[0]: "",
            sources[1]: "",
            sources[2]: "",
            replacements[0]: good_text,
            replacements[1]: good_text,
        }
    )

    def _fake_search(*_: object, **__: object) -> list[str]:
        return replacements

    monkeypatch.setattr(
        "scenarioops.graph.nodes.retrieval_real.search_web",
        _fake_search,
    )

    state = run_retrieval_real_node(
        sources,
        run_id=run_id,
        state=ScenarioOpsState(),
        user_params={"value": "Acme", "scope": "global", "horizon": 12},
        base_dir=tmp_path,
        llm_client=MockLLMClient(),
        settings=settings,
        retriever=retriever,
    )

    units = state.evidence_units.get("evidence_units", []) if state.evidence_units else []
    assert any(unit.get("status") == "failed" for unit in units)
    ok_units = [unit for unit in units if unit.get("status") == "ok"]
    assert len(ok_units) >= 2

    report_path = tmp_path / run_id / "artifacts" / "retrieval_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "partial"

    evidence_id = str(ok_units[0].get("id") or ok_units[0].get("evidence_unit_id"))
    state = run_force_builder_node(
        run_id=run_id,
        state=state,
        user_params={"value": "Acme", "scope": "global", "horizon": 12},
        llm_client=ForceClient(evidence_id),
        base_dir=tmp_path,
        min_forces=1,
        min_per_domain=0,
    )
    forces = state.forces.get("forces", []) if state.forces else []
    assert forces
