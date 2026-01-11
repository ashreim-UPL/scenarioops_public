import hashlib

import pytest

from scenarioops.graph.nodes.drivers import run_drivers_node
from scenarioops.graph.state import ScenarioOpsState
from scenarioops.graph.tools.web_retriever import RetrievedContent


class StubClient:
    def __init__(self, payload) -> None:
        self._payload = payload

    def generate_json(self, prompt: str, schema) -> dict:
        return self._payload

    def generate_markdown(self, prompt: str) -> str:
        raise NotImplementedError


def _fake_retriever(url: str, **_) -> RetrievedContent:
    text = f"Source {url}"
    return RetrievedContent(
        url=url,
        title="Example",
        date="2026-01-01T00:00:00Z",
        text=text,
        excerpt_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )


def _drivers_list(sources: list[str]) -> list[dict]:
    return [
        {
            "id": "drv-1",
            "name": "Regulatory shift",
            "description": "New reporting requirements.",
            "category": "policy",
            "trend": "tightening",
            "impact": "medium",
            "citations": [{"url": sources[0], "excerpt_hash": "hash-a"}],
        }
    ]


def test_drivers_node_accepts_object_payload(tmp_path) -> None:
    sources = ["https://example.com/a", "https://example.com/b", "https://example.com/c"]
    payload = {"drivers": _drivers_list(sources)}

    state = ScenarioOpsState()
    run_drivers_node(
        sources,
        run_id="drivers-ok",
        state=state,
        llm_client=StubClient(payload),
        retriever=_fake_retriever,
        base_dir=tmp_path / "runs",
    )

    assert state.drivers is not None
    assert state.drivers.drivers


def test_drivers_node_rejects_list_payload(tmp_path) -> None:
    sources = ["https://example.com/a", "https://example.com/b", "https://example.com/c"]
    payload = _drivers_list(sources)

    with pytest.raises(TypeError, match="Expected payload\\['drivers'\\] list"):
        run_drivers_node(
            sources,
            run_id="drivers-bad",
            state=ScenarioOpsState(),
            llm_client=StubClient(payload),
            retriever=_fake_retriever,
            base_dir=tmp_path / "runs",
        )
