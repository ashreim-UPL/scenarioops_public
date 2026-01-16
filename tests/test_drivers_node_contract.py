import pytest

from scenarioops.graph.nodes.drivers import run_drivers_node
from scenarioops.graph.state import ScenarioOpsState


class StubClient:
    def __init__(self, payload) -> None:
        self._payload = payload

    def generate_json(self, prompt: str, schema) -> dict:
        return self._payload

    def generate_markdown(self, prompt: str) -> str:
        raise NotImplementedError


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


def _evidence_units(sources: list[str]) -> dict:
    units = []
    for idx, url in enumerate(sources, start=1):
        units.append(
            {
                "id": f"ev-{idx}",
                "title": "Example",
                "url": url,
                "publisher": "example.com",
                "retrieved_at": "2026-01-01T00:00:00Z",
                "excerpt": f"Source {url}",
            }
        )
    return {"evidence_units": units}


def test_drivers_node_accepts_object_payload(tmp_path) -> None:
    sources = ["https://example.com/a", "https://example.com/b", "https://example.com/c"]
    payload = {"drivers": _drivers_list(sources)}

    state = ScenarioOpsState(evidence_units=_evidence_units(sources))
    run_drivers_node(
        run_id="drivers-ok",
        state=state,
        llm_client=StubClient(payload),
        base_dir=tmp_path / "runs",
    )

    assert state.drivers is not None
    assert state.drivers.drivers


def test_drivers_node_rejects_list_payload(tmp_path) -> None:
    sources = ["https://example.com/a", "https://example.com/b", "https://example.com/c"]
    payload = _drivers_list(sources)

    with pytest.raises(TypeError, match="Expected payload\\['drivers'\\] list"):
        run_drivers_node(
            run_id="drivers-bad",
            state=ScenarioOpsState(evidence_units=_evidence_units(sources)),
            llm_client=StubClient(payload),
            base_dir=tmp_path / "runs",
        )
