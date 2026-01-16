import pytest

from scenarioops.app.config import ScenarioOpsSettings
from scenarioops.graph.guards.fixture_guard import validate_or_fail
from scenarioops.graph.state import ScenarioOpsState


def test_fixture_blocked_in_live_mode(tmp_path) -> None:
    payload = {
        "forces": [
            {
                "id": "force-1",
                "name": "Political signal 1",
                "domain": "political",
                "description": "Fixture force.",
                "why_it_matters": "Fixture data.",
                "citations": [
                    {"url": "https://example.com/a", "excerpt_hash": "hash-1"},
                ],
            }
        ]
    }

    settings = ScenarioOpsSettings(mode="live", sources_policy="mixed_reputable")
    state = ScenarioOpsState(driving_forces=payload)
    with pytest.raises(
        RuntimeError,
        match="LIVE run contains fixture content; check retriever and sources policy.",
    ):
        validate_or_fail(
            run_id="fixture-live",
            state=state,
            settings=settings,
            base_dir=tmp_path / "runs",
            command="test",
        )
