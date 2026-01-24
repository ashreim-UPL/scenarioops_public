from __future__ import annotations

from pathlib import Path

from scenarioops.app.config import ScenarioOpsSettings
from scenarioops.graph.nodes.scenario_media import run_scenario_media_node
from scenarioops.graph.state import ScenarioOpsState
from scenarioops.llm.client import MockLLMClient


def _scenario_payload() -> dict[str, object]:
    scenarios = []
    for idx in range(1, 5):
        scenarios.append(
            {
                "scenario_id": f"S{idx}",
                "name": f"Scenario {idx}",
                "axis_states": {"axis-1": "low", "axis-2": "high"},
                "narrative": f"Narrative {idx}",
                "signposts": ["signal"],
                "implications": ["implication"],
                "no_regret_moves": ["move"],
                "contingent_moves": ["contingent"],
                "evidence_touchpoints": {
                    "cluster_ids": ["cluster-1", "cluster-2"],
                    "force_ids": ["force-1", "force-2"],
                },
            }
        )
    return {
        "needs_correction": False,
        "warnings": [],
        "axes": ["axis-1", "axis-2"],
        "scenarios": scenarios,
    }


def test_scenario_media_enriches_story_and_images(tmp_path: Path) -> None:
    run_id = "scenario-media"
    settings = ScenarioOpsSettings(allow_web=False, llm_provider="mock")
    state = ScenarioOpsState(scenarios=_scenario_payload())

    state = run_scenario_media_node(
        run_id=run_id,
        state=state,
        user_params={"value": "Acme", "scope": "global", "horizon": 12},
        llm_client=MockLLMClient(),
        base_dir=tmp_path,
        settings=settings,
    )

    payload = state.scenarios
    assert payload is not None
    scenarios = payload.get("scenarios", [])
    assert scenarios
    for scenario in scenarios:
        assert scenario.get("story_text")
        assert scenario.get("visual_prompt")
        image_rel = scenario.get("image_artifact_path")
        assert image_rel
        image_path = tmp_path / run_id / Path(str(image_rel))
        assert image_path.exists()
        assert image_path.stat().st_size > 0
