from __future__ import annotations

from pathlib import Path

from scenarioops.app.config import LLMConfig
from scenarioops.graph.nodes.logic import run_logic_node
from scenarioops.graph.nodes.narratives import run_narratives_node
from scenarioops.graph.nodes.skeletons import run_skeletons_node
from scenarioops.graph.nodes.drivers import run_drivers_node
from scenarioops.graph.nodes.uncertainties import run_uncertainties_node
from scenarioops.graph.nodes.utils import get_client
from scenarioops.graph.state import ScenarioOpsState
from .types import Gemini3Client
from .client_wrapper import SquadClient
from .telemetry import record_node_event

class Analyst:
    """Analyst agent for logic, skeletons, and narratives."""

    def process(
        self,
        state: ScenarioOpsState,
        run_id: str,
        base_dir: Path | None = None,
        config: LLMConfig | None = None,
        llm_client=None,
    ) -> ScenarioOpsState:
        """Executes the analysis pipeline."""
        
        client = get_client(llm_client, config)
        if not isinstance(client, Gemini3Client):
            client = SquadClient(client, thinking_level="low") # Default for Analyst

        llm_label = f"llm:{getattr(config, 'mode', 'unknown')}"

        state = record_node_event(
            run_id=run_id,
            node_name="drivers",
            inputs=["evidence_units.json"],
            outputs=["drivers.jsonl"],
            tools=[llm_label],
            base_dir=base_dir,
            action=lambda: run_drivers_node(
                run_id=run_id,
                state=state,
                llm_client=client,
                base_dir=base_dir,
                config=config,
            ),
        )

        state = record_node_event(
            run_id=run_id,
            node_name="uncertainties",
            inputs=["drivers.jsonl"],
            outputs=["uncertainties.json"],
            tools=[llm_label],
            base_dir=base_dir,
            action=lambda: run_uncertainties_node(
                run_id=run_id,
                state=state,
                llm_client=client,
                base_dir=base_dir,
                config=config,
            ),
        )

        state = record_node_event(
            run_id=run_id,
            node_name="logic",
            inputs=["uncertainties.json"],
            outputs=["logic.json"],
            tools=[llm_label],
            base_dir=base_dir,
            action=lambda: run_logic_node(
                run_id=run_id,
                state=state,
                llm_client=client,
                base_dir=base_dir,
                config=config,
            ),
        )

        state = record_node_event(
            run_id=run_id,
            node_name="skeletons",
            inputs=["logic.json"],
            outputs=["skeletons.json"],
            tools=[llm_label],
            base_dir=base_dir,
            action=lambda: run_skeletons_node(
                run_id=run_id,
                state=state,
                llm_client=client,
                base_dir=base_dir,
                config=config,
            ),
        )

        state = record_node_event(
            run_id=run_id,
            node_name="narratives",
            inputs=["skeletons.json"],
            outputs=["narrative_*.md"],
            tools=[llm_label],
            base_dir=base_dir,
            action=lambda: run_narratives_node(
                run_id=run_id,
                state=state,
                llm_client=client,
                base_dir=base_dir,
                config=config,
            ),
        )

        return state
