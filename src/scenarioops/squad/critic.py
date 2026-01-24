from __future__ import annotations

from pathlib import Path

from scenarioops.app.config import LLMConfig, ScenarioOpsSettings
from scenarioops.graph.nodes.auditor import run_auditor_node
from scenarioops.graph.nodes.washout import run_washout_node
from scenarioops.graph.nodes.wind_tunnel import run_wind_tunnel_node
from scenarioops.graph.nodes.utils import get_client
from scenarioops.graph.state import ScenarioOpsState
from .types import Gemini3Client
from .client_wrapper import SquadClient
from .telemetry import record_node_event

class Critic:
    """Critic agent for validation and wind-tunneling."""

    def __init__(self, thinking_level: str = "high"):
        self.thinking_level = thinking_level

    def validate(
        self,
        state: ScenarioOpsState,
        run_id: str,
        base_dir: Path | None = None,
        config: LLMConfig | None = None,
        settings: ScenarioOpsSettings | None = None,
        llm_client=None,
    ) -> ScenarioOpsState:
        """Executes validation (auditor, washout) and wind tunnel."""
        
        client = get_client(llm_client, config)
        if not isinstance(client, Gemini3Client):
            client = SquadClient(client, thinking_level=self.thinking_level)
        llm_label = f"llm:{getattr(config, 'mode', 'unknown')}"
        
        # Wind Tunnel (Deep testing)
        # Only run if strategies exist
        if state.strategies:
            state = record_node_event(
                run_id=run_id,
                node_name="wind_tunnel",
                inputs=["strategies.json"],
                outputs=["wind_tunnel.json"],
                tools=[llm_label],
                base_dir=base_dir,
                action=lambda: run_wind_tunnel_node(
                    run_id=run_id,
                    state=state,
                    llm_client=client,
                    base_dir=base_dir,
                    config=config,
                    settings=settings,
                ),
            )

        state = record_node_event(
            run_id=run_id,
            node_name="washout_report",
            inputs=["driving_forces.json"],
            outputs=["washout_report.json"],
            tools=[llm_label],
            base_dir=base_dir,
            action=lambda: run_washout_node(
                run_id=run_id,
                state=state,
                llm_client=client,
                base_dir=base_dir,
                config=config,
                settings=settings,
            ),
        )

        # Final Auditor check
        state = record_node_event(
            run_id=run_id,
            node_name="auditor",
            inputs=["artifacts/*"],
            outputs=["audit_report.json"],
            tools=[llm_label],
            base_dir=base_dir,
            action=lambda: run_auditor_node(
                run_id=run_id,
                state=state,
                base_dir=base_dir,
                settings=settings,
                config=config,
                llm_client=client,
            ),
        )
        
        return state
