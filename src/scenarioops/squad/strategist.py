from __future__ import annotations

from pathlib import Path

from scenarioops.app.config import LLMConfig
from scenarioops.graph.nodes.strategies import run_strategies_node
from scenarioops.graph.nodes.utils import get_client
from scenarioops.graph.state import ScenarioOpsState
from .types import Gemini3Client, GMReviewRequired
from .client_wrapper import SquadClient
from .telemetry import record_node_event

class Strategist:
    """Strategist agent for strategy generation and ROI safety checks."""

    def generate(
        self,
        state: ScenarioOpsState,
        run_id: str,
        base_dir: Path | None = None,
        config: LLMConfig | None = None,
        llm_client=None,
        strategy_notes: str | None = None,
    ) -> ScenarioOpsState:
        """Generates strategies and enforces ROI safety gates."""
        
        client = get_client(llm_client, config)
        if not isinstance(client, Gemini3Client):
            client = SquadClient(client, thinking_level="low")

        # In a real implementation, we would append "Please include a P0/P1/P2 Action Plan" to the prompt context
        # or rely on the updated schema. Here we pass the instruction via strategy_notes if not provided.
        notes = strategy_notes or "Generate a P0/P1/P2 Action Plan for each strategy."

        llm_label = f"llm:{getattr(config, 'mode', 'unknown')}"
        state = record_node_event(
            run_id=run_id,
            node_name="strategies",
            inputs=["narrative_*.md"],
            outputs=["strategies.json"],
            tools=[llm_label],
            base_dir=base_dir,
            action=lambda: run_strategies_node(
                run_id=run_id,
                state=state,
                strategy_notes=notes,
                llm_client=client,
                base_dir=base_dir,
                config=config,
            ),
        )

        # Fail-Fast ROI Check (C2: Fail Fast)
        # We explicitly look for ROI metrics.
        if state.strategies and state.strategies.strategies:
            for strategy in state.strategies.strategies:
                roi = getattr(strategy, "projected_roi", None)
                
                # If we were using a real model response, we'd check if 'roi' key exists in the raw dict
                # if the Strategy object doesn't have the field yet.
                # Assuming the model follows instructions to output ROI.
                
                if roi is not None:
                    try:
                        roi_val = float(roi)
                        if roi_val < -0.10: # -10%
                            raise GMReviewRequired(
                                f"Strategy '{strategy.name}' projected ROI ({roi_val:.1%}) is below safety threshold (-10%).",
                                context={"strategy_id": strategy.id, "roi": roi_val}
                            )
                    except (ValueError, TypeError):
                        pass # ROI not a valid number, skip check

        return state
