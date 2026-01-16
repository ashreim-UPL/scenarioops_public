from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from scenarioops.app.config import load_settings, llm_config_from_settings
from scenarioops.graph.state import ScenarioOpsState
from scenarioops.graph.types import GraphInputs
from scenarioops.graph.setup import default_sources, mock_payloads_for_sources, mock_retriever
from scenarioops.graph.nodes.charter import run_charter_node
from scenarioops.graph.nodes.focal_issue import run_focal_issue_node
from scenarioops.graph.nodes.utils import get_client
from scenarioops.graph.tools.web_retriever import retrieve_url

from .sentinel import Sentinel
from .analyst import Analyst
from .strategist import Strategist
from .critic import Critic
from .client_wrapper import SquadClient
from .telemetry import record_node_event

def run_squad_orchestrator(
    inputs: GraphInputs,
    *,
    run_id: str,
    base_dir: Path | None = None,
    mock_mode: bool = False,
    generate_strategies: bool = True,
) -> ScenarioOpsState:
    """Orchestrates the Dynamic Strategy Team (Squad)."""
    
    overrides: dict[str, Any] = {}
    if mock_mode:
        overrides["mode"] = "demo"
        overrides["sources_policy"] = "fixtures"
        overrides["llm_provider"] = "mock"
    
    settings = load_settings(overrides)
    config = llm_config_from_settings(settings)
    
    # Initialize State
    state = ScenarioOpsState()
    
    # Initialize Base Client (Shared History Holder)
    base_client = get_client(None, config)
    mock_payloads = mock_payloads_for_sources(inputs.sources or default_sources()) if mock_mode else None
    
    # In a real impl, we'd wrap the specific node clients. 
    # Here we create a single shared SquadClient to hold the "Thought Signatures" (history).
    # This fulfills: "Update the orchestrator to pass the Gemini3Client.history... between agents"
    squad_client = SquadClient(base_client, thinking_level="low")
    
    # Pre-Squad Setup (Charter/Focal Issue) - typically part of setup
    # We run these using the squad client to maintain context.
    # Note: Using run_charter_node directly as it wasn't assigned to a specific agent in instructions,
    # or arguably belongs to Analyst or Sentinel. We'll run it as prep.
    llm_label = f"llm:{getattr(config, 'mode', 'unknown')}"
    record_node_event(
        run_id=run_id,
        node_name="charter",
        inputs=["user_params"],
        outputs=["charter.json"],
        tools=[llm_label],
        base_dir=base_dir,
        action=lambda: run_charter_node(
            user_params=inputs.user_params,
            run_id=run_id,
            state=state,
            llm_client=squad_client,
            base_dir=base_dir,
            config=config,
        ),
    )
    record_node_event(
        run_id=run_id,
        node_name="focal_issue",
        inputs=["charter.json"],
        outputs=["focal_issue.json"],
        tools=[llm_label],
        base_dir=base_dir,
        action=lambda: run_focal_issue_node(
            user_params=inputs.user_params,
            state=state,
            llm_client=squad_client,
            config=config,
        ),
    )
    
    # 1. Sentinel (Exploration)
    # "Configure sentinel.py to accept company and country. Use enable_search=True and thinking_level='low'"
    # We extract company/country from user_params or default.
    company = str(inputs.user_params.get("company", "Unknown Company"))
    country = str(inputs.user_params.get("country", inputs.user_params.get("value", "Unknown Country")))
    
    sentinel = Sentinel(
        company=company, 
        country=country, 
        enable_search=True, 
        thinking_level="low"
    )
    # Sentinel uses the squad_client, potentially updating its thinking_level locally if needed,
    # but since we pass the client instance, we should ensure the client respects the agent's pref.
    # Our SquadClient wrapper allows init with level, but here we share the instance. 
    # We might need to update the client's thinking level on the fly or create a child context.
    squad_client.thinking_level = sentinel.thinking_level
    
    retriever = mock_retriever if mock_mode else retrieve_url

    state = sentinel.explore(
        state=state,
        sources=list(inputs.sources) if inputs.sources else [],
        run_id=run_id,
        base_dir=base_dir,
        config=config,
        settings=settings,
        llm_client=squad_client,
        retriever=retriever
    )
    
    # 2. Analyst (Logic/Narratives)
    analyst = Analyst()
    squad_client.thinking_level = "low" # Reset/Default for Analyst
    state = analyst.process(
        state=state,
        run_id=run_id,
        base_dir=base_dir,
        config=config,
        llm_client=squad_client
    )
    
    # 3. Strategist (Strategies + ROI)
    if generate_strategies:
        strategist = Strategist()
        squad_client.thinking_level = "low"
        state = strategist.generate(
            state=state,
            run_id=run_id,
            base_dir=base_dir,
            config=config,
            llm_client=squad_client
        )
    
    # 4. Critic (Wind Tunnel / Validation)
    critic = Critic(thinking_level="high")
    squad_client.thinking_level = critic.thinking_level
    state = critic.validate(
        state=state,
        run_id=run_id,
        base_dir=base_dir,
        config=config,
        settings=settings,
        llm_client=squad_client
    )
    
    return state
