from __future__ import annotations

from pathlib import Path

from scenarioops.app.config import LLMConfig
from scenarioops.graph.nodes.utils import get_client, load_prompt, render_prompt
from scenarioops.graph.state import ScenarioOpsState, Strategies, Strategy
from scenarioops.graph.tools.normalization import stable_id
from scenarioops.graph.tools.schema_validate import load_schema, validate_artifact
from scenarioops.graph.tools.storage import log_normalization, write_artifact
from scenarioops.llm.guards import ensure_dict


def run_strategies_node(
    *,
    run_id: str,
    state: ScenarioOpsState,
    strategy_notes: str | None = None,
    llm_client=None,
    base_dir: Path | None = None,
    config: LLMConfig | None = None,
) -> ScenarioOpsState:
    if state.logic is None:
        raise ValueError("Logic is required to generate strategies.")

    prompt_template = load_prompt("strategies")
    scenarios = [scenario.__dict__ for scenario in state.logic.scenarios]
    context = {"scenarios": scenarios}
    if strategy_notes:
        context["strategy_notes"] = strategy_notes
    prompt = render_prompt(prompt_template, context)

    client = get_client(llm_client, config)
    schema = load_schema("strategies")
    response = client.generate_json(prompt, schema)
    parsed = ensure_dict(response, node_name="strategies")

    if not parsed.get("id"):
        parsed["id"] = stable_id(
            "strategies",
            parsed.get("title"),
            [scenario.id for scenario in state.logic.scenarios],
        )
        log_normalization(
            run_id=run_id,
            node_name="strategies",
            operation="stable_id_assigned",
            details={"field": "id"},
            base_dir=base_dir,
        )

    validate_artifact("strategies", parsed)
    for strategy in parsed.get("strategies", []):
        if not strategy.get("id"):
            strategy["id"] = stable_id(
                "strategy",
                strategy.get("name"),
                strategy.get("objective"),
                strategy.get("actions", []),
            )
            log_normalization(
                run_id=run_id,
                node_name="strategies",
                operation="stable_id_assigned",
                details={"field": "strategy.id", "name": strategy.get("name", "")},
                base_dir=base_dir,
            )
        if not strategy.get("kpis"):
            raise ValueError(f"Strategy {strategy.get('id')} missing KPIs.")

    write_artifact(
        run_id=run_id,
        artifact_name="strategies",
        payload=parsed,
        ext="json",
        input_values={
            "scenario_ids": [scenario.id for scenario in state.logic.scenarios],
            "strategy_notes": strategy_notes or "",
        },
        prompt_values={"prompt": prompt},
        tool_versions={"strategies_node": "0.1.0"},
        base_dir=base_dir,
    )

    strategies = [Strategy(**strategy) for strategy in parsed.get("strategies", [])]
    state.strategies = Strategies(
        id=parsed.get("id", f"strategies-{run_id}"),
        title=parsed.get("title", "Strategies"),
        strategies=strategies,
    )
    return state
