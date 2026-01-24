from __future__ import annotations

from pathlib import Path

from scenarioops.app.config import LLMConfig
from scenarioops.graph.nodes.utils import build_prompt, get_client
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
    scenario_payload = None
    if state.logic is None:
        if state.scenarios is None:
            raise ValueError("Logic or scenarios are required to generate strategies.")
        scenario_payload = state.scenarios.get("scenarios", [])

    scenarios = (
        [scenario.__dict__ for scenario in state.logic.scenarios]
        if state.logic is not None
        else (scenario_payload or [])
    )
    prompt_context = {"scenarios": scenarios}
    if strategy_notes:
        prompt_context["strategy_notes"] = strategy_notes
    prompt_bundle = build_prompt("strategies", prompt_context)
    prompt = prompt_bundle.text

    client = get_client(llm_client, config)
    schema = load_schema("strategies")
    response = client.generate_json(prompt, schema)
    parsed = ensure_dict(response, node_name="strategies")

    if not parsed.get("id"):
        scenario_ids = (
            [scenario.id for scenario in state.logic.scenarios]
            if state.logic is not None
            else [str(entry.get("scenario_id") or entry.get("id")) for entry in scenarios if isinstance(entry, dict)]
        )
        parsed["id"] = stable_id(
            "strategies",
            parsed.get("title"),
            scenario_ids,
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
            "scenario_ids": (
                [scenario.id for scenario in state.logic.scenarios]
                if state.logic is not None
                else [str(entry.get("scenario_id") or entry.get("id")) for entry in scenarios if isinstance(entry, dict)]
            ),
            "strategy_notes": strategy_notes or "",
        },
        prompt_values={
            "prompt_name": prompt_bundle.name,
            "prompt_sha256": prompt_bundle.sha256,
        },
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
