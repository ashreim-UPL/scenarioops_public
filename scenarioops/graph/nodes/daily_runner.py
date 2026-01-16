from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from scenarioops.app.config import LLMConfig
from scenarioops.graph.nodes.utils import get_client, load_prompt, render_prompt
from scenarioops.graph.state import DailyBrief, ScenarioOpsState
from scenarioops.graph.tools.normalization import stable_id
from scenarioops.graph.tools.schema_validate import validate_artifact
from scenarioops.graph.tools.scenario_activation import (
    compute_activation_deltas,
    compute_scenario_activation,
)
from scenarioops.graph.tools.storage import log_normalization, write_artifact


def _wind_tunnel_scores(tests: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    scores: dict[str, list[float]] = {}
    for test in tests:
        scenario_id = str(test.get("scenario_id"))
        score = float(test.get("rubric_score", 0.0))
        scores.setdefault(scenario_id, []).append(score)
    return {
        scenario_id: (sum(values) / len(values)) if values else 0.0
        for scenario_id, values in scores.items()
    }


def _wind_tunnel_deltas(
    today_tests: Sequence[Mapping[str, Any]],
    yesterday_tests: Sequence[Mapping[str, Any]] | None,
) -> dict[str, float]:
    today_scores = _wind_tunnel_scores(today_tests)
    yesterday_scores = _wind_tunnel_scores(yesterday_tests or [])
    scenario_ids = set(today_scores.keys()) | set(yesterday_scores.keys())
    return {
        scenario_id: today_scores.get(scenario_id, 0.0)
        - yesterday_scores.get(scenario_id, 0.0)
        for scenario_id in scenario_ids
    }


def run_daily_runner_node(
    signals: Sequence[Mapping[str, Any]],
    *,
    run_id: str,
    state: ScenarioOpsState,
    llm_client=None,
    base_dir: Path | None = None,
    config: LLMConfig | None = None,
    report_date: str | None = None,
    yesterday_activation=None,
    yesterday_wind_tunnel: Sequence[Mapping[str, Any]] | None = None,
) -> ScenarioOpsState:
    prompt_template = load_prompt("daily_runner")
    client = get_client(llm_client, config)

    missing_reasons = []
    if state.ewi is None:
        missing_reasons.append("EWIs missing")
    if state.wind_tunnel is None:
        missing_reasons.append("wind tunnel tests missing")
    if not signals:
        missing_reasons.append("signals missing")
    missing_data = bool(missing_reasons)

    context = {
        "signals": list(signals),
        "indicators": [indicator.__dict__ for indicator in state.ewi.indicators]
        if state.ewi
        else [],
        "wind_tunnel_tests": [test.__dict__ for test in state.wind_tunnel.tests]
        if state.wind_tunnel
        else [],
    }

    prompt = render_prompt(prompt_template, context)
    if missing_data:
        _ = client.generate_markdown(prompt)
        reason = ", ".join(missing_reasons)
        markdown = f"# Daily Brief\n\nData unavailable: {reason}.\n"
    else:
        markdown = client.generate_markdown(prompt)

    validate_artifact("markdown", markdown)

    if report_date is None:
        report_date = datetime.now(timezone.utc).date().isoformat()
        log_normalization(
            run_id=run_id,
            node_name="daily_runner",
            operation="default_report_date",
            details={"value": report_date},
            base_dir=base_dir,
        )

    write_artifact(
        run_id=run_id,
        artifact_name="daily_brief",
        payload=markdown,
        ext="md",
        input_values={"signals_count": len(signals), "missing_data": missing_data},
        prompt_values={"prompt": prompt},
        tool_versions={"daily_runner_node": "0.1.0"},
        base_dir=base_dir,
    )

    brief_id = stable_id(
        "daily-brief",
        report_date,
        len(signals),
        missing_data,
    )
    log_normalization(
        run_id=run_id,
        node_name="daily_runner",
        operation="stable_id_assigned",
        details={"field": "id"},
        base_dir=base_dir,
    )
    daily_brief = {
        "id": brief_id,
        "date": report_date,
        "headline": "data unavailable" if missing_data else "Daily brief",
        "developments": [],
        "signals": [],
        "implications": [],
        "actions": [],
        "markdown": markdown,
    }
    if missing_data:
        daily_brief["notes"] = "data unavailable: " + ", ".join(missing_reasons)

    if not missing_data:
        activations = compute_scenario_activation(
            indicators=[indicator.__dict__ for indicator in state.ewi.indicators],
            signals=signals,
        )
        deltas = compute_activation_deltas(activations, yesterday_activation)
        wind_deltas = _wind_tunnel_deltas(
            [test.__dict__ for test in state.wind_tunnel.tests],
            yesterday_wind_tunnel,
        )
        daily_brief["signals"] = [
            f"{item.scenario_id}: {item.score_delta:+.2f} ({item.band_from}->{item.band_to})"
            for item in deltas
        ]
        daily_brief["implications"] = [
            f"{scenario_id}: {delta:+.2f}"
            for scenario_id, delta in sorted(wind_deltas.items())
        ]

    validate_artifact("daily_brief", daily_brief)

    write_artifact(
        run_id=run_id,
        artifact_name="daily_brief",
        payload=daily_brief,
        ext="json",
        input_values={"signals_count": len(signals), "missing_data": missing_data},
        prompt_values={"prompt": prompt},
        tool_versions={"daily_runner_node": "0.1.0"},
        base_dir=base_dir,
    )

    state.daily_brief = DailyBrief(
        id=daily_brief["id"],
        date=daily_brief["date"],
        headline=daily_brief["headline"],
        developments=daily_brief["developments"],
        signals=daily_brief["signals"],
        implications=daily_brief["implications"],
        actions=daily_brief["actions"],
        notes=daily_brief.get("notes"),
        markdown=daily_brief["markdown"],
    )
    return state
