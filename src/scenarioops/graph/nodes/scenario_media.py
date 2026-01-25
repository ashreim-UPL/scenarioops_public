from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

from scenarioops.app.config import LLMConfig, ScenarioOpsSettings
from scenarioops.graph.nodes.utils import build_prompt, get_client
from scenarioops.graph.state import ScenarioOpsState
from scenarioops.graph.tools.image_generation import (
    get_image_client,
    placeholder_image_bytes,
)
from scenarioops.graph.tools.schema_validate import load_schema, validate_artifact
from scenarioops.graph.tools.storage import (
    ensure_run_dirs,
    log_normalization,
    write_artifact,
)
from scenarioops.graph.tools.traceability import build_run_metadata
from scenarioops.llm.guards import ensure_dict


def _safe_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", value or "")
    return cleaned.strip("_") or "scenario"


def _truncate(value: str, limit: int) -> str:
    if not value:
        return ""
    return value if len(value) <= limit else value[: limit - 3].rstrip() + "..."


def _summary_prompt_for_image(scenario: Mapping[str, Any]) -> str:
    name = str(scenario.get("name", "")).strip()
    axis_states = scenario.get("axis_states") if isinstance(scenario, Mapping) else None
    axis_text = ""
    if isinstance(axis_states, Mapping):
        parts = [f"{key}: {value}" for key, value in axis_states.items() if value]
        axis_text = "; ".join(parts)
    narrative = str(scenario.get("narrative", "")).strip()
    signposts = scenario.get("signposts") if isinstance(scenario, Mapping) else None
    signpost_text = ""
    if isinstance(signposts, list):
        signpost_text = "; ".join(str(item) for item in signposts[:3] if item)
    implications = scenario.get("implications") if isinstance(scenario, Mapping) else None
    implication_text = ""
    if isinstance(implications, list):
        implication_text = "; ".join(str(item) for item in implications[:2] if item)

    parts = []
    if name:
        parts.append(f"Scenario: {name}.")
    if axis_text:
        parts.append(f"Axis states: {axis_text}.")
    if narrative:
        parts.append(f"Narrative: {_truncate(narrative, 420)}")
    if signpost_text:
        parts.append(f"Signals: {signpost_text}.")
    if implication_text:
        parts.append(f"Implications: {implication_text}.")
    prompt = " ".join(parts).strip()
    return _truncate(prompt, 800)


def run_scenario_media_node(
    *,
    run_id: str,
    state: ScenarioOpsState,
    user_params: Mapping[str, Any],
    llm_client=None,
    base_dir: Path | None = None,
    config: LLMConfig | None = None,
    settings: ScenarioOpsSettings | None = None,
) -> ScenarioOpsState:
    if state.scenarios is None:
        raise ValueError("Scenarios are required before scenario media.")

    scenarios_payload = state.scenarios
    scenarios = scenarios_payload.get("scenarios", [])
    if not isinstance(scenarios, list):
        raise TypeError("Scenarios payload must include scenarios list.")

    metadata = build_run_metadata(run_id=run_id, user_params=user_params)
    client = get_client(llm_client, config)
    resolved_settings = settings or ScenarioOpsSettings()
    image_client = get_image_client(resolved_settings)
    schema = load_schema("scenario_story")

    dirs = ensure_run_dirs(run_id, base_dir=base_dir)
    images_dir = dirs["artifacts_dir"] / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    enriched: list[dict[str, Any]] = []
    prompt_sha = ""
    for idx, scenario in enumerate(scenarios, start=1):
        scenario_id = str(
            scenario.get("scenario_id") or scenario.get("id") or f"S{idx}"
        )
        prompt_bundle = build_prompt(
            "scenario_story",
            {
                "company_name": metadata.get("company_name"),
                "geography": metadata.get("geography"),
                "horizon_months": metadata.get("horizon_months"),
                "axes": scenarios_payload.get("axes", []),
                "scenario": scenario,
            },
        )
        if not prompt_sha:
            prompt_sha = prompt_bundle.sha256
        response = client.generate_json(prompt_bundle.text, schema)
        parsed = ensure_dict(response, node_name="scenario_story")
        story_text = str(parsed.get("story_text", "")).strip()
        visual_prompt = str(parsed.get("visual_prompt", "")).strip()

        image_bytes = None
        image_prompt = _summary_prompt_for_image(scenario)
        if image_prompt:
            log_normalization(
                run_id=run_id,
                node_name="scenario_media",
                operation="image_prompt_built",
                details={
                    "scenario_id": scenario_id,
                    "prompt_len": len(image_prompt),
                    "prompt_excerpt": _truncate(image_prompt, 160),
                },
                base_dir=base_dir,
            )
        else:
            log_normalization(
                run_id=run_id,
                node_name="scenario_media",
                operation="image_prompt_missing",
                details={"scenario_id": scenario_id},
                base_dir=base_dir,
            )
        if image_prompt or visual_prompt:
            try:
                image_bytes = image_client.generate_image(
                    image_prompt or visual_prompt, model=resolved_settings.image_model
                )
                log_normalization(
                    run_id=run_id,
                    node_name="scenario_media",
                    operation="image_generation_ok",
                    details={
                        "scenario_id": scenario_id,
                        "model": resolved_settings.image_model,
                        "bytes": len(image_bytes) if image_bytes else 0,
                    },
                    base_dir=base_dir,
                )
            except Exception as exc:
                log_normalization(
                    run_id=run_id,
                    node_name="scenario_media",
                    operation="image_generation_error",
                    details={
                        "scenario_id": scenario_id,
                        "model": resolved_settings.image_model,
                        "error": str(exc),
                    },
                    base_dir=base_dir,
                )
                image_bytes = None
        if not image_bytes:
            image_bytes = placeholder_image_bytes()
            log_normalization(
                run_id=run_id,
                node_name="scenario_media",
                operation="image_generation_placeholder",
                details={"scenario_id": scenario_id},
                base_dir=base_dir,
            )

        safe_id = _safe_id(scenario_id)
        image_name = f"scenario_{safe_id}.png"
        image_path = images_dir / image_name
        image_path.write_bytes(image_bytes)
        image_rel_path = (Path("artifacts") / "images" / image_name).as_posix()

        scenario_enriched = dict(scenario)
        scenario_enriched["scenario_id"] = scenario_id
        scenario_enriched["story_text"] = story_text
        scenario_enriched["visual_prompt"] = visual_prompt
        scenario_enriched["image_artifact_path"] = image_rel_path
        enriched.append(scenario_enriched)

    payload = {
        **metadata,
        "needs_correction": scenarios_payload.get("needs_correction", False),
        "warnings": scenarios_payload.get("warnings", []),
        "axes": scenarios_payload.get("axes", []),
        "scenarios": enriched,
    }
    validate_artifact("scenarios", payload)

    write_artifact(
        run_id=run_id,
        artifact_name="scenarios_enriched",
        payload=payload,
        ext="json",
        input_values={"scenario_count": len(enriched)},
        prompt_values={
            "prompt_name": "scenario_story",
            "prompt_sha256": prompt_sha,
        },
        tool_versions={"scenario_media_node": "0.1.0"},
        base_dir=base_dir,
    )
    state.scenarios = payload
    return state
