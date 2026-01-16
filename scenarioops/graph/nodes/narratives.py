from __future__ import annotations

import re
from pathlib import Path

from scenarioops.app.config import LLMConfig
from scenarioops.graph.nodes.utils import get_client, load_prompt, render_prompt
from scenarioops.graph.state import ScenarioOpsState
from scenarioops.graph.tools.schema_validate import validate_artifact
from scenarioops.graph.tools.storage import write_artifact


_SECTION_RE = re.compile(r"^##\s+(?P<title>.+)$")
_CITATION_RE = re.compile(r"(\[[0-9]+\]|\[[^\]]+\]\([^)]+\)|https?://)")
_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")


def parse_narratives(markdown: str) -> dict[str, str]:
    narratives: dict[str, list[str]] = {}
    current_id: str | None = None
    for line in markdown.splitlines():
        match = _SECTION_RE.match(line.strip())
        if match:
            title = match.group("title").strip()
            scenario_id = title.replace("Scenario:", "").strip()
            current_id = scenario_id
            narratives.setdefault(current_id, [])
            continue
        if current_id is not None:
            narratives[current_id].append(line)

    return {key: "\n".join(lines).strip() for key, lines in narratives.items() if lines}


def extract_numeric_claims_without_citations(markdown: str) -> list[str]:
    flagged = []
    for line in markdown.splitlines():
        if _NUMBER_RE.search(line) and not _CITATION_RE.search(line):
            flagged.append(line.strip())
    return [line for line in flagged if line]


def run_narratives_node(
    *,
    run_id: str,
    state: ScenarioOpsState,
    llm_client=None,
    base_dir: Path | None = None,
    config: LLMConfig | None = None,
) -> ScenarioOpsState:
    if state.skeleton is None:
        raise ValueError("Skeletons are required to generate narratives.")

    prompt_template = load_prompt("narratives")
    skeletons = [scenario.__dict__ for scenario in state.skeleton.scenarios]
    prompt = render_prompt(prompt_template, {"skeletons": skeletons})

    client = get_client(llm_client, config)
    markdown = client.generate_markdown(prompt)

    narratives = parse_narratives(markdown)
    if len(narratives) != 4:
        raise ValueError("Narratives output must include 4 scenario sections.")

    for scenario_id, narrative in narratives.items():
        validate_artifact("markdown", narrative)
        write_artifact(
            run_id=run_id,
            artifact_name=f"narrative_{scenario_id}",
            payload=narrative,
            ext="md",
            input_values={"scenario_id": scenario_id},
            prompt_values={"prompt": prompt},
            tool_versions={"narratives_node": "0.1.0"},
            base_dir=base_dir,
        )

    state.narratives = narratives
    return state
