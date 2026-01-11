from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from app.config import LLMConfig
from scenarioops.graph.nodes.utils import get_client, load_prompt, render_prompt
from scenarioops.graph.state import DriverEntry, Drivers, ScenarioOpsState
from scenarioops.graph.tools.schema_validate import load_schema, validate_jsonl
from scenarioops.graph.tools.storage import write_artifact
from scenarioops.graph.tools.web_retriever import RetrievedContent, retrieve_url
from scenarioops.llm.guards import ensure_dict, ensure_key, ensure_list, truncate_for_log


Retriever = Callable[..., RetrievedContent]


def _validate_citations(drivers: Sequence[Mapping[str, Any]], sources: Sequence[str]) -> None:
    source_set = set(sources)
    for entry in drivers:
        citations = entry.get("citations", [])
        if not citations:
            raise ValueError(f"Driver {entry.get('id')} missing citations.")
        for citation in citations:
            url = citation.get("url")
            if url not in source_set:
                raise ValueError(f"Citation url not in sources: {url}")


def run_drivers_node(
    sources: Sequence[str],
    *,
    run_id: str,
    state: ScenarioOpsState,
    llm_client=None,
    retriever: Retriever = retrieve_url,
    base_dir: Path | None = None,
    config: LLMConfig | None = None,
) -> ScenarioOpsState:
    prompt_template = load_prompt("drivers")
    retrieved = []
    for url in sources:
        retrieved_content = retriever(url, run_id=run_id, base_dir=base_dir)
        retrieved.append(retrieved_content.to_dict())

    context = {
        "charter": state.charter.__dict__ if state.charter else None,
        "sources": list(sources),
        "retrieved": retrieved,
    }
    prompt = render_prompt(prompt_template, context)
    client = get_client(llm_client, config)
    schema = load_schema("drivers_list")
    response = client.generate_json(prompt, schema)
    try:
        payload = ensure_dict(response, node_name="drivers")
    except TypeError as exc:
        if isinstance(response, list):
            raw = truncate_for_log(getattr(response, "raw", None) or repr(response))
            raise TypeError(
                "drivers: Expected payload['drivers'] list, "
                f"received {type(response)}. raw={raw}"
            ) from exc
        raise

    drivers_payload = ensure_key(payload, "drivers", node_name="drivers")
    if not isinstance(drivers_payload, list):
        raw = truncate_for_log(getattr(payload, "raw", None) or repr(payload))
        raise TypeError(
            "drivers: Expected payload['drivers'] list, "
            f"received {type(drivers_payload)}. raw={raw}"
        )
    drivers_payload = ensure_list(drivers_payload, node_name="drivers")

    validate_jsonl("driver_entry", drivers_payload)
    _validate_citations(drivers_payload, sources)

    write_artifact(
        run_id=run_id,
        artifact_name="drivers",
        payload=drivers_payload,
        ext="jsonl",
        input_values={"sources": list(sources)},
        prompt_values={"prompt": prompt},
        tool_versions={"drivers_node": "0.1.0"},
        base_dir=base_dir,
    )

    driver_entries = [DriverEntry(**entry) for entry in drivers_payload]
    state.drivers = Drivers(id=f"drivers-{run_id}", title="Drivers", drivers=driver_entries)
    return state
