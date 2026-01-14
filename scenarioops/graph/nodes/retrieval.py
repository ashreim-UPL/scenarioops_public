from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence
from urllib.parse import urlparse

from scenarioops.app.config import ScenarioOpsSettings
from scenarioops.graph.gates.source_reputation import (
    classify_publisher,
    validate_reputable_sources,
)
from scenarioops.graph.state import ScenarioOpsState
from scenarioops.graph.tools.schema_validate import load_schema, validate_artifact
from scenarioops.graph.tools.storage import write_artifact
from scenarioops.graph.tools.web_retriever import RetrievedContent, retrieve_url
from scenarioops.sources.policy import policy_for_name


Retriever = Callable[..., RetrievedContent]


def _excerpt(text: str, length: int = 1000) -> str:
    return text[:length]


def _publisher(url: str, title: str | None = None) -> str:
    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    if hostname:
        return hostname
    return title or url


def run_retrieval_node(
    sources: Sequence[str],
    *,
    run_id: str,
    state: ScenarioOpsState,
    retriever: Retriever = retrieve_url,
    base_dir: Path | None = None,
    mode: str | None = None,
    allow_web: bool | None = None,
    settings: ScenarioOpsSettings | None = None,
) -> ScenarioOpsState:
    resolved_mode = (mode or (settings.mode if settings else "demo")).lower()
    if allow_web is None and settings is not None:
        allow_web = settings.allow_web
    if allow_web is None:
        allow_web = False
    policy = policy_for_name(settings.sources_policy) if settings else None
    enforce_allowlist = policy.enforce_allowlist if policy else True
    evidence_units: list[dict[str, Any]] = []
    for idx, url in enumerate(sources, start=1):
        try:
            retrieved = retriever(
                url,
                run_id=run_id,
                base_dir=base_dir,
                allow_web=allow_web,
                enforce_allowlist=enforce_allowlist,
            )
            excerpt = _excerpt(retrieved.text)
            retrieved_at = datetime.now(timezone.utc).isoformat()
            if resolved_mode == "demo" and retrieved.date:
                retrieved_at = retrieved.date
            evidence_units.append(
                {
                    "id": f"ev-{idx}",
                    "title": retrieved.title or url,
                    "url": retrieved.url,
                    "publisher": _publisher(retrieved.url, retrieved.title),
                    "retrieved_at": retrieved_at,
                    "excerpt": excerpt,
                }
            )
        except Exception as exc:
            print(f"Warning: Failed to retrieve {url}: {exc}")
            continue

    if resolved_mode == "live":
        validate_reputable_sources(evidence_units)
        if settings is not None and settings.min_sources_per_domain > 0:
            if len(evidence_units) < settings.min_sources_per_domain:
                raise ValueError(
                    "Insufficient evidence sources: "
                    f"{len(evidence_units)} < {settings.min_sources_per_domain}."
                )
        if policy and policy.allowed_categories:
            allowed = set(policy.allowed_categories)
            for unit in evidence_units:
                category = classify_publisher(str(unit.get("url", "")))
                if category not in allowed:
                    raise ValueError(
                        f"Source category '{category}' not allowed by policy."
                    )

    payload = {"evidence_units": evidence_units}
    schema = load_schema("evidence_units.schema")
    validate_artifact("evidence_units.schema", payload)

    write_artifact(
        run_id=run_id,
        artifact_name="evidence_units",
        payload=payload,
        ext="json",
        input_values={"source_count": len(sources)},
        prompt_values={"prompt": "retrieval"},
        tool_versions={"retrieval_node": "0.1.0"},
        base_dir=base_dir,
    )

    state.evidence_units = payload
    return state
