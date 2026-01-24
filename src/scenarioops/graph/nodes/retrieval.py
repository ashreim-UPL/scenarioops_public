from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence
from urllib.parse import urlparse

from scenarioops.app.config import ScenarioOpsSettings
from scenarioops.graph.gates.source_reputation import classify_publisher
from scenarioops.graph.gates.source_reputation import (
    classify_publisher,
    validate_reputable_sources,
)
from scenarioops.graph.state import ScenarioOpsState
from scenarioops.graph.tools.schema_validate import load_schema, validate_artifact
from scenarioops.graph.tools.storage import write_artifact
from scenarioops.graph.tools.web_retriever import RetrievedContent, retrieve_url
from scenarioops.graph.tools.traceability import build_run_metadata
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
    user_params: Mapping[str, Any] | None = None,
    retriever: Retriever = retrieve_url,
    base_dir: Path | None = None,
    mode: str | None = None,
    allow_web: bool | None = None,
    settings: ScenarioOpsSettings | None = None,
    simulate_evidence: bool | None = None,
) -> ScenarioOpsState:
    resolved_mode = (mode or (settings.mode if settings else "demo")).lower()
    if allow_web is None and settings is not None:
        allow_web = settings.allow_web
    if allow_web is None:
        allow_web = False
    simulate = (
        bool(simulate_evidence)
        if simulate_evidence is not None
        else bool(settings.simulate_evidence if settings else False)
    )
    policy = policy_for_name(settings.sources_policy) if settings else None
    enforce_allowlist = policy.enforce_allowlist if policy else True
    evidence_units: list[dict[str, Any]] = []
    failures: list[str] = []
    metadata = build_run_metadata(
        run_id=run_id,
        user_params=user_params or {},
    )
    
    # Sentinel logic: If sources contain queries (spaces, no scheme), treat as search
    # This requires the retriever to handle it or we preprocess.
    # We'll just pass them through.
    
    for idx, url in enumerate(sources, start=1):
        try:
            # Check if it looks like a query
            if " " in url and "://" not in url:
                if not simulate:
                    raise RuntimeError(
                        "retrieval_failed: search queries require real search "
                        "or --simulate-evidence."
                    )
                retrieved_at = datetime.now(timezone.utc).isoformat()
                evidence_units.append(
                    {
                        "evidence_unit_id": f"sim-{idx}",
                        "source_type": "tertiary",
                        "title": f"Simulated evidence for: {url}",
                        "publisher": "simulated",
                        "date_published": retrieved_at,
                        "url": f"simulated://{url.replace(' ', '+')}",
                        "excerpt": f"Simulated evidence generated for query: {url}.",
                        "claims": [],
                        "metrics": [],
                        "reliability_grade": "D",
                        "reliability_reason": "simulated evidence",
                        "geography_tags": [metadata["geography"]],
                        "domain_tags": [],
                        "simulated": True,
                    }
                )
                continue

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
            publisher = _publisher(retrieved.url, retrieved.title)
            category = classify_publisher(retrieved.url)
            reliability_grade = "B" if category in {"government", "academic"} else "C"
            evidence_units.append(
                {
                    "evidence_unit_id": f"ev-{idx}",
                    "source_type": "secondary",
                    "title": retrieved.title or url,
                    "publisher": publisher,
                    "date_published": retrieved_at,
                    "url": retrieved.url,
                    "excerpt": excerpt,
                    "claims": [],
                    "metrics": [],
                    "reliability_grade": reliability_grade,
                    "reliability_reason": f"{category} source",
                    "geography_tags": [metadata["geography"]],
                    "domain_tags": [],
                    "simulated": False,
                }
            )
        except Exception as exc:
            failures.append(f"{url}: {exc}")
            continue

    if failures and not evidence_units:
        raise RuntimeError(
            "retrieval_failed: " + "; ".join(failures[:3])
        )
    if not evidence_units:
        raise RuntimeError("retrieval_failed: no evidence units retrieved.")

    if resolved_mode == "live":
        # In live mode we might skip strict validations if we generated search results
        pass

    simulated_flag = any(unit.get("simulated") for unit in evidence_units)
    payload = {
        **metadata,
        "simulated": simulated_flag,
        "evidence_units": evidence_units,
    }
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
