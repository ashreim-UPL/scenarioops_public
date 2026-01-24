from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlparse

from scenarioops.app.config import LLMConfig, ScenarioOpsSettings, llm_config_from_settings
from scenarioops.graph.gates.source_reputation import classify_publisher
from scenarioops.graph.state import ScenarioOpsState
from scenarioops.graph.tools.schema_validate import validate_artifact
from scenarioops.graph.tools.storage import write_artifact
from scenarioops.graph.tools.web_retriever import RetrievedContent, retrieve_url
from scenarioops.graph.tools.traceability import build_run_metadata
from scenarioops.graph.tools.evidence_processing import fallback_summary, summarize_text
from scenarioops.llm.client import get_llm_client
from scenarioops.sources.policy import policy_for_name


Retriever = Callable[..., RetrievedContent]


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
    resolved_settings = settings or ScenarioOpsSettings()
    resolved_mode = (mode or resolved_settings.mode).lower()
    if allow_web is None:
        allow_web = resolved_settings.allow_web
    simulate = (
        bool(simulate_evidence)
        if simulate_evidence is not None
        else bool(resolved_settings.simulate_evidence)
    )
    min_ok = int(getattr(resolved_settings, "min_evidence_ok", 10))
    min_total = int(getattr(resolved_settings, "min_evidence_total", 15))
    max_failed_ratio = getattr(resolved_settings, "max_failed_ratio", None)
    if max_failed_ratio is not None:
        max_failed_ratio = float(max_failed_ratio)
    policy = policy_for_name(resolved_settings.sources_policy)
    enforce_allowlist = policy.enforce_allowlist if policy else True
    evidence_units: list[dict[str, Any]] = []
    failures: list[str] = []
    metadata = build_run_metadata(
        run_id=run_id,
        user_params=user_params or {},
    )

    resolved_config = llm_config_from_settings(resolved_settings)
    summarizer_config = LLMConfig(
        model_name=resolved_settings.summarizer_model,
        temperature=resolved_config.temperature,
        timeouts=resolved_config.timeouts,
        mode=resolved_config.mode,
    )
    summarizer_client = get_llm_client(summarizer_config)
    unit_counter = 1

    def _next_unit_id(prefix: str) -> str:
        nonlocal unit_counter
        unit_id = f"{prefix}-{unit_counter}"
        unit_counter += 1
        return unit_id

    def _build_unit(
        *,
        unit_id: str,
        url: str,
        title: str,
        publisher: str,
        date_published: str,
        raw_text: str,
        content_type: str | None,
        http_status: int | None,
        source_method: str,
        simulated_flag: bool,
    ) -> dict[str, Any]:
        status = "ok"
        failure_reason = None
        summary_payload = None
        if not raw_text.strip():
            status = "failed"
            failure_reason = "empty_extracted_text"
        else:
            summary_payload, _, summary_error = summarize_text(
                client=summarizer_client,
                text=raw_text,
                title=title,
                url=url,
            )
            if summary_payload is None:
                status = "partial"
                failure_reason = f"summarizer_failed:{summary_error}"
                summary_payload = fallback_summary(raw_text)

        category = classify_publisher(url)
        reliability_grade = "B" if category in {"government", "academic"} else "C"
        return {
            "id": unit_id,
            "evidence_unit_id": unit_id,
            "url": url,
            "title": title,
            "status": status,
            "failure_reason": failure_reason,
            "summary": summary_payload.get("summary", "") if summary_payload else "",
            "claims": summary_payload.get("claims", []) if summary_payload else [],
            "metrics": summary_payload.get("metrics", []) if summary_payload else [],
            "tags": summary_payload.get("tags", []) if summary_payload else [],
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "content_type": content_type,
            "http_status": http_status,
            "source_method": source_method,
            "embedding_ref": None,
            "raw_text": raw_text[:4000],
            "excerpt": raw_text[:300],
            "publisher": publisher,
            "date_published": date_published,
            "source_type": "secondary",
            "reliability_grade": reliability_grade,
            "reliability_reason": f"{category} source",
            "reliability_notes": summary_payload.get("reliability_notes", "") if summary_payload else "",
            "geography_tags": [metadata["geography"]],
            "domain_tags": [],
            "simulated": simulated_flag,
        }

    for url in sources:
        try:
            if " " in url and "://" not in url:
                if not simulate:
                    raise RuntimeError(
                        "retrieval_failed: search queries require real search "
                        "or --simulate-evidence."
                    )
                unit_id = _next_unit_id("sim")
                raw_text = f"Simulated evidence generated for query: {url}."
                evidence_units.append(
                    _build_unit(
                        unit_id=unit_id,
                        url=f"simulated://{url.replace(' ', '+')}",
                        title=f"Simulated evidence for: {url}",
                        publisher="simulated",
                        date_published=datetime.now(timezone.utc).isoformat(),
                        raw_text=raw_text,
                        content_type=None,
                        http_status=None,
                        source_method="simulation",
                        simulated_flag=True,
                    )
                )
                continue

            retrieved = retriever(
                url,
                run_id=run_id,
                base_dir=base_dir,
                allow_web=allow_web,
                enforce_allowlist=enforce_allowlist,
            )
            retrieved_at = datetime.now(timezone.utc).isoformat()
            if resolved_mode == "demo" and retrieved.date:
                retrieved_at = retrieved.date
            publisher = _publisher(retrieved.url, retrieved.title)
            unit_id = _next_unit_id("ev")
            evidence_units.append(
                _build_unit(
                    unit_id=unit_id,
                    url=retrieved.url,
                    title=retrieved.title or url,
                    publisher=publisher,
                    date_published=retrieved_at,
                    raw_text=retrieved.text,
                    content_type=retrieved.content_type,
                    http_status=retrieved.http_status,
                    source_method="source_url",
                    simulated_flag=False,
                )
            )
        except Exception as exc:
            failures.append(f"{url}: {exc}")
            continue

    if failures and not evidence_units:
        raise RuntimeError("retrieval_failed: " + "; ".join(failures[:3]))
    if not evidence_units:
        raise RuntimeError("retrieval_failed: no evidence units retrieved.")

    counts = {"ok": 0, "partial": 0, "failed": 0, "total": 0}
    for unit in evidence_units:
        status = str(unit.get("status", "ok")).lower()
        counts["total"] += 1
        if status in counts:
            counts[status] += 1
    failed_ratio = counts["failed"] / max(1, counts["total"])
    status = "ok"
    if counts["ok"] < min_ok:
        status = "failed"
    elif counts["total"] < min_total:
        status = "partial"
    if max_failed_ratio is not None and failed_ratio > max_failed_ratio:
        status = "partial" if status == "ok" else status

    retrieval_report = {
        **metadata,
        "schema_version": "1.0",
        "status": status,
        "counts": counts,
        "thresholds": {
            "min_evidence_ok": min_ok,
            "min_evidence_total": min_total,
            "max_failed_ratio": max_failed_ratio,
        },
        "recovery": {
            "attempted_replacements": 0,
            "search_queries_used": 0,
            "exhausted": status == "failed",
            "reasons": failures[:3],
        },
        "notes": [],
    }
    validate_artifact("retrieval_report", retrieval_report)
    write_artifact(
        run_id=run_id,
        artifact_name="retrieval_report",
        payload=retrieval_report,
        ext="json",
        input_values={"status": status},
        tool_versions={"retrieval_node": "0.2.0"},
        base_dir=base_dir,
    )

    simulated_flag = any(unit.get("simulated") for unit in evidence_units)
    payload = {
        **metadata,
        "schema_version": "2.0",
        "simulated": simulated_flag,
        "evidence_units": evidence_units,
    }
    validate_artifact("evidence_units.schema", payload)

    write_artifact(
        run_id=run_id,
        artifact_name="evidence_units",
        payload=payload,
        ext="json",
        input_values={"source_count": len(sources), "simulated": simulated_flag},
        prompt_values={"prompt": "retrieval"},
        tool_versions={"retrieval_node": "0.2.0"},
        base_dir=base_dir,
    )

    state.evidence_units = payload
    if status == "failed":
        raise RuntimeError("retrieval_failed: insufficient evidence in legacy mode.")
    return state
