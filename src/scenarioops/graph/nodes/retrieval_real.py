from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from scenarioops.app.config import (
    LLMConfig,
    ScenarioOpsSettings,
    llm_config_from_settings,
)
from scenarioops.graph.gates.source_reputation import classify_publisher
from scenarioops.graph.nodes.utils import build_prompt, get_client
from scenarioops.graph.state import ScenarioOpsState
from scenarioops.graph.tools.schema_validate import load_schema, validate_artifact
from scenarioops.graph.tools.storage import write_artifact
from scenarioops.graph.tools.traceability import build_run_metadata
from scenarioops.graph.tools.web_retriever import RetrievedContent, retrieve_url
from scenarioops.graph.tools.search import search_web
from scenarioops.graph.tools.evidence_processing import fallback_summary, summarize_text
from scenarioops.llm.client import get_llm_client
from scenarioops.llm.guards import ensure_dict
from scenarioops.sources.policy import PESTEL_QUERY_TEMPLATES


_CACHE_VERSION = "v2"
_MIN_TEXT_CHARS = 200


def _seed_queries(company: str, geography: str) -> list[str]:
    scope = f"{company} {geography}".strip()
    queries: list[str] = []
    for _, templates in PESTEL_QUERY_TEMPLATES.items():
        if not templates:
            continue
        queries.append(templates[0].format(scope=scope))
    return queries


def _grade_for_publisher(publisher_category: str) -> tuple[str, str]:
    category = publisher_category.lower()
    if category in {"government", "academic", "multilateral"}:
        return "A", f"{publisher_category} source"
    if category in {"consulting", "ngo"}:
        return "B", f"{publisher_category} source"
    if category in {"media"}:
        return "C", f"{publisher_category} source"
    return "D", "commercial or unknown source"


def _source_type_for_bucket(bucket: str) -> str:
    if bucket == "primary":
        return "primary"
    if bucket == "secondary":
        return "secondary"
    return "tertiary"


def _dedupe_urls(urls: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for url in urls:
        if url in seen:
            continue
        seen.add(url)
        deduped.append(url)
    return deduped


def _normalize_text(text: str) -> str:
    return " ".join(text.split())


def _is_text_usable(text: str) -> bool:
    return len(_normalize_text(text)) >= _MIN_TEXT_CHARS


def _unit_status(unit: Mapping[str, Any]) -> str:
    return str(unit.get("status", "ok")).lower()


def _unit_id(unit: Mapping[str, Any]) -> str:
    return str(unit.get("id") or unit.get("evidence_unit_id") or unit.get("url") or "")


def _dedupe_units(units: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for unit in units:
        if not isinstance(unit, Mapping):
            continue
        unit_key = _unit_id(unit)
        if not unit_key or unit_key in seen:
            continue
        seen.add(unit_key)
        deduped.append(dict(unit))
    return deduped


def _count_units(units: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts = {"total": 0, "ok": 0, "partial": 0, "failed": 0}
    for unit in units:
        if not isinstance(unit, Mapping):
            continue
        status = _unit_status(unit)
        counts["total"] += 1
        if status in counts:
            counts[status] += 1
    return counts


def _cache_root(base_dir: Path | None) -> Path:
    if base_dir is None:
        return Path(__file__).resolve().parents[4] / "cache"
    return base_dir.parent / "cache"


def _cache_dir(base_dir: Path | None) -> Path:
    return _cache_root(base_dir) / "evidence_units"


def _cache_ttl_days() -> int:
    raw = os.environ.get("EVIDENCE_CACHE_DAYS")
    if raw is None:
        return 7
    try:
        return max(0, int(raw))
    except ValueError:
        return 7


def _cache_key(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _load_cached_evidence(
    cache_key: str,
    *,
    base_dir: Path | None,
    ttl_days: int,
) -> dict[str, Any] | None:
    if ttl_days <= 0:
        return None
    path = _cache_dir(base_dir) / f"{cache_key}.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    cached_at = payload.get("cached_at")
    if isinstance(cached_at, str):
        try:
            cached_ts = datetime.fromisoformat(cached_at.replace("Z", "+00:00"))
            age_days = (datetime.now(timezone.utc) - cached_ts).days
            if age_days > ttl_days:
                return None
        except ValueError:
            pass
    return payload if isinstance(payload, dict) else None


def _write_cached_evidence(
    cache_key: str,
    *,
    evidence_units: list[dict[str, Any]],
    base_dir: Path | None,
    schema_version: str,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    cache_dir = _cache_dir(base_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "cache_key": cache_key,
        "cached_at": datetime.now(timezone.utc).isoformat(),
        "schema_version": schema_version,
        "evidence_units": evidence_units,
    }
    if metadata:
        payload["metadata"] = dict(metadata)
    (cache_dir / f"{cache_key}.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _expand_queries(
    client,
    *,
    company: str,
    geography: str,
    horizon_months: int,
    seed_queries: Sequence[str],
    focal_issue: str | None = None,
) -> tuple[dict[str, list[str]], Any]:
    prompt_bundle = build_prompt(
        "query_expansion",
        {
            "company": company,
            "geography": geography,
            "horizon_months": horizon_months,
            "seed_queries": list(seed_queries),
            "focal_issue": focal_issue or "",
        },
    )
    schema = load_schema("query_expansion")
    response = client.generate_json(prompt_bundle.text, schema)
    parsed = ensure_dict(response, node_name="query_expansion")
    validate_artifact("query_expansion", parsed)
    return parsed, prompt_bundle


def run_retrieval_real_node(
    sources: Sequence[str],
    *,
    run_id: str,
    state: ScenarioOpsState,
    user_params: Mapping[str, Any],
    focal_issue: Mapping[str, Any] | None = None,
    base_dir: Path | None = None,
    llm_client=None,
    config: LLMConfig | None = None,
    settings: ScenarioOpsSettings | None = None,
    simulate_evidence: bool | None = None,
    retriever: Callable[..., RetrievedContent] | None = None,
) -> ScenarioOpsState:
    resolved_settings = settings or ScenarioOpsSettings()
    allow_web = bool(resolved_settings.allow_web)
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

    metadata = build_run_metadata(
        run_id=run_id,
        user_params=user_params,
        focal_issue=focal_issue,
    )
    company = metadata["company_name"]
    geography = metadata["geography"]
    horizon_months = metadata["horizon_months"]

    resolved_config = config or llm_config_from_settings(resolved_settings)
    client = get_client(llm_client, resolved_config)
    summarizer_config = LLMConfig(
        model_name=resolved_settings.summarizer_model,
        temperature=resolved_config.temperature,
        timeouts=resolved_config.timeouts,
        mode=resolved_config.mode,
    )
    summarizer_client = get_llm_client(summarizer_config)
    retriever = retriever or retrieve_url
    seed_queries = _seed_queries(company, geography)
    expanded, prompt_bundle = _expand_queries(
        client,
        company=company,
        geography=geography,
        horizon_months=horizon_months,
        seed_queries=seed_queries,
        focal_issue=str(focal_issue.get("focal_issue", "")) if focal_issue else None,
    )
    cache_key_payload: dict[str, Any] = {
        "company": company,
        "geography": geography,
        "horizon_months": horizon_months,
        "prompt_sha": prompt_bundle.sha256,
        "cache_version": _CACHE_VERSION,
        "summarizer_model": resolved_settings.summarizer_model,
    }
    if sources:
        cache_key_payload["sources"] = sorted({str(url) for url in sources})
    else:
        cache_key_payload["seed_queries"] = seed_queries
        cache_key_payload["focal_issue"] = (
            str(focal_issue.get("focal_issue", "")) if focal_issue else ""
        )
    cache_key = _cache_key(cache_key_payload)
    existing_units: list[dict[str, Any]] = []
    if state.evidence_units:
        prior_units = state.evidence_units.get("evidence_units")
        if isinstance(prior_units, list):
            existing_units = _dedupe_units(prior_units)

    existing_units = _dedupe_units(existing_units)
    cached_payload = None
    cache_hit = False
    cache_notes: list[str] = []
    cached_units: list[dict[str, Any]] = []
    if not simulate:
        cached_payload = _load_cached_evidence(
            cache_key,
            base_dir=base_dir,
            ttl_days=_cache_ttl_days(),
        )
    if cached_payload:
        cached_schema = cached_payload.get("schema_version")
        cached_units_payload = cached_payload.get("evidence_units")
        if cached_schema != "2.0":
            cache_notes.append("cache_schema_mismatch")
        elif isinstance(cached_units_payload, list):
            cached_units = _dedupe_units(cached_units_payload)
    if cached_units:
        cache_hit = True

    retrieved_units: list[dict[str, Any]] = _dedupe_units(existing_units + cached_units)
    for unit in retrieved_units:
        if not isinstance(unit, Mapping):
            continue
        if "id" not in unit:
            unit["id"] = unit.get("evidence_unit_id") or unit.get("url") or ""
        if not unit.get("id"):
            encoded = json.dumps(dict(unit), sort_keys=True, separators=(",", ":"))
            unit["id"] = f"legacy-{hashlib.sha256(encoded.encode('utf-8')).hexdigest()}"
        status = str(unit.get("status", "ok")).lower()
        if status not in {"ok", "partial", "failed"}:
            status = "ok"
        unit["status"] = status
        unit.setdefault("failure_reason", None)
        unit.setdefault("content_type", None)
        unit.setdefault("http_status", None)
        unit.setdefault("source_method", "unknown")
        unit.setdefault("embedding_ref", None)

        summary = str(unit.get("summary") or unit.get("excerpt") or "").strip()
        claims = unit.get("claims")
        metrics = unit.get("metrics")
        tags = unit.get("tags")
        if not isinstance(claims, list):
            claims = []
        if not isinstance(metrics, list):
            metrics = []
        if not isinstance(tags, list):
            tags = []

        if status == "ok" and (not summary or not claims):
            fallback = fallback_summary(summary or str(unit.get("raw_text") or ""))
            if not summary:
                summary = str(fallback.get("summary", "")).strip()
            if not claims:
                claims = fallback.get("claims", [])
            if not metrics:
                metrics = fallback.get("metrics", [])
            if not tags:
                tags = fallback.get("tags", [])
            unit.setdefault(
                "reliability_notes", fallback.get("reliability_notes", "")
            )

        unit["summary"] = summary
        unit["claims"] = claims
        unit["metrics"] = metrics
        unit["tags"] = tags
    sources_used: list[tuple[str, str, str | None]] = []
    search_failures: list[str] = []
    search_queries_used = 0
    attempted_replacements = 0
    if sources:
        sources_used = [(str(url), "tertiary", None) for url in sources]
    if not sources_used:
        for bucket in ("primary", "secondary", "tertiary", "counter"):
            for query in expanded.get(bucket, []):
                if not allow_web:
                    continue
                try:
                    results = search_web(
                        query,
                        max_results=5,
                        run_id=run_id,
                        base_dir=base_dir,
                        model_name=resolved_settings.search_model,
                    )
                    search_queries_used += 1
                except Exception as exc:
                    search_failures.append(f"{bucket}:{query}: {exc}")
                    continue
                for result in results:
                    sources_used.append((result, bucket, query))

    deduped_pairs: list[tuple[str, str, str | None]] = []
    seen: set[str] = set()
    for url, bucket, origin_query in sources_used:
        if url in seen:
            continue
        seen.add(url)
        deduped_pairs.append((url, bucket, origin_query))

    if not deduped_pairs and not simulate and not existing_units:
        detail = "; ".join(search_failures[:3])
        raise RuntimeError(
            "retrieval_failed: no sources available and simulation disabled."
            + (f" Search errors: {detail}" if detail else "")
        )

    failures: list[str] = []
    summary_prompt_name = ""
    summary_prompt_sha = ""
    existing_ids = {_unit_id(unit) for unit in retrieved_units}
    unit_counter = 1
    processed_urls = {
        str(unit.get("url"))
        for unit in retrieved_units
        if isinstance(unit, Mapping) and unit.get("url")
    }

    def _next_unit_id(prefix: str) -> str:
        nonlocal unit_counter
        while True:
            candidate = f"{prefix}-{unit_counter}"
            unit_counter += 1
            if candidate not in existing_ids:
                existing_ids.add(candidate)
                return candidate

    def _record_prompt(bundle: Any | None) -> None:
        nonlocal summary_prompt_name, summary_prompt_sha
        if bundle and not summary_prompt_name:
            summary_prompt_name = bundle.name
            summary_prompt_sha = bundle.sha256

    def _build_unit_from_text(
        *,
        unit_id: str,
        url: str,
        title: str,
        publisher: str,
        source_type: str,
        source_method: str,
        raw_text: str,
        content_type: str | None,
        http_status: int | None,
        date_published: str,
        simulated_flag: bool,
    ) -> dict[str, Any]:
        status = "ok"
        failure_reason = None
        summary_payload = None
        if not _is_text_usable(raw_text):
            status = "failed"
            failure_reason = "empty_extracted_text"
        else:
            summary_payload, bundle, summary_error = summarize_text(
                client=summarizer_client,
                text=raw_text,
                title=title,
                url=url,
            )
            _record_prompt(bundle)
            if summary_payload is None:
                status = "partial"
                failure_reason = f"summarizer_failed:{summary_error}"
                summary_payload = fallback_summary(raw_text)

        summary = summary_payload.get("summary", "") if summary_payload else ""
        claims = summary_payload.get("claims", []) if summary_payload else []
        metrics = summary_payload.get("metrics", []) if summary_payload else []
        tags = summary_payload.get("tags", []) if summary_payload else []
        reliability_notes = summary_payload.get("reliability_notes", "") if summary_payload else ""

        category = classify_publisher(url)
        grade, reason = _grade_for_publisher(category)
        unit = {
            "id": unit_id,
            "evidence_unit_id": unit_id,
            "url": url,
            "title": title,
            "status": status,
            "failure_reason": failure_reason,
            "summary": summary,
            "claims": claims,
            "metrics": metrics,
            "tags": tags,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "content_type": content_type,
            "http_status": http_status,
            "source_method": source_method,
            "embedding_ref": None,
            "raw_text": raw_text[:4000],
            "excerpt": raw_text[:300],
            "publisher": publisher,
            "date_published": date_published,
            "source_type": source_type,
            "reliability_grade": grade,
            "reliability_reason": reason,
            "reliability_notes": reliability_notes or reason,
            "geography_tags": [geography],
            "domain_tags": [],
            "simulated": simulated_flag,
        }
        return unit

    def _build_failed_unit(
        *,
        url: str,
        source_method: str,
        reason: str,
    ) -> dict[str, Any]:
        unit_id = _next_unit_id("ev")
        unit = {
            "id": unit_id,
            "evidence_unit_id": unit_id,
            "url": url,
            "title": url,
            "status": "failed",
            "failure_reason": reason,
            "summary": "",
            "claims": [],
            "metrics": [],
            "tags": [],
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "content_type": None,
            "http_status": None,
            "source_method": source_method,
            "embedding_ref": None,
            "raw_text": "",
            "excerpt": "",
            "publisher": url,
            "date_published": datetime.now(timezone.utc).isoformat(),
            "source_type": "tertiary",
            "reliability_grade": "D",
            "reliability_reason": "unavailable",
            "reliability_notes": "retrieval failed",
            "geography_tags": [geography],
            "domain_tags": [],
            "simulated": False,
        }
        return unit

    def _build_simulated_unit(query: str) -> dict[str, Any]:
        unit_id = _next_unit_id("sim")
        url = f"simulated://{query.replace(' ', '+')}"
        title = f"Simulated evidence for: {query}"
        raw_text = f"Simulated evidence generated for query: {query}."
        summary_payload = fallback_summary(raw_text)
        unit = {
            "id": unit_id,
            "evidence_unit_id": unit_id,
            "url": url,
            "title": title,
            "status": "ok",
            "failure_reason": None,
            "summary": summary_payload.get("summary", ""),
            "claims": summary_payload.get("claims", []),
            "metrics": summary_payload.get("metrics", []),
            "tags": summary_payload.get("tags", []),
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "content_type": None,
            "http_status": None,
            "source_method": "simulation",
            "embedding_ref": None,
            "raw_text": raw_text,
            "excerpt": raw_text[:300],
            "publisher": "simulated",
            "date_published": datetime.now(timezone.utc).isoformat(),
            "source_type": "tertiary",
            "reliability_grade": "D",
            "reliability_reason": "simulated evidence",
            "reliability_notes": summary_payload.get("reliability_notes", ""),
            "geography_tags": [geography],
            "domain_tags": [],
            "simulated": True,
        }
        return unit

    counts = _count_units(retrieved_units)
    need_more = counts["ok"] < min_ok or counts["total"] < min_total

    if simulate and not allow_web:
        for query in _dedupe_urls(seed_queries):
            if not need_more:
                break
            simulated_unit = _build_simulated_unit(query)
            retrieved_units.append(simulated_unit)
            counts = _count_units(retrieved_units)
            need_more = counts["ok"] < min_ok or counts["total"] < min_total
    elif need_more and allow_web and deduped_pairs:
        queue = list(deduped_pairs)
        queue_index = 0
        while queue_index < len(queue) and (counts["ok"] < min_ok or counts["total"] < min_total):
            url, bucket, origin_query = queue[queue_index]
            queue_index += 1
            if url in processed_urls:
                continue
            processed_urls.add(url)
            source_method = "search" if origin_query else "source_url"
            try:
                fetched = retriever(
                    url,
                    run_id=run_id,
                    allow_web=allow_web,
                    enforce_allowlist=False,
                    base_dir=base_dir,
                )
            except Exception as exc:
                failures.append(f"{url}: {exc}")
                retrieved_units.append(
                    _build_failed_unit(
                        url=url,
                        source_method=source_method,
                        reason=str(exc),
                    )
                )
                counts = _count_units(retrieved_units)
                continue
            title = fetched.title or url
            publisher = fetched.title or fetched.url
            date_published = fetched.date or datetime.now(timezone.utc).isoformat()
            unit_id = _next_unit_id("ev")
            unit = _build_unit_from_text(
                unit_id=unit_id,
                url=fetched.url,
                title=title,
                publisher=publisher,
                source_type=_source_type_for_bucket(bucket),
                source_method=source_method,
                raw_text=fetched.text,
                content_type=fetched.content_type,
                http_status=fetched.http_status,
                date_published=date_published,
                simulated_flag=False,
            )
            retrieved_units.append(unit)
            counts = _count_units(retrieved_units)

        if counts["ok"] < min_ok or counts["total"] < min_total:
            recovery_queries = [q for q in seed_queries if q]
            for query in recovery_queries:
                if counts["ok"] >= min_ok and counts["total"] >= min_total:
                    break
                if not allow_web:
                    break
                try:
                    results = search_web(
                        query,
                        max_results=5,
                        run_id=run_id,
                        base_dir=base_dir,
                        model_name=resolved_settings.search_model,
                    )
                    search_queries_used += 1
                except Exception as exc:
                    search_failures.append(f"recovery:{query}: {exc}")
                    continue
                for result in results:
                    if result in processed_urls:
                        continue
                    attempted_replacements += 1
                    queue.append((result, "tertiary", query))

                while queue_index < len(queue) and (counts["ok"] < min_ok or counts["total"] < min_total):
                    url, bucket, origin_query = queue[queue_index]
                    queue_index += 1
                    if url in processed_urls:
                        continue
                    processed_urls.add(url)
                    source_method = "search" if origin_query else "source_url"
                    try:
                        fetched = retriever(
                            url,
                            run_id=run_id,
                            allow_web=allow_web,
                            enforce_allowlist=False,
                            base_dir=base_dir,
                        )
                    except Exception as exc:
                        failures.append(f"{url}: {exc}")
                        retrieved_units.append(
                            _build_failed_unit(
                                url=url,
                                source_method=source_method,
                                reason=str(exc),
                            )
                        )
                        counts = _count_units(retrieved_units)
                        continue
                    title = fetched.title or url
                    publisher = fetched.title or fetched.url
                    date_published = fetched.date or datetime.now(timezone.utc).isoformat()
                    unit_id = _next_unit_id("ev")
                    unit = _build_unit_from_text(
                        unit_id=unit_id,
                        url=fetched.url,
                        title=title,
                        publisher=publisher,
                        source_type=_source_type_for_bucket(bucket),
                        source_method=source_method,
                        raw_text=fetched.text,
                        content_type=fetched.content_type,
                        http_status=fetched.http_status,
                        date_published=date_published,
                        simulated_flag=False,
                    )
                    retrieved_units.append(unit)
                    counts = _count_units(retrieved_units)

    counts = _count_units(retrieved_units)
    failed_ratio = counts["failed"] / max(1, counts["total"])
    status = "ok"
    notes: list[str] = []
    if counts["ok"] < min_ok:
        status = "failed"
        notes.append("ok_count_below_min")
    elif counts["total"] < min_total:
        status = "partial"
        notes.append("total_below_min")
    if max_failed_ratio is not None and failed_ratio > max_failed_ratio:
        if status == "ok":
            status = "partial"
        notes.append("failed_ratio_exceeded")
    notes.extend(cache_notes)
    if search_failures:
        notes.append("search_errors_present")

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
            "attempted_replacements": attempted_replacements,
            "search_queries_used": search_queries_used,
            "exhausted": status == "failed",
            "reasons": failures[:3] + search_failures[:3],
        },
        "notes": notes,
    }
    validate_artifact("retrieval_report", retrieval_report)
    write_artifact(
        run_id=run_id,
        artifact_name="retrieval_report",
        payload=retrieval_report,
        ext="json",
        input_values={"status": status},
        tool_versions={"retrieval_real_node": "0.2.0"},
        base_dir=base_dir,
    )

    simulated_flag = simulate or any(
        unit.get("simulated") for unit in retrieved_units if isinstance(unit, Mapping)
    )
    payload = {
        **metadata,
        "schema_version": "2.0",
        "simulated": simulated_flag,
        "evidence_units": retrieved_units,
    }
    validate_artifact("evidence_units.schema", payload)
    if not simulate and status != "failed":
        _write_cached_evidence(
            cache_key,
            evidence_units=retrieved_units,
            base_dir=base_dir,
            schema_version="2.0",
            metadata=metadata,
        )
    write_artifact(
        run_id=run_id,
        artifact_name="evidence_units",
        payload=payload,
        ext="json",
        input_values={
            "source_count": len(deduped_pairs),
            "simulated": simulated_flag,
            "cache_hit": cache_hit,
            "cache_key": cache_key,
            "ok_count": counts["ok"],
            "total_count": counts["total"],
        },
        prompt_values={
            "query_prompt_name": prompt_bundle.name,
            "query_prompt_sha256": prompt_bundle.sha256,
            "summary_prompt_name": summary_prompt_name or "evidence_summary",
            "summary_prompt_sha256": summary_prompt_sha,
        },
        tool_versions={"retrieval_real_node": "0.2.0"},
        base_dir=base_dir,
    )
    state.evidence_units = payload
    if status == "failed":
        raise RuntimeError(
            f"retrieval_failed: ok={counts['ok']} total={counts['total']} min_ok={min_ok} min_total={min_total}"
        )
    return state
