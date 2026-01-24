from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from scenarioops.app.config import LLMConfig, ScenarioOpsSettings
from scenarioops.graph.gates.source_reputation import classify_publisher
from scenarioops.graph.nodes.utils import build_prompt, get_client
from scenarioops.graph.state import ScenarioOpsState
from scenarioops.graph.tools.schema_validate import load_schema, validate_artifact
from scenarioops.graph.tools.storage import write_artifact
from scenarioops.graph.tools.traceability import build_run_metadata
from scenarioops.graph.tools.web_retriever import RetrievedContent, retrieve_url
from scenarioops.graph.tools.search import search_web
from scenarioops.llm.guards import ensure_dict
from scenarioops.sources.policy import PESTEL_QUERY_TEMPLATES


_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_METRIC_RE = re.compile(r"\b\d+(?:\.\d+)?%?\b")
_CACHE_VERSION = "v1"


def _seed_queries(company: str, geography: str) -> list[str]:
    scope = f"{company} {geography}".strip()
    queries: list[str] = []
    for _, templates in PESTEL_QUERY_TEMPLATES.items():
        if not templates:
            continue
        queries.append(templates[0].format(scope=scope))
    return queries


def _extract_claims(excerpt: str, limit: int = 3) -> list[str]:
    sentences = [s.strip() for s in _SENTENCE_RE.split(excerpt) if s.strip()]
    return sentences[:limit]


def _extract_metrics(excerpt: str, limit: int = 5) -> list[str]:
    metrics = _METRIC_RE.findall(excerpt)
    return metrics[:limit]


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
    metadata: Mapping[str, Any] | None = None,
) -> None:
    cache_dir = _cache_dir(base_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "cache_key": cache_key,
        "cached_at": datetime.now(timezone.utc).isoformat(),
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


def _retrieve_from_sources(
    sources: Sequence[tuple[str, str]],
    *,
    run_id: str,
    allow_web: bool,
    enforce_allowlist: bool,
    base_dir: Path | None,
    retriever: Callable[..., RetrievedContent],
) -> tuple[list[tuple[str, RetrievedContent]], list[str]]:
    retrieved: list[tuple[str, RetrievedContent]] = []
    failures: list[str] = []
    for url, bucket in sources:
        try:
            fetched = retriever(
                url,
                run_id=run_id,
                allow_web=allow_web,
                enforce_allowlist=enforce_allowlist,
                base_dir=base_dir,
            )
            retrieved.append((bucket, fetched))
        except Exception as exc:
            failures.append(f"{url}: {exc}")
    return retrieved, failures


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

    metadata = build_run_metadata(
        run_id=run_id,
        user_params=user_params,
        focal_issue=focal_issue,
    )
    company = metadata["company_name"]
    geography = metadata["geography"]
    horizon_months = metadata["horizon_months"]

    client = get_client(llm_client, config)
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
    min_required = max(8, int(getattr(resolved_settings, "min_sources_per_domain", 8)))

    cache_key_payload: dict[str, Any] = {
        "company": company,
        "geography": geography,
        "horizon_months": horizon_months,
        "prompt_sha": prompt_bundle.sha256,
        "cache_version": _CACHE_VERSION,
    }
    if sources:
        cache_key_payload["sources"] = sorted({str(url) for url in sources})
    else:
        cache_key_payload["seed_queries"] = seed_queries
        cache_key_payload["focal_issue"] = (
            str(focal_issue.get("focal_issue", "")) if focal_issue else ""
        )
    cache_key = _cache_key(cache_key_payload)
    cached_payload = None
    if not simulate:
        cached_payload = _load_cached_evidence(
            cache_key,
            base_dir=base_dir,
            ttl_days=_cache_ttl_days(),
        )
    if cached_payload:
        cached_units = cached_payload.get("evidence_units")
        if isinstance(cached_units, list) and cached_units:
            if len(cached_units) >= min_required:
                payload = {
                    **metadata,
                    "simulated": False,
                    "evidence_units": cached_units,
                }
                validate_artifact("evidence_units.schema", payload)
                write_artifact(
                    run_id=run_id,
                    artifact_name="evidence_units",
                    payload=payload,
                    ext="json",
                    input_values={
                        "source_count": len(cached_units),
                        "simulated": False,
                        "cache_hit": True,
                        "cache_key": cache_key,
                    },
                    prompt_values={
                        "prompt_name": prompt_bundle.name,
                        "prompt_sha256": prompt_bundle.sha256,
                    },
                    tool_versions={"retrieval_real_node": "0.1.0"},
                    base_dir=base_dir,
                )
                state.evidence_units = payload
                return state

    retrieved_units: list[dict[str, Any]] = []
    sources_used: list[tuple[str, str]] = []
    search_failures: list[str] = []
    if sources:
        sources_used = [(str(url), "tertiary") for url in sources]
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
                    )
                except Exception as exc:
                    search_failures.append(f"{bucket}:{query}: {exc}")
                    continue
                for result in results:
                    sources_used.append((result, bucket))

    deduped_pairs: list[tuple[str, str]] = []
    seen: set[str] = set()
    for url, bucket in sources_used:
        if url in seen:
            continue
        seen.add(url)
        deduped_pairs.append((url, bucket))

    if resolved_settings.sources_policy == "fixtures" and deduped_pairs:
        min_required = min(min_required, len(deduped_pairs))

    if not deduped_pairs and not simulate:
        detail = "; ".join(search_failures[:3])
        raise RuntimeError(
            "retrieval_failed: no sources available and simulation disabled."
            + (f" Search errors: {detail}" if detail else "")
        )

    if simulate and not allow_web:
        simulated_units = []
        for idx, query in enumerate(_dedupe_urls(seed_queries), start=1):
            simulated_units.append(
                {
                    "evidence_unit_id": f"sim-{idx}",
                    "source_type": "tertiary",
                    "title": f"Simulated evidence for: {query}",
                    "publisher": "simulated",
                    "date_published": datetime.now(timezone.utc).isoformat(),
                    "url": f"simulated://{query.replace(' ', '+')}",
                    "excerpt": f"Simulated evidence generated for query: {query}.",
                    "claims": [],
                    "metrics": [],
                    "reliability_grade": "D",
                    "reliability_reason": "simulated evidence",
                    "geography_tags": [geography],
                    "domain_tags": [],
                    "simulated": True,
                }
            )
        payload = {
            **metadata,
            "simulated": True,
            "evidence_units": simulated_units,
        }
        validate_artifact("evidence_units.schema", payload)
        write_artifact(
            run_id=run_id,
            artifact_name="evidence_units",
            payload=payload,
            ext="json",
            input_values={"source_count": 0, "simulated": True},
            prompt_values={
                "prompt_name": prompt_bundle.name,
                "prompt_sha256": prompt_bundle.sha256,
            },
            tool_versions={"retrieval_real_node": "0.1.0"},
            base_dir=base_dir,
        )
        state.evidence_units = payload
        return state

    retrieved, failures = _retrieve_from_sources(
        deduped_pairs,
        run_id=run_id,
        allow_web=allow_web,
        enforce_allowlist=False,
        base_dir=base_dir,
        retriever=retriever,
    )
    now = datetime.now(timezone.utc).isoformat()
    for idx, (bucket, item) in enumerate(retrieved, start=1):
        publisher = item.title or item.url
        category = classify_publisher(item.url)
        grade, reason = _grade_for_publisher(category)
        date_published = item.date or now
        retrieved_units.append(
            {
                "evidence_unit_id": f"ev-{idx}",
                "source_type": _source_type_for_bucket(bucket),
                "title": item.title or item.url,
                "publisher": publisher,
                "date_published": date_published,
                "url": item.url,
                "excerpt": item.text[:1000],
                "claims": _extract_claims(item.text),
                "metrics": _extract_metrics(item.text),
                "reliability_grade": grade,
                "reliability_reason": reason,
                "geography_tags": [geography],
                "domain_tags": [],
                "simulated": False,
            }
        )

    if not retrieved_units and not simulate:
        detail = "; ".join(failures[:3])
        search_detail = "; ".join(search_failures[:3])
        raise RuntimeError(
            "retrieval_failed: no evidence units retrieved."
            + (f" Errors: {detail}" if detail else "")
            + (f" Search errors: {search_detail}" if search_detail else "")
        )

    if len(retrieved_units) < min_required and not simulate:
        detail = "; ".join(failures[:3])
        search_detail = "; ".join(search_failures[:3])
        raise RuntimeError(
            f"retrieval_failed: only {len(retrieved_units)} sources found; "
            f"requires at least {min_required}. Use --sources or broaden scope."
            + (f" Errors: {detail}" if detail else "")
            + (f" Search errors: {search_detail}" if search_detail else "")
        )

    payload = {
        **metadata,
        "simulated": False,
        "evidence_units": retrieved_units,
    }
    validate_artifact("evidence_units.schema", payload)
    if not simulate:
        _write_cached_evidence(
            cache_key,
            evidence_units=retrieved_units,
            base_dir=base_dir,
            metadata=metadata,
        )
    write_artifact(
        run_id=run_id,
        artifact_name="evidence_units",
        payload=payload,
        ext="json",
        input_values={
            "source_count": len(deduped_pairs),
            "simulated": False,
            "cache_hit": False,
            "cache_key": cache_key,
        },
        prompt_values={
            "prompt_name": prompt_bundle.name,
            "prompt_sha256": prompt_bundle.sha256,
        },
        tool_versions={"retrieval_real_node": "0.1.0"},
        base_dir=base_dir,
    )
    state.evidence_units = payload
    return state
