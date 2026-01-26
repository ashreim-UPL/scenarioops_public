from __future__ import annotations

import io
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
import re
from urllib.request import Request, urlopen

from scenarioops.app.config import LLMConfig, ScenarioOpsSettings, llm_config_from_settings
from scenarioops.graph.nodes.utils import build_prompt
from scenarioops.graph.state import ScenarioOpsState
from scenarioops.graph.tools.evidence_processing import fallback_summary, summarize_text
from scenarioops.graph.tools.schema_validate import load_schema, validate_artifact
from scenarioops.graph.tools.search import search_web
from scenarioops.graph.tools.traceability import build_run_metadata
from scenarioops.graph.tools.vectordb import open_run_vector_store
from scenarioops.graph.tools.web_retriever import retrieve_url
from scenarioops.graph.types import ArtifactData, NodeResult
from scenarioops.llm.client import get_llm_client
from scenarioops.llm.guards import ensure_dict


_ANNUAL_CHUNK_CHARS = int(os.environ.get("ANNUAL_REPORT_CHUNK_CHARS", "2000"))
_ANNUAL_CHUNK_OVERLAP = int(os.environ.get("ANNUAL_REPORT_CHUNK_OVERLAP", "250"))
_ANNUAL_MAX_CHUNKS = int(os.environ.get("ANNUAL_REPORT_MAX_CHUNKS", "10"))
_URL_RE = re.compile(r"https?://[^\s)\]>\"]+")
_INDUSTRY_BY_COMPANY = {
    "microsoft": "Technology",
    "google": "Technology",
    "alphabet": "Technology",
    "apple": "Technology",
    "amazon": "Technology",
    "meta": "Technology",
    "samsung": "Technology",
    "sap": "Technology",
    "tencent": "Technology",
    "alibaba": "Technology",
    "accor": "Hospitality",
    "marriott": "Hospitality",
    "hilton": "Hospitality",
    "hyatt": "Hospitality",
    "shell": "Energy",
    "bp": "Energy",
    "aramco": "Energy",
    "toyota": "Automotive",
    "tesla": "Automotive",
    "siemens": "Industrial",
}
_INDUSTRY_KEYWORDS = {
    "Technology": [
        "technology",
        "software",
        "cloud",
        "saas",
        "ai",
        "hardware",
        "platform",
        "operating system",
        "semiconductor",
        "computing",
    ],
    "Hospitality": ["hospitality", "hotel", "hotels", "resort", "lodging", "tourism", "travel"],
    "Financial Services": [
        "bank",
        "banking",
        "insurance",
        "investment",
        "asset management",
        "lending",
        "financial services",
    ],
    "Energy": ["oil", "gas", "petroleum", "renewable", "electricity", "utility", "utilities", "power"],
    "Healthcare": ["healthcare", "pharmaceutical", "biotech", "medical", "hospital", "drug", "clinic"],
    "Retail": ["retail", "e-commerce", "store", "consumer goods", "merchandise"],
    "Manufacturing": ["manufacturing", "industrial", "factory", "production"],
    "Transportation": ["logistics", "shipping", "airline", "rail", "transportation", "freight"],
    "Telecom": ["telecom", "telecommunications", "wireless", "5g", "broadband", "carrier"],
}


def _manual_input(user_params: Mapping[str, Any]) -> str:
    value = (
        user_params.get("company_description")
        or user_params.get("org_context")
        or user_params.get("value")
        or ""
    )
    return str(value)


def _industry_from_params(user_params: Mapping[str, Any]) -> str | None:
    for key in ("industry", "sector", "vertical"):
        raw = user_params.get(key)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    return None


def _infer_industry(
    company_name: str, summary: str, manual_input: str
) -> str | None:
    normalized_company = company_name.strip().lower()
    if normalized_company in _INDUSTRY_BY_COMPANY:
        return _INDUSTRY_BY_COMPANY[normalized_company]

    text = " ".join([company_name, summary, manual_input]).lower()
    for label, keywords in _INDUSTRY_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            return label
    return None


def _annual_report_queries(company: str) -> list[str]:
    trimmed = company.strip()
    if not trimmed:
        return []
    return [
        f"{trimmed} annual report PDF",
        f"{trimmed} investor relations annual report",
        f"{trimmed} form 10-k pdf",
    ]


def _score_report_url(url: str) -> int:
    lowered = url.lower()
    score = 0
    if "annual" in lowered:
        score += 2
    if "report" in lowered:
        score += 2
    if "10-k" in lowered or "10k" in lowered:
        score += 2
    if lowered.endswith(".pdf") or ".pdf?" in lowered:
        score += 3
    if "investor" in lowered or "/ir" in lowered:
        score += 1
    return score


def _dedupe_urls(urls: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for url in urls:
        if url in seen:
            continue
        seen.add(url)
        deduped.append(url)
    return deduped


def _clean_candidate_url(value: str) -> str | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    if "](" in raw:
        raw = raw.split("](", 1)[-1]
    match = _URL_RE.search(raw)
    if match:
        raw = match.group(0)
    raw = raw.rstrip(")]")
    if not raw.startswith("http"):
        return None
    return raw


def _clean_candidate_urls(urls: Sequence[str]) -> list[str]:
    cleaned: list[str] = []
    for url in urls:
        fixed = _clean_candidate_url(url)
        if fixed:
            cleaned.append(fixed)
    return cleaned


def _chunk_text(text: str) -> list[str]:
    cleaned = " ".join(text.split())
    if not cleaned:
        return []
    overlap = min(_ANNUAL_CHUNK_OVERLAP, max(_ANNUAL_CHUNK_CHARS - 1, 0))
    step = max(1, _ANNUAL_CHUNK_CHARS - overlap)
    chunks: list[str] = []
    for start in range(0, len(cleaned), step):
        if len(chunks) >= _ANNUAL_MAX_CHUNKS:
            break
        chunk = cleaned[start : start + _ANNUAL_CHUNK_CHARS]
        if chunk:
            chunks.append(chunk)
    return chunks


def _download_pdf_text(url: str, *, timeout_seconds: float = 25.0) -> tuple[str, str | None]:
    try:
        from pypdf import PdfReader
    except Exception:
        return "", "pdf_parser_unavailable"
    try:
        request = Request(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                )
            },
        )
        with urlopen(request, timeout=timeout_seconds) as response:
            content = response.read()
    except Exception as exc:
        return "", f"pdf_download_failed:{exc}"
    try:
        reader = PdfReader(io.BytesIO(content))
    except Exception as exc:
        return "", f"pdf_parse_failed:{exc}"
    parts: list[str] = []
    for page in reader.pages:
        try:
            page_text = page.extract_text() or ""
        except Exception:
            page_text = ""
        if page_text:
            parts.append(page_text)
    text = "\n".join(parts)
    if not text.strip():
        return "", "pdf_text_empty"
    return text, None


def _fetch_report_text(
    url: str,
    *,
    run_id: str,
    base_dir: Path | None,
    allow_web: bool,
) -> tuple[str, str | None, str | None]:
    if not allow_web:
        return "", None, "network_disabled"
    lowered = url.lower()
    if lowered.endswith(".pdf") or ".pdf?" in lowered:
        text, error = _download_pdf_text(url)
        return text, "application/pdf", error
    try:
        retrieved = retrieve_url(
            url,
            run_id=run_id,
            allow_web=allow_web,
            enforce_allowlist=False,
            base_dir=base_dir,
        )
    except Exception as exc:
        return "", None, f"fetch_failed:{exc}"
    content_type = retrieved.content_type
    if content_type and "pdf" in content_type.lower():
        text, error = _download_pdf_text(url)
        return text, content_type, error
    if not retrieved.text.strip():
        return "", content_type, "empty_text"
    return retrieved.text, content_type, None


def _summarize_annual_report(
    *,
    client,
    company: str,
    text: str,
    max_chars: int = 9000,
) -> tuple[dict[str, Any] | None, Any | None, str | None]:
    prompt_bundle = build_prompt(
        "annual_report_summary",
        {"company": company, "text": text[:max_chars]},
    )
    schema = load_schema("annual_report_summary")
    try:
        response = client.generate_json(prompt_bundle.text, schema)
        parsed = ensure_dict(response, node_name="annual_report_summary")
    except Exception as exc:
        return None, prompt_bundle, str(exc)
    return parsed, prompt_bundle, None


def _dedupe_units(units: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for unit in units:
        if not isinstance(unit, Mapping):
            continue
        unit_id = str(unit.get("id") or unit.get("evidence_unit_id") or "")
        if not unit_id or unit_id in seen:
            continue
        seen.add(unit_id)
        deduped.append(dict(unit))
    return deduped


def run_company_profile_node(
    user_params: Mapping[str, Any],
    sources: Sequence[str],
    input_docs: Sequence[str] | None = None,
    *,
    run_id: str,
    state: ScenarioOpsState,
    base_dir: Path | None = None,
    settings: ScenarioOpsSettings | None = None,
    config: LLMConfig | None = None,
) -> NodeResult:
    resolved_settings = settings or ScenarioOpsSettings()
    metadata = build_run_metadata(run_id=run_id, user_params=user_params)
    company_name = metadata.get("company_name", "")
    allow_web = bool(getattr(resolved_settings, "allow_web", False))
    simulated = bool(getattr(resolved_settings, "simulate_evidence", False))

    annual_report_status = "skipped"
    annual_report_source = None
    annual_report_summary = ""
    key_risks: list[str] = []
    strategic_priorities: list[str] = []
    annual_report_text = ""
    annual_report_content_type = None
    manual_input = _manual_input(user_params)
    industry = _industry_from_params(user_params)

    evidence_units: list[dict[str, Any]] = []
    summarizer_client = None
    if allow_web and company_name:
        resolved_config = config or llm_config_from_settings(resolved_settings)
        summarizer_config = LLMConfig(
            model_name=resolved_settings.summarizer_model,
            temperature=resolved_config.temperature,
            timeouts=resolved_config.timeouts,
            mode=resolved_config.mode,
        )
        summarizer_client = get_llm_client(summarizer_config)
        candidates: list[str] = []
        for query in _annual_report_queries(company_name):
            try:
                results = search_web(
                    query,
                    max_results=5,
                    run_id=run_id,
                    base_dir=base_dir,
                    model_name=resolved_settings.search_model,
                )
                candidates.extend(results)
            except Exception:
                continue
        cleaned_candidates = _clean_candidate_urls(candidates)
        ranked = sorted(_dedupe_urls(cleaned_candidates), key=_score_report_url, reverse=True)
        last_error = None
        for url in ranked:
            text, content_type, error = _fetch_report_text(
                url, run_id=run_id, base_dir=base_dir, allow_web=allow_web
            )
            if error:
                last_error = error
                continue
            if text:
                annual_report_text = text
                annual_report_source = url
                annual_report_content_type = content_type
                break
        if annual_report_text:
            summary_payload, _, summary_error = _summarize_annual_report(
                client=summarizer_client,
                company=company_name,
                text=annual_report_text,
            )
            if summary_payload is None:
                annual_report_status = f"summarization_failed:{summary_error}"
                annual_report_summary = "Annual report retrieved but summarization failed."
            else:
                annual_report_status = "ok"
                annual_report_summary = summary_payload.get("annual_report_summary", "")
                key_risks = summary_payload.get("key_risks", [])
                strategic_priorities = summary_payload.get("strategic_priorities", [])
        else:
            if ranked:
                annual_report_status = last_error or "fetch_failed"
            else:
                annual_report_status = "not_found"
    elif not allow_web:
        annual_report_status = "skipped"
    else:
        annual_report_status = "not_found"

    if annual_report_text and summarizer_client:
        vector_store = None
        try:
            vector_store = open_run_vector_store(
                run_id,
                base_dir=base_dir,
                embed_model=resolved_settings.embed_model,
                seed=int(resolved_settings.seed or 0),
            )
        except Exception:
            vector_store = None

        chunks = _chunk_text(annual_report_text)
        published_at = datetime.now(timezone.utc).isoformat()
        for idx, chunk in enumerate(chunks, start=1):
            summary_payload, _, summary_error = summarize_text(
                client=summarizer_client,
                text=chunk,
                title=f"{company_name} annual report (chunk {idx})",
                url=str(annual_report_source or ""),
            )
            status = "ok"
            failure_reason = None
            if summary_payload is None:
                status = "partial"
                failure_reason = f"summarizer_failed:{summary_error}"
                summary_payload = fallback_summary(chunk)
            summary = summary_payload.get("summary", "")
            claims = summary_payload.get("claims", [])
            metrics = summary_payload.get("metrics", [])
            tags = summary_payload.get("tags", [])
            if "company_fundamentals" not in tags:
                tags.append("company_fundamentals")
            reliability_notes = summary_payload.get("reliability_notes", "")

            unit_id = f"ar-{run_id}-{idx}"
            unit = {
                "id": unit_id,
                "evidence_unit_id": unit_id,
                "url": annual_report_source,
                "title": f"{company_name} annual report (chunk {idx})",
                "status": status,
                "failure_reason": failure_reason,
                "summary": summary,
                "claims": claims,
                "metrics": metrics,
                "tags": tags,
                "retrieved_at": published_at,
                "content_type": annual_report_content_type,
                "http_status": None,
                "source_method": "annual_report",
                "embedding_ref": None,
                "raw_text": chunk[:4000],
                "excerpt": chunk[:300],
                "publisher": company_name,
                "date_published": published_at,
                "source_type": "primary",
                "reliability_grade": "A",
                "reliability_reason": "company annual report",
                "reliability_notes": reliability_notes or "company annual report",
                "geography_tags": [metadata.get("geography", "")],
                "domain_tags": [],
                "simulated": False,
            }
            if vector_store and summary:
                doc_id = f"ar:{unit_id}"
                unit["embedding_ref"] = doc_id
                doc = vector_store.build_document(
                    doc_id=doc_id,
                    text=summary,
                    metadata={
                        "evidence_unit": unit,
                        "company_name": company_name,
                        "run_id": run_id,
                    },
                )
                vector_store.add_documents([doc])
            evidence_units.append(unit)

    existing_units: list[Mapping[str, Any]] = []
    if state.evidence_units:
        prior_units = state.evidence_units.get("evidence_units")
        if isinstance(prior_units, list):
            existing_units = prior_units
    combined_units = _dedupe_units([*existing_units, *evidence_units])
    evidence_payload = None
    if combined_units:
        evidence_payload = {
            **metadata,
            "schema_version": "2.0",
            "simulated": False,
            "evidence_units": combined_units,
        }
        validate_artifact("evidence_units.schema", evidence_payload)

    if not industry:
        industry = _infer_industry(company_name, annual_report_summary, manual_input)

    internal_docs = [Path(path).name for path in (input_docs or []) if path]
    payload = {
        **metadata,
        "industry": industry,
        "source_basis": {
            "urls": [str(item) for item in sources],
            "internal_docs": internal_docs,
            "manual_input": manual_input,
        },
        "annual_report_status": annual_report_status,
        "annual_report_source": annual_report_source,
        "annual_report_summary": annual_report_summary,
        "key_risks": key_risks,
        "strategic_priorities": strategic_priorities,
        "simulated": simulated,
        "metadata": metadata,
    }
    validate_artifact("company_profile", payload)
    state_updates = {"company_profile": payload}
    if evidence_payload:
        state_updates["evidence_units"] = evidence_payload
    return NodeResult(
        state_updates=state_updates,
        artifacts=[
            ArtifactData(
                name="company_profile",
                payload=payload,
                ext="json",
                input_values={"source_count": len(sources)},
                tool_versions={"company_profile_node": "0.2.0"},
            )
        ],
    )
