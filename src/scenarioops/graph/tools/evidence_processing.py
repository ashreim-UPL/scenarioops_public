from __future__ import annotations

import re
from typing import Any

from scenarioops.graph.nodes.utils import build_prompt
from scenarioops.graph.tools.schema_validate import load_schema
from scenarioops.llm.guards import ensure_dict


_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_METRIC_RE = re.compile(r"\b\d+(?:\.\d+)?%?\b")


def extract_claims(text: str, *, limit: int = 3) -> list[str]:
    sentences = [item.strip() for item in _SENTENCE_RE.split(text) if item.strip()]
    return sentences[:limit]


def extract_metrics(text: str, *, limit: int = 5) -> list[str]:
    metrics = _METRIC_RE.findall(text)
    return metrics[:limit]


def fallback_summary(text: str) -> dict[str, Any]:
    claims = extract_claims(text, limit=3)
    summary = " ".join(claims).strip()
    if not summary:
        summary = "Source text was too thin to summarize."
    return {
        "summary": summary,
        "claims": claims if claims else [summary],
        "metrics": extract_metrics(text, limit=5),
        "tags": [],
        "reliability_notes": "Fallback summary used due to summarization failure.",
    }


def summarize_text(
    *,
    client,
    text: str,
    title: str,
    url: str,
    prompt_name: str = "evidence_summary",
    schema_name: str = "evidence_summary",
    max_chars: int = 6000,
) -> tuple[dict[str, Any] | None, Any | None, str | None]:
    if not text:
        return None, None, "empty_text"
    prompt_bundle = build_prompt(
        prompt_name,
        {
            "title": title,
            "url": url,
            "text": text[:max_chars],
        },
    )
    schema = load_schema(schema_name)
    try:
        response = client.generate_json(prompt_bundle.text, schema)
        parsed = ensure_dict(response, node_name="evidence_summary")
    except Exception as exc:
        return None, prompt_bundle, str(exc)
    return parsed, prompt_bundle, None
