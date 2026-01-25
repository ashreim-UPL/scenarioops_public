from __future__ import annotations

import json
import os
import re
import time
from urllib.parse import urlparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from scenarioops.app.config import DEFAULT_SEARCH_MODEL
from scenarioops.graph.tools.storage import ensure_run_dirs
from scenarioops.llm.client import get_gemini_api_key
from scenarioops.llm.transport import RequestsTransport, Transport


_LAST_SEARCH_AT: float | None = None
_URL_RE = re.compile(r"https?://[^\s)>\"]+")
_GROUNDING_REDIRECT_HOST = "vertexaisearch.cloud.google.com"


def _log_search(
    *,
    run_id: str,
    query: str,
    status: str,
    base_dir: Path | None = None,
    result_count: int | None = None,
    detail: str | None = None,
    model: str | None = None,
) -> None:
    dirs = ensure_run_dirs(run_id, base_dir=base_dir)
    log_path = dirs["logs_dir"] / "search.log"
    payload: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "status": status,
    }
    if result_count is not None:
        payload["result_count"] = result_count
    if detail:
        payload["detail"] = detail
    if model:
        payload["model"] = model
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def _rate_limit(rate_limit_per_sec: float | None) -> None:
    global _LAST_SEARCH_AT
    if rate_limit_per_sec is None or rate_limit_per_sec <= 0:
        return
    now = time.monotonic()
    if _LAST_SEARCH_AT is None:
        _LAST_SEARCH_AT = now
        return
    min_interval = 1.0 / rate_limit_per_sec
    elapsed = now - _LAST_SEARCH_AT
    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)
    _LAST_SEARCH_AT = time.monotonic()


def _gemini_url(model: str, api_key: str) -> str:
    return (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )


def _extract_urls(value: Any, urls: list[str]) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in {"url", "uri"} and isinstance(item, str) and item.startswith("http"):
                urls.append(item)
                continue
            _extract_urls(item, urls)
        return
    if isinstance(value, list):
        for item in value:
            _extract_urls(item, urls)
        return


def _extract_grounding_urls(response: Mapping[str, Any]) -> list[str]:
    urls: list[str] = []
    candidates = response.get("candidates", [])
    if isinstance(candidates, list):
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                continue
            metadata = candidate.get("groundingMetadata")
            if isinstance(metadata, Mapping):
                _extract_urls(metadata, urls)
            content = candidate.get("content")
            if isinstance(content, Mapping):
                parts = content.get("parts", [])
                if isinstance(parts, list):
                    for part in parts:
                        if isinstance(part, Mapping):
                            text = part.get("text")
                            if isinstance(text, str):
                                urls.extend(_URL_RE.findall(text))
    if not urls:
        _extract_urls(response, urls)
    deduped: list[str] = []
    seen: set[str] = set()
    for url in urls:
        if not url.startswith("http"):
            continue
        if url in seen:
            continue
        seen.add(url)
        deduped.append(url)
    filtered: list[str] = []
    for url in deduped:
        parsed = urlparse(url)
        if parsed.netloc == _GROUNDING_REDIRECT_HOST and parsed.path.startswith(
            "/grounding-api-redirect/"
        ):
            continue
        filtered.append(url)
    return filtered


def _search_prompt(query: str, max_results: int) -> str:
    return (
        "Use the web search tool to find authoritative sources for the query below. "
        f"Return {max_results} results.\n\nQuery: {query}"
    )


def search_web(
    query: str,
    *,
    max_results: int = 5,
    rate_limit_per_sec: float | None = 1.0,
    timeout_seconds: float = 45.0,
    model_name: str | None = None,
    api_key: str | None = None,
    transport: Transport | None = None,
    run_id: str | None = None,
    base_dir: Path | None = None,
) -> list[str]:
    env_rate = os.environ.get("GEMINI_SEARCH_RATE_LIMIT")
    if env_rate:
        try:
            rate_limit_per_sec = float(env_rate)
        except ValueError:
            pass
    _rate_limit(rate_limit_per_sec)
    api_key = api_key or get_gemini_api_key()
    model_name = model_name or os.environ.get("GEMINI_SEARCH_MODEL") or DEFAULT_SEARCH_MODEL
    payload = {
        "contents": [{"parts": [{"text": _search_prompt(query, max_results)}]}],
        "tools": [{"google_search": {}}],
        "generationConfig": {"temperature": 0.0},
    }
    client = transport or RequestsTransport(timeout_seconds=timeout_seconds)
    try:
        response = client.post_json(
            _gemini_url(model_name, api_key),
            {"Content-Type": "application/json"},
            payload,
        )
    except Exception as exc:
        if run_id:
            _log_search(
                run_id=run_id,
                query=query,
                status="error",
                base_dir=base_dir,
                detail=str(exc),
                model=model_name,
            )
        raise
    error = response.get("error")
    if isinstance(error, Mapping):
        message = error.get("message", "Unknown Gemini API error.")
        if run_id:
            _log_search(
                run_id=run_id,
                query=query,
                status="error",
                base_dir=base_dir,
                detail=message,
                model=model_name,
            )
        raise RuntimeError(f"Gemini API error: {message}")
    if not isinstance(response, Mapping):
        if run_id:
            _log_search(
                run_id=run_id,
                query=query,
                status="error",
                base_dir=base_dir,
                detail=f"Expected mapping, got {type(response)}",
                model=model_name,
            )
        raise TypeError(f"Expected response mapping, got {type(response)}.")
    urls = _extract_grounding_urls(response)
    if run_id:
        _log_search(
            run_id=run_id,
            query=query,
            status="ok",
            base_dir=base_dir,
            result_count=len(urls),
            model=model_name,
        )
    return urls[:max_results]
