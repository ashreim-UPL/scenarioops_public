from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
import json
from html.parser import HTMLParser
import re
from pathlib import Path
import time
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .injection_defense import strip_instruction_patterns
from .storage import ensure_run_dirs


@dataclass(frozen=True)
class RetrievedContent:
    url: str
    title: str
    date: str | None
    text: str
    excerpt_hash: str
    content_type: str | None
    http_status: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "date": self.date,
            "text": self.text,
            "excerpt_hash": self.excerpt_hash,
            "content_type": self.content_type,
            "http_status": self.http_status,
        }


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._title_parts: list[str] = []
        self._in_title = False
        self._ignore = 0

    @property
    def text(self) -> str:
        return " ".join(part.strip() for part in self._parts if part.strip())

    @property
    def title(self) -> str:
        return " ".join(part.strip() for part in self._title_parts if part.strip())

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "title":
            self._in_title = True
        if tag.lower() in {"script", "style", "noscript"}:
            self._ignore += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "title":
            self._in_title = False
        if tag.lower() in {"script", "style", "noscript"} and self._ignore > 0:
            self._ignore -= 1

    def handle_data(self, data: str) -> None:
        if self._ignore:
            return
        if self._in_title:
            self._title_parts.append(data)
        else:
            self._parts.append(data)


_LAST_FETCH_AT: float | None = None


def _load_allowlist(path: Path | None = None) -> list[str]:
    if path is None:
        path = Path(__file__).resolve().parents[4] / "data" / "public_sources_allowlist.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    domains = payload.get("domains", [])
    return [domain.lower() for domain in domains if isinstance(domain, str)]


def _is_allowed(url: str, allowlist: list[str]) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return False
    hostname = (parsed.hostname or "").lower()
    if not hostname:
        return False
    for domain in allowlist:
        if hostname == domain or hostname.endswith(f".{domain}"):
            return True
    return False


def _hash_excerpt(text: str, length: int = 1000) -> str:
    excerpt = text[:length]
    return hashlib.sha256(excerpt.encode("utf-8")).hexdigest()


def _parse_date(header_value: str | None) -> str | None:
    if not header_value:
        return None
    try:
        parsed = parsedate_to_datetime(header_value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc).isoformat()
    except (TypeError, ValueError):
        return None


def _cache_dir(base_dir: Path | None = None) -> Path:
    if base_dir:
        return base_dir
    return Path(__file__).resolve().parents[4] / "data" / "retriever_cache"


def _cache_key(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def _cache_path(url: str, cache_dir: Path) -> Path:
    return cache_dir / f"{_cache_key(url)}.json"


def _load_cache(url: str, cache_dir: Path) -> RetrievedContent | None:
    path = _cache_path(url, cache_dir)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return RetrievedContent(
        url=payload.get("url", url),
        title=payload.get("title", ""),
        date=payload.get("date"),
        text=payload.get("text", ""),
        excerpt_hash=payload.get("excerpt_hash", ""),
        content_type=payload.get("content_type"),
        http_status=payload.get("http_status"),
    )


def _write_cache(url: str, content: RetrievedContent, cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload = content.to_dict()
    payload["cached_at"] = datetime.now(timezone.utc).isoformat()
    _cache_path(url, cache_dir).write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _rate_limit(rate_limit_per_sec: float | None) -> None:
    global _LAST_FETCH_AT
    if rate_limit_per_sec is None or rate_limit_per_sec <= 0:
        return
    now = time.monotonic()
    if _LAST_FETCH_AT is None:
        _LAST_FETCH_AT = now
        return
    min_interval = 1.0 / rate_limit_per_sec
    elapsed = now - _LAST_FETCH_AT
    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)
    _LAST_FETCH_AT = time.monotonic()


def _log_retrieval(
    *,
    run_id: str,
    url: str,
    status: str,
    base_dir: Path | None = None,
    detail: str | None = None,
) -> None:
    dirs = ensure_run_dirs(run_id, base_dir=base_dir)
    log_path = dirs["logs_dir"] / "retriever.log"
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "url": url,
        "status": status,
    }
    if detail:
        payload["detail"] = detail
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def retrieve_bytes(
    url: str,
    *,
    run_id: str,
    allowlist_path: Path | None = None,
    base_dir: Path | None = None,
    rate_limit_per_sec: float | None = 0.5,
    allow_web: bool | None = None,
    allow_network: bool | None = None,
    enforce_allowlist: bool = True,
    timeout_seconds: float = 20.0,
) -> tuple[bytes, str | None, int | None]:
    if allow_web is None:
        allow_web = allow_network
    if allow_web is None:
        allow_web = False

    allowlist = _load_allowlist(allowlist_path) if enforce_allowlist else []
    if allowlist and not _is_allowed(url, allowlist):
        _log_retrieval(
            run_id=run_id,
            url=url,
            status="blocked",
            base_dir=base_dir,
            detail="allowlist_blocked",
        )
        raise PermissionError("URL blocked by allowlist.")

    if not allow_web:
        _log_retrieval(
            run_id=run_id,
            url=url,
            status="blocked",
            base_dir=base_dir,
            detail="network disabled",
        )
        raise PermissionError("Network disabled.")

    _rate_limit(rate_limit_per_sec)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    request = Request(url, headers=headers)
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read()
            content_type = response.headers.get("Content-Type")
            http_status = getattr(response, "status", None)
    except Exception as exc:
        _log_retrieval(
            run_id=run_id,
            url=url,
            status="error",
            base_dir=base_dir,
            detail=str(exc),
        )
        raise

    _log_retrieval(
        run_id=run_id,
        url=url,
        status="ok",
        base_dir=base_dir,
        detail=f"bytes={len(raw)} content_type={content_type}",
    )
    return raw, content_type, http_status


def retrieve_url(
    url: str,
    *,
    run_id: str,
    allowlist_path: Path | None = None,
    base_dir: Path | None = None,
    cache_dir: Path | None = None,
    rate_limit_per_sec: float | None = 0.5,
    allow_web: bool | None = None,
    allow_network: bool | None = None,
    enforce_allowlist: bool = True,
    timeout_seconds: float = 20.0,
) -> RetrievedContent:
    if allow_web is None:
        allow_web = allow_network
    if allow_web is None:
        allow_web = False

    allowlist = _load_allowlist(allowlist_path) if enforce_allowlist else []
    if allowlist and not _is_allowed(url, allowlist):
        _log_retrieval(
            run_id=run_id,
            url=url,
            status="blocked",
            base_dir=base_dir,
            detail="domain not allowlisted",
        )
        raise PermissionError(f"Domain not allowlisted for URL: {url}")

    cache_root = _cache_dir(cache_dir)
    cached = _load_cache(url, cache_root)
    if cached:
        sanitized = strip_instruction_patterns(cached.text)
        if sanitized != cached.text:
            cached = RetrievedContent(
                url=cached.url,
                title=cached.title,
                date=cached.date,
                text=sanitized,
                excerpt_hash=_hash_excerpt(sanitized),
                content_type=cached.content_type,
                http_status=cached.http_status,
            )
        _log_retrieval(
            run_id=run_id,
            url=url,
            status="cache_hit",
            base_dir=base_dir,
            detail=f"bytes={len(cached.text)}",
        )
        return cached

    if not allow_web:
        _log_retrieval(
            run_id=run_id,
            url=url,
            status="blocked",
            base_dir=base_dir,
            detail="cache miss and network disabled",
        )
        raise PermissionError("Network disabled and cache miss.")

    _rate_limit(rate_limit_per_sec)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    request = Request(url, headers=headers)
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read()
            content_type = response.headers.get("Content-Type", "")
            encoding = response.headers.get_content_charset() or "utf-8"
            text = raw.decode(encoding, errors="replace")
            title = ""
            extracted_text = text
            if "text/html" in content_type:
                parser = _HTMLTextExtractor()
                parser.feed(text)
                title = parser.title.strip()
                extracted_text = parser.text
                if not extracted_text.strip():
                    extracted_text = re.sub(r"<[^>]+>", " ", text)
            extracted_text = strip_instruction_patterns(extracted_text)
            date = _parse_date(response.headers.get("Last-Modified"))
            http_status = getattr(response, "status", None)
    except Exception as exc:
        _log_retrieval(
            run_id=run_id,
            url=url,
            status="error",
            base_dir=base_dir,
            detail=str(exc),
        )
        raise

    excerpt_hash = _hash_excerpt(extracted_text)
    result = RetrievedContent(
        url=url,
        title=title or urlparse(url).hostname or url,
        date=date,
        text=extracted_text,
        excerpt_hash=excerpt_hash,
        content_type=content_type,
        http_status=http_status,
    )
    _write_cache(url, result, cache_root)
    _log_retrieval(
        run_id=run_id,
        url=url,
        status="ok",
        base_dir=base_dir,
        detail=f"bytes={len(extracted_text)} content_type={content_type}",
    )
    return result
