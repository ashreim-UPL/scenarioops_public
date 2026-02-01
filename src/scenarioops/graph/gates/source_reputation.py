from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlparse


DEFAULT_PUBLISHER_CATEGORIES: dict[str, list[str]] = {
    "government": ["gov", "gov.uk", "gov.ae"],
    "academic": ["edu", "ac.uk"],
    "multilateral": ["worldbank.org", "imf.org", "oecd.org", "un.org", "who.int"],
    "media": ["reuters.com", "bbc.co.uk", "ft.com", "nytimes.com"],
    "consulting": ["mckinsey.com", "bcg.com", "bain.com"],
}


@dataclass(frozen=True)
class SourceReputationConfig:
    min_categories: int = 2
    blocked_domains: tuple[str, ...] = ("example.com", "localhost")
    blocked_domain_patterns: tuple[str, ...] = (
        "raw.githubusercontent.com",
        "gist.githubusercontent.com",
    )
    blocked_blog_hosts: tuple[str, ...] = (
        "medium.com",
        "substack.com",
        "blogspot.com",
        "wordpress.com",
    )
    publisher_categories: dict[str, list[str]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.publisher_categories is None:
            object.__setattr__(self, "publisher_categories", DEFAULT_PUBLISHER_CATEGORIES)


def _load_reputation_db() -> dict[str, Any]:
    default_path = Path(__file__).resolve().parents[4] / "data" / "source_reputation.json"
    path_env = os.environ.get("SCENARIOOPS_SOURCE_REPUTATION_PATH")


def _load_config_payload(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _as_tuple(values: Any, default: tuple[str, ...]) -> tuple[str, ...]:
    if values is None:
        return default
    if isinstance(values, str):
        return (values,)
    if isinstance(values, Iterable):
        items = [str(item).lower() for item in values if str(item).strip()]
        return tuple(items) if items else default
    return default


def _merge_categories(payload: dict[str, Any]) -> dict[str, list[str]]:
    merged = {key: list(values) for key, values in DEFAULT_PUBLISHER_CATEGORIES.items()}
    overrides = payload.get("publisher_categories")
    if not isinstance(overrides, Mapping):
        return merged
    for category, domains in overrides.items():
        if not isinstance(domains, Iterable):
            continue
        merged[str(category)] = [str(domain).lower() for domain in domains if str(domain).strip()]
    return merged


def load_source_reputation_config(path: Path | None = None) -> SourceReputationConfig:
    payload = _load_config_payload(_config_path(path))
    min_categories = payload.get("min_categories")
    try:
        min_categories = int(min_categories)
    except (TypeError, ValueError):
        min_categories = 2
    return SourceReputationConfig(
        min_categories=min_categories if min_categories > 0 else 2,
        blocked_domains=_as_tuple(payload.get("blocked_domains"), ("example.com", "localhost")),
        blocked_domain_patterns=_as_tuple(
            payload.get("blocked_domain_patterns"),
            ("raw.githubusercontent.com", "gist.githubusercontent.com"),
        ),
        blocked_blog_hosts=_as_tuple(
            payload.get("blocked_blog_hosts"),
            ("medium.com", "substack.com", "blogspot.com", "wordpress.com"),
        ),
        publisher_categories=_merge_categories(payload),
    )


def _hostname(url: str) -> str:
    parsed = urlparse(url)
    return (parsed.hostname or "").lower()


def classify_publisher(url: str, config: SourceReputationConfig | None = None) -> str:
    if config is None:
        config = load_source_reputation_config()
    hostname = _hostname(url)
    if not hostname:
        return "unknown"
    for category, domains in config.publisher_categories.items():
        for domain in domains:
            if hostname == domain or hostname.endswith(f".{domain}"):
                return category
    if hostname.endswith(".gov") or ".gov." in hostname:
        return "government"
    if hostname.endswith(".edu") or ".ac." in hostname:
        return "academic"
    if hostname.endswith(".org"):
        return "ngo"
    return "commercial"


def _blocked_reason(url: str, config: SourceReputationConfig) -> str | None:
    hostname = _hostname(url)
    if not hostname:
        return "missing hostname"
    for domain in config.blocked_domains:
        if hostname == domain or hostname.endswith(f".{domain}"):
            return f"blocked domain: {domain}"
    for domain in config.blocked_domain_patterns:
        if hostname == domain or hostname.endswith(f".{domain}"):
            return f"blocked pattern: {domain}"
    for domain in config.blocked_blog_hosts:
        if hostname == domain or hostname.endswith(f".{domain}"):
            return f"blocked blog host: {domain}"
    if "blog" in hostname and classify_publisher(url, config) == "commercial":
        return "blocked blog host: unknown blog"
    return None


def validate_reputable_sources(
    evidence_units: Sequence[Mapping[str, Any]],
    *,
    config: SourceReputationConfig | None = None,
) -> None:
    if config is None:
        config = load_source_reputation_config()
    if not evidence_units:
        raise ValueError("No evidence units provided for source validation.")

    categories: set[str] = set()
    for unit in evidence_units:
        url = str(unit.get("url", ""))
        if not url:
            raise ValueError("Evidence unit missing url.")
        reason = _blocked_reason(url, config)
        if reason:
            raise ValueError(f"Untrusted source: {url} ({reason})")
        categories.add(classify_publisher(url, config))

    if len(categories) < config.min_categories:
        raise ValueError(
            f"Insufficient publisher diversity: {len(categories)} categories, "
            f"requires at least {config.min_categories}."
        )
