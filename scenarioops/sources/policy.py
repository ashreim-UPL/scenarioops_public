from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


PESTEL_QUERY_TEMPLATES: dict[str, list[str]] = {
    "political": [
        "regulatory outlook {scope}",
        "geopolitical risk {scope}",
        "public policy priorities {scope}",
    ],
    "economic": [
        "macro indicators {scope}",
        "inflation outlook {scope}",
        "interest rate expectations {scope}",
    ],
    "social": [
        "labor market trends {scope}",
        "demographic shifts {scope}",
        "consumer sentiment {scope}",
    ],
    "technological": [
        "technology adoption trends {scope}",
        "digital infrastructure investment {scope}",
        "automation and AI impact {scope}",
    ],
    "environmental": [
        "climate risk assessment {scope}",
        "energy transition policy {scope}",
        "sustainability regulation {scope}",
    ],
    "legal": [
        "compliance requirements {scope}",
        "sector regulations {scope}",
        "litigation risk {scope}",
    ],
}


ACADEMIC_ONLY_SOURCES: tuple[str, ...] = (
    "https://www.nature.com",
    "https://www.science.org",
    "https://www.pnas.org",
    "https://www.cell.com",
    "https://onlinelibrary.wiley.com",
    "https://www.springer.com",
    "https://www.tandfonline.com",
    "https://arxiv.org",
)


MIXED_REPUTABLE_SOURCES: tuple[str, ...] = (
    *ACADEMIC_ONLY_SOURCES,
    "https://www.worldbank.org",
    "https://www.imf.org",
    "https://www.oecd.org",
    "https://www.un.org",
    "https://www.who.int",
    "https://www.bis.org",
    "https://www.ecb.europa.eu",
    "https://www.federalreserve.gov",
    "https://www.sec.gov",
    "https://www.weforum.org",
)


FIXTURE_SOURCES: tuple[str, ...] = (
    "https://example.com/a",
    "https://example.com/b",
    "https://example.com/c",
)


@dataclass(frozen=True)
class SourcesPolicy:
    name: str
    sources: tuple[str, ...]
    query_templates: dict[str, list[str]]
    allowed_categories: tuple[str, ...] | None = None
    enforce_allowlist: bool = False

    def default_sources(self) -> list[str]:
        return list(self.sources)


def policy_for_name(name: str) -> SourcesPolicy:
    normalized = name.strip().lower()
    if normalized == "fixtures":
        return SourcesPolicy(
            name="fixtures",
            sources=FIXTURE_SOURCES,
            query_templates=PESTEL_QUERY_TEMPLATES,
            enforce_allowlist=True,
        )
    if normalized == "academic_only":
        return SourcesPolicy(
            name="academic_only",
            sources=ACADEMIC_ONLY_SOURCES,
            query_templates=PESTEL_QUERY_TEMPLATES,
            allowed_categories=("academic",),
            enforce_allowlist=False,
        )
    if normalized == "mixed_reputable":
        return SourcesPolicy(
            name="mixed_reputable",
            sources=MIXED_REPUTABLE_SOURCES,
            query_templates=PESTEL_QUERY_TEMPLATES,
            enforce_allowlist=False,
        )
    raise ValueError(f"Unknown sources policy: {name}")


def default_sources_for_policy(name: str) -> list[str]:
    return policy_for_name(name).default_sources()


def allowed_categories_for_policy(name: str) -> tuple[str, ...] | None:
    return policy_for_name(name).allowed_categories


def query_templates_for_policy(name: str) -> dict[str, list[str]]:
    return policy_for_name(name).query_templates
