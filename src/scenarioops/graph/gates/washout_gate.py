from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


PESTEL_DOMAINS = (
    "political",
    "economic",
    "social",
    "technological",
    "environmental",
    "legal",
)

DOMAIN_ALIASES = {
    "politics": "political",
    "policy": "legal",
    "regulatory": "legal",
    "regulation": "legal",
    "economy": "economic",
    "financial": "economic",
    "finance": "economic",
    "societal": "social",
    "demographic": "social",
    "technology": "technological",
    "tech": "technological",
    "environment": "environmental",
    "climate": "environmental",
    "law": "legal",
}


class WashoutGateError(ValueError):
    """Raised when washout rules fail."""


@dataclass(frozen=True)
class WashoutGateConfig:
    min_total_forces: int = 30
    min_per_domain: int = 5
    max_duplicate_ratio: float = 0.2
    max_iterations: int = 2


def _config_value(config: Mapping[str, Any] | WashoutGateConfig | None, name: str, default: Any) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _normalize_domain(domain: str | None) -> str | None:
    if not domain:
        return None
    value = str(domain).strip().lower()
    if value in PESTEL_DOMAINS:
        return value
    return DOMAIN_ALIASES.get(value)


def _forces_list(driving_forces: Any) -> list[dict[str, Any]]:
    if isinstance(driving_forces, dict):
        forces = driving_forces.get("forces", [])
        return forces if isinstance(forces, list) else []
    if isinstance(driving_forces, list):
        return driving_forces
    return []


def _domain_counts(forces: list[dict[str, Any]]) -> dict[str, int]:
    counts = {domain: 0 for domain in PESTEL_DOMAINS}
    for force in forces:
        raw_domain = force.get("domain") if isinstance(force, dict) else None
        domain = _normalize_domain(raw_domain)
        if domain:
            counts[domain] += 1
    return counts


def washout_deficits(
    driving_forces: Any,
    washout_report: Mapping[str, Any] | None,
    config: Mapping[str, Any] | WashoutGateConfig | None = None,
) -> dict[str, list[str]]:
    min_per_domain = int(_config_value(config, "min_per_domain", 5))
    forces = _forces_list(driving_forces)
    counts = _domain_counts(forces)
    missing_domains = [
        domain for domain, count in counts.items() if count < min_per_domain
    ]

    missing_categories: list[str] = []
    if isinstance(washout_report, Mapping):
        missing = washout_report.get("missing_categories", [])
        if isinstance(missing, list):
            missing_categories = [str(item) for item in missing if item]
    return {
        "missing_domains": missing_domains,
        "missing_categories": missing_categories,
    }


def assert_washout_pass(
    driving_forces: Any,
    washout_report: Mapping[str, Any] | None,
    config: Mapping[str, Any] | WashoutGateConfig | None = None,
) -> None:
    min_total_forces = int(_config_value(config, "min_total_forces", 30))
    min_per_domain = int(_config_value(config, "min_per_domain", 5))
    max_duplicate_ratio = float(_config_value(config, "max_duplicate_ratio", 0.2))

    errors: list[str] = []
    forces = _forces_list(driving_forces)
    if len(forces) < min_total_forces:
        errors.append(f"total_forces<{min_total_forces}")

    counts = _domain_counts(forces)
    for domain, count in counts.items():
        if count < min_per_domain:
            errors.append(f"{domain}_forces<{min_per_domain}")

    for force in forces:
        citations = force.get("citations") if isinstance(force, dict) else None
        if not isinstance(citations, list) or not citations:
            errors.append("missing_citations")
            break

    duplicate_ratio = None
    if isinstance(washout_report, Mapping):
        duplicate_ratio = washout_report.get("duplicate_ratio")
    if not isinstance(duplicate_ratio, (int, float)):
        errors.append("duplicate_ratio_missing")
    elif float(duplicate_ratio) > max_duplicate_ratio:
        errors.append(f"duplicate_ratio>{max_duplicate_ratio}")

    missing_categories = []
    if isinstance(washout_report, Mapping):
        missing_categories = washout_report.get("missing_categories", [])
    if isinstance(missing_categories, list) and missing_categories:
        errors.append("missing_categories")

    if errors:
        raise WashoutGateError("washout_gate_failed: " + "; ".join(errors))
