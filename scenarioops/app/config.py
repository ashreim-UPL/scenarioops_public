from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Mapping

LLMModeLiteral = Literal["mock", "gemini"]


@dataclass(frozen=True)
class LLMTimeouts:
    request_seconds: float = 60.0


@dataclass(frozen=True)
class LLMConfig:
    model_name: str = "gemini-1.5-pro"
    temperature: float = 0.2
    timeouts: LLMTimeouts = field(default_factory=LLMTimeouts)
    mode: LLMModeLiteral = "mock"

ModeLiteral = Literal["demo", "live"]
ProviderLiteral = Literal["gemini", "mock"]
SourcesPolicyLiteral = Literal["fixtures", "academic_only", "mixed_reputable"]

_ALLOWED_MODES = {"demo", "live"}
_ALLOWED_PROVIDERS = {"gemini", "mock"}
_ALLOWED_POLICIES = {"fixtures", "academic_only", "mixed_reputable"}
_BOOL_FIELDS = {"allow_web", "forbid_fixture_citations"}
_INT_FIELDS = {"min_sources_per_domain", "min_citations_per_driver"}


@dataclass(frozen=True)
class ScenarioOpsSettings:
    mode: ModeLiteral = "demo"
    llm_provider: ProviderLiteral = "gemini"
    gemini_model: str = "gemini-2.0-flash"
    sources_policy: SourcesPolicyLiteral = "academic_only"
    allow_web: bool = False
    min_sources_per_domain: int = 8
    min_citations_per_driver: int = 2
    forbid_fixture_citations: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "llm_provider": self.llm_provider,
            "gemini_model": self.gemini_model,
            "sources_policy": self.sources_policy,
            "allow_web": self.allow_web,
            "min_sources_per_domain": self.min_sources_per_domain,
            "min_citations_per_driver": self.min_citations_per_driver,
            "forbid_fixture_citations": self.forbid_fixture_citations,
        }


def _find_repo_root(start: Path) -> Path:
    current = start.resolve()
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    return start.resolve()


def _default_config_path() -> Path:
    root = _find_repo_root(Path(__file__).resolve())
    return root / "config" / "scenarioops.yaml"


def _parse_scalar(value: str) -> Any:
    cleaned = value.strip().strip("'\"")
    lowered = cleaned.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if cleaned.isdigit():
        return int(cleaned)
    return cleaned


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    values: dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            continue
        key, raw_value = stripped.split(":", 1)
        values[key.strip()] = _parse_scalar(raw_value)
    return values


def _apply_overrides(
    settings: ScenarioOpsSettings, values: Mapping[str, Any]
) -> ScenarioOpsSettings:
    if not values:
        return settings
    valid_keys = set(settings.as_dict().keys())
    unknown = [key for key in values.keys() if key not in valid_keys]
    if unknown:
        raise ValueError(f"Unknown settings keys: {', '.join(sorted(unknown))}")
    updated: dict[str, Any] = {}
    for key, value in values.items():
        if key in _BOOL_FIELDS:
            if isinstance(value, str):
                updated[key] = value.strip().lower() in {"1", "true", "yes", "on"}
            else:
                updated[key] = bool(value)
            continue
        if key in _INT_FIELDS:
            try:
                updated[key] = int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid int for {key}: {value}") from exc
            continue
        if key == "mode":
            mode = str(value).strip().lower()
            if mode not in _ALLOWED_MODES:
                raise ValueError(f"Unsupported mode: {mode}")
            updated[key] = mode
            continue
        if key == "llm_provider":
            provider = str(value).strip().lower()
            if provider not in _ALLOWED_PROVIDERS:
                raise ValueError(f"Unsupported llm_provider: {provider}")
            updated[key] = provider
            continue
        if key == "sources_policy":
            policy = str(value).strip().lower()
            if policy not in _ALLOWED_POLICIES:
                raise ValueError(f"Unsupported sources_policy: {policy}")
            updated[key] = policy
            continue
        updated[key] = str(value)
    return replace(settings, **updated)


def apply_overrides(
    settings: ScenarioOpsSettings, values: Mapping[str, Any]
) -> ScenarioOpsSettings:
    return _apply_overrides(settings, values)


def settings_from_dict(values: Mapping[str, Any]) -> ScenarioOpsSettings:
    settings = ScenarioOpsSettings()
    return _apply_overrides(settings, values)


def load_settings(
    overrides: Mapping[str, Any] | None = None, config_path: Path | None = None
) -> ScenarioOpsSettings:
    settings = ScenarioOpsSettings()
    path = config_path or _default_config_path()
    file_values = _load_yaml(path)
    settings = _apply_overrides(settings, file_values)
    if overrides:
        settings = _apply_overrides(settings, overrides)
    return settings


def llm_config_from_settings(settings: ScenarioOpsSettings) -> LLMConfig:
    provider = settings.llm_provider
    if settings.mode == "demo" and provider == "mock":
        mode = "mock"
    elif provider == "gemini":
        mode = "gemini"
    else:
        mode = "mock"
    return LLMConfig(model_name=settings.gemini_model, mode=mode)
