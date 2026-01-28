from __future__ import annotations

from dataclasses import dataclass, field, replace
import os
from pathlib import Path
from typing import Any, Literal, Mapping

LLMModeLiteral = Literal["mock", "gemini"]


@dataclass(frozen=True)
class LLMTimeouts:
    request_seconds: float = 60.0


DEFAULT_LLM_MODEL = "gemini-3-flash-preview"
DEFAULT_SEARCH_MODEL = "gemini-3-flash-preview"
DEFAULT_SUMMARIZER_MODEL = "gemini-3-flash-preview"
DEFAULT_IMAGE_MODEL = "gemini-2.5-flash-image"
DEFAULT_EMBED_MODEL = "local-hash-256"

ALLOWED_TEXT_MODELS = {
    "gemini-3-pro-preview",
    "gemini-3-flash-preview",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-2.5-flash-lite",
    "gemini-1.5-pro-latest",
    "gemini-1.5-flash-latest",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-2.0-pro",
    "gemini-2.0-pro-latest",
}
ALLOWED_IMAGE_MODELS = {
    "imagen-3.0-generate-002",
    "imagen-3.0-fast-generate-001",
    "gemini-2.5-flash-image",
    "gemini-3-pro-image-preview",
}
ALLOWED_EMBED_MODELS = {
    "local-hash-256",
}

@dataclass(frozen=True)
class LLMConfig:
    model_name: str = DEFAULT_LLM_MODEL
    temperature: float = 0.2
    timeouts: LLMTimeouts = field(default_factory=LLMTimeouts)
    mode: LLMModeLiteral = "mock"

ModeLiteral = Literal["demo", "live"]
ProviderLiteral = Literal["gemini", "mock"]
SourcesPolicyLiteral = Literal["fixtures", "academic_only", "mixed_reputable"]

_ALLOWED_MODES = {"demo", "live"}
_ALLOWED_PROVIDERS = {"gemini", "mock"}
_ALLOWED_POLICIES = {"fixtures", "academic_only", "mixed_reputable"}
_BOOL_FIELDS = {"allow_web", "forbid_fixture_citations", "simulate_evidence"}
_INT_FIELDS = {
    "min_sources_per_domain",
    "min_citations_per_driver",
    "seed",
    "min_forces",
    "min_forces_per_domain",
    "min_evidence_ok",
    "min_evidence_total",
}
_FLOAT_FIELDS = {"max_failed_ratio", "temperature"}
_MODEL_FIELDS = {
    "gemini_model",
    "llm_model",
    "search_model",
    "summarizer_model",
    "embed_model",
    "image_model",
}
_TEXT_MODEL_FIELDS = {
    "gemini_model",
    "llm_model",
    "search_model",
    "summarizer_model",
}
_MODEL_ALLOWED = {
    "gemini_model": ALLOWED_TEXT_MODELS,
    "llm_model": ALLOWED_TEXT_MODELS,
    "search_model": ALLOWED_TEXT_MODELS,
    "summarizer_model": ALLOWED_TEXT_MODELS,
    "embed_model": ALLOWED_EMBED_MODELS,
    "image_model": ALLOWED_IMAGE_MODELS,
}
_DYNAMIC_TEXT_MODELS: set[str] | None = None


def refresh_gemini_text_models(api_key: str | None = None) -> set[str]:
    global _DYNAMIC_TEXT_MODELS
    if _DYNAMIC_TEXT_MODELS is not None:
        return set(_DYNAMIC_TEXT_MODELS)
    try:
        from google import genai
    except Exception:
        _DYNAMIC_TEXT_MODELS = set()
        return set()
    if api_key is None:
        try:
            from scenarioops.llm.client import get_gemini_api_key

            api_key = get_gemini_api_key()
        except RuntimeError:
            _DYNAMIC_TEXT_MODELS = set()
            return set()
    try:
        client = genai.Client(api_key=api_key)
        models: set[str] = set()
        for model in client.models.list():
            methods = getattr(model, "supported_generation_methods", None) or []
            if "generateContent" not in methods:
                continue
            name = getattr(model, "name", None)
            if isinstance(name, str) and name.strip():
                models.add(name.strip())
        if models:
            ALLOWED_TEXT_MODELS.update(models)
        _DYNAMIC_TEXT_MODELS = models
        return set(models)
    except Exception:
        _DYNAMIC_TEXT_MODELS = set()
        return set()


def get_allowed_text_models(*, refresh: bool = False) -> set[str]:
    models = set(ALLOWED_TEXT_MODELS)
    if refresh:
        models.update(refresh_gemini_text_models())
    return models


def maybe_refresh_model_catalog(settings: "ScenarioOpsSettings") -> None:
    refresh = settings.llm_provider == "gemini" and bool(settings.allow_web)
    if refresh or os.environ.get("SCENARIOOPS_REFRESH_MODELS") == "1":
        refresh_gemini_text_models()


@dataclass(frozen=True)
class ScenarioOpsSettings:
    mode: ModeLiteral = "live"
    llm_provider: ProviderLiteral = "gemini"
    gemini_model: str = DEFAULT_LLM_MODEL
    llm_model: str = DEFAULT_LLM_MODEL
    search_model: str = DEFAULT_SEARCH_MODEL
    summarizer_model: str = DEFAULT_SUMMARIZER_MODEL
    embed_model: str = DEFAULT_EMBED_MODEL
    image_model: str = DEFAULT_IMAGE_MODEL
    temperature: float = 0.2
    sources_policy: SourcesPolicyLiteral = "mixed_reputable"
    allow_web: bool = False
    min_sources_per_domain: int = 8
    min_citations_per_driver: int = 2
    min_forces: int = 60
    min_forces_per_domain: int = 10
    min_evidence_ok: int = 10
    min_evidence_total: int = 15
    max_failed_ratio: float | None = None
    forbid_fixture_citations: bool = True
    simulate_evidence: bool = False
    seed: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "llm_provider": self.llm_provider,
            "gemini_model": self.gemini_model,
            "llm_model": self.llm_model,
            "search_model": self.search_model,
            "summarizer_model": self.summarizer_model,
            "embed_model": self.embed_model,
            "image_model": self.image_model,
            "temperature": self.temperature,
            "sources_policy": self.sources_policy,
            "allow_web": self.allow_web,
            "min_sources_per_domain": self.min_sources_per_domain,
            "min_citations_per_driver": self.min_citations_per_driver,
            "min_forces": self.min_forces,
            "min_forces_per_domain": self.min_forces_per_domain,
            "min_evidence_ok": self.min_evidence_ok,
            "min_evidence_total": self.min_evidence_total,
            "max_failed_ratio": self.max_failed_ratio,
            "forbid_fixture_citations": self.forbid_fixture_citations,
            "simulate_evidence": self.simulate_evidence,
            "seed": self.seed,
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
        if key in _FLOAT_FIELDS:
            if value is None:
                updated[key] = None
                continue
            try:
                updated[key] = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid float for {key}: {value}") from exc
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
        if key in _MODEL_FIELDS:
            model = str(value).strip()
            if not model:
                raise ValueError(f"Invalid model for {key}: empty value")
            allowed = _MODEL_ALLOWED.get(key)
            if allowed and model not in allowed:
                if key in _TEXT_MODEL_FIELDS:
                    refresh_gemini_text_models()
                    allowed = _MODEL_ALLOWED.get(key)
                if allowed and model not in allowed:
                    raise ValueError(
                        f"Unsupported {key}: {model}. Allowed: {', '.join(sorted(allowed))}"
                    )
            updated[key] = model
            continue
        updated[key] = str(value)
    return replace(settings, **updated)


def apply_overrides(
    settings: ScenarioOpsSettings, values: Mapping[str, Any]
) -> ScenarioOpsSettings:
    return _apply_overrides(settings, values)


def settings_from_dict(values: Mapping[str, Any]) -> ScenarioOpsSettings:
    settings = ScenarioOpsSettings()
    payload = dict(values)
    if "settings" in payload and isinstance(payload["settings"], Mapping):
        payload = dict(payload["settings"])
    settings = _apply_overrides(settings, payload)
    maybe_refresh_model_catalog(settings)
    return settings


def load_settings(
    overrides: Mapping[str, Any] | None = None, config_path: Path | None = None
) -> ScenarioOpsSettings:
    settings = ScenarioOpsSettings()
    path = config_path or _default_config_path()
    file_values = _load_yaml(path)
    settings = _apply_overrides(settings, file_values)
    if overrides:
        settings = _apply_overrides(settings, overrides)
    maybe_refresh_model_catalog(settings)
    return settings


def llm_config_from_settings(settings: ScenarioOpsSettings) -> LLMConfig:
    provider = settings.llm_provider
    if settings.mode == "demo" and provider == "mock":
        mode = "mock"
    elif provider == "gemini":
        mode = "gemini"
    else:
        mode = "mock"
    model_name = settings.llm_model or settings.gemini_model or DEFAULT_LLM_MODEL
    return LLMConfig(model_name=model_name, mode=mode, temperature=settings.temperature)
