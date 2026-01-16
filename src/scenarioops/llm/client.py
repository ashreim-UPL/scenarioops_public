from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from scenarioops.app.config import LLMConfig

from scenarioops.graph.tools.schema_validate import validate_schema
from scenarioops.llm.transport import MockTransport, RequestsTransport, Transport


class LLMClient(Protocol):
    def generate_json(self, prompt: str, schema: Mapping[str, Any]) -> dict[str, Any]:
        ...

    def generate_markdown(self, prompt: str) -> str:
        ...


@dataclass(frozen=True)
class GeminiClient:
    api_key: str
    model: str
    transport: Transport
    temperature: float = 0.2

    def generate_json(self, prompt: str, schema: Mapping[str, Any]) -> dict[str, Any]:
        raw = self._generate_text(prompt)
        if not isinstance(raw, str):
            raise TypeError(f"Expected raw model output as str, got {type(raw)}.")

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            extracted = _extract_first_json_object(raw)
            if extracted is None:
                raise ValueError(
                    f"Unable to locate JSON object in output: {raw[:500]!r}"
                ) from exc
            parsed = json.loads(extracted)

        if not isinstance(parsed, dict):
            raise TypeError(
                f"Expected JSON object, got {type(parsed)}. Raw: {raw[:500]!r}"
            )

        schema_name = "unknown"
        if isinstance(schema, Mapping):
            schema_name = str(schema.get("title") or "unknown")
        validate_schema(parsed, schema, schema_name)
        return _wrap_payload(parsed, raw)

    def generate_markdown(self, prompt: str) -> str:
        return self._generate_text(prompt)

    def _generate_text(self, prompt: str) -> str:
        url = _gemini_url(self.model, self.api_key)
        headers = {"Content-Type": "application/json"}
        # Include thoughtSignature capability implicitly by using the preview model
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": self.temperature},
        }
        response = self.transport.post_json(url, headers, payload)
        if not isinstance(response, Mapping):
            raise TypeError(f"Expected response mapping, got {type(response)}.")
        return _extract_candidate_text(response)


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class _JsonPayload(dict):
    __slots__ = ("raw",)

    def __init__(self, payload: Mapping[str, Any], raw: str) -> None:
        super().__init__()
        self.update(payload)
        self.raw = raw


def _wrap_payload(payload: Mapping[str, Any], raw: str) -> dict[str, Any]:
    if isinstance(payload, _JsonPayload):
        return payload
    return _JsonPayload(payload, raw)


def _extract_first_json_object(raw: str) -> str | None:
    start = raw.find("{")
    if start == -1:
        return None
    depth = 0
    for idx in range(start, len(raw)):
        char = raw[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return raw[start : idx + 1]
    return None


def _gemini_url(model: str, api_key: str) -> str:
    return (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )


def _extract_candidate_text(response: Mapping[str, Any]) -> str:
    error = response.get("error")
    if isinstance(error, Mapping):
        message = error.get("message", "Unknown Gemini API error.")
        raise RuntimeError(f"Gemini API error: {message}")
    candidates = response.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("Gemini response missing candidates.")
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        content = candidate.get("content")
        if isinstance(content, Mapping):
            parts = content.get("parts")
            if isinstance(parts, list):
                for part in parts:
                    if isinstance(part, Mapping):
                        text = part.get("text")
                        if isinstance(text, str):
                            return text
        text = candidate.get("text")
        if isinstance(text, str):
            return text
    raise ValueError("Gemini response missing text output.")


@dataclass
class MockLLMClient:
    json_payload: dict[str, Any] | None = None
    markdown_payload: str | None = None

    def generate_json(self, prompt: str, schema: Mapping[str, Any]) -> dict[str, Any]:
        if self.json_payload is not None:
            if not isinstance(self.json_payload, dict):
                raise TypeError(
                    f"Expected mock json_payload dict, got {type(self.json_payload)}."
                )
            raw = json.dumps(self.json_payload, sort_keys=True)
            return _wrap_payload(self.json_payload, raw)
        schema_title = schema.get("title") if isinstance(schema, Mapping) else None
        
        if schema_title == "Charter":
            payload = {
                "id": "mock-charter-uuid",
                "title": "Mock Charter Title",
                "purpose": "Generated by MockLLMClient for testing.",
                "decision_context": "Mock Decision Context",
                "scope": "Global Mock Scope",
                "time_horizon": "5 years",
                "stakeholders": ["Executive Team"],
                "constraints": ["Budget"],
                "assumptions": ["Market stability"],
                "success_criteria": ["Actionable plan"],
            }
        elif schema_title == "Focal Issue":
            payload = {
                "focal_issue": "Mock Focal Issue Decision",
                "scope": {
                    "geography": "Global",
                    "sectors": ["Technology"],
                    "time_horizon_years": 5
                },
                "decision_type": "Strategic",
                "exclusions": ["Operational details"],
                "success_criteria": "Clear decision path",
            }
        elif schema_title == "Driving Forces":
            payload = {
                "forces": [
                    {
                        "id": "mock-force-1",
                        "name": "Mock AI Regulation",
                        "domain": "Political",
                        "lenses": ["geopolitics"],
                        "description": "Rising calls for AI safety laws.",
                        "why_it_matters": "Could limit deployment speed.",
                        "citations": [{"url": "https://example.com/ai-policy"}]
                    },
                    {
                        "id": "mock-force-2",
                        "name": "GPU Shortage",
                        "domain": "Technological",
                        "lenses": ["macro"],
                        "description": "Supply chain constraints on chips.",
                        "why_it_matters": "Bottleneck for scaling.",
                        "citations": [{"url": "https://example.com/chips"}]
                    }
                ]
            }
        elif schema_title == "Drivers Payload":
            payload = {
                "drivers": [
                    {
                        "id": "drv-1",
                        "name": "AI Regulation",
                        "description": "New compliance.",
                        "category": "political",
                        "trend": "increasing",
                        "impact": "high",
                        "citations": [{"url": "https://example.com/ai-policy"}]
                    },
                    {
                        "id": "drv-2",
                        "name": "Chip Supply",
                        "description": "Shortage.",
                        "category": "technological",
                        "trend": "volatile",
                        "impact": "high",
                        "citations": [{"url": "https://example.com/chips"}]
                    }
                ]
            }
        elif schema_title == "Uncertainties":
            payload = {
                "id": "unc-1",
                "title": "Critical uncertainties",
                "uncertainties": [
                    {
                        "id": "u1",
                        "name": "Reg Pace",
                        "description": "Fast vs Slow",
                        "extremes": ["low", "high"],
                        "driver_ids": ["drv-1", "drv-2"],
                        "criticality": 5,
                        "volatility": 4,
                        "implications": ["Impacts compliance"]
                    },
                    {
                        "id": "u2",
                        "name": "Tech Supply",
                        "description": "Abundant vs Scarce",
                        "extremes": ["plenty", "none"],
                        "driver_ids": ["drv-2", "drv-1"],
                        "criticality": 4,
                        "volatility": 5,
                        "implications": ["Impacts production"]
                    }
                ]
            }
        elif schema_title == "Logic" or schema_title == "Scenario Logic":
            payload = {
                "id": "logic-1",
                "title": "Logic",
                "axes": [
                    {"uncertainty_id": "u1", "low": "Slow Reg", "high": "Fast Reg"},
                    {"uncertainty_id": "u2", "low": "Scarce Chips", "high": "Plenty Chips"}
                ],
                "scenarios": [
                    {"id": "S1", "name": "Slow/Scarce", "logic": "Slow + Scarce"},
                    {"id": "S2", "name": "Fast/Scarce", "logic": "Fast + Scarce"},
                    {"id": "S3", "name": "Slow/Plenty", "logic": "Slow + Plenty"},
                    {"id": "S4", "name": "Fast/Plenty", "logic": "Fast + Plenty"}
                ]
            }
        elif schema_title == "Skeleton" or schema_title == "Skeletons":
            payload = {
                "id": "sk-1",
                "title": "Skeletons",
                "scenarios": [
                    {"id": "S1", "name": "Slow/Scarce", "narrative": "Stagnation.", "operating_rules": {"policy": "a", "market": "b", "operations": "c"}, "key_events": []},
                    {"id": "S2", "name": "Fast/Scarce", "narrative": "Choked.", "operating_rules": {"policy": "a", "market": "b", "operations": "c"}, "key_events": []},
                    {"id": "S3", "name": "Slow/Plenty", "narrative": "Boom.", "operating_rules": {"policy": "a", "market": "b", "operations": "c"}, "key_events": []},
                    {"id": "S4", "name": "Fast/Plenty", "narrative": "Hyper.", "operating_rules": {"policy": "a", "market": "b", "operations": "c"}, "key_events": []}
                ]
            }
        elif schema_title == "Strategies":
            payload = {
                "id": "strat-1",
                "title": "Strategies",
                "strategies": [
                    {
                        "id": "st-1",
                        "name": "Invest in R&D",
                        "objective": "Innovate",
                        "actions": ["Build Lab"],
                        "kpis": ["Patents"],
                        "projected_roi": 0.15
                    },
                    {
                        "id": "st-2",
                        "name": "Lobbying",
                        "objective": "Influence",
                        "actions": ["Hire firm"],
                        "kpis": ["Meetings"],
                        "projected_roi": 0.05
                    }
                ]
            }
        elif schema_title == "Wind Tunnel":
            payload = {
                "id": "wt-1",
                "title": "Wind Tunnel",
                "tests": [
                    {
                        "id": "wt-t1",
                        "strategy_id": "st-1",
                        "scenario_id": "S1",
                        "outcome": "Resilient",
                        "failure_modes": [],
                        "adaptations": [],
                        "feasibility_score": 0.9,
                        "rubric_score": 0.9,
                        "action": "KEEP",
                        "rubric_inputs": {"relevance": 0.9, "credibility": 0.9, "recency": 0.9, "specificity": 0.9}
                    }
                ]
            }
        elif schema_title == "Audit Report":
            payload = {
                "id": "audit-1",
                "title": "Audit Report",
                "findings": [],
                "score": 1.0,
                "status": "PASS"
            }
        elif schema_title == "Washout Report":
            payload = {
                "duplicate_ratio": 0.0,
                "duplicate_groups": [],
                "undercovered_domains": [],
                "missing_categories": [],
                "proposed_forces": [],
                "reason": "Mock audit passed.",
                "notes": "No duplicates found."
            }
        else:
            payload = {
                "mock": True,
                "prompt_hash": _hash_text(prompt),
                "schema_title": schema_title or "unknown",
            }
        raw = json.dumps(payload, sort_keys=True)
        return _wrap_payload(payload, raw)

    def generate_markdown(self, prompt: str) -> str:
        if self.markdown_payload is not None:
            return self.markdown_payload
        
        if "skeletons" in prompt or "Scenario:" in prompt:
            return (
                "## Scenario: S1\nFirst narrative.\n\n"
                "## Scenario: S2\nSecond narrative.\n\n"
                "## Scenario: S3\nThird narrative.\n\n"
                "## Scenario: S4\nFourth narrative.\n"
            )

        return f"# Mock Response\n\nprompt_hash: {_hash_text(prompt)}\n"


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = value.strip().strip("'\"")
        os.environ.setdefault(key, value)


def _try_load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
        return
    except ImportError:
        pass
    # Minimal fallback when python-dotenv isn't installed.
    cwd = Path.cwd()
    for parent in [cwd, *cwd.parents]:
        candidate = parent / ".env"
        if candidate.exists():
            _load_env_file(candidate)
            break


def get_gemini_api_key() -> str:
    # Try loading from .env file first
    _try_load_dotenv()

    try:
        import streamlit as st  # type: ignore
    except Exception:
        st = None

    if st is not None:
        try:
            secret_value = st.secrets.get("GEMINI_API_KEY")
        except Exception:
            secret_value = None
        if isinstance(secret_value, str):
            secret_value = secret_value.strip()
            if secret_value:
                return secret_value

    value = os.environ.get("GEMINI_API_KEY", "").strip()
    if value:
        return value
    raise RuntimeError("Missing GEMINI_API_KEY. Please set it in a .env file or environment variable.")


def get_gemini_api_key_from_env() -> str:
    return get_gemini_api_key()


def get_llm_client(config: "LLMConfig") -> LLMClient:
    mode = getattr(config, "mode", "mock")
    if mode == "mock":
        return MockLLMClient()
    if mode == "gemini":
        api_key = get_gemini_api_key()
        timeout_seconds = None
        timeouts = getattr(config, "timeouts", None)
        if timeouts is not None:
            timeout_seconds = getattr(timeouts, "request_seconds", None)
        transport = RequestsTransport(
            timeout_seconds=timeout_seconds,
            user_agent="ScenarioOps",
        )
        return GeminiClient(
            api_key=api_key,
            model=getattr(config, "model_name", "gemini-3-pro-preview"),
            transport=transport,
            temperature=getattr(config, "temperature", 0.2),
        )
    raise ValueError(f"Unsupported LLM mode: {mode}")


__all__ = [
    "GeminiClient",
    "LLMClient",
    "MockLLMClient",
    "MockTransport",
    "RequestsTransport",
    "Transport",
    "get_gemini_api_key",
    "get_gemini_api_key_from_env",
    "get_llm_client",
]
