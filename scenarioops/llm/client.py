from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
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
                "scope": "Global Mock Scope",
                "time_horizon": "5 years",
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
                        "description": "Rising calls for AI safety laws.",
                        "why_it_matters": "Could limit deployment speed.",
                        "citations": [{"url": "https://example.com/ai-policy"}]
                    },
                    {
                        "id": "mock-force-2",
                        "name": "GPU Shortage",
                        "domain": "Technological",
                        "description": "Supply chain constraints on chips.",
                        "why_it_matters": "Bottleneck for scaling.",
                        "citations": [{"url": "https://example.com/chips"}]
                    }
                ]
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
        return f"# Mock Response\n\nprompt_hash: {_hash_text(prompt)}\n"


def get_gemini_api_key() -> str:
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
    raise RuntimeError("Missing GEMINI_API_KEY")


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
            model=getattr(config, "model_name", "gemini-1.5-pro"),
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
