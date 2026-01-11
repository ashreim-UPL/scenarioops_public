from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from app.config import LLMConfig

from scenarioops.graph.tools.schema_validate import validate_schema


class LLMClient(Protocol):
    def generate_json(self, prompt: str, schema: Mapping[str, Any]) -> dict[str, Any]:
        ...

    def generate_markdown(self, prompt: str) -> str:
        ...


class GeminiTransport(Protocol):
    def generate_json(
        self,
        *,
        prompt: str,
        schema: Mapping[str, Any],
        model_name: str,
        temperature: float,
        timeout_seconds: float | None,
    ) -> str:
        ...

    def generate_markdown(
        self,
        *,
        prompt: str,
        model_name: str,
        temperature: float,
        timeout_seconds: float | None,
    ) -> str:
        ...


@dataclass(frozen=True)
class GeminiClient:
    model_name: str
    temperature: float = 0.2
    timeout_seconds: float | None = None
    transport: GeminiTransport | None = None

    def generate_json(self, prompt: str, schema: Mapping[str, Any]) -> dict[str, Any]:
        if self.transport is None:
            raise RuntimeError("GeminiClient requires a transport implementation.")
        raw = self.transport.generate_json(
            prompt=prompt,
            schema=schema,
            model_name=self.model_name,
            temperature=self.temperature,
            timeout_seconds=self.timeout_seconds,
        )
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
        if self.transport is None:
            raise RuntimeError("GeminiClient requires a transport implementation.")
        return self.transport.generate_markdown(
            prompt=prompt,
            model_name=self.model_name,
            temperature=self.temperature,
            timeout_seconds=self.timeout_seconds,
        )


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


def get_llm_client(config: "LLMConfig") -> LLMClient:
    mode = getattr(config, "mode", "mock")
    if mode == "mock":
        return MockLLMClient()
    if mode == "gemini":
        timeout_seconds = None
        timeouts = getattr(config, "timeouts", None)
        if timeouts is not None:
            timeout_seconds = getattr(timeouts, "request_seconds", None)
        return GeminiClient(
            model_name=getattr(config, "model_name", "gemini-1.5-pro"),
            temperature=getattr(config, "temperature", 0.2),
            timeout_seconds=timeout_seconds,
        )
    raise ValueError(f"Unsupported LLM mode: {mode}")
