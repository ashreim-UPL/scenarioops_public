from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from scenarioops.app.config import ScenarioOpsSettings
from scenarioops.llm.client import get_gemini_api_key
from scenarioops.llm.transport import RequestsTransport, Transport


_PLACEHOLDER_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8"
    "/x8AAwMB/6XprZkAAAAASUVORK5CYII="
)


class ImageClient(Protocol):
    def generate_image(self, prompt: str, *, model: str) -> bytes:
        ...


def placeholder_image_bytes() -> bytes:
    return _PLACEHOLDER_PNG


def _extract_base64(payload: Any) -> str | None:
    if not isinstance(payload, Mapping):
        return None
    for key in ("bytesBase64Encoded", "imageBytes", "data"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    inline = payload.get("inlineData") or payload.get("inline_data")
    if isinstance(inline, Mapping):
        value = inline.get("data") or inline.get("bytesBase64Encoded")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _extract_image_bytes(response: Mapping[str, Any]) -> bytes:
    for key in ("generatedImages", "images", "outputs", "candidates"):
        items = response.get(key)
        if isinstance(items, list):
            for item in items:
                encoded = _extract_base64(item)
                if encoded:
                    return base64.b64decode(encoded)
    encoded = _extract_base64(response)
    if encoded:
        return base64.b64decode(encoded)
    raise ValueError("Image response did not include base64-encoded image data.")


@dataclass(frozen=True)
class GeminiImageClient:
    api_key: str
    transport: Transport
    timeout_seconds: float | None = 30.0

    def generate_image(self, prompt: str, *, model: str) -> bytes:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateImages?key={self.api_key}"
        )
        payload = {"prompt": {"text": prompt}, "image": {"sampleCount": 1}}
        response = self.transport.post_json(
            url,
            {"Content-Type": "application/json"},
            payload,
        )
        error = response.get("error")
        if isinstance(error, Mapping):
            message = error.get("message", "Unknown image generation error.")
            raise RuntimeError(f"Gemini image API error: {message}")
        return _extract_image_bytes(response)


@dataclass(frozen=True)
class GeminiGenAIImageClient:
    api_key: str

    def generate_image(self, prompt: str, *, model: str) -> bytes:
        try:
            from google import genai
            from google.genai import types
        except Exception as exc:
            raise RuntimeError(
                "google-genai package is required for Gemini image models."
            ) from exc

        client = genai.Client(api_key=self.api_key)
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
            ),
        )
        candidates = getattr(response, "candidates", None)
        if not candidates:
            raise ValueError("Gemini image response missing candidates.")
        content = candidates[0].content if candidates else None
        parts = getattr(content, "parts", None) if content else None
        if not parts:
            raise ValueError("Gemini image response missing content parts.")
        for part in parts:
            inline_data = getattr(part, "inline_data", None)
            if inline_data and getattr(inline_data, "data", None):
                return bytes(inline_data.data)
        raise ValueError("Gemini image response did not include inline image data.")


@dataclass(frozen=True)
class MockImageClient:
    def generate_image(self, prompt: str, *, model: str) -> bytes:
        return _PLACEHOLDER_PNG


def get_image_client(settings: ScenarioOpsSettings) -> ImageClient:
    if settings.llm_provider == "mock" or not settings.allow_web:
        return MockImageClient()
    try:
        api_key = get_gemini_api_key()
    except RuntimeError:
        return MockImageClient()
    image_model = getattr(settings, "image_model", "") or ""
    if image_model.startswith("gemini-"):
        return GeminiGenAIImageClient(api_key=api_key)
    transport = RequestsTransport(timeout_seconds=30, user_agent="ScenarioOps")
    return GeminiImageClient(api_key=api_key, transport=transport)
