import pytest

from scenarioops.llm.client import GeminiClient


class StubTransport:
    def __init__(self, raw: str) -> None:
        self._raw = raw

    def post_json(self, url: str, headers: dict, payload: dict) -> dict:
        return {"candidates": [{"content": {"parts": [{"text": self._raw}]}}]}


def _schema() -> dict:
    return {
        "title": "Test Payload",
        "type": "object",
        "required": ["foo"],
        "properties": {"foo": {"type": "string"}},
        "additionalProperties": False,
    }


def test_generate_json_parses_object() -> None:
    raw = '{"foo": "bar"}'
    client = GeminiClient(api_key="test", model="stub", transport=StubTransport(raw))

    payload = client.generate_json("prompt", _schema())

    assert payload["foo"] == "bar"


def test_generate_json_parses_code_fence() -> None:
    raw = "```json\n{\"foo\": \"bar\"}\n```"
    client = GeminiClient(api_key="test", model="stub", transport=StubTransport(raw))

    payload = client.generate_json("prompt", _schema())

    assert payload["foo"] == "bar"


def test_generate_json_parses_embedded_object() -> None:
    raw = "Result:\n{\"foo\": \"bar\"}\nThanks."
    client = GeminiClient(api_key="test", model="stub", transport=StubTransport(raw))

    payload = client.generate_json("prompt", _schema())

    assert payload["foo"] == "bar"


def test_generate_json_rejects_array() -> None:
    raw = '[{"foo": "bar"}]'
    client = GeminiClient(api_key="test", model="stub", transport=StubTransport(raw))

    with pytest.raises(TypeError):
        client.generate_json("prompt", _schema())


def test_generate_json_wraps_single_array_payload() -> None:
    raw = '[{"force_id": "f-1"}]'
    client = GeminiClient(api_key="test", model="stub", transport=StubTransport(raw))
    schema = {
        "title": "Forces Payload",
        "type": "object",
        "required": ["forces"],
        "properties": {"forces": {"type": "array"}},
        "additionalProperties": False,
    }

    payload = client.generate_json("prompt", schema)

    assert isinstance(payload.get("forces"), list)
    assert payload["forces"][0]["force_id"] == "f-1"
