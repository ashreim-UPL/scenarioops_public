import pytest

from scenarioops.llm.client import GeminiClient


class StubTransport:
    def __init__(self, raw: str) -> None:
        self._raw = raw

    def generate_json(
        self,
        *,
        prompt: str,
        schema,
        model_name: str,
        temperature: float,
        timeout_seconds: float | None,
    ) -> str:
        return self._raw

    def generate_markdown(
        self,
        *,
        prompt: str,
        model_name: str,
        temperature: float,
        timeout_seconds: float | None,
    ) -> str:
        raise NotImplementedError


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
    client = GeminiClient(model_name="stub", transport=StubTransport(raw))

    payload = client.generate_json("prompt", _schema())

    assert payload["foo"] == "bar"


def test_generate_json_parses_code_fence() -> None:
    raw = "```json\n{\"foo\": \"bar\"}\n```"
    client = GeminiClient(model_name="stub", transport=StubTransport(raw))

    payload = client.generate_json("prompt", _schema())

    assert payload["foo"] == "bar"


def test_generate_json_parses_embedded_object() -> None:
    raw = "Result:\n{\"foo\": \"bar\"}\nThanks."
    client = GeminiClient(model_name="stub", transport=StubTransport(raw))

    payload = client.generate_json("prompt", _schema())

    assert payload["foo"] == "bar"


def test_generate_json_rejects_array() -> None:
    raw = '[{"foo": "bar"}]'
    client = GeminiClient(model_name="stub", transport=StubTransport(raw))

    with pytest.raises(TypeError):
        client.generate_json("prompt", _schema())
