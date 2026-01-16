import hashlib

from scenarioops.app.config import LLMConfig
from scenarioops.llm.client import (
    GeminiClient,
    MockLLMClient,
    MockTransport,
    get_llm_client,
)


def test_mock_llm_client_is_deterministic() -> None:
    prompt = "Tell me about ScenarioOps."
    schema = {"title": "Note"}
    client = MockLLMClient()

    first = client.generate_json(prompt, schema)
    second = client.generate_json(prompt, schema)

    assert first == second
    assert first["prompt_hash"] == hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    assert first["schema_title"] == "Note"


def test_get_llm_client_returns_mock() -> None:
    config = LLMConfig(mode="mock")
    client = get_llm_client(config)

    markdown = client.generate_markdown("Hello!")
    assert markdown.startswith("# Mock Response")


def test_gemini_client_uses_transport() -> None:
    schema = {
        "title": "Transport Payload",
        "type": "object",
        "required": ["foo"],
        "properties": {"foo": {"type": "string"}},
        "additionalProperties": False,
    }
    response = {
        "candidates": [{"content": {"parts": [{"text": "{\"foo\": \"bar\"}"}]}}]
    }
    transport = MockTransport(response=response)
    client = GeminiClient(api_key="test", model="stub", transport=transport)

    payload = client.generate_json("hello", schema)

    assert payload["foo"] == "bar"
    assert len(transport.calls) == 1
