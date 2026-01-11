import hashlib

from app.config import LLMConfig
from scenarioops.llm.client import MockLLMClient, get_llm_client


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
