from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from app.config import LLMConfig
from scenarioops.llm.client import LLMClient, get_llm_client


def prompts_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "prompts"


def load_prompt(name: str) -> str:
    path = prompts_dir() / f"{name}.txt"
    return path.read_text(encoding="utf-8")


def render_prompt(template: str, context: Mapping[str, Any]) -> str:
    payload = json.dumps(context, indent=2, sort_keys=True)
    return f"{template}\n\nContext:\n{payload}\n"


def get_client(client: LLMClient | None, config: LLMConfig | None = None) -> LLMClient:
    if client is not None:
        return client
    if config is None:
        config = LLMConfig()
    return get_llm_client(config)


def parse_json_response(response: Any) -> Any:
    if isinstance(response, str):
        return json.loads(response)
    return response
