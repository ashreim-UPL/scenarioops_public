from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from scenarioops.app.config import LLMConfig
from scenarioops.llm.client import LLMClient, get_llm_client


def prompts_dir() -> Path:
    # __file__ is src/scenarioops/graph/nodes/utils.py
    # parents[4] is root
    return Path(__file__).resolve().parents[4] / "prompts"


def load_prompt(name: str) -> str:
    prompt_root = prompts_dir()
    candidate = prompt_root / name
    if candidate.suffix:
        return candidate.read_text(encoding="utf-8")
    for ext in (".prompt", ".txt"):
        path = prompt_root / f"{name}{ext}"
        if path.exists():
            return path.read_text(encoding="utf-8")
    raise FileNotFoundError(f"Prompt not found: {prompt_root / name}")


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
