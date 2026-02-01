from __future__ import annotations

import json
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from scenarioops.app.config import LLMConfig
from scenarioops.llm.client import LLMClient, get_llm_client
from scenarioops.graph.tools.prompts import load_prompt_spec


def prompts_dir() -> Path:
    # src/scenarioops/graph/nodes/utils.py -> parents[2] is src/scenarioops
    return Path(__file__).resolve().parents[2] / "prompts"


@dataclass(frozen=True)
class PromptBundle:
    name: str
    sha256: str
    text: str


def load_prompt(name: str) -> str:
    return load_prompt_spec(name).text


def build_prompt(name: str, context: Mapping[str, Any]) -> PromptBundle:
    spec = load_prompt_spec(name)
    prompt = render_prompt(spec.text, context)
    return PromptBundle(name=spec.name, sha256=spec.sha256, text=prompt)


def render_prompt(template: str, context: Mapping[str, Any]) -> str:
    now = datetime.now(timezone.utc)
    enriched = dict(context)
    enriched.setdefault("current_date", now.date().isoformat())
    enriched.setdefault("current_datetime", now.isoformat())
    enriched.setdefault("current_timezone", "UTC")
    payload = json.dumps(
        enriched,
        indent=2,
        sort_keys=True,
        default=_json_default,
    )
    return f"{template}\n\nContext:\n{payload}\n"


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


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
