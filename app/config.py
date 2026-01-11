from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class LLMTimeouts:
    request_seconds: float = 60.0


@dataclass(frozen=True)
class LLMConfig:
    model_name: str = "gemini-1.5-pro"
    temperature: float = 0.2
    timeouts: LLMTimeouts = field(default_factory=LLMTimeouts)
    mode: Literal["mock", "gemini"] = "mock"
