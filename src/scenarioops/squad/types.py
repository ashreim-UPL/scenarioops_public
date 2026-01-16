from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

@runtime_checkable
class Gemini3Client(Protocol):
    history: list[dict[str, Any]]
    thinking_level: str  # 'low' | 'high'

    def generate_json(self, prompt: str, schema: Mapping[str, Any]) -> dict[str, Any]:
        ...

    def generate_markdown(self, prompt: str) -> str:
        ...

class GMReviewRequired(Exception):
    """Raised when critical safety thresholds (e.g., ROI) are breached."""
    def __init__(self, message: str, context: dict[str, Any]):
        super().__init__(message)
        self.context = context

@dataclass
class AgentState:
    history: list[dict[str, Any]] = field(default_factory=list)
