from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

@dataclass(frozen=True)
class ArtifactData:
    name: str
    payload: Any
    ext: str = "json"
    input_values: Mapping[str, Any] | None = None
    prompt_values: Mapping[str, Any] | None = None
    tool_versions: Mapping[str, str] | None = None


@dataclass(frozen=True)
class NodeResult:
    """
    Represents the pure result of a node execution.
    The orchestrator is responsible for applying state_updates to the global state
    and persisting any artifacts.
    """
    state_updates: Mapping[str, Any] = field(default_factory=dict)
    artifacts: list[ArtifactData] = field(default_factory=list)


@dataclass(frozen=True)
class GraphInputs:
    user_params: Mapping[str, Any]
    sources: Sequence[str]
    signals: Sequence[Mapping[str, Any]]