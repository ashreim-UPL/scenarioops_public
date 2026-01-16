from __future__ import annotations

from typing import Any, Mapping

from scenarioops.llm.client import LLMClient
from .types import Gemini3Client


class SquadClient(Gemini3Client):
    """Concrete implementation of Gemini3Client tracking history."""

    def __init__(self, inner: LLMClient, thinking_level: str = "low"):
        self.inner = inner
        self.history: list[dict[str, Any]] = []
        self.thinking_level = thinking_level

    def generate_json(self, prompt: str, schema: Mapping[str, Any]) -> dict[str, Any]:
        # Track the "thought" (prompt)
        entry = {"role": "user", "content": prompt, "schema": schema.get("title")}
        if hasattr(self, "thinking_level"):
             entry["thinking_level"] = self.thinking_level
        self.history.append(entry)
        
        # Execute
        result = self.inner.generate_json(prompt, schema)
        
        # Track the result with implicit thoughtSignature if available
        # In a real Gemini 3 response, we might extract the 'thought' part if we had access to full candidates.
        # Since LLMClient.generate_json returns the parsed dict, we just store that.
        # If we had access to the raw response object including reasoning traces, we'd store it here.
        # For now, we assume the result itself contains the output.
        self.history.append({
            "role": "assistant", 
            "content": result,
            "thoughtSignature": "implicit-v1" # Placeholder for actual signature extraction
        })
        return result

    def generate_markdown(self, prompt: str) -> str:
        entry = {"role": "user", "content": prompt, "type": "markdown"}
        if hasattr(self, "thinking_level"):
             entry["thinking_level"] = self.thinking_level
        self.history.append(entry)

        result = self.inner.generate_markdown(prompt)
        
        self.history.append({
            "role": "assistant", 
            "content": result,
            "thoughtSignature": "implicit-v1"
        })
        return result
