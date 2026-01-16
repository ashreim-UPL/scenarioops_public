from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


class Transport(Protocol):
    def post_json(
        self, url: str, headers: dict[str, str], payload: dict[str, Any]
    ) -> dict[str, Any]:
        ...


@dataclass(frozen=True)
class RequestsTransport:
    timeout_seconds: float | None = None
    user_agent: str | None = None

    def post_json(
        self, url: str, headers: dict[str, str], payload: dict[str, Any]
    ) -> dict[str, Any]:
        import requests

        merged_headers = dict(headers)
        if self.user_agent and "User-Agent" not in merged_headers:
            merged_headers["User-Agent"] = self.user_agent

        response = requests.post(
            url, headers=merged_headers, json=payload, timeout=self.timeout_seconds
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise TypeError(f"Expected JSON object, got {type(data)}.")
        return data


@dataclass
class MockTransport:
    response: dict[str, Any] | None = None
    responses: list[dict[str, Any]] | None = None
    calls: list[dict[str, Any]] = field(default_factory=list)

    def post_json(
        self, url: str, headers: dict[str, str], payload: dict[str, Any]
    ) -> dict[str, Any]:
        self.calls.append({"url": url, "headers": headers, "payload": payload})
        if self.responses:
            return self.responses.pop(0)
        if self.response is not None:
            return self.response
        return {"candidates": [{"content": {"parts": [{"text": ""}]}}]}
