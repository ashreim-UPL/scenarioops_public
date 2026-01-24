from __future__ import annotations

from dataclasses import dataclass, field
import time
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
    max_retries: int = 2
    backoff_seconds: float = 1.0
    retry_statuses: tuple[int, ...] = (429, 500, 502, 503, 504)

    def post_json(
        self, url: str, headers: dict[str, str], payload: dict[str, Any]
    ) -> dict[str, Any]:
        import requests

        merged_headers = dict(headers)
        if self.user_agent and "User-Agent" not in merged_headers:
            merged_headers["User-Agent"] = self.user_agent

        def _sleep_backoff(response, attempt: int) -> None:
            retry_after = response.headers.get("Retry-After") if response else None
            delay = None
            if isinstance(retry_after, str) and retry_after.strip().isdigit():
                delay = float(retry_after.strip())
            if delay is None:
                delay = self.backoff_seconds * (2 ** attempt)
            time.sleep(min(delay, 30.0))

        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(
                    url,
                    headers=merged_headers,
                    json=payload,
                    timeout=self.timeout_seconds,
                )
            except requests.RequestException as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    time.sleep(self.backoff_seconds * (2 ** attempt))
                    continue
                raise

            if response.status_code in self.retry_statuses and attempt < self.max_retries:
                _sleep_backoff(response, attempt)
                continue

            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                raise TypeError(f"Expected JSON object, got {type(data)}.")
            return data

        if last_exc:
            raise last_exc
        raise RuntimeError("Request failed without response.")


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
