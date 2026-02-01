from __future__ import annotations

import os
from contextvars import ContextVar
from typing import Optional

from fastapi import Request

_REQUEST_GEMINI_KEY: ContextVar[Optional[str]] = ContextVar(
    "scenarioops_gemini_api_key", default=None
)


def extract_gemini_api_key(request: Request | None) -> str | None:
    if request is None:
        return None
    header = request.headers.get("X-Gemini-Api-Key") or request.headers.get(
        "X-GEMINI-API-KEY"
    )
    if header:
        return header.strip()
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        return auth.split(" ", 1)[1].strip()
    return None


def set_request_gemini_api_key(value: str | None):
    if value:
        return _REQUEST_GEMINI_KEY.set(value)
    return _REQUEST_GEMINI_KEY.set(None)


def reset_request_gemini_api_key(token) -> None:
    _REQUEST_GEMINI_KEY.reset(token)


def get_request_gemini_api_key() -> str | None:
    return _REQUEST_GEMINI_KEY.get()


def has_env_gemini_api_key() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY", "").strip())

