from __future__ import annotations

import os
import logging
from contextvars import ContextVar
from typing import Tuple

from fastapi import Request


_API_KEY_CTX: ContextVar[str | None] = ContextVar("scenarioops_api_key", default=None)
_API_KEY_SOURCE_CTX: ContextVar[str] = ContextVar("scenarioops_api_key_source", default="none")
_LOGGER = logging.getLogger(__name__)


def set_request_api_key(key: str | None, source: str) -> None:
    _API_KEY_CTX.set(key)
    _API_KEY_SOURCE_CTX.set(source)


def get_request_api_key() -> str | None:
    return _API_KEY_CTX.get()


def get_request_api_key_source() -> str:
    return _API_KEY_SOURCE_CTX.get()


def resolve_api_key(request: Request) -> Tuple[str | None, str]:
    key = request.headers.get("X-Api-Key") or request.headers.get("X-Gemini-Api-Key")
    source = "header"
    if not key:
        auth = request.headers.get("Authorization", "")
        parts = auth.split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            key = parts[1]
            source = "bearer"
    if not key:
        key = request.query_params.get("api_key")
        if key:
            source = "query"
    if not key:
        env_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if env_key:
            key = env_key
            source = "env"
    if not key:
        source = "none"
    set_request_api_key(key, source)
    _LOGGER.info("API key resolved from: %s", source)
    return key, source


__all__ = [
    "resolve_api_key",
    "set_request_api_key",
    "get_request_api_key",
    "get_request_api_key_source",
]
