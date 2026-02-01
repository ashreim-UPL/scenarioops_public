from __future__ import annotations

import os
import random
import time
from typing import Callable, TypeVar

T = TypeVar("T")


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes"}


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def retry_settings() -> dict[str, float | int | bool]:
    return {
        "enabled": _bool_env("SCENARIOOPS_RETRY_ENABLED", True),
        "max_attempts": max(1, _int_env("SCENARIOOPS_RETRY_MAX", 3)),
        "base_delay": max(0.1, _float_env("SCENARIOOPS_RETRY_BASE", 1.5)),
        "max_delay": max(1.0, _float_env("SCENARIOOPS_RETRY_CAP", 12.0)),
        "jitter": max(0.0, _float_env("SCENARIOOPS_RETRY_JITTER", 0.4)),
    }


def is_transient_error(exc: Exception) -> bool:
    transient_types = (
        TimeoutError,
        ConnectionError,
    )
    if isinstance(exc, transient_types):
        return True
    message = str(exc).lower()
    return any(token in message for token in ["timeout", "timed out", "temporarily", "429", "503", "502", "504"])


def retry_call(action: Callable[[], T]) -> tuple[T, int]:
    settings = retry_settings()
    if not settings["enabled"]:
        return action(), 1
    max_attempts = int(settings["max_attempts"])
    base_delay = float(settings["base_delay"])
    max_delay = float(settings["max_delay"])
    jitter = float(settings["jitter"])
    attempt = 0
    while True:
        attempt += 1
        try:
            return action(), attempt
        except Exception as exc:
            if attempt >= max_attempts or not is_transient_error(exc):
                raise
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            delay = delay + random.uniform(0, jitter)
            time.sleep(delay)
