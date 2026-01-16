from __future__ import annotations

from typing import Any


def truncate_for_log(text: str, n: int = 800) -> str:
    if text is None:
        return ""
    snippet = str(text)
    if len(snippet) <= n:
        return snippet
    return f"{snippet[:n]}...[truncated]"


def _raw_snippet(payload: Any) -> str:
    raw = getattr(payload, "raw", None)
    if raw is None:
        return truncate_for_log(repr(payload))
    return truncate_for_log(str(raw))


def ensure_dict(payload: Any, *, node_name: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raw = _raw_snippet(payload)
        raise TypeError(
            f"{node_name}: expected dict, received {type(payload)}. raw={raw}"
        )
    return payload


def ensure_key(payload: Any, key: str, *, node_name: str) -> Any:
    payload = ensure_dict(payload, node_name=node_name)
    if key not in payload:
        raw = _raw_snippet(payload)
        keys = list(payload.keys())
        raise TypeError(
            f"{node_name}: expected dict with key '{key}', received keys={keys}. raw={raw}"
        )
    return payload[key]


def ensure_list(payload: Any, *, node_name: str) -> list[Any]:
    if not isinstance(payload, list):
        raw = _raw_snippet(payload)
        label = getattr(payload, "label", None)
        expected = f"{label} list" if label else "list"
        raise TypeError(
            f"{node_name}: expected {expected}, received {type(payload)}. raw={raw}"
        )
    return payload
