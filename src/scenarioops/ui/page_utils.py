from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import sys

import streamlit as st


def _find_repo_root(start: Path) -> Path:
    current = start.resolve()
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return start.resolve().parent

ROOT = _find_repo_root(Path(__file__).resolve())
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

RUNS_DIR = ROOT / "storage" / "runs"
LATEST_POINTER = RUNS_DIR / "latest.json"


def _get_query_run_id() -> str | None:
    try:
        params = st.query_params
        value = params.get("run_id")
        if isinstance(value, list):
            value = value[0] if value else None
        if isinstance(value, str) and value.strip():
            return value.strip()
    except Exception:
        return None
    return None


def resolve_run_id() -> str | None:
    query_run_id = _get_query_run_id()
    if query_run_id and (RUNS_DIR / query_run_id).exists():
        return query_run_id
    if LATEST_POINTER.exists():
        try:
            return json.loads(LATEST_POINTER.read_text(encoding="utf-8")).get("run_id")
        except Exception:
            return None
    return None


def load_artifact(run_id: str | None, name: str) -> dict[str, Any] | None:
    if not run_id:
        return None
    path = RUNS_DIR / run_id / "artifacts" / f"{name}.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def load_latest_status(run_id: str | None) -> dict[str, Any] | None:
    if not run_id or not LATEST_POINTER.exists():
        return None
    try:
        latest = json.loads(LATEST_POINTER.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(latest, dict) and latest.get("run_id") == run_id:
        return latest
    return None


def page_header(title: str, run_id: str | None) -> None:
    st.header(title)
    if run_id:
        st.caption(f"Run ID: {run_id}")
    status = load_latest_status(run_id)
    if status and status.get("status") != "OK":
        error = status.get("error_summary")
        if error:
            st.error(f"Latest run failed: {error}")


def placeholder_section(title: str, lines: list[str]) -> None:
    st.subheader(title)
    for line in lines:
        st.markdown(f"- {line}")


def metric_row(metrics: dict[str, Any]) -> None:
    cols = st.columns(len(metrics)) if metrics else []
    for col, (label, value) in zip(cols, metrics.items()):
        with col:
            st.metric(label, value)
