from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
import sys

import streamlit as st

from scenarioops.graph.tools.storage import default_runs_dir


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

RUNS_DIR = default_runs_dir()
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
    def _persist(run_id: str) -> None:
        st.session_state["run_id"] = run_id
        st.session_state["last_run_id"] = run_id
        try:
            params = st.query_params
            value = params.get("run_id")
            if isinstance(value, list):
                value = value[0] if value else None
            if value != run_id:
                st.query_params["run_id"] = run_id
        except Exception:
            pass

    def _existing(run_id: str | None) -> str | None:
        if not run_id:
            return None
        candidate = str(run_id).strip()
        if not candidate:
            return None
        return candidate if (RUNS_DIR / candidate).exists() else None

    query_run_id = _get_query_run_id()
    existing = _existing(query_run_id)
    if existing:
        _persist(existing)
        return existing
    session_run_id = _existing(st.session_state.get("run_id"))
    if session_run_id:
        _persist(session_run_id)
        return session_run_id
    last_run_id = _existing(st.session_state.get("last_run_id"))
    if last_run_id:
        _persist(last_run_id)
        return last_run_id
    if LATEST_POINTER.exists():
        try:
            latest = json.loads(LATEST_POINTER.read_text(encoding="utf-8"))
        except Exception:
            return None
        resolved = _existing(latest.get("run_id") if isinstance(latest, dict) else None)
        if resolved:
            _persist(resolved)
        return resolved
    return None


def run_dir(run_id: str) -> Path:
    return RUNS_DIR / run_id


def load_artifact(run_id: str | None, name: str) -> Any | None:
    if not run_id:
        return None
    path = RUNS_DIR / run_id / "artifacts" / f"{name}.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload


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


def resolve_image_path(run_id: str | None, relative_path: str | None) -> Path | None:
    if not run_id or not relative_path:
        return None
    path = run_dir(run_id) / Path(relative_path)
    return path if path.exists() else None


def apply_branding() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700&family=Spectral:wght@600;700&display=swap');
        :root {
          --ink:#1b1f24;
          --muted:#5f6772;
          --brand:#0e7c86;
          --brand-deep:#0a4a52;
          --accent:#f2c94c;
          --bg:#f6f1ea;
          --card:#ffffff;
          --border:#e6dfd4;
        }
        .stApp {
          background: radial-gradient(1200px 600px at 15% 0%, #fbf7f0 0%, #f6f1ea 55%, #efe6da 100%);
        }
        h1, h2, h3, h4 {
          font-family: 'Spectral', serif;
          color: var(--ink);
        }
        p, li, span, div, label {
          font-family: 'Manrope', sans-serif;
          color: var(--ink);
        }
        .so-hero {
          display:flex;
          justify-content:space-between;
          align-items:flex-end;
          padding:24px 26px;
          border:1px solid var(--border);
          background: linear-gradient(120deg, #ffffff 0%, #fbf7f0 55%, #f1e7d7 100%);
          border-radius:16px;
          box-shadow: 0 14px 30px rgba(31, 24, 13, 0.12);
          margin-bottom:20px;
        }
        .so-eyebrow {
          text-transform: uppercase;
          letter-spacing: 0.18em;
          font-size: 11px;
          font-weight: 700;
          color: var(--brand);
        }
        .so-title {
          font-size: 28px;
          font-weight: 700;
          margin-top: 6px;
        }
        .so-subtitle {
          font-size: 14px;
          color: var(--muted);
          margin-top: 4px;
        }
        .so-run {
          font-size: 12px;
          color: var(--muted);
          text-align:right;
        }
        .so-section {
          padding: 18px 22px;
          border: 1px solid var(--border);
          border-radius: 14px;
          background: var(--card);
          margin-bottom: 18px;
          box-shadow: 0 10px 18px rgba(31, 24, 13, 0.08);
        }
        .so-section h3 {
          margin: 0 0 8px 0;
          font-size: 20px;
        }
        .so-section p {
          margin: 0;
          color: var(--muted);
        }
        .so-grid {
          display:grid;
          grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
          gap:14px;
          margin-top:14px;
        }
        .so-card {
          padding: 14px 16px;
          border-radius: 12px;
          border: 1px solid var(--border);
          background: #fffdf8;
        }
        .so-card h4 {
          margin: 0 0 6px 0;
          font-size: 16px;
        }
        .so-pill {
          display:inline-block;
          padding: 4px 10px;
          border-radius: 999px;
          background: #0e7c8620;
          color: var(--brand-deep);
          font-size: 12px;
          font-weight: 600;
          margin-right: 6px;
          margin-bottom: 6px;
        }
        .so-kpi {
          font-size: 26px;
          font-weight: 700;
          color: var(--brand-deep);
        }
        .so-kpi-label {
          font-size: 12px;
          text-transform: uppercase;
          letter-spacing: 0.1em;
          color: var(--muted);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(title: str, run_id: str | None, subtitle: str | None = None) -> None:
    apply_branding()
    subtitle_text = subtitle or "Strategic intelligence output"
    run_line = f"Run ID: {run_id}" if run_id else "Run ID: pending"
    st.markdown(
        f"""
        <div class="so-hero">
          <div>
            <div class="so-eyebrow">ScenarioOps</div>
            <div class="so-title">{title}</div>
            <div class="so-subtitle">{subtitle_text}</div>
          </div>
          <div class="so-run">{run_line}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    status = load_latest_status(run_id)
    if status and status.get("status") != "OK":
        error = status.get("error_summary")
        if error:
            st.error(f"Latest run failed: {error}")


def placeholder_section(title: str, lines: list[str]) -> None:
    st.markdown(
        f"""
        <div class="so-section">
          <h3>{title}</h3>
          <p>{' | '.join(lines)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_row(metrics: dict[str, Any]) -> None:
    cols = st.columns(len(metrics)) if metrics else []
    for col, (label, value) in zip(cols, metrics.items()):
        with col:
            st.metric(label, value)


def section(title: str, body: str | None = None) -> None:
    safe_title = html.escape(title)
    safe_body = html.escape(body) if isinstance(body, str) else ""
    body_html = f"<p>{safe_body}</p>" if safe_body else ""
    st.markdown(
        f"""
        <div class="so-section">
          <h3>{safe_title}</h3>
          {body_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def pill_row(items: Iterable[str]) -> None:
    pills = "".join(
        f"<span class='so-pill'>{html.escape(item)}</span>" for item in items if item
    )
    st.markdown(f"<div>{pills}</div>", unsafe_allow_html=True)


def card_grid(cards: Iterable[tuple[str, str]]) -> None:
    chunks = "".join(
        f"<div class='so-card'><h4>{html.escape(str(title))}</h4><p>{html.escape(str(body))}</p></div>"
        for title, body in cards
    )
    st.markdown(f"<div class='so-grid'>{chunks}</div>", unsafe_allow_html=True)
