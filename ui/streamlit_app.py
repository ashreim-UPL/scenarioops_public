from __future__ import annotations

import json
from datetime import datetime, timezone
import re
import time
import subprocess
import sys
import os
from pathlib import Path
from typing import Any
import queue
import threading
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

def _find_repo_root(start: Path) -> Path:
    current = start.resolve()
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return start.resolve().parent

ROOT = _find_repo_root(Path(__file__).resolve())
SRC_DIR = ROOT / "src"

# Add src to sys.path so we can import scenarioops
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from scenarioops.app.config import (
    ALLOWED_EMBED_MODELS,
    ALLOWED_IMAGE_MODELS,
    ALLOWED_TEXT_MODELS,
    DEFAULT_EMBED_MODEL,
    DEFAULT_IMAGE_MODEL,
    DEFAULT_LLM_MODEL,
    DEFAULT_SEARCH_MODEL,
    DEFAULT_SUMMARIZER_MODEL,
)
from scenarioops.graph.tools.view_model import build_view_model

RUNS_DIR = ROOT / "storage" / "runs"
LATEST_POINTER = RUNS_DIR / "latest.json"

st.set_page_config(
    page_title="ScenarioOps",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
<style>
.pipeline-row {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin: 6px 0 4px;
}
.pipeline-box {
  padding: 6px 10px;
  border-radius: 6px;
  border: 1px solid #d0d7de;
  background: #ffffff;
  font-size: 0.85rem;
  font-weight: 600;
  color: #111827;
}
.pipeline-box.running {
  background: #dbeafe;
  border-color: #93c5fd;
}
.pipeline-box.completed {
  background: #dcfce7;
  border-color: #86efac;
}
.pipeline-box.failed {
  background: #fee2e2;
  border-color: #fca5a5;
}
.pipeline-box.skipped {
  background: #f3f4f6;
  border-color: #e5e7eb;
  color: #6b7280;
}
.pipeline-label {
  margin: 4px 0 8px;
  font-size: 0.85rem;
  color: #4b5563;
}
</style>
""",
    unsafe_allow_html=True,
)

def run_cli_async(args: list[str]):
    """Runs CLI command as a subprocess, returning the Popen object."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC_DIR)
    return subprocess.Popen(
        [sys.executable, "-m", "scenarioops", *args],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,  # Line buffered
        env=env
    )

def _start_log_threads(process: subprocess.Popen) -> None:
    if "log_queue" not in st.session_state:
        st.session_state["log_queue"] = queue.Queue()
    if st.session_state.get("log_threads_started"):
        return
    q = st.session_state["log_queue"]

    def reader(stream, name: str) -> None:
        for line in iter(stream.readline, ""):
            if not line:
                break
            q.put((name, line))

    stdout_thread = threading.Thread(
        target=reader, args=(process.stdout, "stdout"), daemon=True
    )
    stderr_thread = threading.Thread(
        target=reader, args=(process.stderr, "stderr"), daemon=True
    )
    stdout_thread.start()
    stderr_thread.start()
    st.session_state["log_threads_started"] = True
    st.session_state["log_threads"] = [stdout_thread, stderr_thread]


def _drain_log_queue() -> None:
    q = st.session_state.get("log_queue")
    if not q:
        return
    if "logs" not in st.session_state:
        st.session_state["logs"] = ""
    if "stderr" not in st.session_state:
        st.session_state["stderr"] = ""
    while True:
        try:
            stream, line = q.get_nowait()
        except queue.Empty:
            break
        if stream == "stderr":
            st.session_state["stderr"] += line
        else:
            st.session_state["logs"] += line

def load_latest_run_id() -> str | None:
    if LATEST_POINTER.exists():
        try:
            return json.loads(LATEST_POINTER.read_text()).get("run_id")
        except:
            pass
    if RUNS_DIR.exists():
        runs = sorted([p for p in RUNS_DIR.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
        if runs:
            return runs[0].name
    return None

def get_run_status(run_id: str) -> dict[str, Any]:
    """Reads latest status and node events."""
    status = {"state": "UNKNOWN", "nodes": [], "run_config": None}
    
    # Check latest.json
    if LATEST_POINTER.exists():
        try:
            latest = json.loads(LATEST_POINTER.read_text())
            if latest.get("run_id") == run_id:
                status["state"] = latest.get("status", "UNKNOWN")
                status["error"] = latest.get("error_summary")
                status["run_config"] = latest.get("run_config")
        except:
            pass

    # Read node events
    log_path = RUNS_DIR / run_id / "logs" / "node_events.jsonl"
    if log_path.exists():
        nodes = []
        try:
            for line in log_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    nodes.append(json.loads(line))
            status["nodes"] = nodes
        except:
            pass
    return status

LEGACY_STEP_ORDER = [
    ("charter", "Charter"),
    ("focal_issue", "Focal Issue"),
    ("retrieval", "Retrieval"),
    ("scan_pestel", "Scan (PESTEL)"),
    ("drivers", "Drivers"),
    ("uncertainties", "Uncertainties"),
    ("logic", "Logic"),
    ("skeletons", "Skeletons"),
    ("narratives", "Narratives"),
    ("strategies", "Strategies"),
    ("wind_tunnel", "Wind Tunnel"),
    ("auditor", "Auditor"),
]

PRO_STEP_ORDER = [
    ("charter", "Charter"),
    ("focal_issue", "Focal Issue"),
    ("company_profile", "Company Profile"),
    ("ingest_docs", "Uploads"),
    ("retrieval_real", "Retrieval"),
    ("forces", "Forces"),
    ("ebe_rank", "EBE Rank"),
    ("clusters", "Clusters"),
    ("uncertainty_axes", "Uncertainty Axes"),
    ("scenarios", "Scenarios"),
    ("scenario_media", "Scenario Media"),
    ("strategies", "Strategies"),
    ("wind_tunnel", "Wind Tunnel"),
    ("auditor", "Auditor"),
]


def _format_list(values: Any) -> str:
    if not values:
        return "none"
    if isinstance(values, list):
        return ", ".join(str(item) for item in values)
    return str(values)

NODE_FUNCTIONS = {
    "charter": "run_charter_node",
    "focal_issue": "run_focal_issue_node",
    "company_profile": "run_company_profile_node",
    "ingest_docs": "run_ingest_docs_node",
    "retrieval_real": "run_retrieval_real_node",
    "forces": "run_force_builder_node",
    "ebe_rank": "run_ebe_rank_node",
    "clusters": "run_cluster_node",
    "uncertainty_axes": "run_uncertainty_axes_node",
    "scenarios": "run_scenario_synthesis_node",
    "scenario_media": "run_scenario_media_node",
    "strategies": "run_strategies_node",
    "wind_tunnel": "run_wind_tunnel_node",
    "auditor": "run_auditor_node",
    "retrieval": "run_retrieval_node",
    "scan_pestel": "run_scan_node",
    "drivers": "run_drivers_node",
    "uncertainties": "run_uncertainties_node",
    "logic": "run_logic_node",
    "skeletons": "run_skeletons_node",
    "narratives": "run_narratives_node",
}

COUNTRY_OPTIONS = [
    "United States",
    "United Kingdom",
    "Canada",
    "Germany",
    "France",
    "Italy",
    "Spain",
    "Netherlands",
    "Sweden",
    "Norway",
    "Switzerland",
    "India",
    "China",
    "Japan",
    "South Korea",
    "Singapore",
    "Australia",
    "Brazil",
    "Mexico",
    "South Africa",
    "Saudi Arabia",
    "United Arab Emirates",
    "Qatar",
    "Kuwait",
    "Egypt",
    "Nigeria",
]

REGION_OPTIONS = [
    "GCC",
    "MENA",
    "North Africa",
    "Sub-Saharan Africa",
    "Europe",
    "North America",
    "Latin America",
    "Southeast Asia",
    "South Asia",
    "East Asia",
    "Central Asia",
    "Oceania",
]

COMPANY_GEO_HINTS = {
    "microsoft": ("country", "United States"),
    "apple": ("country", "United States"),
    "google": ("country", "United States"),
    "alphabet": ("country", "United States"),
    "amazon": ("country", "United States"),
    "meta": ("country", "United States"),
    "tesla": ("country", "United States"),
    "samsung": ("country", "South Korea"),
    "tencent": ("country", "China"),
    "alibaba": ("country", "China"),
    "toyota": ("country", "Japan"),
    "siemens": ("country", "Germany"),
    "sap": ("country", "Germany"),
    "shell": ("region", "Europe"),
    "bp": ("region", "Europe"),
    "aramco": ("country", "Saudi Arabia"),
}


def _normalize_label(value: str) -> str:
    return " ".join(value.lower().strip().split())


def _safe_filename(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    return cleaned.strip("_") or "document"


def _geo_select(label: str, options: list[str], key_prefix: str) -> str:
    choice = st.selectbox(label, options + ["Custom..."], key=f"{key_prefix}_select")
    if choice == "Custom...":
        custom = st.text_input(f"{label} (custom)", key=f"{key_prefix}_custom")
        return custom.strip()
    return choice


def _infer_scope_from_company(company_name: str) -> tuple[str | None, str | None]:
    normalized = _normalize_label(company_name)
    if not normalized:
        return None, None
    if normalized in COMPANY_GEO_HINTS:
        return COMPANY_GEO_HINTS[normalized]
    countries = {_normalize_label(item) for item in COUNTRY_OPTIONS}
    regions = {_normalize_label(item) for item in REGION_OPTIONS}
    if normalized in countries:
        return "country", company_name.strip()
    if normalized in regions:
        return "region", company_name.strip()
    return None, None


def _parse_iso(ts: str) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    total = int(round(seconds))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:d}:{secs:02d}"


def _run_start_time(run_id: str | None, nodes: list[dict[str, Any]]) -> datetime | None:
    if run_id:
        config_path = RUNS_DIR / run_id / "run_config.json"
        if config_path.exists():
            try:
                run_config = json.loads(config_path.read_text(encoding="utf-8"))
            except Exception:
                run_config = None
            if isinstance(run_config, dict):
                created_at = _parse_iso(str(run_config.get("created_at", "")))
                if created_at:
                    return created_at
    timestamps = []
    for entry in nodes:
        ts = _parse_iso(str(entry.get("timestamp", "")))
        if ts:
            timestamps.append(ts)
    return min(timestamps) if timestamps else None


def _last_event_time(nodes: list[dict[str, Any]]) -> datetime | None:
    timestamps = []
    for entry in nodes:
        ts = _parse_iso(str(entry.get("timestamp", "")))
        if ts:
            timestamps.append(ts)
    return max(timestamps) if timestamps else None


def _load_jsonl_log(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return []
    for line in lines:
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if isinstance(payload, dict):
            entries.append(payload)
    if limit and len(entries) > limit:
        return entries[-limit:]
    return entries


def _latest_log_time(entries: list[dict[str, Any]]) -> datetime | None:
    for entry in reversed(entries):
        ts = _parse_iso(str(entry.get("timestamp", "")))
        if ts:
            return ts
    return None

def _load_artifact_json(run_id: str | None, name: str) -> dict[str, Any] | None:
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


def _build_step_statuses(
    nodes: list[dict[str, Any]],
    process_running: bool,
    step_order: list[tuple[str, str]],
) -> list[dict[str, Any]]:
    last_by_node: dict[str, dict[str, Any]] = {}
    for entry in nodes:
        name = str(entry.get("node", ""))
        if name:
            last_by_node[name] = entry

    order_names = [name for name, _ in step_order]
    failure_node = None
    for entry in nodes:
        if entry.get("status") == "FAIL":
            failure_node = str(entry.get("node", ""))
            break

    last_node = nodes[-1].get("node") if nodes else None
    statuses: list[dict[str, Any]] = []
    failed_seen = False
    for name, label in step_order:
        event = last_by_node.get(name)
        status = "PENDING"
        if event:
            status = str(event.get("status") or "UNKNOWN")
        if failure_node:
            if failed_seen and not event:
                status = "SKIPPED"
            if name == failure_node:
                failed_seen = True
        statuses.append(
            {
                "name": name,
                "label": label,
                "status": status,
                "event": event,
            }
        )

    if process_running and not failure_node:
        if last_node in order_names:
            idx = order_names.index(last_node)
            if idx + 1 < len(order_names):
                next_name = order_names[idx + 1]
                for item in statuses:
                    if item["name"] == next_name and item["status"] == "PENDING":
                        item["status"] = "RUNNING"
                        break
        elif statuses:
            statuses[0]["status"] = "RUNNING"

    return statuses


def _select_step_order(nodes: list[dict[str, Any]], run_id: str | None) -> list[tuple[str, str]]:
    node_names = {str(entry.get("node", "")) for entry in nodes}
    pro_only = {name for name, _ in PRO_STEP_ORDER} - {name for name, _ in LEGACY_STEP_ORDER}
    legacy_only = {name for name, _ in LEGACY_STEP_ORDER} - {name for name, _ in PRO_STEP_ORDER}
    if node_names & pro_only:
        return PRO_STEP_ORDER
    if node_names & legacy_only:
        return LEGACY_STEP_ORDER
    if run_id:
        config_path = RUNS_DIR / run_id / "run_config.json"
        if config_path.exists():
            try:
                run_config = json.loads(config_path.read_text(encoding="utf-8"))
            except Exception:
                run_config = None
            if isinstance(run_config, dict) and run_config.get("legacy_mode") is True:
                return LEGACY_STEP_ORDER
    mode = st.session_state.get("pipeline_mode")
    if mode == "legacy":
        return LEGACY_STEP_ORDER
    return PRO_STEP_ORDER


def _current_step_label(statuses: list[dict[str, Any]]) -> str | None:
    for item in statuses:
        if item["status"] == "RUNNING":
            function = NODE_FUNCTIONS.get(item["name"], "unknown")
            return f"{item['name']} ({item['label']}) -> {function}"
    for item in statuses:
        if item["status"] == "FAIL":
            function = NODE_FUNCTIONS.get(item["name"], "unknown")
            return f"FAILED: {item['name']} ({item['label']}) -> {function}"
    completed = [
        item for item in statuses if item["status"] in {"OK", "HYDRATED"}
    ]
    if completed:
        last = completed[-1]
        function = NODE_FUNCTIONS.get(last["name"], "unknown")
        return f"Last completed: {last['name']} ({last['label']}) -> {function}"
    return None


def _running_step(statuses: list[dict[str, Any]]) -> dict[str, Any] | None:
    for item in statuses:
        if item["status"] == "RUNNING":
            return item
    return None


def render_pipeline_boxes(
    *,
    nodes: list[dict[str, Any]],
    run_id: str | None,
    process_running: bool,
) -> None:
    step_order = _select_step_order(nodes, run_id)
    statuses = _build_step_statuses(nodes, process_running, step_order)
    parts: list[str] = []
    for item in statuses:
        status = item["status"]
        if status in {"OK", "HYDRATED"}:
            cls = "completed"
        elif status == "RUNNING":
            cls = "running"
        elif status == "FAIL":
            cls = "failed"
        elif status == "SKIPPED":
            cls = "skipped"
        else:
            cls = "pending"
        parts.append(f'<div class="pipeline-box {cls}">{item["label"]}</div>')
    st.markdown(
        f'<div class="pipeline-row">{"".join(parts)}</div>',
        unsafe_allow_html=True,
    )
    current = _current_step_label(statuses)
    if current:
        st.markdown(
            f'<div class="pipeline-label">Current step: {current}</div>',
            unsafe_allow_html=True,
        )


def render_run_timing(
    *,
    run_id: str | None,
    nodes: list[dict[str, Any]],
    process_running: bool,
) -> None:
    start_time = _run_start_time(run_id, nodes)
    if not start_time:
        return
    now = datetime.now(timezone.utc)
    elapsed = (now - start_time).total_seconds()
    st.markdown(
        f'<div class="pipeline-label">Run elapsed: {_format_duration(elapsed)}</div>',
        unsafe_allow_html=True,
    )
    last_event = _last_event_time(nodes)
    if last_event:
        idle = (now - last_event).total_seconds()
        st.markdown(
            f'<div class="pipeline-label">Last step completed {_format_duration(idle)} ago.</div>',
            unsafe_allow_html=True,
        )
        if process_running and idle >= 60:
            st.warning(
                f"No new step completed for {_format_duration(idle)}. "
                "The current step may be waiting or rate-limited."
            )
    if process_running:
        running_for = None
        if last_event:
            running_for = (now - last_event).total_seconds()
        else:
            running_for = (now - start_time).total_seconds()
        st.markdown(
            f'<div class="pipeline-label">Current step running for {_format_duration(running_for)}.</div>',
            unsafe_allow_html=True,
        )


def render_context_snapshot(run_id: str | None) -> None:
    charter = _load_artifact_json(run_id, "scenario_charter")
    focal_issue = _load_artifact_json(run_id, "focal_issue")
    company_profile = _load_artifact_json(run_id, "company_profile")
    if not any([charter, focal_issue, company_profile]):
        return
    with st.expander("Context Snapshot", expanded=True):
        if charter:
            st.markdown("**Charter**")
            st.write(
                {
                    "title": charter.get("title"),
                    "purpose": charter.get("purpose"),
                    "decision_context": charter.get("decision_context"),
                    "time_horizon": charter.get("time_horizon"),
                }
            )
            assumptions = charter.get("assumptions", [])
            if assumptions:
                st.markdown("Assumptions")
                st.write(assumptions)
            constraints = charter.get("constraints", [])
            if constraints:
                st.markdown("Constraints")
                st.write(constraints)
        if focal_issue:
            st.markdown("**Focal Issue**")
            st.write(
                {
                    "focal_issue": focal_issue.get("focal_issue"),
                    "decision_type": focal_issue.get("decision_type"),
                    "scope": focal_issue.get("scope"),
                }
            )
        if company_profile:
            st.markdown("**Company Profile**")
            st.write(
                {
                    "company_name": company_profile.get("company_name"),
                    "geography": company_profile.get("geography"),
                    "horizon_months": company_profile.get("horizon_months"),
                    "source_basis": company_profile.get("source_basis"),
                    "simulated": company_profile.get("simulated"),
                }
            )


def render_retrieval_activity(
    run_id: str | None,
    *,
    process_running: bool,
    statuses: list[dict[str, Any]],
) -> None:
    if not run_id:
        return
    search_path = RUNS_DIR / run_id / "logs" / "search.log"
    retriever_path = RUNS_DIR / run_id / "logs" / "retriever.log"
    search_entries = _load_jsonl_log(search_path, limit=200)
    retriever_entries = _load_jsonl_log(retriever_path, limit=200)
    retrieval_status = next(
        (item for item in statuses if item["name"] in {"retrieval_real", "retrieval"}),
        None,
    )
    if not retrieval_status and not search_entries and not retriever_entries:
        return
    if not search_entries and not retriever_entries and not process_running:
        return

    def _count_by_status(entries: list[dict[str, Any]]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for entry in entries:
            status = str(entry.get("status", "unknown"))
            counts[status] = counts.get(status, 0) + 1
        return counts

    now = datetime.now(timezone.utc)
    last_search_time = _latest_log_time(search_entries)
    last_retrieval_time = _latest_log_time(retriever_entries)
    last_activity = None
    for ts in [last_search_time, last_retrieval_time]:
        if ts and (last_activity is None or ts > last_activity):
            last_activity = ts

    with st.expander("Retrieval Activity", expanded=bool(process_running)):
        if last_activity:
            idle = (now - last_activity).total_seconds()
            st.caption(f"Last retrieval activity {_format_duration(idle)} ago.")
            if process_running and idle >= 60:
                st.warning(
                    f"No retrieval activity for {_format_duration(idle)}. "
                    "The retriever may be waiting or rate-limited."
                )
        cols = st.columns(2)
        with cols[0]:
            st.markdown("**Search**")
            if search_entries:
                counts = _count_by_status(search_entries)
                st.write(
                    {
                        "queries": len(search_entries),
                        "ok": counts.get("ok", 0),
                        "error": counts.get("error", 0),
                    }
                )
                last_entry = search_entries[-1]
                st.caption(f"Last query: {last_entry.get('query', '')}")
            else:
                st.caption("No search activity yet.")
        with cols[1]:
            st.markdown("**Retriever**")
            if retriever_entries:
                counts = _count_by_status(retriever_entries)
                st.write(
                    {
                        "fetches": len(retriever_entries),
                        "ok": counts.get("ok", 0),
                        "cache_hit": counts.get("cache_hit", 0),
                        "error": counts.get("error", 0),
                        "blocked": counts.get("blocked", 0),
                    }
                )
                last_entry = retriever_entries[-1]
                st.caption(f"Last URL: {last_entry.get('url', '')}")
            else:
                st.caption("No retrieval activity yet.")

        if search_entries:
            st.markdown("**Recent search queries**")
            df = pd.DataFrame(search_entries[-5:])
            display_cols = [
                col
                for col in ["timestamp", "status", "query", "result_count", "detail"]
                if col in df.columns
            ]
            if display_cols:
                st.dataframe(df[display_cols], width="stretch", height=160)
        if retriever_entries:
            st.markdown("**Recent retrievals**")
            df = pd.DataFrame(retriever_entries[-5:])
            display_cols = [
                col
                for col in ["timestamp", "status", "url", "detail"]
                if col in df.columns
            ]
            if display_cols:
                st.dataframe(df[display_cols], width="stretch", height=160)

def render_step_panel(
    container: Any,
    nodes: list[dict[str, Any]],
    *,
    process_running: bool,
) -> None:
    run_id = st.session_state.get("run_id") or load_latest_run_id()
    run_start = _run_start_time(run_id, nodes)
    step_order = _select_step_order(nodes, run_id)
    statuses = _build_step_statuses(nodes, process_running, step_order)
    with container:
        st.subheader("Run Steps")
        for item in statuses:
            label = f"[{item['status']}] {item['label']}"
            expanded = item["status"] in {"FAIL", "RUNNING"}
            with st.expander(label, expanded=expanded):
                event = item.get("event")
                if not event:
                    st.caption("No data yet.")
                    continue
                st.write(f"timestamp: {event.get('timestamp', '')}")
                st.write(f"duration_seconds: {event.get('duration_seconds', '')}")
                if run_start:
                    timestamp = _parse_iso(str(event.get("timestamp", "")))
                    if timestamp:
                        elapsed = (timestamp - run_start).total_seconds()
                        st.write(f"elapsed_since_start: {_format_duration(elapsed)}")
                st.write(f"tools: {_format_list(event.get('tools'))}")
                st.write(f"inputs: {_format_list(event.get('inputs'))}")
                st.write(f"outputs: {_format_list(event.get('outputs'))}")
                if event.get("error"):
                    st.error(str(event.get("error")))

# --- Sidebar Controls ---
with st.sidebar:
    st.title("🔮 ScenarioOps")
    st.markdown("Dynamic Strategy Squad")
    
    with st.expander("Configuration", expanded=True):
        mode = st.selectbox("Mode", ["live", "demo"], index=1)
        
        # Defaults based on mode
        default_web = True if mode == "live" else False
        allow_web_choice = st.checkbox("Allow web retrieval", value=default_web)

        simulate_evidence = st.checkbox(
            "Simulate evidence (demo only)",
            value=False,
            help="Only use when running offline demos. Real evidence is required otherwise.",
        )
        legacy_mode = st.checkbox(
            "Legacy mode",
            value=False,
            help="Use pre-upgrade pipeline for comparison or fallback.",
        )
        generate_strategies = st.checkbox(
            "Generate strategies + wind tunnel",
            value=True,
        )
        seed = st.number_input("Seed (optional)", min_value=0, value=0, step=1)
        min_evidence_ok = st.number_input(
            "Min evidence (ok)",
            min_value=0,
            value=10,
            step=1,
        )
        min_evidence_total = st.number_input(
            "Min evidence (total)",
            min_value=0,
            value=15,
            step=1,
        )
        max_failed_ratio = st.number_input(
            "Max failed ratio",
            min_value=0.0,
            max_value=1.0,
            value=0.4,
            step=0.05,
            format="%.2f",
        )

        text_models = sorted(ALLOWED_TEXT_MODELS)
        embed_models = sorted(ALLOWED_EMBED_MODELS)
        image_models = sorted(ALLOWED_IMAGE_MODELS)

        def _model_index(options: list[str], default: str) -> int:
            return options.index(default) if default in options else 0

        llm_model = st.selectbox(
            "LLM model",
            text_models,
            index=_model_index(text_models, DEFAULT_LLM_MODEL),
        )
        search_model = st.selectbox(
            "Search model",
            text_models,
            index=_model_index(text_models, DEFAULT_SEARCH_MODEL),
        )
        summarizer_model = st.selectbox(
            "Summarizer model",
            text_models,
            index=_model_index(text_models, DEFAULT_SUMMARIZER_MODEL),
        )
        embed_model = st.selectbox(
            "Embedding model",
            embed_models,
            index=_model_index(embed_models, DEFAULT_EMBED_MODEL),
        )
        image_model = st.selectbox(
            "Image model",
            image_models,
            index=_model_index(image_models, DEFAULT_IMAGE_MODEL),
        )

        upload_files = st.file_uploader(
            "Upload documents",
            type=["pdf", "docx", "txt", "csv"],
            accept_multiple_files=True,
            help="Documents are ingested before retrieval.",
        )
        
        planning_target = st.selectbox(
            "Planning Target",
            ["Company", "Geography"],
            index=0,
            help="Choose Company to keep organization inputs separate from geography.",
        )
        company_label = "Company" if planning_target == "Company" else "Company (optional)"
        default_company = "Microsoft" if mode == "live" else "Acme Corp"
        company_name = st.text_input(company_label, default_company).strip()

        scope_choice = st.selectbox(
            "Geographic Scope",
            ["auto", "country", "region", "world"],
            index=0,
        )
        geography = ""
        effective_scope = scope_choice
        if scope_choice == "country":
            geography = _geo_select("Country", COUNTRY_OPTIONS, "geo_country")
        elif scope_choice == "region":
            geography = _geo_select("Region", REGION_OPTIONS, "geo_region")
        elif scope_choice == "world":
            geography = "Global"
        else:
            inferred_scope, inferred_geo = _infer_scope_from_company(company_name)
            if inferred_scope and inferred_geo:
                effective_scope = inferred_scope
                geography = inferred_geo
                st.info(f"Auto scope detected: {inferred_scope} ({inferred_geo})")
            else:
                effective_scope = "world"
                geography = "Global"
                st.caption("Auto scope defaulted to world. Select a scope to limit geography.")

        horizon = st.slider("Horizon (Months)", 6, 60, 12)
        value = company_name or geography or "Unknown"
        can_launch = True
        if planning_target == "Company" and not company_name:
            st.error("Company name is required for company-focused planning.")
            can_launch = False
        if company_name and geography and _normalize_label(company_name) == _normalize_label(geography):
            st.error("Company name must differ from geography.")
            can_launch = False

        resume_mode = st.checkbox(
            "Resume last run",
            value=False,
            help="Resume from a specific node using existing artifacts.",
        )
        resume_from = None
        resume_run_id = None
        if resume_mode:
            resume_choices = [label for _, label in PRO_STEP_ORDER]
            resume_default = "Forces" if "Forces" in resume_choices else resume_choices[0]
            resume_label = st.selectbox(
                "Resume from node",
                resume_choices,
                index=resume_choices.index(resume_default),
            )
            resume_from = next(
                name for name, label in PRO_STEP_ORDER if label == resume_label
            )
            resume_run_id = st.text_input(
                "Resume run id",
                value=st.session_state.get("last_run_id") or load_latest_run_id() or "",
            ).strip()
            if not resume_run_id:
                st.error("Resume run id is required.")
                can_launch = False
            else:
                run_path = RUNS_DIR / resume_run_id
                if not run_path.exists():
                    st.error(f"Run id not found: {resume_run_id}")
                    can_launch = False
        
    st.divider()
    
    if st.button("Launch Exploration", width="stretch", disabled=not can_launch):
        st.session_state["running"] = True
        run_id = (
            resume_run_id
            if resume_mode
            else datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        )
        st.session_state["run_id"] = run_id
        st.session_state["last_run_id"] = run_id
        st.session_state["logs"] = ""
        st.session_state["stderr"] = ""
        st.session_state["log_queue"] = queue.Queue()
        st.session_state["log_threads_started"] = False
        st.session_state["pipeline_mode"] = "legacy" if legacy_mode else "pro"

        upload_paths: list[str] = []
        if upload_files:
            uploads_dir = RUNS_DIR / run_id / "inputs"
            uploads_dir.mkdir(parents=True, exist_ok=True)
            for upload in upload_files:
                safe_name = _safe_filename(upload.name)
                target_path = uploads_dir / safe_name
                counter = 1
                while target_path.exists():
                    target_path = uploads_dir / f"{target_path.stem}-{counter}{target_path.suffix}"
                    counter += 1
                target_path.write_bytes(upload.getbuffer())
                upload_paths.append(str(target_path))
        
        args = [
            "build-scenarios",
            "--run-id", run_id,
            "--scope", effective_scope,
            "--value", value,
            "--horizon", str(horizon),
            "--mode", mode,
        ]
        if resume_mode and resume_from:
            args.extend(["--resume-from", resume_from])
        if company_name:
            args.extend(["--company", company_name])
        if geography:
            args.extend(["--geography", geography])

        if legacy_mode:
            args.append("--legacy-mode")
        if simulate_evidence:
            args.append("--simulate-evidence")
        if not generate_strategies:
            args.append("--no-strategies")
        if seed:
            args.extend(["--seed", str(seed)])
        args.extend(["--min-evidence-ok", str(min_evidence_ok)])
        args.extend(["--min-evidence-total", str(min_evidence_total)])
        args.extend(["--max-failed-ratio", str(max_failed_ratio)])
        if llm_model:
            args.extend(["--llm-model", llm_model])
        if search_model:
            args.extend(["--search-model", search_model])
        if summarizer_model:
            args.extend(["--summarizer-model", summarizer_model])
        if embed_model:
            args.extend(["--embed-model", embed_model])
        if image_model:
            args.extend(["--image-model", image_model])
        if upload_paths:
            args.append("--input-docs")
            args.extend(upload_paths)
        
        # Logic: If live, force allow-web. If demo, allow toggle.
        # Also ensure fixtures policy for demo if needed, or academic/mixed for live.
        
        if mode == "live":
            args.append("--allow-web")
            # Default to mixed_reputable for live if not set, or let config decide.
            # Using academic_only as sensible default or respecting config.
        else:
            if allow_web_choice:
                args.append("--allow-web")
            else:
                args.append("--no-allow-web")
            args.extend(["--sources-policy", "fixtures"])

        # Start process
        process = run_cli_async(args)
        st.session_state["process"] = process
        st.session_state["command"] = " ".join(args)

# --- Main Area ---

if st.session_state.get("running"):
    st.subheader("?? Operation in Progress")

    main_col, side_col = st.columns([3, 1], gap="large")
    process = st.session_state.get("process")
    status_data: dict[str, Any] = {"nodes": []}
    process_running = False
    should_rerun = False
    should_clear_running = False

    with main_col:
        status_container = st.container()
        log_container = st.empty()

        if process:
            _start_log_threads(process)
            _drain_log_queue()

            # Check for new run ID if not set
            if not st.session_state.get("run_id"):
                rid = load_latest_run_id()
                if rid:
                    st.session_state["run_id"] = rid

            run_id = st.session_state.get("run_id")
            status_data = get_run_status(run_id) if run_id else {"nodes": []}
            process_running = process.poll() is None

            with status_container:
                nodes = status_data.get("nodes", [])
                step_order = _select_step_order(nodes, run_id)
                statuses = _build_step_statuses(nodes, process_running, step_order)
                render_pipeline_boxes(
                    nodes=nodes,
                    run_id=run_id,
                    process_running=process_running,
                )
                render_run_timing(
                    run_id=run_id,
                    nodes=nodes,
                    process_running=process_running,
                )
                render_context_snapshot(run_id)
                render_retrieval_activity(
                    run_id,
                    process_running=process_running,
                    statuses=statuses,
                )
                current_label = _current_step_label(statuses)
                if current_label:
                    st.info(f"Current step: {current_label}")
                elif nodes:
                    last_node = nodes[-1].get("node", "unknown")
                    st.info(f"Last event: `{last_node}`")

                    # Progress Bar based on known nodes count (approx 15)
                    progress = min(len(nodes) / 15.0, 0.95)
                    st.progress(progress)

                    # Active Log Table
                    with st.expander("Live Execution Log", expanded=True):
                        log_df = pd.DataFrame(nodes)
                        if not log_df.empty:
                            if "tools" in log_df.columns:
                                log_df["tools"] = log_df["tools"].apply(
                                    lambda tools: ", ".join(tools)
                                    if isinstance(tools, list)
                                    else (tools or "")
                                )
                            display_cols = [
                                "timestamp",
                                "node",
                                "status",
                                "duration_seconds",
                                "tools",
                                "error",
                            ]
                            display_df = log_df.reindex(columns=display_cols)
                            if "timestamp" in display_df.columns:
                                display_df = display_df.sort_values(
                                    by="timestamp", ascending=False
                                )
                            st.dataframe(
                                display_df,
                                width='stretch',
                                height=220,
                            )
                else:
                    st.info("Initializing Squad...")
                    st.spinner("Spinning up agents...")
                if status_data.get("error"):
                    st.error(status_data["error"])

            if st.session_state.get("logs"):
                log_container.code(st.session_state["logs"][-2000:], language="json")
            if st.session_state.get("stderr"):
                with st.expander("stderr", expanded=False):
                    st.code(st.session_state["stderr"][-2000:], language="text")

            cmd = st.session_state.get("command", "")
            proc_status = "running" if process_running else f"exit {process.returncode}"
            st.caption(f"Process PID: {process.pid} | Status: {proc_status}")
            if cmd:
                st.caption(f"Command: {cmd}")

            if process_running:
                should_rerun = True
            else:
                if process.returncode == 0:
                    st.success("Exploration Complete!")
                else:
                    st.error("Exploration Failed.")
                    if st.session_state.get("stderr"):
                        st.code(st.session_state["stderr"][-2000:], language="text")
                st.session_state["running"] = False
                should_clear_running = True
        else:
            st.error("No process found. Please relaunch the exploration.")

    render_step_panel(side_col, status_data.get("nodes", []), process_running=process_running)
    if should_rerun:
        time.sleep(0.2)
        st.rerun()
    if should_clear_running:
        st.rerun()

# --- Dashboard View (Post-Run) ---
run_id = st.session_state.get("last_run_id") or load_latest_run_id()
if run_id and not st.session_state.get("running"):
    main_col, side_col = st.columns([3, 1], gap="large")
    status_data = get_run_status(run_id)

    with main_col:
        st.header(f"Strategy Dashboard: {run_id}")
        if LATEST_POINTER.exists():
            try:
                latest = json.loads(LATEST_POINTER.read_text())
            except Exception:
                latest = None
            if isinstance(latest, dict) and latest.get("run_id") == run_id:
                if latest.get("status") != "OK" and latest.get("error_summary"):
                    st.error(f"Latest run failed: {latest.get('error_summary')}")

        render_pipeline_boxes(
            nodes=status_data.get("nodes", []),
            run_id=run_id,
            process_running=False,
        )
        render_run_timing(
            run_id=run_id,
            nodes=status_data.get("nodes", []),
            process_running=False,
        )
        render_context_snapshot(run_id)

        # Load Data
        run_dir = RUNS_DIR / run_id
        try:
            view_model_path = run_dir / "artifacts" / "view_model.json"
            if view_model_path.exists():
                view_model = json.loads(view_model_path.read_text(encoding="utf-8"))
            else:
                view_model = build_view_model(run_dir)
        except Exception as e:
            st.error(f"Could not load view model: {e}")
            st.stop()

        # Metrics Row
        m1, m2, m3, m4 = st.columns(4)
        forces_count = len(view_model.get("forces") or view_model.get("driving_forces", []))
        axes_count = len(view_model.get("uncertainty_axes", [])) or len(view_model.get("uncertainties", []))
        m1.metric("Forces Identified", forces_count)
        m2.metric("Uncertainty Axes", axes_count)
        m3.metric("Scenarios", len(view_model.get("scenarios", [])))
        m4.metric("Strategies", len(view_model.get("strategies", [])))

        # Main Tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Strategy Map", "Scenarios", "Wind Tunnel", "Raw Data"])

        with tab1:
            st.subheader("Strategic Radar")
            # Placeholder for Force Graph if we had pyvis
            evidence_units = view_model.get("evidence_units", [])
            forces = view_model.get("forces") or view_model.get("driving_forces", [])
            clusters = view_model.get("clusters", [])
            axes = view_model.get("uncertainty_axes", [])

            st.caption(f"Evidence Units: {len(evidence_units)}")
            if forces:
                df = pd.DataFrame(forces)
                if "domain" in df.columns:
                    fig = px.pie(df, names="domain", title="Forces by Domain (PESTEL)")
                    st.plotly_chart(fig, width="stretch")
                if "layer" in df.columns:
                    layer_counts = df["layer"].value_counts().reset_index()
                    layer_counts.columns = ["layer", "count"]
                    fig = px.bar(layer_counts, x="layer", y="count", title="Force Layer Distribution")
                    st.plotly_chart(fig, width="stretch")

                st.markdown("### Top Forces")
                for f in forces[:5]:
                    label = f.get("label") or f.get("name")
                    detail = f.get("mechanism") or f.get("description")
                    st.markdown(f"- **{label}**: {detail}")

            if clusters:
                st.markdown("### Cluster Summary")
                cluster_df = pd.DataFrame(clusters)
                display_cols = [col for col in ["cluster_id", "cluster_label", "coherence_score"] if col in cluster_df.columns]
                st.dataframe(cluster_df[display_cols], height=220)

            if axes:
                st.markdown("### Uncertainty Axes")
                axes_df = pd.DataFrame(axes)
                display_cols = [col for col in ["axis_id", "axis_name", "impact_score", "uncertainty_score"] if col in axes_df.columns]
                st.dataframe(axes_df[display_cols], height=220)

        with tab2:
            cols = st.columns(2)
            scenarios = view_model.get("scenarios", [])
            for i, scen in enumerate(scenarios):
                with cols[i % 2]:
                    with st.container(border=True):
                        st.markdown(f"### {scen.get('name')}")
                        story_text = scen.get("story_text") or scen.get("narrative", "")
                        if story_text:
                            st.write(story_text)
                        image_path = scen.get("image_artifact_path")
                        if image_path:
                            full_path = RUNS_DIR / run_id / Path(image_path)
                            if full_path.exists():
                                st.image(str(full_path), use_column_width=True)
                        signposts = scen.get("signposts", [])
                        if signposts:
                            st.markdown("**Signposts:**")
                            for sp in signposts[:3]:
                                st.markdown(f"- {sp}")
                        implications = scen.get("implications", [])
                        if implications:
                            st.markdown("**Implications:**")
                            for imp in implications[:3]:
                                st.markdown(f"- {imp}")

        with tab3:
            st.subheader("Wind Tunnel Results")
            wt = view_model.get("wind_tunnel") or {}
            if not isinstance(wt, dict):
                wt = {}
            tests = wt.get("tests", [])
            if tests:
                df_wt = pd.DataFrame(tests)
                fig = px.bar(
                    df_wt,
                    x="strategy_id",
                    y="feasibility_score",
                    color="outcome",
                    title="Strategy Feasibility",
                )
                st.plotly_chart(fig, width="stretch")
                st.dataframe(df_wt[["strategy_id", "scenario_id", "outcome", "action"]])
                matrix = wt.get("matrix", [])
                if matrix:
                    st.markdown("### Strategy x Scenario Matrix")
                    matrix_df = pd.DataFrame(matrix)
                    st.dataframe(matrix_df, height=240)
            else:
                st.info("No wind tunnel tests available.")

        with tab4:
            st.json(view_model)

    render_step_panel(side_col, status_data.get("nodes", []), process_running=False)
