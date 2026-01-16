from __future__ import annotations

import json
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

from scenarioops.graph.tools.view_model import build_view_model

RUNS_DIR = ROOT / "storage" / "runs"
LATEST_POINTER = RUNS_DIR / "latest.json"

st.set_page_config(
    page_title="ScenarioOps",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ... (CSS remains same) ...

def run_cli_async(args: list[str]):
    """Runs CLI command as a subprocess, returning the Popen object."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC_DIR)
    return subprocess.Popen(
        [sys.executable, "-m", "scenarioops.app.main", *args],
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

STEP_ORDER = [
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


def _format_list(values: Any) -> str:
    if not values:
        return "none"
    if isinstance(values, list):
        return ", ".join(str(item) for item in values)
    return str(values)


def _build_step_statuses(nodes: list[dict[str, Any]], process_running: bool) -> list[dict[str, Any]]:
    last_by_node: dict[str, dict[str, Any]] = {}
    for entry in nodes:
        name = str(entry.get("node", ""))
        if name:
            last_by_node[name] = entry

    order_names = [name for name, _ in STEP_ORDER]
    failure_node = None
    for entry in nodes:
        if entry.get("status") == "FAIL":
            failure_node = str(entry.get("node", ""))
            break

    last_node = nodes[-1].get("node") if nodes else None
    statuses: list[dict[str, Any]] = []
    failed_seen = False
    for name, label in STEP_ORDER:
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


def render_step_panel(
    container: Any,
    nodes: list[dict[str, Any]],
    *,
    process_running: bool,
) -> None:
    statuses = _build_step_statuses(nodes, process_running)
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
        
        scope = st.selectbox("Scope", ["country", "region", "world"], index=0)
        value = st.text_input("Entity Value", "NEOM" if mode == "live" else "UAE")
        horizon = st.slider("Horizon (Months)", 6, 60, 12)
        
    st.divider()
    
    if st.button("🚀 Launch Exploration", use_container_width=True):
        st.session_state["running"] = True
        st.session_state["run_id"] = None
        st.session_state["logs"] = ""
        st.session_state["stderr"] = ""
        st.session_state["log_queue"] = queue.Queue()
        st.session_state["log_threads_started"] = False
        
        args = [
            "build-scenarios",
            "--scope", scope,
            "--value", value,
            "--horizon", str(horizon),
            "--mode", mode
        ]
        
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
                if nodes:
                    current_node = nodes[-1].get("node", "unknown")
                    st.info(f"**? Executing:** `{current_node}`")

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
                                use_container_width=True,
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
                time.sleep(0.2)
                st.rerun()
            else:
                if process.returncode == 0:
                    st.success("Exploration Complete!")
                else:
                    st.error("Exploration Failed.")
                    if st.session_state.get("stderr"):
                        st.code(st.session_state["stderr"][-2000:], language="text")
                st.session_state["running"] = False
                st.rerun()
        else:
            st.error("No process found. Please relaunch the exploration.")

    render_step_panel(side_col, status_data.get("nodes", []), process_running=process_running)

# --- Dashboard View (Post-Run) ---
run_id = load_latest_run_id()
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
        m1.metric("Forces Identified", len(view_model.get("driving_forces", [])))
        m2.metric("Uncertainties", len(view_model.get("uncertainties", [])))
        m3.metric("Scenarios", len(view_model.get("scenarios", [])))
        m4.metric("Strategies", len(view_model.get("strategies", [])))

        # Main Tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Strategy Map", "Scenarios", "Wind Tunnel", "Raw Data"])

        with tab1:
            st.subheader("Strategic Radar")
            # Placeholder for Force Graph if we had pyvis
            forces = view_model.get("driving_forces", [])
            if forces:
                df = pd.DataFrame(forces)
                if "domain" in df.columns:
                    fig = px.pie(df, names="domain", title="Forces by Domain (PESTEL)")
                    st.plotly_chart(fig, use_container_width=True)

                st.markdown("### Top Drivers")
                for f in forces[:5]:
                    st.markdown(f"- **{f.get('name')}**: {f.get('description')}")

        with tab2:
            cols = st.columns(2)
            scenarios = view_model.get("scenarios", [])
            for i, scen in enumerate(scenarios):
                with cols[i % 2]:
                    with st.container(border=True):
                        st.markdown(f"### {scen.get('name')}")
                        st.caption(scen.get("narrative", "")[:200] + "...")
                        st.markdown("**Implications:**")
                        st.markdown("* Impact on supply chain") # Placeholder logic
                        st.markdown("* Regulatory pressure")

        with tab3:
            st.subheader("Wind Tunnel Results")
            wt = view_model.get("wind_tunnel", {})
            tests = wt.get("tests", [])
            if tests:
                df_wt = pd.DataFrame(tests)
                fig = px.bar(df_wt, x="strategy_id", y="feasibility_score", color="outcome", title="Strategy Feasibility")
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(df_wt[["strategy_id", "scenario_id", "outcome", "action"]])
            else:
                st.info("No wind tunnel tests available.")

        with tab4:
            st.json(view_model)

    render_step_panel(side_col, status_data.get("nodes", []), process_running=False)
