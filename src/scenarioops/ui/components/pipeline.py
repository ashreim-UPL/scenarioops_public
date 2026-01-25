from __future__ import annotations
import json
from typing import Any
from datetime import datetime, timezone
import streamlit as st

from scenarioops.ui.utils import RUNS_DIR, _parse_iso, _format_duration

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
