# ui/streamlit_app.py
from __future__ import annotations

import json
from pathlib import Path
import streamlit as st
import subprocess

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "storage" / "runs"
LATEST_POINTER = RUNS_DIR / "latest.json"

st.title("ScenarioOps Live")

scope = st.selectbox("Scope", ["world", "region", "country"], index=0)
value = st.text_input("Value", "UAE")
horizon = st.slider("Horizon (months)", 6, 60, 24)

def run_cli(args: list[str]) -> tuple[int, str]:
    p = subprocess.run(
        ["python", "-m", "scenarioops.app.main", *args],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    out = (p.stdout or "") + ("\n" + p.stderr if p.stderr else "")
    return p.returncode, out.strip()

def load_latest_status() -> dict | None:
    if not LATEST_POINTER.exists():
        return None
    try:
        return json.loads(LATEST_POINTER.read_text(encoding="utf-8"))
    except Exception:
        return None

def find_most_recent_run_id() -> str | None:
    if not RUNS_DIR.exists():
        return None
    candidates = [p for p in RUNS_DIR.iterdir() if p.is_dir()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0].name

def resolve_run_context() -> tuple[str | None, dict | None]:
    status = load_latest_status()
    run_id = status.get("run_id") if status else None
    if not run_id:
        run_id = find_most_recent_run_id()
    return run_id, status

def load_daily_brief_text(run_id: str) -> tuple[str | None, Path]:
    run_dir = RUNS_DIR / run_id
    brief_path = run_dir / "artifacts" / "daily_brief.md"
    if brief_path.exists():
        return brief_path.read_text(encoding="utf-8"), brief_path
    return None, run_dir

def load_run_logs(run_id: str) -> str | None:
    logs_dir = RUNS_DIR / run_id / "logs"
    if not logs_dir.exists():
        return None
    log_files = sorted([p for p in logs_dir.iterdir() if p.is_file()])
    if not log_files:
        return None
    parts: list[str] = []
    for path in log_files:
        content = path.read_text(encoding="utf-8")
        parts.append(f"--- {path.name} ---\n{content.strip()}")
    return "\n\n".join(parts)

if st.button("Build Scenarios"):
    rc, out = run_cli(["build-scenarios", "--scope", scope, "--value", value, "--horizon", str(horizon)])
    if rc == 0:
        st.success("Scenario build complete.")
    else:
        st.error("Build failed.")
    st.code(out)

st.header("Latest Run Status")

run_id, status = resolve_run_context()
if status:
    st.write(f"Status: {status.get('status', 'unknown')}")
    if status.get("error_summary"):
        st.warning(f"Error: {status['error_summary']}")
elif run_id:
    st.write("Status: unknown (latest.json missing)")
else:
    st.info("No runs found yet. Click **Build Scenarios** first.")

st.header("Build Logs")
if run_id:
    logs = load_run_logs(run_id)
    if logs:
        st.code(logs)
    else:
        st.info("No logs yet for the latest run.")

st.header("Latest Daily Brief")

if run_id:
    brief_text, path_info = load_daily_brief_text(run_id)
    if brief_text:
        st.caption(f"Loaded from: {path_info}")
        st.markdown(brief_text)
    else:
        st.info("No brief yet.")
else:
    st.info("No brief yet.")
