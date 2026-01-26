from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

import streamlit as st

from scenarioops.ui.page_utils import (
    card_grid,
    load_artifact,
    page_header,
    placeholder_section,
    resolve_run_id,
    section,
)

st.set_page_config(page_title="Auditor", page_icon="A", layout="wide")
run_id = resolve_run_id()
page_header("Auditor", run_id, subtitle="Quality checks and audit findings")

report = load_artifact(run_id, "audit_report")
if not report:
    placeholder_section("Audit Summary", ["Findings", "Lessons", "Actions"])
    st.stop()

card_grid(
    [
        ("Audit ID", str(report.get("id", ""))),
        ("Period Start", str(report.get("period_start", ""))),
        ("Period End", str(report.get("period_end", ""))),
    ]
)

section("Summary", report.get("summary", ""))

findings = report.get("findings", [])
if findings:
    section("Findings", " | ".join(findings))

lessons = report.get("lessons", [])
if lessons:
    section("Lessons", " | ".join(lessons))

actions = report.get("actions", [])
if actions:
    section("Actions", " | ".join(actions))
