from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))


from collections import Counter
import streamlit as st

from scenarioops.ui.page_utils import load_artifact, metric_row, page_header, placeholder_section, resolve_run_id

st.set_page_config(page_title="Retrieval Summary", page_icon="R", layout="wide")
run_id = resolve_run_id()
page_header("Retrieval Summary", run_id)

report = load_artifact(run_id, "retrieval_report")
units = load_artifact(run_id, "evidence_units")
if not report:
    placeholder_section("Evidence Counts", ["ok", "total", "failed"])
    placeholder_section("Notes", ["Any retrieval notes"])
    st.stop()

counts = report.get("counts", {}) if isinstance(report.get("counts"), dict) else {}
metric_row({
    "Evidence OK": counts.get("ok", 0),
    "Evidence Total": counts.get("total", 0),
    "Evidence Failed": counts.get("failed", 0),
})

notes = report.get("notes", [])
if notes:
    st.subheader("Notes")
    st.write(notes)

if units and isinstance(units.get("evidence_units"), list):
    publishers = [u.get("publisher") for u in units.get("evidence_units", []) if u.get("publisher")]
    if publishers:
        st.subheader("Top Publishers")
        counts = Counter(publishers).most_common(10)
        st.table({"publisher": [c[0] for c in counts], "count": [c[1] for c in counts]})
