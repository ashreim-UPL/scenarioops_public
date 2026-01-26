from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

import pandas as pd
import streamlit as st

from scenarioops.ui.page_utils import (
    card_grid,
    load_artifact,
    page_header,
    placeholder_section,
    resolve_run_id,
    section,
)

st.set_page_config(page_title="Uploads", page_icon="U", layout="wide")
run_id = resolve_run_id()
page_header("Uploads", run_id, subtitle="Ingested documents and internal sources")

payload = load_artifact(run_id, "evidence_units_uploads")
if not payload:
    placeholder_section("Uploads Intake", ["Uploaded documents", "Parsed evidence units"])
    st.stop()

units = payload.get("evidence_units", []) if isinstance(payload, dict) else []
if not units:
    section("Uploads Intake", "No evidence units extracted from uploads.")
    st.stop()

publishers = {u.get("publisher") for u in units if u.get("publisher")}
card_grid(
    [
        ("Uploaded Evidence Units", str(len(units))),
        ("Unique Publishers", str(len(publishers))),
        ("Run ID", str(payload.get("run_id", ""))),
    ]
)

rows = []
for unit in units:
    rows.append(
        {
            "title": unit.get("title", ""),
            "publisher": unit.get("publisher", ""),
            "date": unit.get("date", ""),
            "url": unit.get("url", ""),
        }
    )

df = pd.DataFrame(rows)
if not df.empty:
    section("Uploaded Evidence Inventory", "Primary documents and key citations.")
    st.dataframe(df, height=420)
