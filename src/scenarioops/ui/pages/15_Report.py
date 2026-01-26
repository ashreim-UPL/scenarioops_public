from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

import streamlit as st

from scenarioops.reporting import build_management_report
from scenarioops.ui.page_utils import page_header, placeholder_section, resolve_run_id

st.set_page_config(page_title="Management Report", page_icon="R", layout="wide")
run_id = resolve_run_id()
page_header("Management Report", run_id, subtitle="PDF narrative for leadership review")

if not run_id:
    placeholder_section("Report", ["Load or run a scenario pipeline to generate a report."])
    st.stop()

st.markdown("### Generate Report")
st.write("Build a PDF report that compiles each node into a management narrative.")

if st.button("Generate PDF Report"):
    try:
        output_path = build_management_report(run_id)
        st.success(f"Report generated: {output_path.name}")
        with open(output_path, "rb") as handle:
            st.download_button(
                label="Download Report",
                data=handle,
                file_name=output_path.name,
                mime="application/pdf",
            )
    except Exception as exc:
        st.error(f"Report generation failed: {exc}")
