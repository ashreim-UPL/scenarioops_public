from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

import streamlit as st

from scenarioops.ui.page_utils import (
    load_artifact,
    page_header,
    placeholder_section,
    resolve_image_path,
    resolve_run_id,
    section,
)

st.set_page_config(page_title="Scenario Media", page_icon="M", layout="wide")
run_id = resolve_run_id()
page_header("Scenario Media", run_id, subtitle="Visual narrative support for scenarios")

payload = load_artifact(run_id, "scenarios_enriched")
if not payload:
    placeholder_section("Scenario Media", ["Scenario imagery", "Narrative summaries"])
    st.stop()

scenarios = payload.get("scenarios", []) if isinstance(payload, dict) else []
if not scenarios:
    section("Scenario Media", "No enriched scenario assets available yet.")
    st.stop()

for scenario in scenarios:
    section(scenario.get("name", "Scenario"), scenario.get("summary", "") or "")
    image_path = resolve_image_path(run_id, scenario.get("image_artifact_path"))
    if image_path:
        st.image(str(image_path), use_container_width=True)
    else:
        st.info("Image not yet generated for this scenario.")
