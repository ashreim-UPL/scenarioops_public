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

st.set_page_config(page_title="Forces", page_icon="S", layout="wide")
run_id = resolve_run_id()
page_header("Forces", run_id, subtitle="Signals shaping the strategic landscape")

payload = load_artifact(run_id, "forces")
if not payload:
    placeholder_section("Force Summary", ["Top forces by domain", "Key mechanisms", "Confidence levels"])
    st.stop()

forces = payload.get("forces", []) if isinstance(payload.get("forces"), list) else []
if not forces:
    st.info("No forces found.")
    st.stop()

card_grid(
    [
        ("Total Forces", str(len(forces))),
        ("Domains", str(len({f.get("domain") for f in forces if f.get("domain")}))),
        ("Layers", str(len({f.get("layer") for f in forces if f.get("layer")}))),
    ]
)

section("Force Inventory", "Core drivers organized by domain and layer.")

df = pd.DataFrame(forces)
columns = [c for c in ["label", "domain", "layer", "mechanism", "directionality", "confidence"] if c in df.columns]
st.dataframe(df[columns], height=420)
