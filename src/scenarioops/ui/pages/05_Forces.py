from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))


import pandas as pd
import streamlit as st

from scenarioops.ui.page_utils import load_artifact, page_header, placeholder_section, resolve_run_id

st.set_page_config(page_title="Forces", page_icon="S", layout="wide")
run_id = resolve_run_id()
page_header("Forces", run_id)

payload = load_artifact(run_id, "forces")
if not payload:
    placeholder_section("Force Summary", ["Top forces by domain", "Key mechanisms", "Confidence levels"])
    st.stop()

forces = payload.get("forces", []) if isinstance(payload.get("forces"), list) else []
if not forces:
    st.info("No forces found.")
    st.stop()

st.subheader("Force Overview")

cols = st.columns(3)
cols[0].metric("Total Forces", len(forces))
cols[1].metric("Domains", len({f.get("domain") for f in forces if f.get("domain")}))
cols[2].metric("Layers", len({f.get("layer") for f in forces if f.get("layer")}))

st.subheader("Forces Table")

df = pd.DataFrame(forces)
columns = [c for c in ["label", "domain", "layer", "mechanism", "directionality", "confidence"] if c in df.columns]
st.dataframe(df[columns], height=420)
