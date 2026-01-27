from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from html import escape
import textwrap

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

domain_order = ["economic", "social", "technological", "legal", "political", "environmental"]
layer_order = ["primary", "secondary", "tertiary"]

def _norm(value: str | None) -> str:
    return (value or "").strip().lower()

domain_groups: dict[str, dict[str, list[str]]] = {d: {l: [] for l in layer_order} for d in domain_order}
domain_counts: dict[str, int] = {d: 0 for d in domain_order}
for _, row in df.iterrows():
    domain = _norm(row.get("domain"))
    layer = _norm(row.get("layer"))
    label = str(row.get("label") or "").strip()
    if not label:
        continue
    if domain not in domain_groups:
        domain_groups.setdefault(domain, {l: [] for l in layer_order})
        domain_counts.setdefault(domain, 0)
    if layer not in layer_order:
        layer = "tertiary"
    domain_groups[domain][layer].append(label)
    domain_counts[domain] = domain_counts.get(domain, 0) + 1

palette = {
    "economic": {"accent": "#007eb5", "bg": "#d4e6f1"},
    "social": {"accent": "#00a3a1", "bg": "#d5f4e6"},
    "technological": {"accent": "#ff9800", "bg": "#fff4e6"},
    "legal": {"accent": "#702f8a", "bg": "#f4ecf7"},
    "political": {"accent": "#d63426", "bg": "#fadbd8"},
    "environmental": {"accent": "#00a896", "bg": "#e8f8f5"},
}

sections_html = []
for domain in domain_order:
    layers_html = []
    for layer in layer_order:
        items = domain_groups.get(domain, {}).get(layer, [])
        if not items:
            continue
        items_html = "\n".join(
            f'<div class="force-item" style="border-left-color:{palette[domain]["accent"]};">{escape(item)}</div>'
            for item in sorted(items)
        )
        layers_html.append(
            f"<div class=\"priority-section\">"
            f"<div class=\"priority-label\">{escape(layer)} forces</div>"
            f"<div class=\"forces-grid\">{items_html}</div>"
            f"</div>"
        )
    if not layers_html:
        continue
    sections_html.append(
        f"<div class=\"domain-card {domain}\">"
        f"<div class=\"domain-header\" style=\"border-color:{palette[domain]['accent']}; background:{palette[domain]['bg']};\">"
        f"<div class=\"domain-title\">{escape(domain)}</div>"
        f"<div class=\"domain-count\">{domain_counts.get(domain, 0)} forces</div>"
        f"</div>"
        f"<div class=\"domain-content\">{''.join(layers_html)}</div>"
        f"</div>"
    )

st.markdown(
    textwrap.dedent(
        """
<style>
.forces-wrap {
  background: white;
  border: 1px solid #d0d0d0;
  padding: 24px;
}
.domains-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 18px;
}
.domain-card {
  background: white;
  border: 1px solid #d0d0d0;
  overflow: hidden;
}
.domain-header {
  padding: 16px 20px;
  border-bottom: 3px solid;
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.domain-title {
  font-family: "Lora", Georgia, serif;
  font-size: 18px;
  font-weight: 600;
  text-transform: capitalize;
  color: #1a1a1a;
}
.domain-count {
  font-size: 12px;
  color: #4a5568;
  background: #f5f5f5;
  padding: 4px 10px;
  border-radius: 14px;
  font-weight: 600;
}
.domain-content {
  padding: 16px 20px 20px;
}
.priority-section {
  margin-bottom: 16px;
}
.priority-section:last-child {
  margin-bottom: 0;
}
.priority-label {
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 1.2px;
  color: #667085;
  font-weight: 700;
  margin-bottom: 8px;
  padding-bottom: 6px;
  border-bottom: 1px solid #e6e6e6;
}
.forces-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(210px, 1fr));
  gap: 10px;
}
.force-item {
  background: #fafafa;
  padding: 10px 12px;
  border-left: 3px solid;
  font-size: 13px;
  line-height: 1.35;
  color: #1f2937;
}
@media (max-width: 1200px) {
  .domains-grid { grid-template-columns: 1fr; }
}
</style>
"""
    ).strip(),
    unsafe_allow_html=True,
)

st.markdown(
    f"<div class='forces-wrap'><div class='domains-grid'>{''.join(sections_html)}</div></div>",
    unsafe_allow_html=True,
)

st.dataframe(df[columns], height=420)
