from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

import html
import streamlit as st

from scenarioops.ui.page_utils import (
    card_grid,
    load_artifact,
    page_header,
    placeholder_section,
    resolve_run_id,
    section,
)

st.set_page_config(page_title="Uncertainty Axes", page_icon="U", layout="wide")
run_id = resolve_run_id()
page_header("Uncertainty Axes", run_id, subtitle="Primary uncertainties defining the scenario space")

payload = load_artifact(run_id, "uncertainty_axes")
if not payload:
    placeholder_section("Axes", ["Two core uncertainties", "Pole definitions", "Evidence basis"])
    st.stop()

axes = payload.get("axes", []) if isinstance(payload, dict) else []
scenarios_payload = load_artifact(run_id, "scenarios") or {}
clusters_payload = load_artifact(run_id, "clusters") or {}
card_grid(
    [
        ("Axes Selected", str(len(axes))),
        ("Company", str(payload.get("company_name", ""))),
        ("Horizon (months)", str(payload.get("horizon_months", ""))),
    ]
)

if not axes:
    st.stop()

axis_x = axes[0] if len(axes) > 0 else {}
axis_y = axes[1] if len(axes) > 1 else {}

section("Axes Overview", "Two primary uncertainties defining the scenario space.")
st.markdown(f"**X-axis:** {axis_x.get('axis_name', '')}")
st.markdown(f"**Y-axis:** {axis_y.get('axis_name', '')}")

scenarios = scenarios_payload.get("scenarios", []) if isinstance(scenarios_payload, dict) else []
axis_ids = scenarios_payload.get("axes", []) if isinstance(scenarios_payload, dict) else []
axis_x_id = axis_ids[0] if len(axis_ids) > 0 else axis_x.get("axis_id")
axis_y_id = axis_ids[1] if len(axis_ids) > 1 else axis_y.get("axis_id")

def _quadrant_key(axis_states: dict[str, str]) -> str | None:
    if not axis_states:
        return None
    x_state = axis_states.get(axis_x_id)
    y_state = axis_states.get(axis_y_id)
    if not x_state or not y_state:
        return None
    x_right = x_state == axis_x.get("pole_a")
    y_top = y_state == axis_y.get("pole_a")
    if x_right and y_top:
        return "tr"
    if not x_right and y_top:
        return "tl"
    if not x_right and not y_top:
        return "bl"
    return "br"


def _truncate(text: str, limit: int = 160) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "…"


quadrant_titles = {
    "tr": f"{axis_y.get('pole_a', '')} / {axis_x.get('pole_a', '')}",
    "tl": f"{axis_y.get('pole_a', '')} / {axis_x.get('pole_b', '')}",
    "bl": f"{axis_y.get('pole_b', '')} / {axis_x.get('pole_b', '')}",
    "br": f"{axis_y.get('pole_b', '')} / {axis_x.get('pole_a', '')}",
}

quadrant_scenarios: dict[str, list[dict[str, str]]] = {"tr": [], "tl": [], "bl": [], "br": []}
for scenario in scenarios:
    axis_states = scenario.get("axis_states", {})
    if not isinstance(axis_states, dict):
        continue
    key = _quadrant_key(axis_states)
    if not key:
        continue
    quadrant_scenarios[key].append(
        {
            "name": str(scenario.get("name", "Scenario")),
            "summary": str(scenario.get("summary") or ""),
        }
    )

quadrant_html = []
for key in ("tl", "tr", "bl", "br"):
    title = html.escape(quadrant_titles.get(key, ""))
    cards = []
    for item in quadrant_scenarios.get(key, []):
        summary = html.escape(_truncate(item.get("summary", "")))
        cards.append(
            f"""
<div class="quad-card">
  <div class="quad-title">{html.escape(item['name'])}</div>
  <div class="quad-summary">{summary}</div>
</div>
"""
        )
    cards_html = "".join(cards) if cards else "<div class='quad-empty'>No scenario</div>"
    quadrant_html.append(
        f"""
<div class="quad-cell">
  <div class="quad-bg">{title}</div>
  <div class="quad-cards">{cards_html}</div>
</div>
"""
    )

st.markdown(
    """
<style>
.quad-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 18px;
}
.quad-cell {
  position: relative;
  min-height: 260px;
  border: 1px solid #e6dfd4;
  border-radius: 16px;
  background: #fffdf8;
  padding: 24px;
  overflow: hidden;
}
.quad-bg {
  position: absolute;
  inset: 14px;
  display: flex;
  align-items: center;
  justify-content: center;
  text-align: center;
  font-size: 28px;
  font-weight: 600;
  letter-spacing: 0.02em;
  color: rgba(80,80,80,0.12);
  line-height: 1.25;
  pointer-events: none;
}
.quad-cards {
  position: relative;
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 12px;
  z-index: 2;
}
.quad-card {
  background: rgba(14,124,134,0.08);
  border: 1px solid rgba(14,124,134,0.25);
  border-radius: 12px;
  padding: 12px 14px;
}
.quad-title {
  font-weight: 600;
  font-size: 15px;
  color: #0a4a52;
}
.quad-summary {
  margin-top: 6px;
  font-size: 12px;
  color: #285e66;
  line-height: 1.45;
}
.quad-empty {
  font-size: 12px;
  color: #7a7a7a;
}
@media (max-width: 900px) {
  .quad-grid { grid-template-columns: 1fr; }
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(f"<div class='quad-grid'>{''.join(quadrant_html)}</div>", unsafe_allow_html=True)

for axis in axes:
    section(axis.get("axis_name", "Axis"), axis.get("independence_notes", ""))
    st.markdown(f"**Pole A:** {axis.get('pole_a', '')}")
    st.markdown(f"**Pole B:** {axis.get('pole_b', '')}")
    st.markdown(
        f"**Impact / Uncertainty:** {axis.get('impact_score', '')} / {axis.get('uncertainty_score', '')}"
    )
    changes = axis.get("what_would_change_mind", [])
    if changes:
        st.markdown("**What would change our mind**")
        st.write(changes)
