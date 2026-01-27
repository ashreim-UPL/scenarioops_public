from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

import plotly.graph_objects as go
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
st.markdown(f"**X-axis (left/right):** {axis_x.get('axis_name', '')}")
st.markdown(f"**Left:** {axis_x.get('pole_b', '')}")
st.markdown(f"**Right:** {axis_x.get('pole_a', '')}")
st.markdown(f"**Y-axis (bottom/top):** {axis_y.get('axis_name', '')}")
st.markdown(f"**Bottom:** {axis_y.get('pole_b', '')}")
st.markdown(f"**Top:** {axis_y.get('pole_a', '')}")

scenarios = scenarios_payload.get("scenarios", []) if isinstance(scenarios_payload, dict) else []
axis_ids = scenarios_payload.get("axes", []) if isinstance(scenarios_payload, dict) else []
axis_x_id = axis_ids[0] if len(axis_ids) > 0 else axis_x.get("axis_id")
axis_y_id = axis_ids[1] if len(axis_ids) > 1 else axis_y.get("axis_id")

def _axis_value(axis, state_value):
    if not state_value:
        return 0
    if state_value == axis.get("pole_a"):
        return 1
    if state_value == axis.get("pole_b"):
        return -1
    return 0

scenario_positions = {}
for scenario in scenarios:
    axis_states = scenario.get("axis_states", {})
    if not isinstance(axis_states, dict):
        continue
    x_val = _axis_value(axis_x, axis_states.get(axis_x_id))
    y_val = _axis_value(axis_y, axis_states.get(axis_y_id))
    scenario_positions[scenario.get("name", "")] = (x_val, y_val, scenario)

clusters = clusters_payload.get("clusters", []) if isinstance(clusters_payload, dict) else []
cluster_points = []
for cluster in clusters:
    cluster_id = cluster.get("cluster_id")
    if not cluster_id:
        continue
    hits = []
    for scenario in scenarios:
        evidence = scenario.get("evidence_touchpoints", {})
        if not isinstance(evidence, dict):
            continue
        cluster_ids = evidence.get("cluster_ids", [])
        if cluster_id in cluster_ids:
            axis_states = scenario.get("axis_states", {})
            if not isinstance(axis_states, dict):
                continue
            hits.append(
                (
                    _axis_value(axis_x, axis_states.get(axis_x_id)),
                    _axis_value(axis_y, axis_states.get(axis_y_id)),
                )
            )
    if not hits:
        continue
    x_avg = sum(item[0] for item in hits) / len(hits)
    y_avg = sum(item[1] for item in hits) / len(hits)
    force_count = len(cluster.get("force_ids", [])) if isinstance(cluster.get("force_ids"), list) else 0
    cluster_points.append(
        {
            "label": cluster.get("cluster_label", cluster_id),
            "x": x_avg,
            "y": y_avg,
            "size": max(10, force_count * 3),
        }
    )

fig = go.Figure()
top_labels = {item["label"] for item in sorted(cluster_points, key=lambda p: p["size"], reverse=True)[:6]}
for point in cluster_points:
    label = point["label"] if point["label"] in top_labels else ""
    fig.add_trace(
        go.Scatter(
            x=[point["x"]],
            y=[point["y"]],
            mode="markers+text",
            text=[label],
            textposition="top center",
            hovertext=[point["label"]],
            marker=dict(size=point["size"], color="#0e7c86", opacity=0.65),
        )
    )
if scenario_positions:
    scenario_names = list(scenario_positions.keys())
    scenario_x = [scenario_positions[name][0] for name in scenario_names]
    scenario_y = [scenario_positions[name][1] for name in scenario_names]
    fig.add_trace(
        go.Scatter(
            x=scenario_x,
            y=scenario_y,
            mode="markers+text",
            text=scenario_names,
            textposition="bottom center",
            marker=dict(size=8, color="#666", opacity=0.8),
            hovertext=scenario_names,
            showlegend=False,
        )
    )
fig.add_shape(type="line", x0=0, x1=0, y0=-1.2, y1=1.2, line=dict(color="#999", dash="dot"))
fig.add_shape(type="line", x0=-1.2, x1=1.2, y0=0, y1=0, line=dict(color="#999", dash="dot"))
fig.update_layout(
    height=520,
    xaxis=dict(range=[-1.2, 1.2], tickvals=[-1, 1], ticktext=[axis_x.get("pole_b", ""), axis_x.get("pole_a", "")]),
    yaxis=dict(range=[-1.2, 1.2], tickvals=[-1, 1], ticktext=[axis_y.get("pole_b", ""), axis_y.get("pole_a", "")]),
    xaxis_title=axis_x.get("axis_name", ""),
    yaxis_title=axis_y.get("axis_name", ""),
    showlegend=False,
)
st.plotly_chart(fig, width="stretch")

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
