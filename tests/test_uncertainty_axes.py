from __future__ import annotations

from typing import Any, Mapping

from scenarioops.graph.nodes.uncertainty_axes import run_uncertainty_axes_node
from scenarioops.graph.state import ScenarioOpsState


def _axis(idx: int, *, include_name: bool = True) -> dict[str, Any]:
    axis = {
        "axis_id": f"axis-{idx}",
        "pole_a": f"Low {idx}",
        "pole_b": f"High {idx}",
        "impact_score": 4,
        "uncertainty_score": 4,
        "tension_basis": {"cluster_ids": ["cluster-1"], "force_ids": ["force-1"]},
        "what_would_change_mind": [f"signal-{idx}"],
        "independence_notes": "Distinct enough.",
    }
    if include_name:
        axis["axis_name"] = f"Axis {idx}"
    return axis


class AxisCorrectionClient:
    def generate_json(self, prompt: str, schema: Mapping[str, Any]) -> dict[str, Any]:
        title = schema.get("title") if isinstance(schema, Mapping) else ""
        if title == "Uncertainty Axes Payload":
            axes = [_axis(1, include_name=False)]
            axes.extend(_axis(idx) for idx in range(2, 7))
            return {"axes": axes}
        if title == "Uncertainty Axis Item":
            fixed = _axis(1, include_name=True)
            fixed["axis_name"] = "Axis 1"
            return fixed
        return {}


def test_uncertainty_axes_repairs_missing_axis_name() -> None:
    state = ScenarioOpsState()
    state.clusters = {"clusters": [{"cluster_id": "cluster-1"}]}
    state.forces = {"forces": [{"force_id": "force-1"}]}

    state = run_uncertainty_axes_node(
        run_id="axis-run",
        state=state,
        user_params={"value": "Acme", "scope": "world", "horizon": 12},
        llm_client=AxisCorrectionClient(),
        allow_needs_correction=True,
    )

    axes = state.uncertainty_axes.get("axes", []) if state.uncertainty_axes else []
    assert axes
    assert axes[0]["axis_name"]
