from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping

from scenarioops.graph.tools.storage import get_run_timestamp

def build_run_metadata(
    *,
    run_id: str,
    user_params: Mapping[str, Any],
    focal_issue: Mapping[str, Any] | None = None,
    timestamp: str | None = None,
) -> dict[str, Any]:
    company_name = (
        user_params.get("company")
        or user_params.get("value")
        or "unknown"
    )
    geography = user_params.get("geography") or user_params.get("scope") or "unspecified"
    horizon_months = user_params.get("horizon")
    if focal_issue and isinstance(focal_issue, Mapping):
        scope = focal_issue.get("scope")
        if isinstance(scope, Mapping):
            geography = scope.get("geography") or geography
            horizon_years = scope.get("time_horizon_years")
            if isinstance(horizon_years, int) and horizon_years > 0:
                horizon_months = horizon_years * 12
    if horizon_months is None:
        horizon_months = 0
    if timestamp is None:
        registered = get_run_timestamp(run_id)
        if registered:
            timestamp = registered
    return {
        "run_id": run_id,
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        "company_name": str(company_name),
        "geography": str(geography),
        "horizon_months": int(horizon_months),
    }
