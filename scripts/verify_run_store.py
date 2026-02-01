from __future__ import annotations

import os
import sys
import time
from typing import Any

import requests


def _api_base() -> str:
    return os.environ.get("SCENARIOOPS_API_BASE", "http://localhost:8502").rstrip("/")


def _headers() -> dict[str, str]:
    headers: dict[str, str] = {}
    api_key = os.environ.get("SCENARIOOPS_API_KEY")
    gemini_key = os.environ.get("SCENARIOOPS_GEMINI_API_KEY")
    if api_key:
        headers["X-Api-Key"] = api_key
    if gemini_key:
        headers["X-Gemini-Api-Key"] = gemini_key
    return headers


def _post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    resp = requests.post(f"{_api_base()}{path}", json=payload, headers=_headers(), timeout=60)
    resp.raise_for_status()
    return resp.json()


def _get(path: str) -> dict[str, Any]:
    resp = requests.get(f"{_api_base()}{path}", headers=_headers(), timeout=30)
    resp.raise_for_status()
    return resp.json()


def main() -> int:
    print("Health:", _get("/health"))
    payload = {
        "scope": "country",
        "value": "United States",
        "company": "ScenarioOps Demo Co",
        "geography": "United States",
        "horizon": 60,
        "mock": True,
        "generate_strategies": False,
    }
    run = _post("/build", payload)
    run_id = run.get("run_id")
    if not run_id:
        print("No run_id returned.")
        return 1
    print("Run:", run_id)
    time.sleep(2)
    view_model = _get(f"/artifact/{run_id}/view_model.json")
    if not view_model:
        print("Missing view_model.")
        return 1
    print("View model OK.")
    node_events = _get(f"/run/{run_id}/node_events")
    print(f"Node events: {len(node_events.get('entries', []))}")
    action_console = _get(f"/action-console/{run_id}")
    print(f"Action console items: {len(action_console.get('items', []))}")
    print("Verify complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
