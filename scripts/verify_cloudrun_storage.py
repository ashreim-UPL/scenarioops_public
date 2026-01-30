from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

from scenarioops.storage.run_store import get_run_store, runs_root


def _http_json(url: str, *, headers: dict[str, str] | None = None) -> dict:
    req = Request(url, headers=headers or {})
    with urlopen(req, timeout=30) as resp:
        data = resp.read().decode("utf-8")
    return json.loads(data)


def _now_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=os.environ.get("SCENARIOOPS_API_BASE", ""))
    parser.add_argument("--tenant", default=os.environ.get("SCENARIOOPS_DEFAULT_TENANT", "public"))
    parser.add_argument("--run-id", default="")
    parser.add_argument("--delete", action="store_true")
    args = parser.parse_args()

    run_id = args.run_id or f"verify-{_now_id()}"
    tenant_prefix = f"{args.tenant}/{run_id}"
    store = get_run_store()

    run_json = {
        "run_id": run_id,
        "status": "COMPLETED",
        "is_final": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "node_status": {},
        "signature": "verify",
    }
    store.put_bytes(f"{tenant_prefix}/run.json", json.dumps(run_json).encode("utf-8"), content_type="application/json")
    store.put_bytes(f"{tenant_prefix}/inputs.json", json.dumps({"test": True}).encode("utf-8"), content_type="application/json")
    store.put_bytes(f"{tenant_prefix}/events.jsonl", b"", content_type="application/jsonl")

    if args.base_url:
        health = _http_json(f"{args.base_url.rstrip('/')}/health")
        print("health:", health)
        runs = _http_json(f"{args.base_url.rstrip('/')}/runs")
        print("runs:", runs)
        meta = _http_json(f"{args.base_url.rstrip('/')}/runs/{run_id}", headers={"Accept": "application/json"})
        print("run meta:", meta)
        if args.delete:
            api_key = os.environ.get("SCENARIOOPS_API_KEY", "")
            if api_key:
                req = Request(
                    f"{args.base_url.rstrip('/')}/runs/{run_id}",
                    headers={"X-Api-Key": api_key},
                    method="DELETE",
                )
                with urlopen(req, timeout=30) as resp:
                    print("delete:", resp.read().decode("utf-8"))
            else:
                print("delete skipped: SCENARIOOPS_API_KEY not set")

    local_root = runs_root()
    print("local_runs_root:", local_root)
    print("store_ready:", True)
    print("run_id:", run_id)


if __name__ == "__main__":
    main()
