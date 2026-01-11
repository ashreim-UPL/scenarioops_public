from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .provenance import ArtifactProvenance, build_provenance


def default_runs_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "storage" / "runs"


def latest_pointer_path(base_dir: Path | None = None) -> Path:
    if base_dir is None:
        base_dir = default_runs_dir()
    return base_dir / "latest.json"


def read_latest_status(base_dir: Path | None = None) -> dict[str, Any] | None:
    path = latest_pointer_path(base_dir)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def write_latest_status(
    *,
    run_id: str,
    status: str,
    command: str | None = None,
    base_dir: Path | None = None,
    error_summary: str | None = None,
) -> Path:
    if base_dir is None:
        base_dir = default_runs_dir()
    base_dir.mkdir(parents=True, exist_ok=True)
    command_name = command or "unknown"
    payload: dict[str, Any] = {
        "run_id": run_id,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "command": command_name,
    }
    if error_summary:
        payload["error_summary"] = error_summary
    path = latest_pointer_path(base_dir)
    _write_json(path, payload)
    return path


def ensure_run_dirs(run_id: str, base_dir: Path | None = None) -> dict[str, Path]:
    if base_dir is None:
        base_dir = default_runs_dir()
    run_dir = base_dir / run_id
    artifacts_dir = run_dir / "artifacts"
    logs_dir = run_dir / "logs"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    return {"run_dir": run_dir, "artifacts_dir": artifacts_dir, "logs_dir": logs_dir}


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_md(path: Path, payload: Any) -> None:
    if not isinstance(payload, str):
        raise TypeError("Markdown artifacts must be plain strings.")
    path.write_text(payload, encoding="utf-8")


def _write_jsonl(path: Path, payload: Any) -> None:
    if not isinstance(payload, list):
        raise TypeError("JSONL artifacts must be a list of JSON-serializable items.")
    lines = [
        json.dumps(item, sort_keys=True, separators=(",", ":")) for item in payload
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_artifact(
    *,
    run_id: str,
    artifact_name: str,
    payload: Any,
    ext: str = "json",
    input_values: Mapping[str, Any] | None = None,
    prompt_values: Mapping[str, Any] | None = None,
    tool_versions: Mapping[str, str] | None = None,
    base_dir: Path | None = None,
) -> tuple[Path, Path, ArtifactProvenance]:
    dirs = ensure_run_dirs(run_id, base_dir=base_dir)
    artifacts_dir = dirs["artifacts_dir"]

    ext = ext.lstrip(".")
    artifact_filename = f"{artifact_name}.{ext}"
    artifact_path = artifacts_dir / artifact_filename
    if ext == "json":
        _write_json(artifact_path, payload)
    elif ext == "md":
        _write_md(artifact_path, payload)
    elif ext == "jsonl":
        _write_jsonl(artifact_path, payload)
    else:
        raise ValueError("Artifact extension must be 'json', 'jsonl', or 'md'.")

    provenance = build_provenance(
        artifact_name=artifact_name,
        artifact_path=artifact_filename,
        run_id=run_id,
        input_values=input_values,
        prompt_values=prompt_values,
        tool_versions=tool_versions,
    )

    meta_path = artifacts_dir / f"{artifact_name}.meta.json"
    _write_json(meta_path, provenance.to_dict())
    return artifact_path, meta_path, provenance


def run_dummy_hello(run_id: str, base_dir: Path | None = None) -> tuple[Path, Path]:
    payload = {"message": "hello"}
    artifact_path, meta_path, _ = write_artifact(
        run_id=run_id,
        artifact_name="hello",
        payload=payload,
        ext="json",
        input_values={"payload": payload},
        prompt_values={"prompt": "Write a hello artifact."},
        tool_versions={"dummy_node": "0.1.0"},
        base_dir=base_dir,
    )
    return artifact_path, meta_path
