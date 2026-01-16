from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from scenarioops.graph.tools.artifact_contracts import schema_for_artifact
from scenarioops.graph.tools.provenance import hash_value
from scenarioops.graph.tools.schema_validate import (
    SchemaValidationError,
    validate_artifact,
    validate_jsonl,
)
from scenarioops.graph.tools.storage import default_runs_dir


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _hash_file(path: Path) -> str:
    return _hash_text(path.read_text(encoding="utf-8"))


def _repo_root(start: Path) -> Path:
    current = start.resolve()
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    return start.resolve()


def _read_git_commit(repo_root: Path) -> str:
    head_path = repo_root / ".git" / "HEAD"
    if not head_path.exists():
        return "unknown"
    content = head_path.read_text(encoding="utf-8").strip()
    if content.startswith("ref:"):
        ref = content.split(" ", 1)[-1].strip()
        ref_path = repo_root / ".git" / ref
        if ref_path.exists():
            return ref_path.read_text(encoding="utf-8").strip()
        return "unknown"
    return content or "unknown"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[Any]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _collect_versions(paths: Iterable[Path]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for path in paths:
        versions[path.name] = _hash_file(path)
    return versions


def collect_schema_versions(root: Path) -> dict[str, str]:
    schemas_dir = root / "schemas"
    if not schemas_dir.exists():
        return {}
    return _collect_versions(sorted(schemas_dir.glob("*.json")))


def collect_prompt_versions(root: Path) -> dict[str, str]:
    prompts_dir = root / "prompts"
    if not prompts_dir.exists():
        return {}
    prompts = sorted([path for path in prompts_dir.iterdir() if path.is_file()])
    return _collect_versions(prompts)


def build_artifact_index(
    run_id: str,
    base_dir: Path | None = None,
    *,
    strict: bool = True,
) -> dict[str, Any]:
    runs_dir = base_dir if base_dir is not None else default_runs_dir()
    run_dir = runs_dir / run_id
    artifacts_dir = run_dir / "artifacts"
    trace_dir = run_dir / "trace"
    if not artifacts_dir.exists():
        raise FileNotFoundError(f"Artifacts directory missing: {artifacts_dir}")

    entries: list[dict[str, Any]] = []
    errors: list[str] = []

    def handle_path(path: Path, relative_root: Path) -> None:
        if path.name.endswith(".meta.json"):
            return
        if path.name == "index.json":
            return
        name = path.stem
        suffix = path.suffix
        relative = str(path.relative_to(run_dir))
        schema_name = schema_for_artifact(name, suffix)
        entry: dict[str, Any] = {
            "name": name,
            "path": relative,
            "sha256": _hash_file(path),
            "schema": schema_name,
            "validated": False,
        }
        meta_path = path.with_suffix(".meta.json")
        if meta_path.exists():
            entry["meta_path"] = str(meta_path.relative_to(run_dir))
            try:
                meta_payload = _load_json(meta_path)
            except json.JSONDecodeError:
                meta_payload = {}
            if isinstance(meta_payload, dict):
                tool_versions = meta_payload.get("tool_versions")
                if isinstance(tool_versions, dict):
                    entry["tool_versions"] = tool_versions

        if schema_name:
            try:
                if suffix == ".json":
                    payload = _load_json(path)
                    validate_artifact(schema_name, payload)
                elif suffix == ".jsonl":
                    payload = _load_jsonl(path)
                    validate_jsonl(schema_name, payload)
                elif suffix == ".md":
                    validate_artifact(schema_name, path.read_text(encoding="utf-8"))
                entry["validated"] = True
            except (SchemaValidationError, ValueError, json.JSONDecodeError) as exc:
                message = f"{relative}: {exc}"
                entry["validation_errors"] = [message]
                errors.append(message)
                if strict:
                    raise
        else:
            message = f"{relative}: schema mapping missing"
            entry["validation_errors"] = [message]
            errors.append(message)
            if strict:
                raise ValueError(message)

        entries.append(entry)

    for path in sorted(artifacts_dir.iterdir()):
        if path.is_file():
            handle_path(path, artifacts_dir)
    if trace_dir.exists():
        for path in sorted(trace_dir.iterdir()):
            if path.is_file():
                handle_path(path, trace_dir)

    return {"artifacts": entries, "validation_errors": errors}


def write_artifact_index(
    run_id: str,
    base_dir: Path | None = None,
    *,
    strict: bool = True,
) -> Path:
    runs_dir = base_dir if base_dir is not None else default_runs_dir()
    run_dir = runs_dir / run_id
    artifacts_dir = run_dir / "artifacts"
    index_path = artifacts_dir / "index.json"
    payload = build_artifact_index(run_id, base_dir=base_dir, strict=strict)
    index_payload = {"artifacts": payload.get("artifacts", [])}
    index_path.write_text(
        json.dumps(index_payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    validate_artifact("artifact_index", index_payload)
    return index_path


def write_run_manifest(
    *,
    run_id: str,
    run_timestamp: str,
    status: str,
    input_parameters: dict[str, Any],
    node_sequence: list[dict[str, Any]],
    artifact_index_path: Path,
    trace_map_path: Path | None,
    run_config: dict[str, Any] | None,
    base_dir: Path | None = None,
    errors: list[str] | None = None,
) -> Path:
    runs_dir = base_dir if base_dir is not None else default_runs_dir()
    run_dir = runs_dir / run_id
    repo_root = _repo_root(run_dir)
    manifest_path = run_dir / "manifest.json"
    manifest = {
        "run_id": run_id,
        "timestamp": run_timestamp,
        "status": status,
        "code_commit": _read_git_commit(repo_root),
        "config_hash": hash_value(run_config or {}),
        "schema_versions": collect_schema_versions(repo_root),
        "prompt_versions": collect_prompt_versions(repo_root),
        "input_parameters": input_parameters,
        "node_sequence": node_sequence,
        "artifact_index_path": str(artifact_index_path.relative_to(run_dir)),
        "trace_map_path": str(trace_map_path.relative_to(run_dir))
        if trace_map_path is not None
        else None,
    }
    if errors:
        manifest["errors"] = errors

    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    validate_artifact("run_manifest", manifest)
    return manifest_path
