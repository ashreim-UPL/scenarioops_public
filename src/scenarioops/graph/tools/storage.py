from __future__ import annotations

import json
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
from typing import Any, Mapping

from .provenance import ArtifactProvenance, build_provenance
from scenarioops.storage.run_store import get_run_store, run_store_mode, runs_root

_RUN_CREATED_AT: dict[str, str] = {}
_DB_READY = False
_S3_CLIENT = None
_RUN_STORE_READY = False


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _db_enabled() -> bool:
    return bool(os.environ.get("DATABASE_URL"))


def _db_required() -> bool:
    return os.environ.get("SCENARIOOPS_DB_REQUIRED", "").lower() in {"1", "true", "yes"}


def _s3_enabled() -> bool:
    return bool(os.environ.get("S3_BUCKET"))


def _s3_required() -> bool:
    return os.environ.get("SCENARIOOPS_S3_REQUIRED", "").lower() in {"1", "true", "yes"}


def _s3_prefix() -> str:
    prefix = os.environ.get("S3_PREFIX", "scenarioops")
    return prefix.strip().strip("/")


def _s3_key(*parts: str) -> str:
    prefix = _s3_prefix()
    clean = [part.strip("/").replace("\\", "/") for part in parts if part]
    if prefix:
        return "/".join([prefix, *clean])
    return "/".join(clean)


def _s3_client():
    global _S3_CLIENT
    if _S3_CLIENT is not None:
        return _S3_CLIENT
    try:
        import boto3  # type: ignore
    except Exception as exc:
        if _s3_required():
            raise RuntimeError("boto3 is required for S3 storage.") from exc
        return None
    kwargs: dict[str, Any] = {}
    endpoint = os.environ.get("S3_ENDPOINT")
    if endpoint:
        kwargs["endpoint_url"] = endpoint
    region = os.environ.get("S3_REGION")
    if region:
        kwargs["region_name"] = region
    _S3_CLIENT = boto3.client("s3", **kwargs)
    return _S3_CLIENT


def _s3_put_bytes(key: str, data: bytes, content_type: str | None = None) -> None:
    bucket = os.environ.get("S3_BUCKET")
    if not bucket:
        return
    client = _s3_client()
    if client is None:
        return
    kwargs: dict[str, Any] = {"Bucket": bucket, "Key": key, "Body": data}
    if content_type:
        kwargs["ContentType"] = content_type
    client.put_object(**kwargs)


def _s3_put_file(key: str, path: Path, content_type: str | None = None) -> None:
    _s3_put_bytes(key, path.read_bytes(), content_type=content_type)


def _s3_uri(key: str) -> str:
    bucket = os.environ.get("S3_BUCKET", "")
    return f"s3://{bucket}/{key}"


def _runs_root(base_dir: Path | None = None) -> Path:
    return base_dir if base_dir is not None else default_runs_dir()


def _store_path_for(local_path: Path, base_dir: Path | None = None) -> str:
    runs_dir = _runs_root(None)
    try:
        relative = local_path.resolve().relative_to(runs_dir.resolve())
        return str(relative).replace("\\", "/")
    except Exception:
        pass
    if base_dir is not None:
        try:
            relative = local_path.resolve().relative_to(base_dir.resolve())
            return str(relative).replace("\\", "/")
        except Exception:
            pass
    return str(local_path.name).replace("\\", "/")


def _store_put(local_path: Path, *, base_dir: Path | None = None, content_type: str | None = None) -> None:
    if run_store_mode() != "gcs":
        return
    store = get_run_store()
    if not local_path.exists():
        return
    store_path = _store_path_for(local_path, base_dir=base_dir)
    store.put_bytes(store_path, local_path.read_bytes(), content_type=content_type)


def _store_get(local_path: Path, *, base_dir: Path | None = None) -> bool:
    if run_store_mode() != "gcs":
        return False
    store = get_run_store()
    store_path = _store_path_for(local_path, base_dir=base_dir)
    if not store.exists(store_path):
        return False
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_bytes(store.get_bytes(store_path))
    return True


def ensure_local_file(path: Path, *, base_dir: Path | None = None) -> Path:
    if path.exists():
        return path
    _store_get(path, base_dir=base_dir)
    return path


def write_bytes(path: Path, data: bytes, *, base_dir: Path | None = None, content_type: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    _store_put(path, base_dir=base_dir, content_type=content_type)


def write_text(path: Path, data: str, *, base_dir: Path | None = None, content_type: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data, encoding="utf-8")
    _store_put(path, base_dir=base_dir, content_type=content_type)


def _db_connect():
    try:
        import psycopg  # type: ignore
    except Exception as exc:
        if _db_required():
            raise RuntimeError("psycopg is required for Postgres storage.") from exc
        return None
    return psycopg.connect(os.environ["DATABASE_URL"])


def _db_init(conn) -> None:
    global _DB_READY
    if _DB_READY:
        return
    with conn.cursor() as cur:
        cur.execute(
            """
            create table if not exists scenarioops_runs (
                run_id text primary key,
                created_at timestamptz,
                updated_at timestamptz,
                status text,
                command text,
                error_summary text,
                run_config jsonb
            );
            """
        )
        cur.execute(
            """
            create table if not exists scenarioops_artifacts (
                run_id text,
                artifact_name text,
                artifact_path text,
                ext text,
                sha256 text,
                size_bytes bigint,
                created_at timestamptz,
                storage_uri text,
                meta_json jsonb,
                primary key (run_id, artifact_name, artifact_path)
            );
            """
        )
        cur.execute(
            """
            create table if not exists scenarioops_events (
                run_id text,
                event_type text,
                payload jsonb,
                created_at timestamptz
            );
            """
        )
    conn.commit()
    _DB_READY = True


def _db_upsert_run(
    *,
    run_id: str,
    created_at: str | None = None,
    updated_at: str | None = None,
    status: str | None = None,
    command: str | None = None,
    error_summary: str | None = None,
    run_config: Mapping[str, Any] | None = None,
) -> None:
    if not _db_enabled():
        return
    conn = _db_connect()
    if conn is None:
        return
    try:
        _db_init(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
                insert into scenarioops_runs (run_id, created_at, updated_at, status, command, error_summary, run_config)
                values (%s, %s, %s, %s, %s, %s, %s)
                on conflict (run_id) do update set
                    created_at = coalesce(excluded.created_at, scenarioops_runs.created_at),
                    updated_at = coalesce(excluded.updated_at, scenarioops_runs.updated_at),
                    status = coalesce(excluded.status, scenarioops_runs.status),
                    command = coalesce(excluded.command, scenarioops_runs.command),
                    error_summary = coalesce(excluded.error_summary, scenarioops_runs.error_summary),
                    run_config = coalesce(excluded.run_config, scenarioops_runs.run_config);
                """,
                (
                    run_id,
                    created_at,
                    updated_at,
                    status,
                    command,
                    error_summary,
                    dict(run_config) if run_config else None,
                ),
            )
        conn.commit()
    finally:
        conn.close()


def _db_insert_event(run_id: str, event_type: str, payload: Mapping[str, Any]) -> None:
    if not _db_enabled():
        return
    conn = _db_connect()
    if conn is None:
        return
    try:
        _db_init(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
                insert into scenarioops_events (run_id, event_type, payload, created_at)
                values (%s, %s, %s, %s);
                """,
                (
                    run_id,
                    event_type,
                    dict(payload),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
        conn.commit()
    finally:
        conn.close()


def _db_upsert_artifact(
    *,
    run_id: str,
    artifact_name: str,
    artifact_path: str,
    ext: str,
    sha256: str,
    size_bytes: int,
    created_at: str | None,
    storage_uri: str | None,
    meta_json: Mapping[str, Any] | None,
) -> None:
    if not _db_enabled():
        return
    conn = _db_connect()
    if conn is None:
        return
    try:
        _db_init(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
                insert into scenarioops_artifacts
                (run_id, artifact_name, artifact_path, ext, sha256, size_bytes, created_at, storage_uri, meta_json)
                values (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                on conflict (run_id, artifact_name, artifact_path) do update set
                    sha256 = excluded.sha256,
                    size_bytes = excluded.size_bytes,
                    created_at = coalesce(excluded.created_at, scenarioops_artifacts.created_at),
                    storage_uri = coalesce(excluded.storage_uri, scenarioops_artifacts.storage_uri),
                    meta_json = coalesce(excluded.meta_json, scenarioops_artifacts.meta_json);
                """,
                (
                    run_id,
                    artifact_name,
                    artifact_path,
                    ext,
                    sha256,
                    size_bytes,
                    created_at,
                    storage_uri,
                    dict(meta_json) if meta_json else None,
                ),
            )
        conn.commit()
    finally:
        conn.close()


def _find_repo_root(start: Path) -> Path:
    current = start.resolve()
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    return start.resolve()


def default_runs_dir() -> Path:
    return runs_root()


def latest_pointer_path(base_dir: Path | None = None) -> Path:
    if base_dir is None:
        base_dir = default_runs_dir()
    return base_dir / "latest.json"


def read_latest_status(base_dir: Path | None = None) -> dict[str, Any] | None:
    path = latest_pointer_path(base_dir)
    ensure_local_file(path, base_dir=base_dir)
    if not path.exists():
        if _db_enabled():
            conn = _db_connect()
            if conn is None:
                return None
            try:
                _db_init(conn)
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        select run_id, updated_at, status, command, error_summary, run_config
                        from scenarioops_runs
                        where updated_at is not null
                        order by updated_at desc
                        limit 1;
                        """
                    )
                    row = cur.fetchone()
                if row:
                    return {
                        "run_id": row[0],
                        "updated_at": row[1].isoformat() if row[1] else None,
                        "status": row[2],
                        "command": row[3],
                        "error_summary": row[4],
                        "run_config": row[5],
                    }
            finally:
                conn.close()
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
    run_config: Mapping[str, Any] | None = None,
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
    if run_config:
        payload["run_config"] = dict(run_config)
    path = latest_pointer_path(base_dir)
    _write_json(path, payload, base_dir=base_dir)
    _db_upsert_run(
        run_id=run_id,
        updated_at=payload["updated_at"],
        status=status,
        command=command_name,
        error_summary=error_summary,
        run_config=run_config,
    )
    if _s3_enabled():
        key = _s3_key("runs", run_id, "latest.json")
        _s3_put_bytes(
            key,
            json.dumps(payload, indent=2, sort_keys=True).encode("utf-8"),
            content_type="application/json",
        )
    final = status in {"OK", "FAIL"}
    run_updates = {
        "run_id": run_id,
        "status": "COMPLETED" if status == "OK" else "FAILED" if status == "FAIL" else status,
        "updated_at": payload["updated_at"],
        "is_final": final,
    }
    if final:
        run_updates["completed_at"] = payload["updated_at"]
    update_run_json(run_id=run_id, updates=run_updates, base_dir=base_dir)
    return path


def ensure_run_dirs(run_id: str, base_dir: Path | None = None) -> dict[str, Path]:
    if base_dir is None:
        base_dir = default_runs_dir()
    run_dir = base_dir / run_id
    artifacts_dir = run_dir / "artifacts"
    inputs_dir = run_dir / "inputs"
    derived_dir = run_dir / "derived"
    logs_dir = run_dir / "logs"
    trace_dir = run_dir / "trace"
    reports_dir = run_dir / "reports"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    inputs_dir.mkdir(parents=True, exist_ok=True)
    derived_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    return {
        "run_dir": run_dir,
        "artifacts_dir": artifacts_dir,
        "inputs_dir": inputs_dir,
        "derived_dir": derived_dir,
        "logs_dir": logs_dir,
        "trace_dir": trace_dir,
        "reports_dir": reports_dir,
    }


def run_json_path(run_id: str, base_dir: Path | None = None) -> Path:
    if base_dir is None:
        base_dir = default_runs_dir()
    return base_dir / run_id / "run.json"


def read_run_json(run_id: str, base_dir: Path | None = None) -> dict[str, Any] | None:
    path = run_json_path(run_id, base_dir)
    ensure_local_file(path, base_dir=base_dir)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def write_run_json(
    *, run_id: str, payload: Mapping[str, Any], base_dir: Path | None = None
) -> Path:
    dirs = ensure_run_dirs(run_id, base_dir=base_dir)
    path = dirs["run_dir"] / "run.json"
    _write_json(path, dict(payload), base_dir=base_dir)
    return path


def update_run_json(
    *, run_id: str, updates: Mapping[str, Any], base_dir: Path | None = None
) -> dict[str, Any]:
    existing = read_run_json(run_id, base_dir) or {}
    payload = {**existing, **dict(updates)}
    write_run_json(run_id=run_id, payload=payload, base_dir=base_dir)
    return payload


def append_run_event(
    *, run_id: str, payload: Mapping[str, Any], base_dir: Path | None = None
) -> Path:
    if base_dir is None:
        base_dir = default_runs_dir()
    path = base_dir / run_id / "events.jsonl"
    _append_jsonl(path, payload, base_dir=base_dir)
    return path


def write_run_inputs(
    *, run_id: str, payload: Mapping[str, Any], base_dir: Path | None = None
) -> Path:
    if base_dir is None:
        base_dir = default_runs_dir()
    path = base_dir / run_id / "inputs.json"
    _write_json(path, dict(payload), base_dir=base_dir)
    return path


def register_run_timestamp(run_id: str, created_at: str) -> None:
    _RUN_CREATED_AT[run_id] = created_at


def _created_at_for_run(run_id: str) -> str | None:
    return _RUN_CREATED_AT.get(run_id)


def get_run_timestamp(run_id: str) -> str | None:
    return _created_at_for_run(run_id)


def write_run_config(
    *,
    run_id: str,
    run_config: Mapping[str, Any],
    base_dir: Path | None = None,
) -> Path:
    dirs = ensure_run_dirs(run_id, base_dir=base_dir)
    path = dirs["run_dir"] / "run_config.json"
    _write_json(path, dict(run_config), base_dir=base_dir)
    _db_upsert_run(
        run_id=run_id,
        created_at=run_config.get("created_at"),
        run_config=run_config,
    )
    if _s3_enabled():
        key = _s3_key("runs", run_id, "run_config.json")
        _s3_put_file(key, path, content_type="application/json")
    return path


def _write_json(path: Path, payload: Any, *, base_dir: Path | None = None) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _store_put(path, base_dir=base_dir, content_type="application/json")


def _write_md(path: Path, payload: Any, *, base_dir: Path | None = None) -> None:
    if not isinstance(payload, str):
        raise TypeError("Markdown artifacts must be plain strings.")
    path.write_text(payload, encoding="utf-8")
    _store_put(path, base_dir=base_dir, content_type="text/markdown")


def _write_jsonl(path: Path, payload: Any, *, base_dir: Path | None = None) -> None:
    if not isinstance(payload, list):
        raise TypeError("JSONL artifacts must be a list of JSON-serializable items.")
    lines = [
        json.dumps(item, sort_keys=True, separators=(",", ":")) for item in payload
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _store_put(path, base_dir=base_dir, content_type="application/jsonl")


def _content_type(ext: str) -> str:
    if ext == "json":
        return "application/json"
    if ext == "jsonl":
        return "application/json"
    if ext == "md":
        return "text/markdown"
    return "application/octet-stream"


def write_artifact(
    *,
    run_id: str,
    artifact_name: str,
    payload: Any,
    ext: str = "json",
    input_values: Mapping[str, Any] | None = None,
    prompt_values: Mapping[str, Any] | None = None,
    tool_versions: Mapping[str, str] | None = None,
    created_at: str | None = None,
    base_dir: Path | None = None,
) -> tuple[Path, Path, ArtifactProvenance]:
    dirs = ensure_run_dirs(run_id, base_dir=base_dir)
    artifacts_dir = dirs["artifacts_dir"]

    ext = ext.lstrip(".")
    artifact_filename = f"{artifact_name}.{ext}"
    artifact_path = artifacts_dir / artifact_filename
    if ext == "json":
        _write_json(artifact_path, payload, base_dir=base_dir)
    elif ext == "md":
        _write_md(artifact_path, payload, base_dir=base_dir)
    elif ext == "jsonl":
        _write_jsonl(artifact_path, payload, base_dir=base_dir)
    else:
        raise ValueError("Artifact extension must be 'json', 'jsonl', or 'md'.")

    created_at = created_at or _created_at_for_run(run_id)
    provenance = build_provenance(
        artifact_name=artifact_name,
        artifact_path=artifact_filename,
        run_id=run_id,
        input_values=input_values,
        prompt_values=prompt_values,
        tool_versions=tool_versions,
        created_at=created_at,
    )

    meta_path = artifacts_dir / f"{artifact_name}.meta.json"
    _write_json(meta_path, provenance.to_dict(), base_dir=base_dir)
    sha256 = _sha256_file(artifact_path)
    size_bytes = artifact_path.stat().st_size
    storage_uri = None
    if _s3_enabled():
        key = _s3_key("runs", run_id, "artifacts", artifact_filename)
        _s3_put_file(key, artifact_path, content_type=_content_type(ext))
        meta_key = _s3_key("runs", run_id, "artifacts", f"{artifact_name}.meta.json")
        _s3_put_file(meta_key, meta_path, content_type="application/json")
        storage_uri = _s3_uri(key)
    _db_upsert_artifact(
        run_id=run_id,
        artifact_name=artifact_name,
        artifact_path=artifact_filename,
        ext=ext,
        sha256=sha256,
        size_bytes=size_bytes,
        created_at=created_at,
        storage_uri=storage_uri,
        meta_json=provenance.to_dict(),
    )
    return artifact_path, meta_path, provenance


def write_trace_artifact(
    *,
    run_id: str,
    artifact_name: str,
    payload: Any,
    input_values: Mapping[str, Any] | None = None,
    prompt_values: Mapping[str, Any] | None = None,
    tool_versions: Mapping[str, str] | None = None,
    created_at: str | None = None,
    base_dir: Path | None = None,
) -> tuple[Path, Path, ArtifactProvenance]:
    dirs = ensure_run_dirs(run_id, base_dir=base_dir)
    trace_dir = dirs["trace_dir"]
    artifact_filename = f"{artifact_name}.json"
    artifact_path = trace_dir / artifact_filename
    _write_json(artifact_path, payload, base_dir=base_dir)

    created_at = created_at or _created_at_for_run(run_id)
    provenance = build_provenance(
        artifact_name=artifact_name,
        artifact_path=str(Path("trace") / artifact_filename),
        run_id=run_id,
        input_values=input_values,
        prompt_values=prompt_values,
        tool_versions=tool_versions,
        created_at=created_at,
    )
    meta_path = trace_dir / f"{artifact_name}.meta.json"
    _write_json(meta_path, provenance.to_dict(), base_dir=base_dir)
    sha256 = _sha256_file(artifact_path)
    size_bytes = artifact_path.stat().st_size
    storage_uri = None
    if _s3_enabled():
        key = _s3_key("runs", run_id, "trace", artifact_filename)
        _s3_put_file(key, artifact_path, content_type="application/json")
        meta_key = _s3_key("runs", run_id, "trace", f"{artifact_name}.meta.json")
        _s3_put_file(meta_key, meta_path, content_type="application/json")
        storage_uri = _s3_uri(key)
    _db_upsert_artifact(
        run_id=run_id,
        artifact_name=artifact_name,
        artifact_path=str(Path("trace") / artifact_filename),
        ext="json",
        sha256=sha256,
        size_bytes=size_bytes,
        created_at=created_at,
        storage_uri=storage_uri,
        meta_json=provenance.to_dict(),
    )
    return artifact_path, meta_path, provenance


def _append_jsonl(path: Path, payload: Mapping[str, Any], *, base_dir: Path | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, sort_keys=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    _store_put(path, base_dir=base_dir, content_type="application/jsonl")


def log_node_event(
    *,
    run_id: str,
    node_name: str,
    inputs: list[str],
    outputs: list[str],
    schema_validated: bool,
    duration_seconds: float,
    base_dir: Path | None = None,
    error: str | None = None,
    tools: list[str] | None = None,
    status: str | None = None,
    timestamp: str | None = None,
) -> Path:
    if base_dir is None:
        base_dir = default_runs_dir()
    log_path = base_dir / run_id / "logs" / "node_events.jsonl"
    timestamp_value = timestamp or datetime.now(timezone.utc).isoformat()
    run_payload = read_run_json(run_id, base_dir=base_dir) or {"run_id": run_id}
    node_status = run_payload.get("node_status")
    if not isinstance(node_status, dict):
        node_status = {}
    previous = node_status.get(node_name) if isinstance(node_status.get(node_name), dict) else {}
    attempt = int(previous.get("attempt") or 0) + 1
    payload: dict[str, Any] = {
        "timestamp": timestamp_value,
        "node": node_name,
        "inputs": inputs,
        "outputs": outputs,
        "schema_validated": schema_validated,
        "duration_seconds": round(duration_seconds, 6),
        "attempt": attempt,
    }
    if error:
        payload["error"] = error
    if tools:
        payload["tools"] = tools
    if status:
        payload["status"] = status
    _append_jsonl(log_path, payload, base_dir=base_dir)
    append_run_event(run_id=run_id, payload=payload, base_dir=base_dir)
    node_status[node_name] = {
        "status": payload.get("status") or "UNKNOWN",
        "attempt": attempt,
        "updated_at": timestamp_value,
        "duration_seconds": payload.get("duration_seconds"),
        "error": payload.get("error"),
    }
    run_updates = {
        "run_id": run_id,
        "node_status": node_status,
        "updated_at": timestamp_value,
    }
    update_run_json(run_id=run_id, updates=run_updates, base_dir=base_dir)
    _db_insert_event(run_id, "node_event", payload)
    return log_path


def log_normalization(
    *,
    run_id: str,
    node_name: str,
    operation: str,
    details: Mapping[str, Any] | None = None,
    base_dir: Path | None = None,
) -> Path:
    if base_dir is None:
        base_dir = default_runs_dir()
    log_path = base_dir / run_id / "logs" / "normalization.jsonl"
    payload: dict[str, Any] = {"node": node_name, "operation": operation}
    if details:
        payload["details"] = dict(details)
    _append_jsonl(log_path, payload, base_dir=base_dir)
    _db_insert_event(run_id, "normalization", payload)
    return log_path


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
