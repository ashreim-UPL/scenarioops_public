from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


def _find_repo_root(start: Path) -> Path:
    current = start.resolve()
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    return start.resolve()


def runs_root() -> Path:
    env_root = os.environ.get("RUNS_DIR", "").strip()
    if env_root:
        return Path(env_root)
    return _find_repo_root(Path(__file__).resolve()) / "storage" / "runs"


def runs_prefix() -> str:
    prefix = os.environ.get("RUNS_PREFIX", "scenarioforge/runs")
    return prefix.strip().strip("/")


def run_store_mode() -> str:
    return os.environ.get("RUN_STORE", "local").strip().lower()


def _join_prefix(prefix: str, path: str) -> str:
    clean = path.strip().lstrip("/")
    if not prefix:
        return clean
    if not clean:
        return prefix
    return f"{prefix}/{clean}"


@dataclass
class LocalRunStore:
    root_dir: Path

    def _full_path(self, path: str) -> Path:
        return self.root_dir / path

    def put_bytes(self, path: str, data: bytes, content_type: str | None = None) -> None:
        target = self._full_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    def get_bytes(self, path: str) -> bytes:
        return self._full_path(path).read_bytes()

    def exists(self, path: str) -> bool:
        return self._full_path(path).exists()

    def list(self, prefix: str) -> list[str]:
        root = self._full_path(prefix)
        if not root.exists():
            return []
        items: list[str] = []
        if root.is_file():
            return [prefix]
        for path in root.rglob("*"):
            if path.is_file():
                items.append(str(path.relative_to(self.root_dir)).replace("\\", "/"))
        return items

    def delete_prefix(self, prefix: str) -> None:
        root = self._full_path(prefix)
        if not root.exists():
            return
        if root.is_file():
            root.unlink()
            return
        for path in sorted(root.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            else:
                path.rmdir()
        root.rmdir()

    def atomic_json_write(self, path: str, obj: dict) -> None:
        payload = json.dumps(obj, indent=2, sort_keys=True).encode("utf-8")
        self.put_bytes(path, payload, content_type="application/json")


@dataclass
class GCSRunStore:
    bucket_name: str
    prefix: str

    def _client(self):
        try:
            from google.cloud import storage  # type: ignore
        except Exception as exc:
            raise RuntimeError("google-cloud-storage is required for GCS run storage.") from exc
        return storage.Client()

    def _bucket(self):
        return self._client().bucket(self.bucket_name)

    def _key(self, path: str) -> str:
        return _join_prefix(self.prefix, path)

    def put_bytes(self, path: str, data: bytes, content_type: str | None = None) -> None:
        blob = self._bucket().blob(self._key(path))
        blob.upload_from_string(data, content_type=content_type)

    def get_bytes(self, path: str) -> bytes:
        blob = self._bucket().blob(self._key(path))
        return blob.download_as_bytes()

    def exists(self, path: str) -> bool:
        blob = self._bucket().blob(self._key(path))
        return blob.exists()

    def list(self, prefix: str) -> list[str]:
        client = self._client()
        key_prefix = _join_prefix(self.prefix, prefix)
        blobs: Iterable = client.list_blobs(self.bucket_name, prefix=key_prefix)
        items: list[str] = []
        for blob in blobs:
            name = getattr(blob, "name", "")
            if not name:
                continue
            if self.prefix and name.startswith(self.prefix + "/"):
                name = name[len(self.prefix) + 1 :]
            items.append(name)
        return items

    def delete_prefix(self, prefix: str) -> None:
        client = self._client()
        key_prefix = _join_prefix(self.prefix, prefix)
        blobs = list(client.list_blobs(self.bucket_name, prefix=key_prefix))
        if not blobs:
            return
        bucket = self._bucket()
        for blob in blobs:
            bucket.blob(blob.name).delete()

    def atomic_json_write(self, path: str, obj: dict) -> None:
        payload = json.dumps(obj, indent=2, sort_keys=True).encode("utf-8")
        self.put_bytes(path, payload, content_type="application/json")


_RUN_STORE = None


def get_run_store():
    global _RUN_STORE
    if _RUN_STORE is not None:
        return _RUN_STORE
    mode = run_store_mode()
    if mode == "gcs":
        bucket = os.environ.get("GCS_BUCKET", "").strip()
        if not bucket:
            raise RuntimeError("GCS_BUCKET is required when RUN_STORE=gcs.")
        _RUN_STORE = GCSRunStore(bucket_name=bucket, prefix=runs_prefix())
    else:
        _RUN_STORE = LocalRunStore(root_dir=runs_root())
    return _RUN_STORE


__all__ = [
    "LocalRunStore",
    "GCSRunStore",
    "get_run_store",
    "runs_root",
    "runs_prefix",
    "run_store_mode",
]
