from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import sys
from typing import Any, Mapping

from scenarioops import __version__ as scenarioops_version


def _canonical_json(value: Any) -> str:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return json.dumps(value, ensure_ascii=True)
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    except TypeError:
        return json.dumps(str(value), ensure_ascii=True)


def hash_value(value: Any) -> str:
    canonical = _canonical_json(value)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def hash_mapping(values: Mapping[str, Any] | None) -> dict[str, str]:
    if not values:
        return {}
    return {key: hash_value(value) for key, value in values.items()}


def default_tool_versions(extra: Mapping[str, str] | None = None) -> dict[str, str]:
    versions = {
        "python": sys.version.split()[0],
        "scenarioops": scenarioops_version,
    }
    if extra:
        versions.update(extra)
    return versions


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class ArtifactProvenance:
    artifact_name: str
    artifact_path: str
    run_id: str
    timestamps: dict[str, str]
    input_hashes: dict[str, str]
    prompt_hashes: dict[str, str]
    tool_versions: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_name": self.artifact_name,
            "artifact_path": self.artifact_path,
            "run_id": self.run_id,
            "timestamps": self.timestamps,
            "input_hashes": self.input_hashes,
            "prompt_hashes": self.prompt_hashes,
            "tool_versions": self.tool_versions,
        }


def build_provenance(
    *,
    artifact_name: str,
    artifact_path: str,
    run_id: str,
    input_values: Mapping[str, Any] | None = None,
    prompt_values: Mapping[str, Any] | None = None,
    tool_versions: Mapping[str, str] | None = None,
    created_at: str | None = None,
) -> ArtifactProvenance:
    if created_at is None:
        created_at = utc_now_iso()
    return ArtifactProvenance(
        artifact_name=artifact_name,
        artifact_path=artifact_path,
        run_id=run_id,
        timestamps={"created_at": created_at},
        input_hashes=hash_mapping(input_values),
        prompt_hashes=hash_mapping(prompt_values),
        tool_versions=default_tool_versions(tool_versions),
    )
