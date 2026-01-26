from __future__ import annotations
import re
from datetime import datetime
from pathlib import Path

from scenarioops.graph.tools.storage import default_runs_dir

def _find_repo_root(start: Path) -> Path:
    current = start.resolve()
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return start.resolve().parent

ROOT = _find_repo_root(Path(__file__).resolve())
SRC_DIR = ROOT / "src"
RUNS_DIR = default_runs_dir()
LATEST_POINTER = RUNS_DIR / "latest.json"

def _normalize_label(value: str) -> str:
    return " ".join(value.lower().strip().split())

def _safe_filename(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    return cleaned.strip("_") or "document"

def _parse_iso(ts: str) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None

def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    total = int(round(seconds))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:d}:{secs:02d}"

def _format_list(values: any) -> str:
    if not values:
        return "none"
    if isinstance(values, list):
        return ", ".join(str(item) for item in values)
    return str(values)
