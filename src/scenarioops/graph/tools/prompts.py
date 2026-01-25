from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class PromptSpec:
    name: str
    path: Path
    sha256: str
    text: str


def prompts_dir() -> Path:
    # __file__ is src/scenarioops/graph/tools/prompts.py
    # parents[4] is repo root
    return Path(__file__).resolve().parents[4] / "prompts"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_prompt_spec(name: str) -> PromptSpec:
    prompt_root = prompts_dir()
    requested = name.strip()
    if not requested:
        raise FileNotFoundError("Prompt not found: empty name")
    candidate = prompt_root / requested
    if candidate.suffix:
        if candidate.exists():
            text = candidate.read_text(encoding="utf-8")
            return PromptSpec(
                name=candidate.stem,
                path=candidate,
                sha256=_sha256(text),
                text=text,
            )
        requested = candidate.stem
    for ext in (".prompt", ".txt"):
        path = prompt_root / f"{requested}{ext}"
        if path.exists():
            text = path.read_text(encoding="utf-8")
            return PromptSpec(
                name=requested,
                path=path,
                sha256=_sha256(text),
                text=text,
            )
    lowered = requested.lower()
    for path in prompt_root.glob("*"):
        if not path.is_file() or path.suffix not in {".prompt", ".txt"}:
            continue
        if path.stem.lower() == lowered:
            text = path.read_text(encoding="utf-8")
            return PromptSpec(
                name=path.stem,
                path=path,
                sha256=_sha256(text),
                text=text,
            )
    available = ", ".join(sorted(p.stem for p in prompt_root.glob("*.prompt")))
    raise FileNotFoundError(
        f"Prompt not found: {prompt_root / requested}. Available: {available}"
    )


def iter_prompt_specs() -> Iterable[PromptSpec]:
    prompt_root = prompts_dir()
    for path in sorted(prompt_root.glob("*")):
        if not path.is_file():
            continue
        if path.suffix not in {".prompt", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8")
        yield PromptSpec(
            name=path.stem,
            path=path,
            sha256=_sha256(text),
            text=text,
        )


def build_prompt_manifest() -> dict[str, list[dict[str, str]]]:
    prompts = []
    for spec in iter_prompt_specs():
        prompts.append(
            {
                "name": spec.name,
                "path": str(spec.path),
                "sha256": spec.sha256,
            }
        )
    return {"prompts": prompts}
