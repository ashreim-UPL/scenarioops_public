from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def _cache_dir(root: Path | None = None) -> Path:
    if root is not None:
        return root
    return Path(__file__).resolve().parents[4] / "cache"


def _cache_path(root: Path | None = None) -> Path:
    return _cache_dir(root) / "embeddings.parquet"


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _tokenize(text: str) -> list[str]:
    return [token for token in re.split(r"[^a-zA-Z0-9]+", text.lower()) if token]


def _hash_embedding(text: str, *, dims: int = 256, seed: int = 0) -> list[float]:
    tokens = _tokenize(text)
    vec = np.zeros(dims, dtype=np.float32)
    for token in tokens:
        digest = hashlib.md5(f"{token}:{seed}".encode("utf-8")).hexdigest()
        idx = int(digest, 16) % dims
        vec[idx] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec.tolist()


def load_embedding_cache(root: Path | None = None) -> dict[tuple[str, str], list[float]]:
    path = _cache_path(root)
    if not path.exists():
        return {}
    try:
        df = pd.read_parquet(path)
    except Exception:
        return {}
    cache: dict[tuple[str, str], list[float]] = {}
    for _, row in df.iterrows():
        force_id = str(row.get("force_id", ""))
        content_hash = str(row.get("content_hash", ""))
        embedding = row.get("embedding")
        if force_id and content_hash and isinstance(embedding, list):
            cache[(force_id, content_hash)] = embedding
    return cache


def save_embedding_cache(
    entries: Iterable[dict[str, object]], root: Path | None = None
) -> None:
    path = _cache_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(entries)
    df.to_parquet(path, index=False)


def embed_texts(
    items: Iterable[tuple[str, str]],
    *,
    dims: int = 256,
    seed: int = 0,
    cache_root: Path | None = None,
) -> dict[str, dict[str, object]]:
    cache = load_embedding_cache(cache_root)
    output: dict[str, dict[str, object]] = {}
    updated_entries: list[dict[str, object]] = []
    for force_id, text in items:
        text_hash = _content_hash(text)
        cached = cache.get((force_id, text_hash))
        if cached is None:
            embedding = _hash_embedding(text, dims=dims, seed=seed)
            updated_entries.append(
                {
                    "force_id": force_id,
                    "content_hash": text_hash,
                    "embedding": embedding,
                }
            )
        else:
            embedding = cached
        output[force_id] = {"content_hash": text_hash, "embedding": embedding}
    if updated_entries:
        existing = [
            {"force_id": fid, "content_hash": h, "embedding": emb}
            for (fid, h), emb in cache.items()
        ]
        save_embedding_cache(existing + updated_entries, cache_root)
    return output
