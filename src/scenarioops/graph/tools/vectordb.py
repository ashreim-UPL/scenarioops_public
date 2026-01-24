from __future__ import annotations

from dataclasses import dataclass
import json
import math
import sqlite3
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol

from scenarioops.graph.tools.embeddings import embed_texts
from scenarioops.graph.tools.storage import default_runs_dir


@dataclass(frozen=True)
class VectorDocument:
    doc_id: str
    text: str
    embedding: list[float]
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class VectorMatch:
    doc_id: str
    score: float
    text: str
    metadata: Mapping[str, Any]


class VectorStoreClient(Protocol):
    def add_documents(self, documents: Iterable[VectorDocument]) -> None:
        ...

    def query(self, text: str, *, top_k: int = 5) -> list[VectorMatch]:
        ...

    def get_by_ids(self, ids: Iterable[str]) -> list[VectorDocument]:
        ...


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    norm_left = math.sqrt(sum(a * a for a in left))
    norm_right = math.sqrt(sum(b * b for b in right))
    if norm_left == 0.0 or norm_right == 0.0:
        return 0.0
    return dot / (norm_left * norm_right)


def _embed_text(
    text: str,
    *,
    embed_model: str,
    seed: int,
    cache_root: Path | None,
) -> list[float]:
    if embed_model != "local-hash-256":
        raise ValueError(f"Unsupported embed_model: {embed_model}")
    output = embed_texts([("query", text)], seed=seed, cache_root=cache_root)
    embedding = output.get("query", {}).get("embedding", [])
    return list(embedding) if isinstance(embedding, list) else []


def _encode(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


def _decode(value: str) -> Any:
    return json.loads(value)


class LocalVectorStore:
    def __init__(
        self,
        db_path: Path,
        *,
        embed_model: str,
        seed: int = 0,
        cache_root: Path | None = None,
    ) -> None:
        self.db_path = db_path
        self.embed_model = embed_model
        self.seed = seed
        self.cache_root = cache_root
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    metadata TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "INSERT OR IGNORE INTO meta (key, value) VALUES (?, ?)",
                ("embed_model", self.embed_model),
            )
            conn.commit()

    def add_documents(self, documents: Iterable[VectorDocument]) -> None:
        rows = []
        for doc in documents:
            rows.append(
                (
                    doc.doc_id,
                    doc.text,
                    _encode(doc.embedding),
                    _encode(doc.metadata),
                )
            )
        if not rows:
            return
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO documents (doc_id, text, embedding, metadata)
                VALUES (?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()

    def build_document(
        self,
        *,
        doc_id: str,
        text: str,
        metadata: Mapping[str, Any],
    ) -> VectorDocument:
        embedding = _embed_text(
            text, embed_model=self.embed_model, seed=self.seed, cache_root=self.cache_root
        )
        return VectorDocument(
            doc_id=doc_id,
            text=text,
            embedding=embedding,
            metadata=metadata,
        )

    def query(self, text: str, *, top_k: int = 5) -> list[VectorMatch]:
        embedding = _embed_text(
            text, embed_model=self.embed_model, seed=self.seed, cache_root=self.cache_root
        )
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT doc_id, text, embedding, metadata FROM documents"
            ).fetchall()
        matches: list[VectorMatch] = []
        for doc_id, doc_text, doc_embedding, metadata in rows:
            score = _cosine_similarity(embedding, _decode(doc_embedding))
            matches.append(
                VectorMatch(
                    doc_id=doc_id,
                    score=score,
                    text=doc_text,
                    metadata=_decode(metadata),
                )
            )
        matches.sort(key=lambda item: (-item.score, item.doc_id))
        return matches[:top_k]

    def get_by_ids(self, ids: Iterable[str]) -> list[VectorDocument]:
        id_list = [str(value) for value in ids if str(value)]
        if not id_list:
            return []
        placeholders = ",".join("?" for _ in id_list)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT doc_id, text, embedding, metadata FROM documents WHERE doc_id IN ({placeholders})",
                id_list,
            ).fetchall()
        documents: list[VectorDocument] = []
        for doc_id, doc_text, doc_embedding, metadata in rows:
            documents.append(
                VectorDocument(
                    doc_id=doc_id,
                    text=doc_text,
                    embedding=_decode(doc_embedding),
                    metadata=_decode(metadata),
                )
            )
        return documents


def run_vectordb_dir(run_id: str, base_dir: Path | None = None) -> Path:
    runs_dir = base_dir if base_dir is not None else default_runs_dir()
    return runs_dir / run_id / "vectordb"


def open_run_vector_store(
    run_id: str,
    *,
    base_dir: Path | None = None,
    embed_model: str,
    seed: int = 0,
) -> LocalVectorStore:
    db_dir = run_vectordb_dir(run_id, base_dir=base_dir)
    cache_root = db_dir / "embed_cache"
    db_path = db_dir / "embeddings.sqlite"
    return LocalVectorStore(
        db_path,
        embed_model=embed_model,
        seed=seed,
        cache_root=cache_root,
    )
