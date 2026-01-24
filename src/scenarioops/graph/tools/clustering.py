from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.cluster import AgglomerativeClustering


def _cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = vectors / norms
    return normalized @ normalized.T


def cluster_vectors(
    vectors: np.ndarray,
    *,
    distance_threshold: float = 0.4,
    min_cluster_size: int = 3,
) -> tuple[list[int], dict[str, Any]]:
    if vectors.shape[0] < 2:
        adjusted = [-1 for _ in range(vectors.shape[0])]
        stats = {
            "distance_threshold": distance_threshold,
            "min_cluster_size": min_cluster_size,
            "cluster_count": 0,
            "singleton_count": len(adjusted),
        }
        return adjusted, stats
    try:
        model = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            linkage="average",
            metric="cosine",
        )
    except TypeError:
        model = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            linkage="average",
            affinity="cosine",
        )
    labels = model.fit_predict(vectors)
    sizes: dict[int, int] = {}
    for label in labels:
        sizes[label] = sizes.get(label, 0) + 1

    adjusted = []
    for label in labels:
        if sizes.get(label, 0) < min_cluster_size:
            adjusted.append(-1)
        else:
            adjusted.append(label)

    stats = {
        "distance_threshold": distance_threshold,
        "min_cluster_size": min_cluster_size,
        "cluster_count": len({label for label in adjusted if label >= 0}),
        "singleton_count": sum(1 for label in adjusted if label == -1),
    }
    return adjusted, stats


def coherence_scores(
    vectors: np.ndarray, labels: list[int]
) -> dict[int, float]:
    scores: dict[int, float] = {}
    sim = _cosine_similarity_matrix(vectors)
    for label in set(labels):
        if label == -1:
            continue
        idx = [i for i, lab in enumerate(labels) if lab == label]
        if len(idx) < 2:
            scores[label] = 0.0
            continue
        sub = sim[np.ix_(idx, idx)]
        scores[label] = float(np.mean(sub))
    return scores
