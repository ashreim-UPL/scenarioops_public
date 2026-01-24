from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from scenarioops.app.config import LLMConfig, ScenarioOpsSettings
from scenarioops.graph.nodes.utils import build_prompt, get_client
from scenarioops.graph.state import ScenarioOpsState
from scenarioops.graph.tools.clustering import cluster_vectors, coherence_scores
from scenarioops.graph.tools.embeddings import embed_texts
from scenarioops.graph.tools.schema_validate import load_schema, validate_artifact
from scenarioops.graph.tools.storage import write_artifact
from scenarioops.graph.tools.traceability import build_run_metadata
from scenarioops.llm.guards import ensure_dict


def _force_text(force: Mapping[str, Any]) -> str:
    parts = [
        str(force.get("label", "")),
        str(force.get("mechanism", "")),
        str(force.get("directionality", "")),
        str(force.get("domain", "")),
        " ".join(str(item) for item in force.get("affected_dimensions", [])),
    ]
    return " ".join(part for part in parts if part)


def _centroid_summary(forces: list[Mapping[str, Any]]) -> str:
    tokens: list[str] = []
    for force in forces:
        text = _force_text(force).lower()
        tokens.extend([token for token in text.split() if len(token) > 3])
    common = [token for token, _ in Counter(tokens).most_common(3)]
    return " / ".join(common) if common else "miscellaneous drivers"


def _allow_needs_correction(
    settings: ScenarioOpsSettings | None, config: LLMConfig | None
) -> bool:
    if settings is not None:
        if settings.sources_policy == "fixtures":
            return True
        if settings.mode == "demo":
            return True
        if settings.llm_provider == "mock":
            return True
    if config is not None and getattr(config, "mode", None) == "mock":
        return True
    return False


def run_cluster_node(
    *,
    run_id: str,
    state: ScenarioOpsState,
    user_params: Mapping[str, Any],
    llm_client=None,
    base_dir: Path | None = None,
    config: LLMConfig | None = None,
    settings: ScenarioOpsSettings | None = None,
    seed: int = 0,
    allow_needs_correction: bool | None = None,
) -> ScenarioOpsState:
    if state.forces is None:
        raise ValueError("Forces are required before clustering.")

    forces_payload = state.forces
    forces = forces_payload.get("forces", [])
    if not isinstance(forces, list):
        raise TypeError("Forces payload must include a list of forces.")

    items = []
    for force in forces:
        if not isinstance(force, Mapping):
            continue
        force_id = str(force.get("force_id"))
        if not force_id:
            continue
        items.append((force_id, _force_text(force)))

    embedded = embed_texts(items, seed=seed)
    vectors = np.array(
        [embedded[force_id]["embedding"] for force_id, _ in items], dtype=np.float32
    )

    warnings: list[str] = []
    distance_threshold = 0.4
    labels, stats = cluster_vectors(
        vectors, distance_threshold=distance_threshold, min_cluster_size=3
    )
    singleton_ratio = stats["singleton_count"] / max(1, len(labels))
    attempts = 0
    while singleton_ratio > 0.4 and attempts < 2:
        distance_threshold += 0.1
        warnings.append(f"cluster_threshold_adjusted:{distance_threshold:.2f}")
        labels, stats = cluster_vectors(
            vectors, distance_threshold=distance_threshold, min_cluster_size=3
        )
        singleton_ratio = stats["singleton_count"] / max(1, len(labels))
        attempts += 1
    if singleton_ratio > 0.4:
        warnings.append(f"high_singleton_ratio: {singleton_ratio:.2f}")

    coherence = coherence_scores(vectors, labels)
    clusters_by_label: dict[int, list[str]] = {}
    for (force_id, _), label in zip(items, labels):
        clusters_by_label.setdefault(label, []).append(force_id)

    cluster_entries: list[dict[str, Any]] = []
    for label, force_ids in clusters_by_label.items():
        cluster_id = (
            "cluster-singletons" if label == -1 else f"cluster-{label + 1}"
        )
        cluster_forces = [
            force for force in forces if isinstance(force, Mapping) and force.get("force_id") in force_ids
        ]
        cluster_entries.append(
            {
                "cluster_id": cluster_id,
                "force_ids": force_ids,
                "coherence_score": coherence.get(label, 0.0),
                "centroid_summary": _centroid_summary(cluster_forces),
                "cluster_label": "",
                "underlying_dynamic": "",
                "why_these_forces_belong_together": "",
            }
        )

    prompt_bundle = build_prompt(
        "cluster_labeling",
        {
            "clusters": [
                {
                    "cluster_id": entry["cluster_id"],
                    "forces": [
                        {
                            "force_id": force.get("force_id"),
                            "label": force.get("label"),
                            "mechanism": force.get("mechanism"),
                        }
                        for force in forces
                        if isinstance(force, Mapping)
                        and force.get("force_id") in entry["force_ids"]
                    ][:6],
                }
                for entry in cluster_entries
            ]
        },
    )
    client = get_client(llm_client, config)
    schema = load_schema("clusters_payload")
    needs_correction = False
    try:
        response = client.generate_json(prompt_bundle.text, schema)
        labeled = ensure_dict(response, node_name="clusters")
        labels_payload = labeled.get("clusters", [])
    except Exception as exc:
        needs_correction = True
        warnings.append(f"cluster_labeling_failed: {exc}")
        labels_payload = []

    label_map: dict[str, Mapping[str, Any]] = {}
    if isinstance(labels_payload, list):
        for entry in labels_payload:
            if isinstance(entry, Mapping):
                label_map[str(entry.get("cluster_id"))] = entry

    for entry in cluster_entries:
        label_payload = label_map.get(entry["cluster_id"])
        if not label_payload:
            needs_correction = True
            entry["cluster_label"] = entry["centroid_summary"]
            entry["underlying_dynamic"] = "needs_correction"
            entry["why_these_forces_belong_together"] = "labeling unavailable"
            continue
        entry["cluster_label"] = str(label_payload.get("cluster_label", "")).strip()
        entry["underlying_dynamic"] = str(label_payload.get("underlying_dynamic", "")).strip()
        entry["why_these_forces_belong_together"] = str(
            label_payload.get("why_these_forces_belong_together", "")
        ).strip()
        if not entry["cluster_label"] or not entry["underlying_dynamic"]:
            needs_correction = True
            entry["cluster_label"] = entry["cluster_label"] or entry["centroid_summary"]
            entry["underlying_dynamic"] = entry["underlying_dynamic"] or "needs_correction"
            entry["why_these_forces_belong_together"] = (
                entry["why_these_forces_belong_together"] or "labeling incomplete"
            )

    metadata = build_run_metadata(run_id=run_id, user_params=user_params)
    payload = {
        **metadata,
        "needs_correction": needs_correction,
        "warnings": warnings,
        "clusters": cluster_entries,
    }
    validate_artifact("clusters", payload)

    write_artifact(
        run_id=run_id,
        artifact_name="clusters",
        payload=payload,
        ext="json",
        input_values={
            "cluster_count": len(cluster_entries),
            "distance_threshold": distance_threshold,
            "seed": seed,
        },
        prompt_values={
            "prompt_name": prompt_bundle.name,
            "prompt_sha256": prompt_bundle.sha256,
        },
        tool_versions={"cluster_node": "0.1.0"},
        base_dir=base_dir,
    )
    state.clusters = payload
    if needs_correction:
        allow = (
            allow_needs_correction
            if allow_needs_correction is not None
            else _allow_needs_correction(settings, config)
        )
        if not allow:
            raise RuntimeError("cluster_node_needs_correction")
    return state
