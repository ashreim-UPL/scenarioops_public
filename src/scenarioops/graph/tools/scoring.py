from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping


Action = str


@dataclass(frozen=True)
class RubricThresholds:
    keep: float = 0.75
    modify: float = 0.55
    hedge: float = 0.35


DEFAULT_WEIGHTS: dict[str, float] = {
    "relevance": 0.3,
    "credibility": 0.3,
    "recency": 0.2,
    "specificity": 0.2,
}


@dataclass(frozen=True)
class ScoringResult:
    total: float
    normalized_total: float
    breakdown: dict[str, dict[str, float]]
    action: Action
    thresholds: RubricThresholds

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "normalized_total": self.normalized_total,
            "breakdown": self.breakdown,
            "action": self.action,
            "thresholds": {
                "keep": self.thresholds.keep,
                "modify": self.thresholds.modify,
                "hedge": self.thresholds.hedge,
            },
        }


def _clamp(value: float) -> float:
    if value < 0:
        return 0.0
    if value > 1:
        return 1.0
    return value


def _decide_action(score: float, thresholds: RubricThresholds) -> Action:
    if score >= thresholds.keep:
        return "KEEP"
    if score >= thresholds.modify:
        return "MODIFY"
    if score >= thresholds.hedge:
        return "HEDGE"
    return "DROP"


def score_with_rubric(
    scores: Mapping[str, float],
    *,
    weights: Mapping[str, float] | None = None,
    thresholds: RubricThresholds | None = None,
) -> ScoringResult:
    if weights is None:
        weights = DEFAULT_WEIGHTS
    if thresholds is None:
        thresholds = RubricThresholds()

    weight_sum = sum(weights.values())
    if weight_sum <= 0:
        raise ValueError("Rubric weights must sum to a positive value.")

    breakdown: dict[str, dict[str, float]] = {}
    total = 0.0
    for key in sorted(weights.keys()):
        weight = weights[key]
        raw_score = scores.get(key, 0.0)
        clamped = _clamp(float(raw_score))
        weighted = clamped * weight
        breakdown[key] = {
            "score": clamped,
            "weight": weight,
            "weighted": weighted,
        }
        total += weighted

    normalized_total = total / weight_sum
    action = _decide_action(normalized_total, thresholds)
    return ScoringResult(
        total=total,
        normalized_total=normalized_total,
        breakdown=breakdown,
        action=action,
        thresholds=thresholds,
    )


def hash_scoring_result(result: ScoringResult) -> str:
    payload = json.dumps(result.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
