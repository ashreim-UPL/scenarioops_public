from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class ActivationBand:
    label: str
    minimum: float


DEFAULT_BANDS = [
    ActivationBand(label="none", minimum=0.0),
    ActivationBand(label="watch", minimum=0.25),
    ActivationBand(label="elevated", minimum=0.5),
    ActivationBand(label="active", minimum=0.75),
]


@dataclass(frozen=True)
class IndicatorSignal:
    indicator_id: str
    score: float


@dataclass(frozen=True)
class ScenarioActivation:
    scenario_id: str
    score: float
    band: str
    indicator_count: int
    active_indicators: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "score": self.score,
            "band": self.band,
            "indicator_count": self.indicator_count,
            "active_indicators": self.active_indicators,
        }


@dataclass(frozen=True)
class ScenarioActivationDelta:
    scenario_id: str
    score_delta: float
    band_from: str
    band_to: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "score_delta": self.score_delta,
            "band_from": self.band_from,
            "band_to": self.band_to,
        }


def _clamp(value: float) -> float:
    if value < 0:
        return 0.0
    if value > 1:
        return 1.0
    return value


def _band_for_score(score: float, bands: Sequence[ActivationBand]) -> str:
    ordered = sorted(bands, key=lambda item: item.minimum)
    label = ordered[0].label
    for band in ordered:
        if score >= band.minimum:
            label = band.label
    return label


def _coerce_signal(signal: IndicatorSignal | Mapping[str, Any]) -> IndicatorSignal:
    if isinstance(signal, IndicatorSignal):
        return signal
    indicator_id = str(signal.get("indicator_id"))
    score = float(signal.get("score", 0.0))
    return IndicatorSignal(indicator_id=indicator_id, score=_clamp(score))


def compute_scenario_activation(
    indicators: Sequence[Mapping[str, Any]],
    signals: Sequence[IndicatorSignal | Mapping[str, Any]],
    *,
    bands: Sequence[ActivationBand] | None = None,
) -> list[ScenarioActivation]:
    if bands is None:
        bands = DEFAULT_BANDS

    indicator_to_scenarios: dict[str, list[str]] = {}
    for indicator in indicators:
        indicator_id = str(indicator.get("id"))
        linked = indicator.get("linked_scenarios", []) or []
        scenario_ids = [str(item) for item in linked]
        indicator_to_scenarios[indicator_id] = scenario_ids

    signal_map: dict[str, float] = {}
    for signal in signals:
        coerced = _coerce_signal(signal)
        signal_map[coerced.indicator_id] = _clamp(coerced.score)

    scenario_scores: dict[str, list[float]] = {}
    for indicator_id, scenario_ids in indicator_to_scenarios.items():
        score = signal_map.get(indicator_id, 0.0)
        for scenario_id in scenario_ids:
            scenario_scores.setdefault(scenario_id, []).append(score)

    activations: list[ScenarioActivation] = []
    for scenario_id in sorted(scenario_scores.keys()):
        scores = scenario_scores[scenario_id]
        indicator_count = len(scores)
        total = sum(scores)
        average = total / indicator_count if indicator_count else 0.0
        active_count = sum(1 for value in scores if value >= 0.5)
        band = _band_for_score(average, bands)
        activations.append(
            ScenarioActivation(
                scenario_id=scenario_id,
                score=average,
                band=band,
                indicator_count=indicator_count,
                active_indicators=active_count,
            )
        )

    return activations


def compute_activation_deltas(
    today: Sequence[ScenarioActivation],
    yesterday: Sequence[ScenarioActivation] | None = None,
) -> list[ScenarioActivationDelta]:
    today_map = {item.scenario_id: item for item in today}
    yesterday_map = {item.scenario_id: item for item in (yesterday or [])}
    scenario_ids = sorted(set(today_map.keys()) | set(yesterday_map.keys()))

    deltas: list[ScenarioActivationDelta] = []
    for scenario_id in scenario_ids:
        today_activation = today_map.get(
            scenario_id,
            ScenarioActivation(
                scenario_id=scenario_id,
                score=0.0,
                band="none",
                indicator_count=0,
                active_indicators=0,
            ),
        )
        yesterday_activation = yesterday_map.get(
            scenario_id,
            ScenarioActivation(
                scenario_id=scenario_id,
                score=0.0,
                band="none",
                indicator_count=0,
                active_indicators=0,
            ),
        )
        deltas.append(
            ScenarioActivationDelta(
                scenario_id=scenario_id,
                score_delta=today_activation.score - yesterday_activation.score,
                band_from=yesterday_activation.band,
                band_to=today_activation.band,
            )
        )

    return deltas
