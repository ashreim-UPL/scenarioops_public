from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Charter:
    id: str
    title: str
    purpose: str
    decision_context: str
    scope: str
    time_horizon: str
    stakeholders: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    success_criteria: list[str] = field(default_factory=list)
    notes: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class DriverEntry:
    id: str
    name: str
    description: str
    category: str
    trend: str
    impact: str
    evidence: list[str] = field(default_factory=list)
    signals: list[str] = field(default_factory=list)
    confidence: float | None = None
    citations: list[dict[str, str]] = field(default_factory=list)


@dataclass(frozen=True)
class Drivers:
    id: str
    title: str
    drivers: list[DriverEntry] = field(default_factory=list)


@dataclass(frozen=True)
class UncertaintyEntry:
    id: str
    name: str
    description: str
    extremes: list[str]
    driver_ids: list[str] = field(default_factory=list)
    criticality: int | None = None
    volatility: int | None = None
    implications: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Uncertainties:
    id: str
    title: str
    uncertainties: list[UncertaintyEntry] = field(default_factory=list)


@dataclass(frozen=True)
class ScenarioAxis:
    uncertainty_id: str
    low: str
    high: str
    rationale: str | None = None


@dataclass(frozen=True)
class ScenarioLogic:
    id: str
    name: str
    logic: str
    axis_assumptions: dict[str, str] = field(default_factory=dict)
    summary: str | None = None


@dataclass(frozen=True)
class Logic:
    id: str
    title: str
    axes: list[ScenarioAxis] = field(default_factory=list)
    scenarios: list[ScenarioLogic] = field(default_factory=list)


@dataclass(frozen=True)
class ScenarioEvent:
    date: str
    event: str
    implications: str | None = None


@dataclass(frozen=True)
class ScenarioSkeleton:
    id: str
    name: str
    narrative: str
    key_events: list[ScenarioEvent] = field(default_factory=list)
    operating_rules: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Skeleton:
    id: str
    title: str
    scenarios: list[ScenarioSkeleton] = field(default_factory=list)


@dataclass(frozen=True)
class EwiIndicator:
    id: str
    name: str
    description: str
    signal: str
    metric: str
    threshold: str | None = None
    monitoring_frequency: str | None = None
    linked_scenarios: list[str] = field(default_factory=list)
    unit: str | None = None


@dataclass(frozen=True)
class Ewi:
    id: str
    title: str
    indicators: list[EwiIndicator] = field(default_factory=list)


@dataclass(frozen=True)
class Strategy:
    id: str
    name: str
    objective: str
    actions: list[str] = field(default_factory=list)
    owners: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    fit: list[str] = field(default_factory=list)
    kpis: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Strategies:
    id: str
    title: str
    strategies: list[Strategy] = field(default_factory=list)


@dataclass(frozen=True)
class WindTunnelTest:
    id: str
    strategy_id: str
    scenario_id: str
    outcome: str
    failure_modes: list[str] = field(default_factory=list)
    adaptations: list[str] = field(default_factory=list)
    feasibility_score: float | None = None
    rubric_score: float | None = None
    action: str | None = None
    rating: int | None = None
    notes: str | None = None


@dataclass(frozen=True)
class WindTunnel:
    id: str
    title: str
    tests: list[WindTunnelTest] = field(default_factory=list)


@dataclass(frozen=True)
class DailyBrief:
    id: str
    date: str
    headline: str
    developments: list[str] = field(default_factory=list)
    signals: list[str] = field(default_factory=list)
    implications: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    notes: str | None = None
    markdown: str | None = None


@dataclass(frozen=True)
class AuditFinding:
    id: str
    finding: str
    evidence: list[str] = field(default_factory=list)
    impact: str | None = None
    recommendations: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class AuditReport:
    id: str
    period_start: str
    period_end: str
    summary: str
    findings: list[AuditFinding] = field(default_factory=list)
    lessons: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)


@dataclass
class ScenarioOpsState:
    charter: Charter | None = None
    focal_issue: dict[str, Any] | None = None
    company_profile: dict[str, Any] | None = None
    driving_forces: dict[str, Any] | None = None
    forces: dict[str, Any] | None = None
    forces_ranked: dict[str, Any] | None = None
    clusters: dict[str, Any] | None = None
    uncertainty_axes: dict[str, Any] | None = None
    scenarios: dict[str, Any] | None = None
    washout_report: dict[str, Any] | None = None
    evidence_units: dict[str, Any] | None = None
    certainty_uncertainty: dict[str, Any] | None = None
    belief_sets: dict[str, Any] | None = None
    effects: dict[str, Any] | None = None
    coverage_report: dict[str, Any] | None = None
    epistemic_summary: dict[str, Any] | None = None
    drivers: Drivers | None = None
    uncertainties: Uncertainties | None = None
    logic: Logic | None = None
    skeleton: Skeleton | None = None
    narratives: dict[str, str] | None = None
    ewi: Ewi | None = None
    strategies: Strategies | None = None
    wind_tunnel: WindTunnel | None = None
    scenario_profiles: dict[str, Any] | None = None
    trace_map: dict[str, Any] | None = None
    daily_brief: DailyBrief | None = None
    audit_report: AuditReport | None = None
