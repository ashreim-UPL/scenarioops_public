import pytest

from scenarioops.graph.gates.washout_gate import (
    WashoutGateConfig,
    WashoutGateError,
    assert_washout_pass,
)


def _force(force_id: int, domain: str, *, with_citations: bool = True) -> dict:
    return {
        "id": f"force-{force_id}",
        "name": f"{domain} force {force_id}",
        "domain": domain,
        "description": "Example description.",
        "why_it_matters": "It matters for planning.",
        "citations": [{"url": "https://example.com/a", "excerpt_hash": "hash"}]
        if with_citations
        else [],
    }


def _washout_report(duplicate_ratio: float, missing_categories: list[str] | None = None) -> dict:
    return {
        "duplicate_ratio": duplicate_ratio,
        "duplicate_groups": [],
        "undercovered_domains": [],
        "missing_categories": missing_categories or [],
        "proposed_forces": [],
    }


def test_washout_gate_passes() -> None:
    domains = [
        "political",
        "economic",
        "social",
        "technological",
        "environmental",
        "legal",
    ]
    forces = []
    force_id = 1
    for domain in domains:
        for _ in range(5):
            forces.append(_force(force_id, domain))
            force_id += 1
    report = _washout_report(duplicate_ratio=0.1)
    assert_washout_pass({"forces": forces}, report, WashoutGateConfig())


def test_washout_gate_fails_on_missing_citations() -> None:
    forces = [_force(1, "political", with_citations=False)]
    report = _washout_report(duplicate_ratio=0.1)
    with pytest.raises(WashoutGateError):
        assert_washout_pass({"forces": forces}, report, WashoutGateConfig(min_total_forces=1, min_per_domain=0))
