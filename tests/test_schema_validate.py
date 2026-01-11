import pytest

from scenarioops.graph.tools.schema_validate import SchemaValidationError, validate_artifact


def test_charter_schema_accepts_valid_artifact() -> None:
    valid = {
        "id": "charter-001",
        "title": "ScenarioOps Pilot",
        "purpose": "Assess operational resilience for the next planning cycle.",
        "scope": "Global supply chain for Q3-Q4.",
        "time_horizon": "12 months",
        "stakeholders": ["Operations", "Finance"],
        "constraints": ["No headcount increase"],
        "assumptions": ["Stable demand"],
        "success_criteria": ["Decision-ready risk map"],
    }

    validate_artifact("charter", valid)


def test_charter_schema_rejects_invalid_artifact() -> None:
    invalid = {
        "id": "charter-002",
        "title": "Incomplete Charter"
    }

    with pytest.raises(SchemaValidationError) as exc:
        validate_artifact("charter", invalid)

    assert "required property" in str(exc.value)
