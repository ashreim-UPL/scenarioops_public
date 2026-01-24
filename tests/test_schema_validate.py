import pytest

from scenarioops.graph.tools.schema_validate import SchemaValidationError, validate_artifact


def test_charter_schema_accepts_valid_artifact() -> None:
    valid = {
        "id": "charter-001",
        "title": "ScenarioOps Pilot",
        "purpose": "Assess operational resilience for the next planning cycle.",
        "decision_context": "Capital allocation for resilience investments.",
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


def test_evidence_units_schema_accepts_ok_unit() -> None:
    payload = {
        "run_id": "run-1",
        "timestamp": "2026-01-01T00:00:00+00:00",
        "company_name": "Acme",
        "geography": "Global",
        "horizon_months": 12,
        "schema_version": "2.0",
        "simulated": False,
        "evidence_units": [
            {
                "id": "ev-1",
                "url": "https://example.com/a",
                "title": "Example Source",
                "status": "ok",
                "failure_reason": None,
                "summary": "Evidence summary for validation.",
                "claims": ["Evidence claim one."],
                "metrics": ["12%"],
                "tags": ["test"],
                "retrieved_at": "2026-01-01T00:00:00+00:00",
                "content_type": "text/html",
                "http_status": 200,
                "source_method": "source_url",
                "embedding_ref": None,
            }
        ],
    }

    validate_artifact("evidence_units.schema", payload)


def test_evidence_units_schema_accepts_failed_unit_without_summary() -> None:
    payload = {
        "run_id": "run-2",
        "timestamp": "2026-01-01T00:00:00+00:00",
        "company_name": "Acme",
        "geography": "Global",
        "horizon_months": 12,
        "schema_version": "2.0",
        "simulated": False,
        "evidence_units": [
            {
                "id": "ev-2",
                "url": "https://example.com/b",
                "title": "Broken Source",
                "status": "failed",
                "failure_reason": "empty_extracted_text",
                "retrieved_at": "2026-01-01T00:00:00+00:00",
                "content_type": None,
                "http_status": None,
                "source_method": "source_url",
                "embedding_ref": None,
            }
        ],
    }

    validate_artifact("evidence_units.schema", payload)
