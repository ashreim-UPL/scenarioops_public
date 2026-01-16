# Verification Plan

## Objectives
Validate that ScenarioOps enforces the scenario-planning discipline, produces decision-ready scenario outputs, and emits traceable, reproducible evidence bundles.

## Acceptance criteria (pass/fail)
PASS only if all are true:
- All node outputs validate against their JSON Schemas.
- End-to-end fixtures reproduce identical artifact hashes across 3 consecutive runs.
- Coverage checks: every STEEP dimension and every study lens is covered.
- Scenario profiles include required fields and are differentiated (non-trivial narrative delta).
- Trace map links each scenario claim to upstream artifacts or explicit assumptions.
- Verification harness exits 0 and writes the evidence bundle.

FAIL if any are true:
- Any schema validation warning is ignored.
- Any node writes untracked output or bypasses artifact registry.
- Any scenario claim lacks provenance and is not marked as assumption.
- Any test depends on external network/API calls.

## Test matrix

### Unit tests
- Schema enforcement:
  - `tests/test_schema_validate.py` (charter validation)
  - `tests/test_verification_contracts.py::test_charter_rejects_extra_keys`
- Fail-closed enforcement:
  - `tests/test_verification_contracts.py::test_retrieval_fails_closed`
  - `tests/test_verification_contracts.py::test_scan_fails_on_unknown_citation`
- Deterministic normalization:
  - `tests/test_verification_contracts.py::test_stable_id_is_deterministic`
- Epistemic labeling:
  - `tests/test_verification_contracts.py::test_epistemic_summary_labels`

### Integration tests
- End-to-end fixtures:
  - `tests/test_fixtures_golden.py` (small + medium)
- Washout termination:
  - `tests/test_washout_gate.py` (max iteration guard)
- Scenario outputs:
  - `tests/test_nodes.py` (logic, skeletons, narratives, EWIs, strategies)

### Golden-run tests
- Canonical artifacts stored under:
  - `tests/fixtures/small/expected`
  - `tests/fixtures/medium/expected`
- Hash/structured diff comparison:
  - `tests/test_fixtures_golden.py`

## Fixtures
- Small fixture: 3 sources (example.com/a,b,c)
- Medium fixture: 5 sources (example.com/a-e)
- Inputs are defined in `tests/fixtures/<name>/inputs.json`.

## Verification harness
- Script: `scripts/verify_pipeline.py`
- Runs both fixtures with deterministic timestamps.
- Produces evidence bundle per run:
  - `runs/<run_id>/manifest.json`
  - `runs/<run_id>/artifacts/index.json`
  - `runs/<run_id>/trace/trace_map.json`
  - `runs/<run_id>/reports/verification_summary.md`
- Exits non-zero on any failure.

## Evidence bundle checks
- Manifest includes run metadata, schema/prompt hashes, node sequence.
- Artifact index includes schema validation status for each artifact.
- Trace map links scenario claims to upstream artifacts.
- Verification summary lists failing criterion, location, and fix.

## CI hook
No CI configuration found in repo; add a workflow later if CI is introduced.
