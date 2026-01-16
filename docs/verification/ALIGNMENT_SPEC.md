# Alignment Spec

## Scope and priority
- Scope: scenario planning process and outputs only.
- Priority: Accuracy > Traceability > Reproducibility > Coverage > Performance.

## Graph discipline (process alignment)
The graph must execute the following bounded nodes with explicit input/output contracts and schema validation:

- charter -> `scenario_charter.json`, `scenario_charter_raw_prevalidate.json` (schema: `charter`)
- focal_issue -> `focal_issue.json` (schema: `focal_issue.schema`)
- retrieval -> `evidence_units.json` (schema: `evidence_units.schema`)
- scan -> `driving_forces.json` (schema: `driving_forces.schema`)
- washout -> `washout_report.json` (schema: `washout_report.schema`)
- coverage -> `coverage_report.json` (schema: `coverage_report`)
- classify -> `certainty_uncertainty.json` (schema: `certainty_uncertainty.schema`)
- beliefs -> `belief_sets.json` (schema: `belief_sets.schema`)
- effects -> `effects.json` (schema: `effects.schema`)
- epistemic_summary -> `epistemic_summary.json` (schema: `epistemic_summary`)
- drivers -> `drivers.jsonl` (schema: `driver_entry`)
- uncertainties -> `uncertainties.json` (schema: `uncertainties`)
- logic -> `logic.json` (schema: `logic`)
- skeletons -> `skeletons.json` (schema: `skeleton`)
- narratives -> `narrative_<scenario>.md` (schema: `markdown`)
- ewis -> `ewi.json` (schema: `ewi`)
- strategies -> `strategies.json` (schema: `strategies`) when enabled
- wind_tunnel -> `wind_tunnel.json` (schema: `wind_tunnel`) when strategies exist
- daily_runner -> `daily_brief.json`, `daily_brief.md` (schemas: `daily_brief`, `markdown`) when wind tunnel exists
- scenario_profiles -> `scenario_profiles.json` (schema: `scenario_profiles`)
- trace_map -> `trace/trace_map.json` (schema: `trace_map`)
- auditor -> `audit_report.json` (schema: `audit_report`)

Nodes are fail-closed: missing inputs, invalid citations, schema violations, or missing coverage halt the run.

## Charter contract
A valid charter must include:
- purpose (why)
- decision_context (decision to support)
- scope (domain/geography)
- constraints
- assumptions
- success_criteria
- stakeholders
- time_horizon

All fields are required by schema. Any omission is a hard failure.

## Epistemic alignment
Explicit epistemic labeling is produced by `epistemic_summary`:
- facts: `certainty_uncertainty.predetermined_elements`
- unknowns: `certainty_uncertainty.uncertainties`
- assumptions: `belief_sets` (dominant/counter, grouped by uncertainty_id)
- interpretations: `effects` linked to belief_id

Confidence rules (explicit in `epistemic_summary.confidence_rules`):
- facts: 0.8 (predetermined_element)
- assumptions: 0.5 (belief_statement)
- interpretations: 0.4 (effect_chain)
- unknowns: 1 - uncertainty

Conflicts are preserved by retaining both dominant and counter beliefs per uncertainty.

## Coverage alignment
Coverage is enforced by `coverage_report` and must be fully covered:
- STEEP dimensions: political, economic, social, technological, environmental, legal
- Study lenses: macro, law, geopolitics, ethics, culture

Missing dimensions or lenses cause a hard failure.

## Outcome alignment (scenario quality)
Scenario outputs are consolidated in `scenario_profiles`, each containing:
- drivers (with citations and confidence)
- assumptions (dominant/counter beliefs)
- unknowns (uncertainties)
- causal_chain (effects ordered by belief link)
- signals/indicators (EWIs)
- risks/opportunities (wind tunnel failure_modes/adaptations)
- implications (operating rules and key events)
- options (wind tunnel tests per strategy)
- what_would_change_my_mind (indicator triggers)
- epistemic_refs (facts/assumptions/interpretations/unknowns)

Scenarios must be differentiated by narrative and decision hooks.

## Governance and traceability
Every run must emit:
- Run Manifest: `runs/<run_id>/manifest.json`
- Artifact Index: `runs/<run_id>/artifacts/index.json`
- Trace Map: `runs/<run_id>/trace/trace_map.json`
- Verification Summary: `runs/<run_id>/reports/verification_summary.md`

Traceability rules:
- Artifact index enumerates all artifacts and validation status.
- Trace map links scenario claims to upstream artifacts or explicit assumptions.
- Node logs include input references, output references, schema validation status, and timing.

## Determinism and normalization
- Stable IDs are derived via deterministic hashing when missing.
- JSON serialization is canonicalized (sorted keys).
- Run timestamp is fixed for a run and used in provenance; run_id is treated as an explicit input for reproducibility.
- Any normalization is logged (`logs/normalization.jsonl`).
