# UI Spec

## Goals
- Make reasoning visible across ScenarioOps outputs with clear, stable visuals.
- Aggregate run artifacts into a single view model for UI consumption.
- Keep the UI resilient when artifacts are missing or incomplete.
- Ensure deterministic layouts (fixed random seed, stable ordering).

## Data Inputs
Primary: `storage/runs/{run_id}/artifacts/view_model.json`

Fallbacks (if view model missing):
- `scenario_charter.json`
- `drivers.jsonl`
- `uncertainties.json`
- `logic.json`
- `skeletons.json`
- `narrative_{scenario_id}.md`
- `ewi.json`
- `daily_brief.md`
- `storage/runs/latest.json`

Expected view model fields:
- `charter` (object)
- `drivers` (list)
- `drivers_by_domain` (object of lists)
- `uncertainties` (list)
- `scenario_logic` (object)
- `scenarios` (list)
- `narratives` (list of `{scenario_id, markdown}`)
- `ewis` (list)
- `daily_brief_md` (string or null)
- `run_meta` (object with `run_id`, `status`)

## Tabs (Order)
1. Overview
2. Driving Forces
3. Critical Uncertainties
4. Brainstorm Map (EBE)
5. Scenario Logic (2x2)
6. Scenarios
7. Daily Brief

### Overview
- Charter summary: scope, value (title), horizon, domains.
- Counts: drivers, uncertainties, scenarios, EWIs.
- Last updated timestamp from `latest.json`.

### Driving Forces
- Clustered list: drivers grouped by domain (category), shown in expanders.
- Word map: keywords extracted from driver text, stopwords removed, weighted by confidence.
- Optional network graph if `pyvis` is installed.

### Critical Uncertainties
- Plotly scatter: x = uncertainty score (volatility), y = impact (criticality).
- Highlight axis uncertainties referenced by scenario logic axes.

### Brainstorm Map (EBE)
- For top uncertainties:
  - Evidence: supporting drivers + citations.
  - Beliefs: uncertainty description + extremes.
  - Effects: scenario logic summaries.
- Raw JSON view for each uncertainty in an expander.

### Scenario Logic (2x2)
- Axis A and B definitions with low/high poles.
- 2x2 grid of scenarios with one-line premise.

### Scenarios
- Cards display name, premise bullets, operating rules, winners/losers, top 3 EWIs.
- Narrative shown in collapsed expander.

### Daily Brief
- Render `daily_brief.md` if present.
- If missing, show "No daily brief yet" and a Run Daily Update button.

## Fallback Behavior
- If no runs exist: show empty state, keep UI functional.
- If artifacts missing: show "No data yet" placeholders.
- If `view_model.json` missing: build view model from raw artifacts at runtime.

## Error States
- Latest run status FAIL: show red banner and `error_summary`.
- Missing files or malformed JSON: skip the artifact and continue rendering.
