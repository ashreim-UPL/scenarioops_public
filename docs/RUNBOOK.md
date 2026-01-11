# ScenarioOps Runbook

## Build Scenarios
1. Create a new run with core artifacts:

```sh
python -m scenarioops.app.main build-scenarios --scope country --value UAE --horizon 24
```

Outputs:
- `storage/runs/{run_id}/artifacts/*.json`
- `storage/runs/latest.json`

## Inject Strategies (Optional)
If you want to add strategies and run the wind tunnel:

```sh
python -m scenarioops.app.main add-strategies demo/demo_strategies.txt --run-id <run_id>
```

## Run Daily
Run the daily brief workflow:

```sh
python -m scenarioops.app.main run-daily --run-id <run_id>
```

Outputs:
- `storage/runs/{run_id}/artifacts/daily_brief.json`
- `storage/runs/{run_id}/artifacts/daily_brief.md`
- `storage/runs/latest.json`

## Export View Model
Bundle artifacts into a UI-friendly view model:

```sh
python -m scenarioops.app.main export-view --run-id <run_id>
```

Output:
- `storage/runs/{run_id}/artifacts/view_model.json`

## Scheduling
Example cron (daily at 06:00):

```sh
0 6 * * * /usr/bin/python -m scenarioops.app.main run-daily --run-id <run_id>
```

Cloud schedulers can trigger the same command with the repo environment activated.

## Interpreting Outputs
- Drivers: grouped by domain; check citations for source coverage.
- Uncertainties: focus on high impact and volatility.
- Scenario logic: confirm axes are distinct and map to four scenarios.
- EWIs: ensure each scenario has adequate measurable indicators.
- Daily brief: monitor deltas and narrative shifts over time.
