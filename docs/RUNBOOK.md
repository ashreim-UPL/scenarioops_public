# ScenarioOps Runbook

## Install
```sh
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

For fully pinned installs:

```sh
pip install -r requirements.lock
```

## Run the API + UI
```sh
uvicorn scenarioops.app.api:app --host 0.0.0.0 --port 8502
streamlit run src/scenarioops/ui/streamlit_app.py
```

## Build Scenarios
1. Create a new run with core artifacts:

```sh
scenarioops build-scenarios --scope country --value UAE --horizon 24
```

Outputs:
- `storage/runs/{run_id}/artifacts/*.json`
- `storage/runs/latest.json`

## Inject Strategies (Optional)
If you want to add strategies and run the wind tunnel:

```sh
$RUN_ID="RUN_ID_HERE"
scenarioops add-strategies demo/demo_strategies.txt --run-id $RUN_ID
```

## Run Daily
Run the daily brief workflow:

```sh
scenarioops run-daily --run-id $RUN_ID
```

Outputs:
- `storage/runs/{run_id}/artifacts/daily_brief.json`
- `storage/runs/{run_id}/artifacts/daily_brief.md`
- `storage/runs/latest.json`

## Export View Model
Bundle artifacts into a UI-friendly view model:

```sh
scenarioops export-view --run-id $RUN_ID
```

Output:
- `storage/runs/{run_id}/artifacts/view_model.json`

## Scheduling
Example cron (daily at 06:00):

```sh
0 6 * * * /usr/bin/scenarioops run-daily --run-id RUN_ID_HERE
```

Cloud schedulers can trigger the same command with the repo environment activated.

## Interpreting Outputs
- Drivers: grouped by domain; check citations for source coverage.
- Uncertainties: focus on high impact and volatility.
- Scenario logic: confirm axes are distinct and map to four scenarios.
- EWIs: ensure each scenario has adequate measurable indicators.
- Daily brief: monitor deltas and narrative shifts over time.
