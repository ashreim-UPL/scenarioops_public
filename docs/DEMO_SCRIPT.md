# ScenarioOps Demo Script (3 minutes)

## Scene 1 (0:00–0:20) — Setup

Command:

```sh
python -m scenarioops.app.main build-scenarios --scope country --value UAE --horizon 24 --run-id demo-run
```

Say: "We generate scenario fundamentals with a fixed run ID so the demo is repeatable."

## Scene 2 (0:20–0:45) — Strategies + Wind Tunnel

Command:

```sh
python -m scenarioops.app.main add-strategies demo/demo_strategies.txt --run-id demo-run
```

Say: "Strategies are normalized, then stress-tested in the wind tunnel."

## Scene 3 (0:45–1:05) — Daily Runner

Command:

```sh
python -m scenarioops.app.main run-daily --run-id demo-run
```

Say: "Daily runner computes activation and deltas from EWIs."

## Scene 4 (1:05–1:45) — UI: Scenario Cards

Command:

```sh
uvicorn app.api:app --reload
```

Click:

1. Open `http://127.0.0.1:8000/`
2. Click any scenario card to expand the narrative detail.

Say: "Each card shows the 2x2 scenario, operating rules, and narrative."

## Scene 5 (1:45–2:20) — UI: Gauges + Keep/Modify/Drop

Click:

1. Scroll to "Activation Gauges" and point at before/after markers.
2. Scroll to "Keep / Modify / Drop" to show actions by scenario.

Say: "Activation bands show the shift from yesterday to today, and wind tunnel actions are grouped for fast decisions."

## Scene 6 (2:20–3:00) — UI: Daily Delta Brief

Click:

1. Scroll to "Daily Delta Brief".
2. Read one activation delta and one wind tunnel delta.

Say: "The brief summarizes activation movement alongside strategic stress-test changes."
