# ScenarioOps Demo Script (3 minutes)

## 0:00-0:30 Build Scenarios
Command:

```sh
python -m scenarioops.app.main build-scenarios --scope country --value UAE --horizon 24 --run-id demo-run
```

Say: "We generate the base scenario artifacts and set a fixed run ID for repeatability."

## 0:30-1:00 Driving Forces Map
Command:

```sh
streamlit run ui/streamlit_app.py
```

Click:
- Open the Driving Forces tab.
- Expand a domain cluster and point to citations.
- Show the word map for dominant keywords.

Say: "Drivers are grouped by domain and surfaced as a weighted word map."

## 1:00-1:30 Critical Uncertainties
Click:
- Open the Critical Uncertainties tab.
- Point to axis uncertainties highlighted in the scatter.

Say: "Impact and uncertainty are plotted together, with axis uncertainties emphasized."

## 1:30-2:00 Scenario Logic (2x2)
Click:
- Open the Scenario Logic (2x2) tab.
- Walk through the axis poles and the grid placement.

Say: "The 2x2 grid aligns scenarios to the two critical axes."

## 2:00-2:30 Scenario Cards
Click:
- Open the Scenarios tab.
- Show premise bullets, operating rules, and top EWIs.
- Expand one narrative panel.

Say: "Scenario cards summarize premises, rules, and leading indicators."

## 2:30-3:00 Daily Update + Brief
Command:

```sh
python -m scenarioops.app.main run-daily --run-id demo-run
```

Click:
- Open the Daily Brief tab.
- Show the updated brief contents.

Say: "Daily updates produce a brief and keep the run pointer current."
