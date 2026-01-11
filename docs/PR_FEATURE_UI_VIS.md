# PR: Feature UI Visualizations

## Summary
- Added latest run pointer with command/status metadata.
- Introduced view_model.json export for UI consumption.
- Upgraded Streamlit UI with multi-tab scenario visualizations.
- Added verify demo pipeline and new documentation.

## How to Run
```sh
python -m scenarioops.app.main build-scenarios --scope country --value UAE --horizon 24
python -m scenarioops.app.main export-view
python -m scenarioops.app.main run-daily
streamlit run ui/streamlit_app.py
```

## Screenshots
- Overview tab: [screenshot]
- Driving Forces tab: [screenshot]
- Critical Uncertainties tab: [screenshot]
- Brainstorm Map tab: [screenshot]
- Scenario Logic tab: [screenshot]
- Scenarios tab: [screenshot]
- Daily Brief tab: [screenshot]

## Known Limitations
- Scenario winners/losers are placeholders unless provided by upstream data.
- Network graph renders only if `pyvis` is installed.
- Some tabs rely on mock data when artifacts are missing.
