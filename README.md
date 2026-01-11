# ScenarioOps

ScenarioOps is a Python project scaffold for scenario planning and execution.

## Layout

- `scenarioops/` - Python package
- `app/` - application entrypoints
- `schemas/` - schema definitions
- `prompts/` - prompt templates
- `docs/` - documentation
- `data/` - local data assets
- `storage/runs/` - runtime outputs

## Development

Create a virtual environment and install dev tooling:

```sh
python -m venv venv
venv\Scripts\activate
pip install -e ".[dev]"
```

## UI

The Streamlit UI is installed with the base package requirements. Optional network
visualization support can be added with:

```sh
pip install -e ".[ui]"
```

Run the UI:

```sh
streamlit run ui/streamlit_app.py
```

Common tasks:

```sh
make format
make lint
make test
```
