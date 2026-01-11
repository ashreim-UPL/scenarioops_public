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

Common tasks:

```sh
make format
make lint
make test
```
