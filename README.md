# ScenarioOps

ScenarioOps is a Python project scaffold for scenario planning and execution with
retrieval-first evidence collection, scenario synthesis, and run-local provenance.

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

## Configuration

ScenarioOps reads settings from `config/scenarioops.yaml`. UI and CLI overrides apply on top
of that file so mode and source policies stay explicit without relying on environment variables.
See `docs/config.md` for model selection, retrieval thresholds, and ingestion options.

Key settings:

- `mode`: `demo` or `live`
- `sources_policy`: `fixtures`, `academic_only`, or `mixed_reputable`
- `allow_web`: allow network retrieval when using live sources
- `min_sources_per_domain`: minimum evidence sources required in live runs
- `min_citations_per_driver`: minimum citations per driver in live runs
- `llm_model`, `search_model`, `summarizer_model`, `embed_model`, `image_model`: model selection
- `min_evidence_ok`, `min_evidence_total`, `max_failed_ratio`: retrieval grading thresholds

Gemini API keys are loaded from Streamlit secrets at runtime. Use
`.streamlit/secrets.example.toml` as a template.

## Data provenance

Each run writes provenance artifacts into `storage/runs/<run_id>/`:

- `run_config.json` captures the resolved settings for the run.
- `evidence_units.json` contains structured evidence with summaries, claims, and status.
- `retrieval_report.json` captures evidence grading and recovery decisions.
- `driving_forces.json` and `drivers.jsonl` cite evidence by URL, publisher, and evidence id.
- `scenarios_enriched.json` includes story text, visual prompts, and image paths.
- `artifacts/images/` stores generated scenario images.
- `vectordb/` stores the run-local embedding index.
- `inputs/` and `derived/` store uploaded documents and extracted text.
- `latest.json` mirrors run status and the resolved run configuration.

Live runs enforce reputable sources and fail fast if fixture markers are detected.
See `docs/artifacts.md` for the full artifact index.

Common tasks:

```sh
make format
make lint
make test
```
