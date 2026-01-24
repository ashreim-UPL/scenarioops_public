# Configuration

ScenarioOps loads defaults from `config/scenarioops.yaml` and applies CLI or UI
overrides on top of that file. Keep the YAML as the source of truth for run
settings and use overrides for per-run changes.

## Model selection

You can select models for each task using the following settings or CLI flags:

- `llm_model` (`--llm-model`): general reasoning and synthesis.
- `search_model` (`--search-model`): web search queries.
- `summarizer_model` (`--summarizer-model`): evidence summarization.
- `embed_model` (`--embed-model`): embedding generation for the vector store.
- `image_model` (`--image-model`): scenario image generation.

Defaults point to the latest stable Gemini models in the config file. Unknown
model values are rejected at startup.

## Retrieval thresholds

Retrieval is graded at the run level:

- `min_evidence_ok`: minimum evidence units with `status == "ok"`.
- `min_evidence_total`: minimum total evidence units (including failed).
- `max_failed_ratio`: optional ratio of failed units; above this marks status
  as `partial`.

If `min_evidence_ok` is not met, retrieval stops with a failed report. Otherwise
the pipeline continues, even with partial failures.

## Retrieval and sources

- `allow_web`: enable network retrieval.
- `sources_policy`: allowlisted source policy for live runs.
- `simulate_evidence`: explicit opt-in to simulated evidence units.
- `min_sources_per_domain` and `min_citations_per_driver`: coverage thresholds.

## Ingestion and uploads

Document uploads are supported before retrieval:

- CLI: `scenarioops build-scenarios --input-docs <path1> <path2>`
- UI: upload panel on the run setup screen.

Uploaded documents are stored under `storage/runs/<run_id>/inputs/` and their
extracted text under `storage/runs/<run_id>/derived/`.

## Embeddings and vector store

Run-local embeddings are stored under `storage/runs/<run_id>/vectordb/`. The
`embed_model` setting controls the embedding backend.

## Environment

- `GEMINI_API_KEY`: required for live Gemini calls.
- `GEMINI_MAX_OUTPUT_TOKENS` / `GEMINI_JSON_MAX_OUTPUT_TOKENS`: optional output
  caps for Gemini text generation.
