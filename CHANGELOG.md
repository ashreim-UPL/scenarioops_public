# Changelog

## 2026-01

### Breaking/Behavior Changes
- Evidence units are now v2 with `summary` and `claims` required for ok status,
  plus new status fields (`ok`/`partial`/`failed`) and retrieval metadata.
- Retrieval emits `retrieval_report.json` and continues on partial evidence when
  thresholds are met.
- Vector store is now run-local (`vectordb/`) with embeddings for evidence
  summaries.
- Scenario media adds story text and images to `scenarios_enriched.json`.
- Model selection is configurable per task (`llm_model`, `search_model`,
  `summarizer_model`, `embed_model`, `image_model`).

### Added
- Pre-run document ingestion with `inputs/` and `derived/` artifacts.
- Annual report fetching and summaries in `company_profile.json`.
- Scenario image generation with artifacts under `artifacts/images/`.
- Retrieval recovery logic with replacement sources.

### Fixed
- Scenario synthesis now normalizes `scenario_name` into `name` and drops unexpected keys before schema validation.
- Search URL extraction skips Google grounding redirect links that frequently return 403s.

### Migration Notes
- See `docs/migrations/2026-01-retrieval-v2.md` for evidence schema upgrades.
