# Run Artifacts

All run artifacts live under `storage/runs/<run_id>/`. The most important
entries are listed below.

## Core artifacts

- `run_config.json`: resolved settings for the run.
- `evidence_units.json`: evidence units v2 with summary, claims, status, and
  provenance fields.
- `evidence_units_uploads.json`: evidence units created from uploaded documents.
- `retrieval_report.json`: retrieval grading, thresholds, and recovery details.
- `company_profile.json`: company profile enriched with annual report summary.
- `forces.json`, `drivers.jsonl`, `driving_forces.json`: PESTEL synthesis outputs.
- `scenarios.json`: synthesized scenarios (narrative + touchpoints).
- `scenarios_enriched.json`: story text, visual prompts, and image paths.
- `view_model.json`: UI-ready aggregated view of run outputs.

## Media and storage

- `artifacts/images/`: scenario images (one per scenario).
- `inputs/`: uploaded documents (original files).
- `derived/`: extracted text from uploads and annual reports.
- `vectordb/`: run-local vector store for embeddings.

## Indexes and provenance

- `index.json`: artifact index with schemas and metadata.
- `*.meta.json`: provenance metadata for each artifact.
- `logs/` and `trace/`: step logs and trace maps.
