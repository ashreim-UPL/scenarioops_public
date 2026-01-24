# Migration: Retrieval v2 (2026-01)

This release upgrades evidence retrieval to v2 and introduces graded retrieval
outcomes. Evidence units now carry summaries and explicit status fields instead
of raw excerpt slices.

## What changed

- Evidence units schema is now v2 (`schema_version: "2.0"`).
- `summary` and `claims` are required when `status == "ok"`.
- New fields: `status`, `failure_reason`, `content_type`, `http_status`,
  `source_method`, `embedding_ref`, `tags`, `metrics`.
- `url` can be `null` for uploaded documents (use `file_name`/`sha256`).
- Retrieval produces `retrieval_report.json` with thresholds and recovery notes.
- Partial failures no longer stop the pipeline unless thresholds are missed.

## Behavioral changes

- Evidence is summarized by the LLM; raw excerpts are optional.
- Retrieval continues with partial failures when thresholds are satisfied.
- Run-local vector store (`vectordb/`) stores embeddings for evidence summaries.

## Migration steps

1. Update any downstream validation to accept `status`, `failure_reason`, and
   `summary`/`claims` requirements for ok units.
2. If you persist old evidence units, map them into the v2 shape:
   - `schema_version` -> `"2.0"`.
   - `status` -> `"ok"`.
   - `summary` -> existing `excerpt` (or a short fallback summary).
   - `claims` -> array with at least one claim (use the summary if needed).
   - `metrics`, `tags` -> empty arrays when unavailable.
   - `content_type`, `http_status`, `embedding_ref` -> `null` defaults.
   - `source_method` -> `"unknown"` or `"upload"`/`"source_url"` as appropriate.
3. Handle `retrieval_report.json` in run consumers if you surface retrieval
   status or diagnostics.

## Backward compatibility

Retrieval will attempt to backfill missing `summary` and `claims` for any
pre-existing evidence units loaded from cache or artifacts, but full v2
compliance is recommended for downstream tools.
