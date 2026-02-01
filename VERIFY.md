# ScenarioOps Verification (Stability & Judge-Ready Hardening)

## Local Run
1) Install deps:
   - `pip install -r requirements.txt`
2) Start API:
   - `uvicorn scenarioops.app.api:app --reload --port 8502`
3) Optional API/Auth:
   - `export SCENARIOOPS_AUTH_REQUIRED=1`
   - `export SCENARIOOPS_DEFAULT_API_KEY=...`
4) Optional Gemini server key:
   - `export GEMINI_API_KEY=...`
5) Verify:
   - `python scripts/verify_run_store.py`

## Cloud Run Env Vars
Required (server key mode):
- `GEMINI_API_KEY`

Optional (storage durability):
- `RUN_STORE=gcs`
- `GCS_BUCKET=<bucket>`
- `RUNS_PREFIX=scenarioforge/runs`
- `VECTORDB_PREFIX=scenarioforge/vectordb`
- `CACHE_PREFIX=scenarioforge/cache`

Optional (auth):
- `SCENARIOOPS_AUTH_REQUIRED=1`
- `SCENARIOOPS_DEFAULT_API_KEY=...`
- `SCENARIOOPS_DEFAULT_TENANT=public`

Optional (retry tuning):
- `SCENARIOOPS_RETRY_ENABLED=true`
- `SCENARIOOPS_RETRY_MAX=3`
- `SCENARIOOPS_RETRY_BASE=1.5`
- `SCENARIOOPS_RETRY_CAP=12`
- `SCENARIOOPS_RETRY_JITTER=0.4`

Optional (vector DB):
- `SCENARIOOPS_VECTORDB_SCOPE=global|run`
- `SCENARIOOPS_VECTORDB_REQUIRED=false`

## Verification Checklist
- View existing runs without Access key.
- Launch a run using:
  - server key only (no UI key)
  - UI key only (no server key)
- Runs load after container restart (GCS mode).
- Node status shows last attempt + cached outputs on failure.
- Vector DB failures do not break pipeline.

## Health Endpoint
- `GET /health` returns:
  - `run_store`
  - `vector_db`
  - `api_key_mode` (user/server/none)
