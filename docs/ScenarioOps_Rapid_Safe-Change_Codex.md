ScenarioOps: Rapid Safe-Change Codex
30-Minute Code Modifications Without Functional Risk

Version: 1.1
Target: Safe, production-compatible micro-changes
Estimated Time: 30 minutes or less (single focused PR)
Risk Level: VERY LOW (UI consolidation + internal refactor only, behavior preserved)

What Changed vs v1.0 (Critical Corrections)
The original guide is a multi-week refactor plan. That is not compatible with "done by code in 30 minutes." Therefore:

- Keep existing endpoints and existing UI behavior intact.
- Make small, reversible changes that remove redundancy without breaking anything.
- No framework migrations (no FastAPI router split, no Vue rewrite, no Redis, no Postgres pool).
- No async conversions, no auth redesign, no caching layers, no deployment rebuilds.
- Focus only on redundant Run controls and duplicate run listing UI, using thin wrappers.

Hard Rule: "No Functional Change" Definition
A change is allowed only if it satisfies all:

- Same API calls happen for the same user actions (or a strict superset that is harmless).
- Same run selection result for Latest / Load / Rerun as before.
- Same DOM-visible outputs except: removed duplicate controls OR consolidated into one control.
- Same file paths and run IDs used.
- No changes to auth, storage layout, run JSON structure, workflow logic.

If any condition is violated, the change is out of scope for the 30-minute patch.

30-Minute Patch Scope
A) UI: Replace redundant buttons with a single "Run Actions" control
Goal: keep behavior, reduce duplication.

Before:
- Load Run
- Latest
- Rerun
- Recent runs list
- Run library dropdown (duplicates list)

After (safe consolidation):
- One Run Selector (dropdown)
- One Action button group:
  - Load (loads selected)
  - Latest (sets dropdown to latest and loads)
  - Rerun (reruns selected)
- Optional: show "Recent runs" as shortcuts but driven from the same data source as the dropdown.

Key point: you are not removing capabilities, only consolidating UI entry points.

B) Backend: NO structural refactor
Do not introduce:
- /api/v1 router split
- New dependency injection modules
- DB pooling
- Redis caching
- Auth hashing changes

Allowed backend changes (only if needed to support UI consolidation):
- Add one lightweight endpoint alias that calls existing logic (optional).
- Or better: do not touch backend. Change frontend to call existing endpoints.

Implementation Steps (Designed for 30 Minutes)
Step 1 - Identify the "source of truth" functions (2-5 min)
Find the existing JS functions in your current UI (likely in ui.html / commercial_ui.html):
- loadRun(runId)
- loadLatest()
- rerun(runId) or similar
- fetchRuns() or similar

Rule: do not change these functions' internal behavior. Only wrap them.

Step 2 - Create one wrapper module (5-10 min)
Create a small file (or inline script block) like:
- frontend/legacy/runActions.js (or equivalent in current structure)

It exposes:
- selectRun(runId)
- actionLoadSelected()
- actionLoadLatest()
- actionRerunSelected()
- getRunsModel() (returns the same data used by dropdown + recent list)

Rule: wrapper functions must call the original handlers.

Step 3 - Replace duplicate UI wiring (10-15 min)
- Keep existing HTML/CSS.
- Remove duplicate buttons OR hide them via a single container switch.
- Wire new consolidated controls to wrapper functions.
- Do not change styling logic beyond moving blocks around.

Step 4 - Minimal regression checks (5 min)
Manual checks (fast, deterministic):
- Latest loads the same run as before.
- Selecting a run + Load loads the same run as before.
- Selecting a run + Rerun triggers the same rerun behavior as before.
- The run dropdown and recent runs show the same list ordering as before.
- No console errors.

Codex Safety Guardrails (Mandatory)
1) Feature flag for instant rollback (UI-only)
Add a constant toggle:

const CONSOLIDATED_RUN_UI = true;

If false:
- show old buttons as-is
- bypass new wrapper wiring

This makes rollback a 5-second change.

2) No API Contract Changes
- Do not rename endpoints.
- Do not change response schemas.
- Do not introduce /api/v1 unless it already exists.
- Do not change auth headers, cookies, or CSRF behavior.

3) No Workflow Changes
- Do not change run directory structure.
- Do not change run JSON keys.
- Do not change run ID generation.
- Do not change how "latest" is computed.

Future Enhancements (Out of Scope for 30-Min Patch)
- Modular backend routers
- SPA migration
- Caching
- Auth hardening
- Monitoring
- CI/CD + tests

30-Minute Checklist
- Identify existing run functions (load/latest/rerun/list)
- Add wrapper module (no logic changes)
- Consolidate UI controls into one panel
- Keep old UI behind feature flag
- Run 5 manual regression checks
