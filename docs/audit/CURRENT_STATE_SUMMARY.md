# Current State Summary & Audit Report

**Date:** 2026-01-16
**Auditor:** Gemini Agent

## 1. Functional Summary
The `ScenarioOps` system is a scenario planning framework that orchestrates a pipeline of "nodes" to generate strategic scenarios from inputs (user parameters, sources, signals). 

**Key Workflows:**
*   **Build Scenarios:** Takes a scope (Country/Region) and sources, retrieves evidence, scans for drivers (PESTEL), clusters them into uncertainties, builds scenario logic (2x2 matrices), and generates narratives.
*   **Daily Runner:** Monitors signals (Early Warning Indicators) against established scenarios.
*   **Strategies & Wind Tunnel:** Generates strategies and tests them against the scenarios.

The core execution engine uses a graph-based approach (`run_graph`) where state is passed and mutated through a sequence of functional nodes.

## 2. Logic Mapping

### Entry Points
*   **CLI:** `scenarioops/__main__.py` redirects to `scenarioops/app/main.py`. Handles commands like `build-scenarios`, `run-daily`, `verify`.
*   **API:** `scenarioops/app/api.py` (FastAPI). Exposes endpoints `/build`, `/strategies`, `/daily`.
*   **UI:** `ui/streamlit_app.py` (Streamlit). Provides an interactive dashboard to trigger runs and visualize artifacts.

### Orchestration
*   **Core Engine:** `scenarioops/graph/build_graph.py`. The `run_graph` function is the primary orchestrator. It explicitly calls node functions (e.g., `run_charter_node`, `run_scan_node`) in a defined sequence.
*   **State Management:** `scenarioops/graph/state.py` defines `ScenarioOpsState`, which accumulates artifacts across the pipeline.
*   **Workflow Utils:** `scenarioops/app/workflow.py` handles run persistence, retrieving latest runs, and state hydration.

## 3. Dependency Review

**External Libraries (Production):**
*   `fastapi`, `uvicorn`: API server.
*   `streamlit`: UI dashboard.
*   `pandas`, `plotly`: Data manipulation and visualization.
*   `jsonschema`: Validation.
*   `requests`: HTTP client (likely for retrieval).
*   `pyvis`: Network graph visualization (optional UI dependency).

**Internal Modules:**
*   `scenarioops.app`: Configuration, API, CLI entry.
*   `scenarioops.graph`: Core business logic, nodes, gates, guards.
*   `scenarioops.llm`: LLM client abstraction.
*   `scenarioops.sources`: Source policies.

## 4. Codex Compliance Audit

### C1: Structural Integrity
*   **Status:** 🔴 **FAIL**
*   **Findings:**
    *   **Layout:** The source code resides in `scenarioops/` at the root, violating the C1 requirement for a `/src/` directory.
    *   **Layering:** The `domain` logic (conceptually in `graph/nodes`) is not pure. Nodes import `llm_client` and perform I/O (invoking LLMs), mixing infrastructure with domain logic.
    *   **Ghost Files:** `ui/streamlit_app.py` exists outside the package structure but is a key entry point.

### C2: Logic & Safety
*   **Status:** 🟡 **PARTIAL**
*   **Findings:**
    *   **Safety Gating:** Input validation uses Pydantic in `api.py` and Dataclasses in `build_graph.py`. `GraphInputs` provides some structure.
    *   **Deterministic Execution:** Randomness is managed in some areas (e.g., `streamlit_app.py` sets seed), but LLM calls are inherently non-deterministic without strict seeding/temperature controls (not fully verified in deep inspection).
    *   **Fail Fast:** `run_graph` uses broad `try...except Exception` blocks, which might mask specific errors, though it does log the error summary.

### C3: Observability & Reliability
*   **Status:** 🔴 **FAIL**
*   **Findings:**
    *   **Logging:** The codebase uses `print()` statements extensively in `scenarioops/app/main.py` (e.g., `print(json.dumps(...))`, `print(f"verify failed: ...")`). This violates the "No `print()` statements in production code" rule.
    *   **Metrics:** While `log_node_event` captures duration, explicit RED (Rate, Error, Duration) metrics emission for external monitoring is not clearly standardized across all paths.

### C4: Performance & Cost
*   **Status:** ⚪ **UNKNOWN** (Requires runtime profiling)
*   **Findings:**
    *   Batching and concurrency limits are not explicitly visible in the high-level orchestration logic reviewed so far.

## Recommendations
1.  **Refactor Directory Structure:** Move `scenarioops/` into `src/scenarioops/`.
2.  **Purify Domain:** Decouple nodes from direct `LLMClient` usage. Pass "decisions" or "content" into domain entities, or use strict interfaces (Ports & Adapters) where the domain requests information without knowing the provider.
3.  **Replace Print with Logger:** Replace all `print()` calls with a structured JSON logger (e.g., `structlog` or standard `logging` configured for JSON).
4.  **Standardize Error Handling:** Remove broad catch blocks where possible or ensure they re-raise after logging structured error events.
