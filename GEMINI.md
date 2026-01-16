\# ScenarioOps: Project Intelligence \& Codex Guidelines

\*\*Version:\*\* 1.1 (Agent-Enhanced)

\*\*Status:\*\* Enforced



\## 1. Project Mission \& Context

ScenarioOps is a Python-based framework designed for high-stakes scenario planning and execution (Telecom, Construction, Data Centers). It leverages AI-driven automation and Model Context Protocol (MCP) to provide deterministic outcomes in complex enterprise environments.



\## 2. Agentic Directives (The "Gemini Rules")

When operating within this repository, the AI Agent must:

\* \*\*Audit Against Codex:\*\* Every analysis must flag deviations from the "Unified Engineering Standard" (Section 3).

\* \*\*Root-Cause Prioritization:\*\* Do not suggest "patches." Identify the underlying architectural flaw or "ghost" logic.

\* \*\*Production-Ready Output:\*\* All code generated must be standards-compliant, typed, and include observability hooks.



---



\## 3. The Codex: Unified Engineering Standard



\### C1: Structural Integrity

\* \*\*Standard Layout:\*\* `/src/` (Logic), `/tests/` (Mirroring src), `/docs/` (ADRs \& Runbooks), `/scripts/` (Dev Utils).

\* \*\*Layered Architecture:\*\* `interfaces` → `application` → `domain`. 

&nbsp;   \* \*Rule:\* The `domain` layer must remain "pure" (zero I/O or infrastructure imports).

\* \*\*Ghost File Policy:\*\* Any file not explicitly imported or referenced in the `main` execution path or tests is considered "technical debt" and must be flagged for deletion.



\### C2: Logic \& Safety

\* \*\*Deterministic Execution:\*\* Core functions must be idempotent. Randomness (AI outputs) must be wrapped in a controlled seed or validation layer.

\* \*\*Fail Fast, Fail Loud:\*\* No `try-except: pass`. Errors must carry an `operation\_id` and actionable context.

\* \*\*Safety Gating:\*\* Input validation at all boundaries (APIs, CLI, Worker entry) is non-negotiable.



\### C3: Observability \& Reliability

\* \*\*Structured Logging:\*\* JSON logs only. No `print()` statements in production code.

\* \*\*Metrics:\*\* Every critical path must emit Rate, Error, and Duration (RED) signals.

\* \*\*Resiliency:\*\* External calls require: 1) Timeouts, 2) Exponential Backoff, 3) Circuit Breakers.



\### C4: Performance \& Cost

\* \*\*Constraint Discipline:\*\* Explicitly define batch sizes and concurrency limits.

\* \*\*N+1 Prevention:\*\* Audit all DB/API loops. Measure latency before and after "optimizations."



---



\## 4. Definition of Done (DoD)

A task is finished only when:

1\.  \*\*Tests:\*\* Unit tests exist for domain logic; Integration tests exist for adapters.

2\.  \*\*Codex Check:\*\* The code complies with all C1-C4 rules.

3\.  \*\*Docs:\*\* The `README.md` or `docs/adr/` reflects the new behavior.

4\.  \*\*Observability:\*\* New flows include logging and correlation IDs.

