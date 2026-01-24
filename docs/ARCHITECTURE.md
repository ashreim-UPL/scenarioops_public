# Architecture

## Component Diagram

```
+---------------------+        +-------------------------+
|  CLI / Streamlit UI | -----> | scenarioops CLI          |
+---------------------+        +-------------------------+
                                      |
                                      v
                             +-------------------------+
                             | scenarioops.graph       |
                             | build_graph.run_graph   |
                             +-------------------------+
                                      |
                                      v
+-------------------+       +-------------------------+       +------------------+
| Retriever (URLs)  | ----> | Graph Nodes (LLM + logic)| ----> | Artifacts + Logs |
+-------------------+       +-------------------------+       +------------------+
                                      |
                                      v
                             +-------------------------+
                             | Auditor + Provenance    |
                             +-------------------------+
```

## LangGraph Nodes
- Charter
- Drivers
- Uncertainties
- Logic
- Skeletons
- Narratives
- EWIs
- Strategies
- Wind Tunnel
- Daily Runner
- Auditor

## Deterministic Gates
- JSON schema validation for each artifact.
- Scoring rubric normalization (wind tunnel) with deterministic hashing.
- Consistent run IDs and ordered payloads for reproducible outputs.

## Provenance and Audit Logic
- Each artifact is written with a companion `.meta.json` provenance file.
- Auditor validates schemas, citations, and narrative numeric claims.
- Audit findings fail the run and surface in `latest.json`.
