# ScenarioOps Architecture (Text Diagram)

```
User/CLI/API
   |
   v
build_graph.py (orchestration)
   |
   +--> charter -> drivers -> uncertainties -> logic -> skeletons -> narratives -> ewis
   |
   +--> strategies -> wind_tunnel -> daily_runner
   |
   +--> auditor (hard gate)
   |
   v
storage/runs/{run_id}/artifacts + logs
   |
   v
FastAPI + UI
```
