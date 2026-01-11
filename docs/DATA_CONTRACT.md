# Data Contract

## Run Layout
- `storage/runs/{run_id}/artifacts/`
- `storage/runs/{run_id}/logs/`
- `storage/runs/latest.json`

## Artifacts
| Artifact | Path | Schema |
| --- | --- | --- |
| Charter | `artifacts/scenario_charter.json` | `schemas/charter.json` |
| Drivers | `artifacts/drivers.jsonl` | `schemas/driver_entry.json` (per line) |
| Uncertainties | `artifacts/uncertainties.json` | `schemas/uncertainties.json` |
| Logic | `artifacts/logic.json` | `schemas/logic.json` |
| Skeletons | `artifacts/skeletons.json` | `schemas/skeleton.json` |
| Narratives | `artifacts/narrative_{scenario_id}.md` | `schemas/markdown.json` |
| EWIs | `artifacts/ewi.json` | `schemas/ewi.json` |
| Strategies | `artifacts/strategies.json` | `schemas/strategies.json` |
| Wind Tunnel | `artifacts/wind_tunnel.json` | `schemas/wind_tunnel.json` |
| Daily Brief JSON | `artifacts/daily_brief.json` | `schemas/daily_brief.json` |
| Daily Brief Markdown | `artifacts/daily_brief.md` | `schemas/markdown.json` |
| Audit Report | `artifacts/audit_report.json` | `schemas/audit_report.json` |
| View Model | `artifacts/view_model.json` | n/a (UI bundle) |

Each artifact has a provenance file: `artifacts/{artifact}.meta.json`.

## latest.json
Location: `storage/runs/latest.json`

Format:
```
{
  "run_id": "<string>",
  "updated_at": "<ISO8601>",
  "status": "OK|FAIL",
  "command": "build-scenarios|run-daily|...",
  "error_summary": "<string>"
}
```

## view_model.json
Location: `storage/runs/{run_id}/artifacts/view_model.json`

Fields:
- `charter`: object or null
- `drivers`: list
- `drivers_by_domain`: object mapping domain -> drivers
- `uncertainties`: list
- `scenario_logic`: object
- `scenarios`: list
- `narratives`: list of `{scenario_id, markdown}`
- `ewis`: list
- `daily_brief_md`: string or null
- `run_meta`: `{run_id, status}`
