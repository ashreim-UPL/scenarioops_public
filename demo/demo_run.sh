#!/usr/bin/env bash
set -euo pipefail

RUN_ID="demo-run"

python -m scenarioops.app.main build-scenarios \
  --scope country \
  --value UAE \
  --horizon 24 \
  --run-id "${RUN_ID}"

python -m scenarioops.app.main add-strategies demo/demo_strategies.txt \
  --run-id "${RUN_ID}"

python -m scenarioops.app.main run-daily \
  --run-id "${RUN_ID}"

echo "Demo run complete: ${RUN_ID}"
