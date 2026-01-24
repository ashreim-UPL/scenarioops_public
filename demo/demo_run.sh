#!/usr/bin/env bash
set -euo pipefail

RUN_ID="demo-run"

scenarioops build-scenarios \
  --scope country \
  --value UAE \
  --horizon 24 \
  --run-id "${RUN_ID}"

scenarioops add-strategies demo/demo_strategies.txt \
  --run-id "${RUN_ID}"

scenarioops run-daily \
  --run-id "${RUN_ID}"

echo "Demo run complete: ${RUN_ID}"
