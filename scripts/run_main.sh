#!/bin/sh
set -eu

RUN_ID=${RUN_ID:-main-v1}
CONFIG=${CONFIG:-configs/main.yaml}
PYTHON=${PYTHON:-python}

if [ ! -f "results/pilot-v1/pilot_report.json" ]; then
  echo "Pilot decision report is required before the causal main study." >&2
  exit 2
fi

"$PYTHON" -m dynamic_rubric validate-config --config "$CONFIG"
"$PYTHON" -m dynamic_rubric preflight --config "$CONFIG" --run-id "$RUN_ID"
echo "Main causal schedule execution remains gated by the pilot go/no-go decision and fresh split lock." >&2

