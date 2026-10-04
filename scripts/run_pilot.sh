#!/bin/sh
set -eu

RUN_ID=${RUN_ID:-pilot-v1}
CONFIG=${CONFIG:-configs/pilot.yaml}
PYTHON=${PYTHON:-python}

"$PYTHON" -m dynamic_rubric validate-config --config "$CONFIG"
"$PYTHON" -m dynamic_rubric prepare-data --config "$CONFIG"
"$PYTHON" -m dynamic_rubric preflight --config "$CONFIG" --run-id "$RUN_ID"
"$PYTHON" -m dynamic_rubric generate-static --config "$CONFIG" --run-id "$RUN_ID"
"$PYTHON" -m dynamic_rubric train-static --config "$CONFIG" --run-id "$RUN_ID"
"$PYTHON" -m dynamic_rubric replay-dynamic --split development --config "$CONFIG" --run-id "$RUN_ID"
"$PYTHON" -m dynamic_rubric freeze-updater --config "$CONFIG" --run-id "$RUN_ID"
"$PYTHON" -m dynamic_rubric replay-dynamic --split final --config "$CONFIG" --run-id "$RUN_ID"
"$PYTHON" -m dynamic_rubric generate-bon --config "$CONFIG" --run-id "$RUN_ID"
"$PYTHON" -m dynamic_rubric score-proxy --config "$CONFIG" --run-id "$RUN_ID"
"$PYTHON" -m dynamic_rubric select-bon --config "$CONFIG" --run-id "$RUN_ID"
"$PYTHON" -m dynamic_rubric export-audit-package --config "$CONFIG" --run-id "$RUN_ID"
"$PYTHON" -m dynamic_rubric audit-gold --config "$CONFIG" --run-id "$RUN_ID"
"$PYTHON" -m dynamic_rubric analyze --config "$CONFIG" --run-id "$RUN_ID"
"$PYTHON" -m dynamic_rubric validate-inventory --config "$CONFIG" --run-id "$RUN_ID"

