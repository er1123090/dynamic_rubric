#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
RUN_ID=pilot-static-r0-100step-20260821
STAGE_ROOT="$PROJECT_ROOT/artifacts/runs/$RUN_ID/dynamic-batch"
LOG="$STAGE_ROOT/monitor.log"

cd "$PROJECT_ROOT"
while true; do
  printf '%s refreshing Batch status\n' "$(date --iso-8601=seconds)" >> "$LOG"
  PYTHONPATH=src python -m dynamic_rubric status-dynamic-batch \
    --config configs/pilot.yaml --run-id "$RUN_ID" >> "$LOG" 2>&1
  state=$(python - <<'PY_STATUS'
import json
from pathlib import Path
jobs = json.loads(Path("artifacts/runs/pilot-static-r0-100step-20260821/dynamic-batch/status.json").read_text())["jobs"]
statuses = [job["status"] for job in jobs]
if statuses and all(status == "completed" for status in statuses):
    print("completed")
elif any(status in {"failed", "expired", "cancelled"} for status in statuses):
    print("terminal_error")
else:
    print("running")
PY_STATUS
)
  if [[ "$state" == completed ]]; then
    PYTHONPATH=src python -m dynamic_rubric collect-dynamic-batch \
      --config configs/pilot.yaml --run-id "$RUN_ID" >> "$LOG" 2>&1
    printf '%s Batch collection completed\n' "$(date --iso-8601=seconds)" >> "$LOG"
    exit 0
  fi
  if [[ "$state" == terminal_error ]]; then
    printf '%s Batch reached a terminal error state\n' "$(date --iso-8601=seconds)" >> "$LOG"
    exit 2
  fi
  sleep 300
done
