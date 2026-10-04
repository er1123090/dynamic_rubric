#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
RUN_ID=${RUN_ID:-pilot-static-r0-v1}
TARGET_STEP=${TARGET_STEP:-100}
WATCH_PID=${WATCH_PID:-}
POLL_SECONDS=${POLL_SECONDS:-30}
MAX_STALLED_RESTARTS=${MAX_STALLED_RESTARTS:-3}
LATEST_POINTER="${PROJECT_ROOT}/artifacts/runs/${RUN_ID}/train-static/verl-run/checkpoints/latest_checkpointed_iteration.txt"

latest_step() {
  if [[ ! -f "${LATEST_POINTER}" ]]; then
    echo 0
    return
  fi
  local step
  step=$(<"${LATEST_POINTER}")
  if [[ ! "${step}" =~ ^[0-9]+$ ]]; then
    echo "Invalid checkpoint pointer: ${LATEST_POINTER}" >&2
    exit 2
  fi
  echo "${step}"
}

wait_for_watched_run() {
  local pid=$1
  while kill -0 "${pid}" 2>/dev/null; do
    sleep "${POLL_SECONDS}"
  done
}

cd "${PROJECT_ROOT}"
export RESUME_STATIC_ONLY=true

stalled_restarts=0
if [[ -n "${WATCH_PID}" ]]; then
  wait_for_watched_run "${WATCH_PID}"
fi

while true; do
  before=$(latest_step)
  if ((before >= TARGET_STEP)); then
    echo "Static RL reached checkpoint ${before}."
    exit 0
  fi

  echo "Restarting static RL from checkpoint ${before}."
  status=0
  bash scripts/run_stage4_stage5.sh || status=$?
  after=$(latest_step)

  if ((after >= TARGET_STEP)); then
    echo "Static RL reached checkpoint ${after}."
    exit 0
  fi
  if ((after > before)); then
    stalled_restarts=0
  else
    stalled_restarts=$((stalled_restarts + 1))
  fi
  if ((stalled_restarts >= MAX_STALLED_RESTARTS)); then
    echo "Static RL made no checkpoint progress across ${stalled_restarts} restarts; last exit=${status}." >&2
    exit 1
  fi
  sleep "${POLL_SECONDS}"
done
