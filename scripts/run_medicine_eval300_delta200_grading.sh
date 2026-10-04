#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
CONFIG_PATH=${CONFIG_PATH:-${PROJECT_ROOT}/configs/horizon_medicine_eval300_no_sham_low.yaml}
DELTA_PROMPTS=${DELTA_PROMPTS:-${PROJECT_ROOT}/data/rar/medicine/eval300/final_delta200.jsonl}
FINAL_ARTIFACT_ROOT=${FINAL_ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts/horizon/medicine/eval300-no-sham-low}
FINAL_RUBRIC_ROOT=${FINAL_RUBRIC_ROOT:-${FINAL_ARTIFACT_ROOT}/rubrics/seed-11}
RUBRICS_COMPLETE=${RUBRICS_COMPLETE:-${FINAL_ARTIFACT_ROOT}/rubrics.complete}
DELTA_POOL_ROOT=${DELTA_POOL_ROOT:-${PROJECT_ROOT}/artifacts/horizon/medicine/eval300-no-sham/delta200/pools}
DELTA_ARTIFACT_ROOT=${DELTA_ARTIFACT_ROOT:-${FINAL_ARTIFACT_ROOT}/delta200}
DELTA_RUBRIC_ROOT=${DELTA_RUBRIC_ROOT:-${DELTA_ARTIFACT_ROOT}/rubrics/seed-11}
SCORE_ROOT=${SCORE_ROOT:-${DELTA_ARTIFACT_ROOT}/scores/seed-11}
LOG_ROOT=${LOG_ROOT:-${DELTA_ARTIFACT_ROOT}/grading-logs}
JUDGE_BASE_URL=${JUDGE_BASE_URL:-http://127.0.0.1:8108}
GRADE_WORKERS=${GRADE_WORKERS:-3}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-3}
WAIT_INTERVAL_SECONDS=${WAIT_INTERVAL_SECONDS:-60}
PYTHON_BIN=${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}

checkpoint_steps=(0 3 6 9 13 16 24 32 40 48)
checkpoint_epochs=(0.0 0.2 0.4 0.6 0.8 1.0 1.5 2.0 2.5 3.0)

mkdir -p "${DELTA_RUBRIC_ROOT}" "${SCORE_ROOT}" "${LOG_ROOT}"
rm -f "${DELTA_ARTIFACT_ROOT}/grading.failed"

log() {
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$*"
}

run_cli() {
  PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
    PYTHONUNBUFFERED=1 \
    uv run --project "${PROJECT_ROOT}" python -m dynamic_rubric "$@"
}

require_lines() {
  local path=$1
  local expected=$2
  local label=$3
  local actual
  [[ -f "${path}" ]] || { echo "missing ${label}: ${path}" >&2; return 1; }
  actual=$(wc -l < "${path}")
  [[ "${actual}" -eq "${expected}" ]] || {
    echo "wrong ${label} row count: expected=${expected}, actual=${actual}, path=${path}" >&2
    return 1
  }
}

on_exit() {
  local status=$?
  if [[ "${status}" -ne 0 ]]; then
    printf '%s exit_status=%s\n' "$(date --iso-8601=seconds)" "${status}" \
      > "${DELTA_ARTIFACT_ROOT}/grading.failed"
  fi
}
trap on_exit EXIT

wait_for_rubrics() {
  while [[ ! -f "${RUBRICS_COMPLETE}" ]]; do
    log "waiting for final eval300 rubric completion marker"
    sleep "${WAIT_INTERVAL_SECONDS}"
  done
  local step
  for step in "${checkpoint_steps[@]:1}"; do
    require_lines "${FINAL_RUBRIC_ROOT}/step-${step}.jsonl" 300 "step ${step} final rubric"
  done
}

wait_for_judges() {
  while true; do
    if curl -fsS --max-time 5 "${JUDGE_BASE_URL}/health" >/dev/null 2>&1 && \
      "${PYTHON_BIN}" - "${JUDGE_BASE_URL}" <<'PY' >/dev/null 2>&1
import json
import sys
import urllib.request

base_url = sys.argv[1].rstrip("/")
with urllib.request.urlopen(f"{base_url}/dynamic-rubric/routing", timeout=5) as response:
    routing = json.loads(response.read())
if routing.get("upstream_count") != 3 or routing.get("upstream_weights") != [1, 1, 1]:
    raise SystemExit(1)
PY
    then
      return 0
    fi
    log "waiting for Trainer1 + Inference A0/1 aggregate judge proxy: ${JUDGE_BASE_URL}"
    sleep "${WAIT_INTERVAL_SECONDS}"
  done
}

prepare_delta_rubrics() {
  local step
  require_lines "${DELTA_PROMPTS}" 200 "delta prompts"
  for step in "${checkpoint_steps[@]:1}"; do
    PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
      "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/slice_horizon_rubric.py" \
        --prompts "${DELTA_PROMPTS}" \
        --rubric "${FINAL_RUBRIC_ROOT}/step-${step}.jsonl" \
        --checkpoint-id "step${step}" \
        --expected-source-count 300 \
        --output "${DELTA_RUBRIC_ROOT}/step-${step}.jsonl" >/dev/null
    require_lines "${DELTA_RUBRIC_ROOT}/step-${step}.jsonl" 200 "step ${step} delta rubric"
  done
}

validate_pools() {
  local step
  for step in "${checkpoint_steps[@]}"; do
    require_lines "${DELTA_POOL_ROOT}/seed-11-step-${step}-pool-b.jsonl" 3200 \
      "step ${step} delta Pool-B"
  done
}

grade_one() {
  local step=$1
  local epoch=$2
  local output_dir="${SCORE_ROOT}/epoch-${epoch}"
  local log_path="${LOG_ROOT}/step-${step}.log"
  local attempt=1
  local args=(
    grade-horizon
    --config "${CONFIG_PATH}"
    --run-id "rar-horizon-v1-medicine-seed11-step${step}-eval300-delta200-no-sham-low-grade"
    --prompts "${DELTA_PROMPTS}"
    --pool-b "${DELTA_POOL_ROOT}/seed-11-step-${step}-pool-b.jsonl"
    --checkpoint "${epoch}"
    --r0-current-only
    --base-url "${JUDGE_BASE_URL}"
    --output-dir "${output_dir}"
  )
  if [[ "${step}" -ne 0 ]]; then
    args+=(--rubrics "${DELTA_RUBRIC_ROOT}/step-${step}.jsonl")
  fi

  if [[ -f "${output_dir}/score_seal.json" ]]; then
    log "step ${step} epoch ${epoch} already sealed"
    return 0
  fi
  while (( attempt <= MAX_ATTEMPTS )); do
    log "grading step ${step} epoch ${epoch}, attempt ${attempt}"
    if run_cli "${args[@]}" >"${log_path}" 2>&1; then
      [[ -f "${output_dir}/score_seal.json" ]]
      log "sealed step ${step} epoch ${epoch}"
      return 0
    fi
    log "grading step ${step} epoch ${epoch} failed; see ${log_path}"
    attempt=$((attempt + 1))
    sleep 10
  done
  echo "step ${step} failed after ${MAX_ATTEMPTS} attempts" >&2
  return 1
}

run_grading_queue() {
  local queue_root
  queue_root=$(mktemp -d "${LOG_ROOT}/queue.XXXXXX")
  local index
  for index in "${!checkpoint_steps[@]}"; do
    printf '%s:%s\n' "${checkpoint_steps[$index]}" "${checkpoint_epochs[$index]}"
  done > "${queue_root}/tasks"
  printf '0\n' > "${queue_root}/cursor"

  claim_task() {
    local cursor task
    exec 9>"${queue_root}/lock"
    flock 9
    cursor=$(<"${queue_root}/cursor")
    task=$(sed -n "$((cursor + 1))p" "${queue_root}/tasks")
    if [[ -n "${task}" ]]; then
      printf '%s\n' "$((cursor + 1))" > "${queue_root}/cursor"
    fi
    flock -u 9
    printf '%s' "${task}"
  }

  worker() {
    local item step epoch
    while item=$(claim_task) && [[ -n "${item}" ]]; do
      step=${item%%:*}
      epoch=${item#*:}
      grade_one "${step}" "${epoch}"
    done
  }

  local pids=()
  local worker_index
  for worker_index in $(seq 1 "${GRADE_WORKERS}"); do
    worker >"${LOG_ROOT}/worker-${worker_index}.log" 2>&1 &
    pids+=("$!")
  done
  local status=0
  local pid
  for pid in "${pids[@]}"; do
    wait "${pid}" || status=1
  done
  return "${status}"
}

verify_seals() {
  PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
    "${PYTHON_BIN}" - "${SCORE_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected_epochs = ("0.0", "0.2", "0.4", "0.6", "0.8", "1.0", "1.5", "2.0", "2.5", "3.0")
for epoch in expected_epochs:
    seal_path = root / f"epoch-{epoch}" / "score_seal.json"
    if not seal_path.is_file():
        raise SystemExit(f"missing score seal: {seal_path}")
    seal = json.loads(seal_path.read_text())
    expected = {
        "pool_family": "pool_b",
        "comparison_scope": "r0_current",
        "prompt_count": 200,
        "response_count": 3200,
    }
    for key, value in expected.items():
        if seal.get(key) != value:
            raise SystemExit(
                f"invalid score seal field {key}: expected={value!r}, "
                f"actual={seal.get(key)!r}, path={seal_path}"
            )
    summaries = (seal_path.parent / "prompt_summary.jsonl").read_text().splitlines()
    if len(summaries) != 200:
        raise SystemExit(f"invalid prompt summary count: {seal_path.parent}")
print("verified 10 delta200 Pool-B score seals")
PY
}

[[ "${GRADE_WORKERS}" =~ ^[1-9][0-9]*$ ]] || {
  echo "GRADE_WORKERS must be a positive integer" >&2
  exit 2
}
[[ "${MAX_ATTEMPTS}" =~ ^[1-9][0-9]*$ ]] || {
  echo "MAX_ATTEMPTS must be a positive integer" >&2
  exit 2
}

log "supervisor started; existing baseline100 scores will be preserved"
wait_for_rubrics
log "final eval300 rubrics are complete"
prepare_delta_rubrics
validate_pools
run_cli validate-config --config "${CONFIG_PATH}" >/dev/null
wait_for_judges
log "Trainer1 + Inference A0/1 judges are ready; starting delta200 grading"
run_grading_queue
verify_seals
touch "${DELTA_ARTIFACT_ROOT}/grading.complete"
rm -f "${DELTA_ARTIFACT_ROOT}/grading.failed"
log "delta200 grading complete: 10/10 score seals"
