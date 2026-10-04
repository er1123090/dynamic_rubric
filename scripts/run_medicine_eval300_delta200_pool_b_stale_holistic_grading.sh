#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
CONFIG_PATH=${CONFIG_PATH:-${PROJECT_ROOT}/configs/horizon_medicine_eval300_no_sham_low.yaml}
DELTA_PROMPTS=${DELTA_PROMPTS:-${PROJECT_ROOT}/data/rar/medicine/eval300/final_delta200.jsonl}
DELTA_ARTIFACT_ROOT=${DELTA_ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts/horizon/medicine/eval300-no-sham-low/delta200}
POOL_ROOT=${POOL_ROOT:-${PROJECT_ROOT}/artifacts/horizon/medicine/eval300-no-sham/delta200/pools}
STALE_RUBRIC_ROOT=${STALE_RUBRIC_ROOT:-${DELTA_ARTIFACT_ROOT}/rubrics-stale/seed-11}
SCORE_ROOT=${SCORE_ROOT:-${DELTA_ARTIFACT_ROOT}/scores-pool-b-stale/seed-11}
LOG_ROOT=${LOG_ROOT:-${DELTA_ARTIFACT_ROOT}/grading-logs-pool-b-stale-holistic}
CRITERION_CACHE_DIR=${CRITERION_CACHE_DIR:-${PROJECT_ROOT}/artifacts/provider_cache/horizon-vllm-score-proxy}
HOLISTIC_CACHE_DIR=${HOLISTIC_CACHE_DIR:-${PROJECT_ROOT}/artifacts/provider_cache/horizon-vllm-holistic-online-v1}
PYTHON_BIN=${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}
GRADE_WORKERS=${GRADE_WORKERS:-3}
RESPONSE_CONCURRENCY=${RESPONSE_CONCURRENCY:-16}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-3}
TARGET_ENCODING=hybrid_cached_yes_no_then_onlinerubric_full_rubric_json_v1

JUDGE_BASE_URLS=(
  "${TRAINER1_BASE_URL:-http://127.0.0.1:8015}"
  "${INFERENCE_A0_BASE_URL:-http://127.0.0.1:18002}"
  "${INFERENCE_A1_BASE_URL:-http://127.0.0.1:18013}"
)
checkpoint_steps=(6 9 13 16 24 32 40 48)
checkpoint_epochs=(0.4 0.6 0.8 1.0 1.5 2.0 2.5 3.0)

mkdir -p "${SCORE_ROOT}" "${LOG_ROOT}" "${HOLISTIC_CACHE_DIR}"
rm -f "${DELTA_ARTIFACT_ROOT}/pool-b-stale-grading.failed"

log() {
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$*"
}

require_lines() {
  local path=$1 expected=$2 label=$3 actual
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
      > "${DELTA_ARTIFACT_ROOT}/pool-b-stale-grading.failed"
  fi
}
trap on_exit EXIT

wait_for_inputs() {
  require_lines "${DELTA_PROMPTS}" 200 "delta prompts"
  local step
  for step in "${checkpoint_steps[@]}"; do
    require_lines "${POOL_ROOT}/seed-11-step-${step}-pool-b.jsonl" 3200 \
      "step ${step} delta Pool B"
    require_lines "${STALE_RUBRIC_ROOT}/step-${step}.jsonl" 200 \
      "step ${step} true-stale rubric"
  done
}

wait_for_judges() {
  local base_url
  for base_url in "${JUDGE_BASE_URLS[@]}"; do
    curl -fsS --max-time 5 "${base_url}/v1/models" >/dev/null
  done
}

seal_is_compatible() {
  local seal_path=$1
  "${PYTHON_BIN}" - "${seal_path}" "${TARGET_ENCODING}" <<'PY'
import json
import sys
from pathlib import Path

seal = json.loads(Path(sys.argv[1]).read_text())
expected = {
    "pool_family": "pool_b",
    "comparison_scope": "full",
    "prompt_count": 200,
    "response_count": 3200,
    "target_encoding_version": sys.argv[2],
}
raise SystemExit(0 if all(seal.get(k) == v for k, v in expected.items()) else 1)
PY
}

grade_one() {
  local step=$1 epoch=$2
  local output_dir="${SCORE_ROOT}/epoch-${epoch}"
  local log_path base_url
  local args=(
    --config "${CONFIG_PATH}"
    --prompts "${DELTA_PROMPTS}"
    --pool-b "${POOL_ROOT}/seed-11-step-${step}-pool-b.jsonl"
    --rubrics "${STALE_RUBRIC_ROOT}/step-${step}.jsonl"
    --checkpoint "${epoch}"
    --output-dir "${output_dir}"
    --criterion-cache-dir "${CRITERION_CACHE_DIR}"
    --holistic-cache-dir "${HOLISTIC_CACHE_DIR}"
    --concurrency "${RESPONSE_CONCURRENCY}"
    --max-attempts "${MAX_ATTEMPTS}"
    --include-control
  )
  for base_url in "${JUDGE_BASE_URLS[@]}"; do
    args+=(--base-url "${base_url}")
  done

  if [[ -f "${output_dir}/score_seal.json" ]]; then
    if seal_is_compatible "${output_dir}/score_seal.json"; then
      log "step ${step} epoch ${epoch} already sealed"
      return 0
    fi
    echo "incompatible pre-existing score seal: ${output_dir}/score_seal.json" >&2
    return 1
  fi

  local attempt=1
  while (( attempt <= MAX_ATTEMPTS )); do
    log_path="${LOG_ROOT}/step-${step}.attempt-${attempt}.log"
    log "grading Pool B true-stale step ${step} epoch ${epoch}, attempt ${attempt}"
    if PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
      PYTHONUNBUFFERED=1 \
      "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/grade_horizon_hybrid_holistic.py" \
      "${args[@]}" >"${log_path}" 2>&1; then
      [[ -f "${output_dir}/score_seal.json" ]]
      log "sealed Pool B true-stale step ${step} epoch ${epoch}"
      return 0
    fi
    log "Pool B true-stale step ${step} failed; see ${log_path}"
    attempt=$((attempt + 1))
    sleep 10
  done
  return 1
}

run_queue() {
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

  local pids=() worker_index status=0 pid
  for worker_index in $(seq 1 "${GRADE_WORKERS}"); do
    worker >"${LOG_ROOT}/worker-resume-${worker_index}.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "${pid}" || status=1
  done
  return "${status}"
}

verify_seals() {
  "${PYTHON_BIN}" - "${SCORE_ROOT}" "${TARGET_ENCODING}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
target_encoding = sys.argv[2]
epochs = ("0.4", "0.6", "0.8", "1.0", "1.5", "2.0", "2.5", "3.0")
for epoch in epochs:
    directory = root / f"epoch-{epoch}"
    seal = json.loads((directory / "score_seal.json").read_text())
    expected = {
        "pool_family": "pool_b",
        "comparison_scope": "full",
        "prompt_count": 200,
        "response_count": 3200,
        "target_encoding_version": target_encoding,
    }
    for key, value in expected.items():
        if seal.get(key) != value:
            raise SystemExit(f"invalid {key} in {directory}: {seal.get(key)!r}")
    execution = seal.get("grader_execution", {})
    if execution.get("execution_mode") != "onlinerubric_full_rubric_one_call_per_response_v1":
        raise SystemExit(f"invalid execution mode in {directory}")
    summaries = [json.loads(line) for line in (directory / "prompt_summary.jsonl").read_text().splitlines()]
    if len(summaries) != 200 or any(row["response_count"] != 16 for row in summaries):
        raise SystemExit(f"invalid prompt summaries in {directory}")
    valid = [row for row in summaries if row.get("analysis_status") == "valid"]
    if any(not {"r0", "current", "control"} <= set(row.get("variants", {})) for row in valid):
        raise SystemExit(f"missing true-stale comparison in {directory}")
print(json.dumps({"verified_pool_b_true_stale_score_seals": len(epochs)}, sort_keys=True))
PY
}

wait_for_inputs
wait_for_judges
run_queue
verify_seals
touch "${DELTA_ARTIFACT_ROOT}/pool-b-stale-grading.complete"
rm -f "${DELTA_ARTIFACT_ROOT}/pool-b-stale-grading.failed"
log "delta200 Pool B true-stale holistic grading complete: 8/8 score seals"
