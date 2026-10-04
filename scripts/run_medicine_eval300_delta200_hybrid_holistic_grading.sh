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
LOG_ROOT=${LOG_ROOT:-${DELTA_ARTIFACT_ROOT}/grading-logs-holistic}
CRITERION_CACHE_DIR=${CRITERION_CACHE_DIR:-${PROJECT_ROOT}/artifacts/provider_cache/horizon-vllm-score-proxy}
HOLISTIC_CACHE_DIR=${HOLISTIC_CACHE_DIR:-${PROJECT_ROOT}/artifacts/provider_cache/horizon-vllm-holistic-online-v1}
PYTHON_BIN=${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}
GRADE_WORKERS=${GRADE_WORKERS:-3}
RESPONSE_CONCURRENCY=${RESPONSE_CONCURRENCY:-16}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-3}
WAIT_INTERVAL_SECONDS=${WAIT_INTERVAL_SECONDS:-30}

JUDGE_BASE_URLS=(
  "${TRAINER1_BASE_URL:-http://127.0.0.1:8015}"
  "${INFERENCE_A0_BASE_URL:-http://127.0.0.1:18002}"
  "${INFERENCE_A1_BASE_URL:-http://127.0.0.1:18013}"
)
checkpoint_steps=(0 3 6 9 13 16 24 32 40 48)
checkpoint_epochs=(0.0 0.2 0.4 0.6 0.8 1.0 1.5 2.0 2.5 3.0)
TARGET_ENCODING=hybrid_cached_yes_no_then_onlinerubric_full_rubric_json_v1

mkdir -p "${DELTA_RUBRIC_ROOT}" "${SCORE_ROOT}" "${LOG_ROOT}" "${HOLISTIC_CACHE_DIR}"
rm -f "${DELTA_ARTIFACT_ROOT}/grading.failed"

log() {
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$*"
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

wait_for_inputs() {
  [[ -f "${RUBRICS_COMPLETE}" ]] || {
    echo "missing completed rubric marker: ${RUBRICS_COMPLETE}" >&2
    return 1
  }
  require_lines "${DELTA_PROMPTS}" 200 "delta prompts"
  local step
  for step in "${checkpoint_steps[@]}"; do
    require_lines "${DELTA_POOL_ROOT}/seed-11-step-${step}-pool-b.jsonl" 3200 \
      "step ${step} delta Pool-B"
    if [[ "${step}" -ne 0 ]]; then
      require_lines "${DELTA_RUBRIC_ROOT}/step-${step}.jsonl" 200 \
        "step ${step} delta rubric"
    fi
  done
}

wait_for_judges() {
  local ready base_url
  while true; do
    ready=1
    for base_url in "${JUDGE_BASE_URLS[@]}"; do
      if ! curl -fsS --max-time 5 "${base_url}/v1/models" >/dev/null 2>&1; then
        ready=0
      fi
    done
    if [[ "${ready}" -eq 1 ]]; then
      return 0
    fi
    log "waiting for Trainer1 + Inference A0/1 raw Qwen judge endpoints"
    sleep "${WAIT_INTERVAL_SECONDS}"
  done
}

seal_is_compatible() {
  local seal_path=$1
  "${PYTHON_BIN}" - "${seal_path}" "${TARGET_ENCODING}" <<'PY'
import json
import sys
from pathlib import Path

seal = json.loads(Path(sys.argv[1]).read_text())
raise SystemExit(0 if seal.get("target_encoding_version") == sys.argv[2] else 1)
PY
}

grade_one() {
  local step=$1
  local epoch=$2
  local output_dir="${SCORE_ROOT}/epoch-${epoch}"
  local log_path="${LOG_ROOT}/step-${step}.log"
  local attempt=1
  local args=(
    --config "${CONFIG_PATH}"
    --prompts "${DELTA_PROMPTS}"
    --pool-b "${DELTA_POOL_ROOT}/seed-11-step-${step}-pool-b.jsonl"
    --checkpoint "${epoch}"
    --output-dir "${output_dir}"
    --criterion-cache-dir "${CRITERION_CACHE_DIR}"
    --holistic-cache-dir "${HOLISTIC_CACHE_DIR}"
    --concurrency "${RESPONSE_CONCURRENCY}"
    --max-attempts "${MAX_ATTEMPTS}"
  )
  local base_url
  for base_url in "${JUDGE_BASE_URLS[@]}"; do
    args+=(--base-url "${base_url}")
  done
  if [[ "${step}" -ne 0 ]]; then
    args+=(--rubrics "${DELTA_RUBRIC_ROOT}/step-${step}.jsonl")
  fi

  if [[ -f "${output_dir}/score_seal.json" ]]; then
    if seal_is_compatible "${output_dir}/score_seal.json"; then
      log "step ${step} epoch ${epoch} already sealed with holistic mode"
      return 0
    fi
    echo "incompatible pre-existing score seal: ${output_dir}/score_seal.json" >&2
    return 1
  fi

  while (( attempt <= MAX_ATTEMPTS )); do
    log "grading step ${step} epoch ${epoch}, attempt ${attempt}"
    if PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
      PYTHONUNBUFFERED=1 \
      "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/grade_horizon_hybrid_holistic.py" \
      "${args[@]}" >"${log_path}" 2>&1; then
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
    "${PYTHON_BIN}" - "${SCORE_ROOT}" "${TARGET_ENCODING}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
target_encoding = sys.argv[2]
expected_epochs = ("0.0", "0.2", "0.4", "0.6", "0.8", "1.0", "1.5", "2.0", "2.5", "3.0")
totals = {
    "reused_complete_responses": 0,
    "holistic_responses": 0,
    "holistic_api_calls": 0,
    "holistic_cache_hits": 0,
}
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
        "target_encoding_version": target_encoding,
    }
    for key, value in expected.items():
        if seal.get(key) != value:
            raise SystemExit(
                f"invalid score seal field {key}: expected={value!r}, "
                f"actual={seal.get(key)!r}, path={seal_path}"
            )
    execution = seal.get("grader_execution", {})
    if execution.get("execution_mode") != "onlinerubric_full_rubric_one_call_per_response_v1":
        raise SystemExit(f"invalid holistic execution provenance: {seal_path}")
    stats = execution.get("stats", {})
    if stats.get("reused_complete_responses", 0) + stats.get("holistic_responses", 0) != 3200:
        raise SystemExit(f"response accounting mismatch: {seal_path}")
    if sum(stats.get("endpoint_api_calls", {}).values()) != stats.get("holistic_api_calls", 0):
        raise SystemExit(f"endpoint API accounting mismatch: {seal_path}")
    summaries = (seal_path.parent / "prompt_summary.jsonl").read_text().splitlines()
    if len(summaries) != 200:
        raise SystemExit(f"invalid prompt summary count: {seal_path.parent}")
    for key in totals:
        totals[key] += int(stats.get(key, 0))
print(json.dumps({"verified_score_seals": 10, **totals}, sort_keys=True))
PY
}

[[ "${GRADE_WORKERS}" =~ ^[1-9][0-9]*$ ]] || {
  echo "GRADE_WORKERS must be a positive integer" >&2
  exit 2
}
[[ "${RESPONSE_CONCURRENCY}" =~ ^[1-9][0-9]*$ ]] || {
  echo "RESPONSE_CONCURRENCY must be a positive integer" >&2
  exit 2
}

log "hybrid supervisor started; complete legacy responses will be reused"
wait_for_inputs
PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON_BIN}" -m dynamic_rubric validate-config --config "${CONFIG_PATH}" >/dev/null
wait_for_judges
log "Trainer1 + Inference A0/1 ready; incomplete responses will use one full-rubric call"
run_grading_queue
verify_seals
touch "${DELTA_ARTIFACT_ROOT}/grading.complete"
rm -f "${DELTA_ARTIFACT_ROOT}/grading.failed"
log "delta200 hybrid holistic grading complete: 10/10 score seals"
