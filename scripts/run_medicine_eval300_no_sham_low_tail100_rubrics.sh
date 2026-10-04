#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
CONFIG_PATH=${CONFIG_PATH:-${PROJECT_ROOT}/configs/horizon_medicine_eval300_no_sham_low.yaml}
PROMPTS=${PROMPTS:-${PROJECT_ROOT}/data/rar/medicine/eval300/final.jsonl}
DELTA_PROMPTS=${DELTA_PROMPTS:-${PROJECT_ROOT}/data/rar/medicine/eval300/final_delta100_tail.jsonl}
SOURCE_RUBRIC_ROOT=${SOURCE_RUBRIC_ROOT:-${PROJECT_ROOT}/artifacts/horizon/medicine/eval200-no-sham-low/rubrics/seed-11}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts/horizon/medicine/eval300-no-sham-low}
DELTA_POOL_ROOT=${DELTA_POOL_ROOT:-${ARTIFACT_ROOT}/tail100/pools}
DELTA_RUBRIC_ROOT=${DELTA_RUBRIC_ROOT:-${ARTIFACT_ROOT}/tail100/rubrics}
RUBRIC_ROOT=${RUBRIC_ROOT:-${ARTIFACT_ROOT}/rubrics/seed-11}
RUBRIC_STATE_ROOT=${RUBRIC_STATE_ROOT:-${ARTIFACT_ROOT}/tail100/batch/seed-11}
BATCH_POLL_INTERVAL_SECONDS=${BATCH_POLL_INTERVAL_SECONDS:-60}

checkpoint_steps=(3 6 9 13 16 24 32 40 48)

mkdir -p \
  "${DELTA_RUBRIC_ROOT}" "${RUBRIC_ROOT}" "${RUBRIC_STATE_ROOT}"

log() {
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$*"
}

run_cli() {
  PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
    uv run --project "${PROJECT_ROOT}" python -m dynamic_rubric "$@"
}

require_lines() {
  local path=$1
  local expected=$2
  local actual
  [[ -f "${path}" ]] || { echo "missing file: ${path}" >&2; return 1; }
  actual=$(wc -l < "${path}")
  [[ "${actual}" -eq "${expected}" ]] || {
    echo "wrong row count: expected=${expected}, actual=${actual}, path=${path}" >&2
    return 1
  }
}

build_delta_rubric() {
  local step=$1
  local output="${DELTA_RUBRIC_ROOT}/step-${step}.jsonl"
  if [[ ! -f "${output}" ]]; then
    run_cli build-horizon-rubrics \
      --config "${CONFIG_PATH}" \
      --run-id "rar-horizon-v1-medicine-seed11-step${step}-eval300-tail100-no-sham-low" \
      --prompts "${DELTA_PROMPTS}" \
      --current-pool "${DELTA_POOL_ROOT}/seed-11-step-${step}-pool-a.jsonl" \
      --control-pool "${DELTA_POOL_ROOT}/fixed.jsonl" \
      --checkpoint-id "step${step}" \
      --batch-poll-interval-seconds "${BATCH_POLL_INTERVAL_SECONDS}" \
      --api-mode batch \
      --rubric-state-root "${RUBRIC_STATE_ROOT}/step-${step}" \
      --output "${output}"
  fi
  require_lines "${output}" 100
}

merge_rubric() {
  local step=$1
  local prefix_rubric="${SOURCE_RUBRIC_ROOT}/step-${step}.jsonl"
  while [[ ! -f "${prefix_rubric}" ]]; do
    log "waiting for eval200 prefix rubric: step ${step}"
    sleep 60
  done
  require_lines "${prefix_rubric}" 200
  PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
    uv run --project "${PROJECT_ROOT}" python \
      "${PROJECT_ROOT}/scripts/merge_horizon_rubric_extension.py" \
      --prompts "${PROMPTS}" \
      --prefix-prompts 200 \
      --existing-rubric "${prefix_rubric}" \
      --delta-rubric "${DELTA_RUBRIC_ROOT}/step-${step}.jsonl" \
      --checkpoint-id "step${step}" \
      --output "${RUBRIC_ROOT}/step-${step}.jsonl"
  require_lines "${RUBRIC_ROOT}/step-${step}.jsonl" 300
}


[[ -f "${ARTIFACT_ROOT}/responses.complete" ]] || {
  echo "response generation is not complete: ${ARTIFACT_ROOT}/responses.complete" >&2
  exit 1
}
: "${OPENAI_API_KEY:?OPENAI_API_KEY is required for GPT-5-mini rubric extraction}"
require_lines "${PROMPTS}" 300
require_lines "${DELTA_PROMPTS}" 100
run_cli validate-config --config "${CONFIG_PATH}" >/dev/null

for step in "${checkpoint_steps[@]}"; do
  log "building no-sham GPT-5-mini delta rubric for step ${step}"
  build_delta_rubric "${step}"
  merge_rubric "${step}"
done

touch "${ARTIFACT_ROOT}/rubrics.complete"
log "Medicine eval300 no-sham low tail100 rubric extraction complete; grading not started"
