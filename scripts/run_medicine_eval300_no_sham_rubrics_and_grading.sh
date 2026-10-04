#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
CONFIG_PATH=${CONFIG_PATH:-${PROJECT_ROOT}/configs/horizon_medicine_eval300_no_sham.yaml}
PROMPTS=${PROMPTS:-${PROJECT_ROOT}/data/rar/medicine/eval300/final.jsonl}
DELTA_PROMPTS=${DELTA_PROMPTS:-${PROJECT_ROOT}/data/rar/medicine/eval300/final_delta200.jsonl}
SOURCE_RUBRIC_ROOT=${SOURCE_RUBRIC_ROOT:-${PROJECT_ROOT}/artifacts/horizon/medicine/rubrics/seed-11}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts/horizon/medicine/eval300-no-sham}
POOL_ROOT=${POOL_ROOT:-${ARTIFACT_ROOT}/pools}
DELTA_POOL_ROOT=${DELTA_POOL_ROOT:-${ARTIFACT_ROOT}/delta200/pools}
DELTA_RUBRIC_ROOT=${DELTA_RUBRIC_ROOT:-${ARTIFACT_ROOT}/delta200/rubrics}
RUBRIC_ROOT=${RUBRIC_ROOT:-${ARTIFACT_ROOT}/rubrics/seed-11}
RUBRIC_STATE_ROOT=${RUBRIC_STATE_ROOT:-${ARTIFACT_ROOT}/delta200/batch/seed-11}
SCORE_ROOT=${SCORE_ROOT:-${ARTIFACT_ROOT}/scores/seed-11}
RESULT_ROOT=${RESULT_ROOT:-${PROJECT_ROOT}/results/horizon/medicine/eval300-no-sham}
JUDGE_BASE_URL=${JUDGE_BASE_URL:-http://127.0.0.1:8106}
BATCH_POLL_INTERVAL_SECONDS=${BATCH_POLL_INTERVAL_SECONDS:-60}

checkpoint_steps=(3 6 9 13 16 24 32 40 48)
checkpoint_epochs=(0.2 0.4 0.6 0.8 1.0 1.5 2.0 2.5 3.0)

mkdir -p \
  "${DELTA_RUBRIC_ROOT}" "${RUBRIC_ROOT}" "${RUBRIC_STATE_ROOT}" \
  "${SCORE_ROOT}" "${RESULT_ROOT}"

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
      --run-id "rar-horizon-v1-medicine-seed11-step${step}-eval300-delta200-no-sham" \
      --prompts "${DELTA_PROMPTS}" \
      --current-pool "${DELTA_POOL_ROOT}/seed-11-step-${step}-pool-a.jsonl" \
      --control-pool "${DELTA_POOL_ROOT}/fixed.jsonl" \
      --checkpoint-id "step${step}" \
      --batch-poll-interval-seconds "${BATCH_POLL_INTERVAL_SECONDS}" \
      --api-mode batch \
      --rubric-state-root "${RUBRIC_STATE_ROOT}/step-${step}" \
      --output "${output}"
  fi
  require_lines "${output}" 200
}

merge_rubric() {
  local step=$1
  local prefix_rubric="${SOURCE_RUBRIC_ROOT}/independent/step-${step}.jsonl"
  if [[ "${step}" -eq 3 ]]; then
    prefix_rubric="${SOURCE_RUBRIC_ROOT}/step-3.jsonl"
  fi
  require_lines "${prefix_rubric}" 100
  PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
    uv run --project "${PROJECT_ROOT}" python \
      "${PROJECT_ROOT}/scripts/merge_horizon_rubric_extension.py" \
      --prompts "${PROMPTS}" \
      --prefix-prompts 100 \
      --existing-rubric "${prefix_rubric}" \
      --delta-rubric "${DELTA_RUBRIC_ROOT}/step-${step}.jsonl" \
      --checkpoint-id "step${step}" \
      --output "${RUBRIC_ROOT}/step-${step}.jsonl"
  require_lines "${RUBRIC_ROOT}/step-${step}.jsonl" 300
}

grade_checkpoint() {
  local step=$1
  local epoch=$2
  local output_dir="${SCORE_ROOT}/epoch-${epoch}"
  local args=(
    grade-horizon
    --config "${CONFIG_PATH}"
    --run-id "rar-horizon-v1-medicine-seed11-step${step}-eval300-no-sham-grade"
    --prompts "${PROMPTS}"
    --pool-b "${POOL_ROOT}/seed-11-step-${step}-pool-b.jsonl"
    --checkpoint "${epoch}"
    --base-url "${JUDGE_BASE_URL}"
    --output-dir "${output_dir}"
  )
  if [[ "${step}" -ne 0 ]]; then
    args+=(--rubrics "${RUBRIC_ROOT}/step-${step}.jsonl")
  fi
  if [[ ! -f "${output_dir}/score_seal.json" ]]; then
    run_cli "${args[@]}"
  fi
  [[ -f "${output_dir}/score_seal.json" ]]
}

[[ -f "${ARTIFACT_ROOT}/responses.complete" ]] || {
  echo "response generation is not complete: ${ARTIFACT_ROOT}/responses.complete" >&2
  exit 1
}
: "${OPENAI_API_KEY:?OPENAI_API_KEY is required for GPT-5-mini rubric extraction}"
curl -fsS --max-time 10 "${JUDGE_BASE_URL}/health" >/dev/null
require_lines "${PROMPTS}" 300
require_lines "${DELTA_PROMPTS}" 200
run_cli validate-config --config "${CONFIG_PATH}" >/dev/null

for step in "${checkpoint_steps[@]}"; do
  log "building no-sham GPT-5-mini delta rubric for step ${step}"
  build_delta_rubric "${step}"
  merge_rubric "${step}"
done

log "grading checkpoint step 0 on Inference A GPUs 0 and 1"
grade_checkpoint 0 0.0
for index in "${!checkpoint_steps[@]}"; do
  step=${checkpoint_steps[$index]}
  epoch=${checkpoint_epochs[$index]}
  log "grading checkpoint step ${step} on Inference A GPUs 0 and 1"
  grade_checkpoint "${step}" "${epoch}"
done

summaries=()
for epoch in 0.0 0.2 0.4 0.6 0.8 1.0 1.5 2.0 2.5 3.0; do
  summaries+=("${SCORE_ROOT}/epoch-${epoch}/prompt_summary.jsonl")
done
run_cli build-horizon-observations \
  --config "${CONFIG_PATH}" \
  --prompts "${PROMPTS}" \
  --summaries "${summaries[@]}" \
  --output "${ARTIFACT_ROOT}/observations-seed-11.jsonl" >/dev/null
run_cli analyze-horizon \
  --config "${CONFIG_PATH}" \
  --observations "${ARTIFACT_ROOT}/observations-seed-11.jsonl" \
  --output "${RESULT_ROOT}/horizon_report_seed-11.json" >/dev/null
touch "${ARTIFACT_ROOT}/audit.complete"
log "Medicine eval300 no-sham rubric extraction and Inference A grading complete"
