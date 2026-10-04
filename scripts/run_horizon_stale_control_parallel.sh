#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
DOMAIN=${DOMAIN:-medicine}
TRAINING_SEED=${TRAINING_SEED:-11}
CONFIG_PATH=${CONFIG_PATH:-${PROJECT_ROOT}/configs/horizon_${DOMAIN}.yaml}
PROMPTS=${PROMPTS:-${PROJECT_ROOT}/data/rar/${DOMAIN}/public/final.jsonl}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts/horizon/${DOMAIN}}
POOL_ROOT=${POOL_ROOT:-${ARTIFACT_ROOT}/pools}
RUBRIC_ROOT=${RUBRIC_ROOT:-${ARTIFACT_ROOT}/rubrics/seed-${TRAINING_SEED}}
SCORE_ROOT=${SCORE_ROOT:-${ARTIFACT_ROOT}/scores/seed-${TRAINING_SEED}}
IMMEDIATE_SCORE_ROOT=${IMMEDIATE_SCORE_ROOT:-${SCORE_ROOT}/immediate}
AUDIT_LOG_ROOT=${AUDIT_LOG_ROOT:-${ARTIFACT_ROOT}/audit/logs/stale-control}
MAX_GRADE_ATTEMPTS=${MAX_GRADE_ATTEMPTS:-6}
GRADE_RETRY_DELAY_SECONDS=${GRADE_RETRY_DELAY_SECONDS:-30}
EXPECTED_PROMPT_COUNT=${EXPECTED_PROMPT_COUNT:-100}

# Workload-balanced for the observed Qwen3-32B judge rates. Existing full seals
# are reused, so these assignments cover only the unfinished Medicine shards.
judge_urls=(
  "${INFERENCE_A_TP2_URL:-http://127.0.0.1:8102}"
  "${INFERENCE_A_GPU2_URL:-http://127.0.0.1:8104}"
  "${INFERENCE_C_URL:-http://127.0.0.1:8103}"
)
judge_steps=(
  "6 13 32"
  "9 24"
  "48"
)

checkpoint_steps=(3 6 9 13 16 24 32 40 48)
checkpoint_epochs=(0.2 0.4 0.6 0.8 1.0 1.5 2.0 2.5 3.0)

[[ "${DOMAIN}" == "medicine" ]] || {
  echo "this workload assignment is currently validated only for medicine" >&2
  exit 2
}
[[ "${#judge_urls[@]}" -eq "${#judge_steps[@]}" ]] || {
  echo "judge endpoint and assignment counts differ" >&2
  exit 2
}
mkdir -p "${SCORE_ROOT}" "${AUDIT_LOG_ROOT}"

log() {
  printf "%s %s\n" "$(date --iso-8601=seconds)" "$*"
}

run_cli() {
  PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
    uv run --project "${PROJECT_ROOT}" python -m dynamic_rubric "$@"
}

require_lines() {
  local path=$1
  local expected=$2
  local label=$3
  [[ -f "${path}" ]] || { echo "missing ${label}: ${path}" >&2; return 1; }
  local actual
  actual=$(wc -l <"${path}")
  [[ "${actual}" -eq "${expected}" ]] || {
    echo "wrong ${label} row count: expected=${expected} actual=${actual} path=${path}" >&2
    return 1
  }
}

epoch_for_step() {
  local target=$1
  local index
  for index in "${!checkpoint_steps[@]}"; do
    if [[ "${checkpoint_steps[$index]}" -eq "${target}" ]]; then
      printf "%s\n" "${checkpoint_epochs[$index]}"
      return 0
    fi
  done
  echo "unknown checkpoint step: ${target}" >&2
  return 1
}

grade_checkpoint() {
  local step=$1
  local judge_url=$2
  local epoch
  epoch=$(epoch_for_step "${step}")
  local rubric="${RUBRIC_ROOT}/step-${step}.jsonl"
  local pool_b="${POOL_ROOT}/seed-${TRAINING_SEED}-step-${step}-pool-b.jsonl"
  local reuse_dir="${IMMEDIATE_SCORE_ROOT}/epoch-${epoch}"
  local output_dir="${SCORE_ROOT}/epoch-${epoch}"

  if [[ -f "${output_dir}/score_seal.json" ]]; then
    log "reuse full stale-control seal step=${step} epoch=${epoch}"
    return 0
  fi
  require_lines "${rubric}" "${EXPECTED_PROMPT_COUNT}" "step ${step} canonical rubric"
  require_lines "${pool_b}" "$((EXPECTED_PROMPT_COUNT * 16))" "step ${step} Pool B"
  [[ -f "${reuse_dir}/score_seal.json" ]] || {
    echo "missing R0/current reuse seal: ${reuse_dir}/score_seal.json" >&2
    return 1
  }

  local attempt=1
  local active_url=${judge_url}
  while true; do
    log "start stale-control step=${step} epoch=${epoch} judge=${active_url} attempt=${attempt}"
    if run_cli grade-horizon \
      --config "${CONFIG_PATH}" \
      --run-id "rar-horizon-v1-${DOMAIN}-seed${TRAINING_SEED}-step${step}-grade-stale-control" \
      --prompts "${PROMPTS}" \
      --pool-b "${pool_b}" \
      --rubrics "${rubric}" \
      --checkpoint "${epoch}" \
      --reuse-score-dir "${reuse_dir}" \
      --base-url "${active_url}" \
      --output-dir "${output_dir}"
    then
      [[ -f "${output_dir}/score_seal.json" ]]
      log "complete stale-control step=${step} epoch=${epoch}"
      return 0
    fi
    if [[ "${attempt}" -ge "${MAX_GRADE_ATTEMPTS}" ]]; then
      if [[ "${active_url}" != "${judge_urls[0]}" ]]; then
        active_url=${judge_urls[0]}
        attempt=1
        log "fail over stale-control step=${step} judge=${active_url}"
        continue
      fi
      echo "stale-control grading failed after ${attempt} attempts: step=${step}" >&2
      return 1
    fi
    attempt=$((attempt + 1))
    sleep "${GRADE_RETRY_DELAY_SECONDS}"
  done
}

require_lines "${PROMPTS}" "${EXPECTED_PROMPT_COUNT}" "final prompts"
for judge_url in "${judge_urls[@]}"; do
  curl -fsS --max-time 10 "${judge_url}/health" >/dev/null
done

pids=()
for worker_index in "${!judge_urls[@]}"; do
  (
    for step in ${judge_steps[$worker_index]}; do
      grade_checkpoint "${step}" "${judge_urls[$worker_index]}"
    done
  ) >"${AUDIT_LOG_ROOT}/worker-${worker_index}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
[[ "${status}" -eq 0 ]] || {
  echo "one or more stale-control workers failed; inspect ${AUDIT_LOG_ROOT}" >&2
  exit 1
}

for index in "${!checkpoint_steps[@]}"; do
  epoch=${checkpoint_epochs[$index]}
  [[ -f "${SCORE_ROOT}/epoch-${epoch}/score_seal.json" ]] || {
    echo "missing full stale-control seal: epoch=${epoch}" >&2
    exit 1
  }
done
printf "complete\n" >"${SCORE_ROOT}/stale-control.complete"
log "verified stale-control seals=9/9"
