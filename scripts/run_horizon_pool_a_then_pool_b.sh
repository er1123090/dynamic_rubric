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
BASE_RUBRIC_ROOT=${BASE_RUBRIC_ROOT:-${RUBRIC_ROOT}/independent}
AUX_SCORE_ROOT=${AUX_SCORE_ROOT:-${ARTIFACT_ROOT}/scores_pool_a_auxiliary/seed-${TRAINING_SEED}}
IMMEDIATE_SCORE_ROOT=${IMMEDIATE_SCORE_ROOT:-${ARTIFACT_ROOT}/scores/seed-${TRAINING_SEED}/immediate}
AUDIT_LOG_ROOT=${AUDIT_LOG_ROOT:-${ARTIFACT_ROOT}/audit/logs/pool-a-then-pool-b}
JUDGE_BASE_URLS=${JUDGE_BASE_URLS:-http://127.0.0.1:8102 http://127.0.0.1:8103}
JUDGE_TASK_WEIGHTS=${JUDGE_TASK_WEIGHTS:-4 1}
EXPECTED_PROMPT_COUNT=${EXPECTED_PROMPT_COUNT:-100}
MAX_GRADE_ATTEMPTS=${MAX_GRADE_ATTEMPTS:-6}
GRADE_RETRY_DELAY_SECONDS=${GRADE_RETRY_DELAY_SECONDS:-30}

checkpoint_steps=(3 6 9 13 16 24 32 40 48)
checkpoint_epochs=(0.2 0.4 0.6 0.8 1.0 1.5 2.0 2.5 3.0)
read -r -a judge_urls <<<"${JUDGE_BASE_URLS}"
read -r -a judge_task_weights <<<"${JUDGE_TASK_WEIGHTS}"

[[ "${DOMAIN}" == "medicine" || "${DOMAIN}" == "science" ]] || {
  echo "DOMAIN must be medicine or science" >&2
  exit 2
}
((${#judge_urls[@]} > 0)) || { echo "JUDGE_BASE_URLS is empty" >&2; exit 2; }
[[ "${#judge_task_weights[@]}" -eq "${#judge_urls[@]}" ]] || {
  echo "JUDGE_TASK_WEIGHTS must align one-to-one with JUDGE_BASE_URLS" >&2
  exit 2
}
total_task_weight=0
for weight in "${judge_task_weights[@]}"; do
  [[ "${weight}" =~ ^[1-9][0-9]*$ ]] || {
    echo "JUDGE_TASK_WEIGHTS values must be positive integers" >&2
    exit 2
  }
  total_task_weight=$((total_task_weight + weight))
done
[[ "${MAX_GRADE_ATTEMPTS}" =~ ^[1-9][0-9]*$ ]] || {
  echo "MAX_GRADE_ATTEMPTS must be a positive integer" >&2
  exit 2
}
[[ "${GRADE_RETRY_DELAY_SECONDS}" =~ ^[0-9]+$ ]] || {
  echo "GRADE_RETRY_DELAY_SECONDS must be a non-negative integer" >&2
  exit 2
}
mkdir -p "${AUX_SCORE_ROOT}" "${IMMEDIATE_SCORE_ROOT}" "${AUDIT_LOG_ROOT}"

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

rubric_for_step() {
  local step=$1
  if [[ "${step}" -eq 3 || ! -f "${BASE_RUBRIC_ROOT}/step-${step}.jsonl" ]]; then
    printf "%s\n" "${RUBRIC_ROOT}/step-${step}.jsonl"
  else
    printf "%s\n" "${BASE_RUBRIC_ROOT}/step-${step}.jsonl"
  fi
}

stage_output_dir() {
  local stage=$1
  local epoch=$2
  if [[ "${stage}" == "pool-a" ]]; then
    printf "%s\n" "${AUX_SCORE_ROOT}/epoch-${epoch}"
  else
    printf "%s\n" "${IMMEDIATE_SCORE_ROOT}/epoch-${epoch}"
  fi
}

grade_checkpoint() {
  local stage=$1
  local index=$2
  local judge_url=$3
  local step=${checkpoint_steps[$index]}
  local epoch=${checkpoint_epochs[$index]}
  local rubric
  rubric=$(rubric_for_step "${step}")
  local output_dir
  output_dir=$(stage_output_dir "${stage}" "${epoch}")
  if [[ -f "${output_dir}/score_seal.json" ]]; then
    log "reuse ${stage} seal step=${step} epoch=${epoch}"
    return 0
  fi

  local pool_flag pool_path expected_responses run_suffix
  if [[ "${stage}" == "pool-a" ]]; then
    pool_flag=--pool-a
    pool_path="${POOL_ROOT}/seed-${TRAINING_SEED}-step-${step}-pool-a.jsonl"
    expected_responses=$((EXPECTED_PROMPT_COUNT * 8))
    run_suffix=grade-pool-a-aux
  else
    pool_flag=--pool-b
    pool_path="${POOL_ROOT}/seed-${TRAINING_SEED}-step-${step}-pool-b.jsonl"
    expected_responses=$((EXPECTED_PROMPT_COUNT * 16))
    run_suffix=grade-current
  fi
  require_lines "${rubric}" "${EXPECTED_PROMPT_COUNT}" "step ${step} rubric"
  require_lines "${pool_path}" "${expected_responses}" "step ${step} ${stage} responses"
  log "start ${stage} step=${step} epoch=${epoch} judge=${judge_url}"
  local attempt=1
  local active_judge_url=${judge_url}
  local fallback_judge_url=${judge_urls[0]}
  local fallback_used=0
  while ! run_cli grade-horizon \
    --config "${CONFIG_PATH}" \
    --run-id "rar-horizon-v1-${DOMAIN}-seed${TRAINING_SEED}-step${step}-${run_suffix}" \
    --prompts "${PROMPTS}" \
    "${pool_flag}" "${pool_path}" \
    --rubrics "${rubric}" \
    --checkpoint "${epoch}" \
    --r0-current-only \
    --base-url "${active_judge_url}" \
    --output-dir "${output_dir}"
  do
    if [[ "${attempt}" -ge "${MAX_GRADE_ATTEMPTS}" ]]; then
      if [[ "${fallback_used}" -eq 0 && "${active_judge_url}" != "${fallback_judge_url}" ]]; then
        fallback_used=1
        active_judge_url=${fallback_judge_url}
        attempt=1
        log "fail over ${stage} step=${step} judge=${active_judge_url}"
        continue
      fi
      echo "${stage} grading failed after ${attempt} attempts: step=${step}" >&2
      return 1
    fi
    attempt=$((attempt + 1))
    log "retry ${stage} step=${step} attempt=${attempt}/${MAX_GRADE_ATTEMPTS}"
    sleep "${GRADE_RETRY_DELAY_SECONDS}"
  done
  [[ -f "${output_dir}/score_seal.json" ]] || {
    echo "missing ${stage} score seal after grading: step=${step}" >&2
    return 1
  }
  log "complete ${stage} step=${step} epoch=${epoch}"
}

verify_stage() {
  local stage=$1
  local complete=0
  local index epoch output_dir
  for index in "${!checkpoint_steps[@]}"; do
    epoch=${checkpoint_epochs[$index]}
    output_dir=$(stage_output_dir "${stage}" "${epoch}")
    [[ -f "${output_dir}/score_seal.json" ]] && complete=$((complete + 1))
  done
  [[ "${complete}" -eq "${#checkpoint_steps[@]}" ]] || {
    echo "${stage} incomplete: seals=${complete}/${#checkpoint_steps[@]}" >&2
    return 1
  }
  log "verified ${stage}: seals=${complete}/${#checkpoint_steps[@]}"
}

worker_for_checkpoint_index() {
  local index=$1
  local slot=$((index % total_task_weight))
  local worker_index cumulative=0
  for worker_index in "${!judge_task_weights[@]}"; do
    cumulative=$((cumulative + judge_task_weights[worker_index]))
    if ((slot < cumulative)); then
      printf "%s\n" "${worker_index}"
      return 0
    fi
  done
  return 1
}

run_stage() {
  local stage=$1
  local worker_index index
  local pids=()
  log "begin ${stage} with ${#judge_urls[@]} judge workers"
  for worker_index in "${!judge_urls[@]}"; do
    (
      for index in "${!checkpoint_steps[@]}"; do
        [[ "$(worker_for_checkpoint_index "${index}")" -eq "${worker_index}" ]] || continue
        grade_checkpoint "${stage}" "${index}" "${judge_urls[$worker_index]}"
      done
    ) >"${AUDIT_LOG_ROOT}/${stage}-worker-${worker_index}.log" 2>&1 &
    pids+=("$!")
  done
  local status=0 pid
  for pid in "${pids[@]}"; do
    wait "${pid}" || status=1
  done
  [[ "${status}" -eq 0 ]] || {
    echo "${stage} worker failed; inspect ${AUDIT_LOG_ROOT}/${stage}-worker-*.log" >&2
    return 1
  }
  verify_stage "${stage}"
}

require_lines "${PROMPTS}" "${EXPECTED_PROMPT_COUNT}" "final prompts"
for judge_url in "${judge_urls[@]}"; do
  curl -fsS --max-time 10 "${judge_url}/health" >/dev/null
done

run_stage pool-a
run_stage pool-b
log "Pool A and Pool B judging complete"
