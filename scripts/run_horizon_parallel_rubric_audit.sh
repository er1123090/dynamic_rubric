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
SCORE_ROOT=${SCORE_ROOT:-${ARTIFACT_ROOT}/scores/seed-${TRAINING_SEED}}
IMMEDIATE_SCORE_ROOT=${IMMEDIATE_SCORE_ROOT:-${SCORE_ROOT}/immediate}
AUX_SCORE_ROOT=${AUX_SCORE_ROOT:-${ARTIFACT_ROOT}/scores_pool_a_auxiliary/seed-${TRAINING_SEED}}
RUBRIC_STATE_ROOT=${RUBRIC_STATE_ROOT:-${ARTIFACT_ROOT}/batch-active/seed-${TRAINING_SEED}}
PARALLEL_STATE_ROOT=${PARALLEL_STATE_ROOT:-${RUBRIC_STATE_ROOT}/parallel}
AUDIT_LOG_ROOT=${AUDIT_LOG_ROOT:-${ARTIFACT_ROOT}/audit/logs/parallel}
JUDGE_LOCK_ROOT=${JUDGE_LOCK_ROOT:-${ARTIFACT_ROOT}/audit/judge-locks}
JUDGE_BASE_URL=${JUDGE_BASE_URL:-http://127.0.0.1:8102}
JUDGE_WORKERS=${JUDGE_WORKERS:-2}
BATCH_POLL_INTERVAL_SECONDS=${BATCH_POLL_INTERVAL_SECONDS:-30}
OPENAI_BATCH_REQUESTS_PER_MINUTE=${OPENAI_BATCH_REQUESTS_PER_MINUTE:-240}
EXPECTED_PROMPT_COUNT=${EXPECTED_PROMPT_COUNT:-100}

checkpoint_steps=(3 6 9 13 16 24 32 40 48)
checkpoint_epochs=(0.2 0.4 0.6 0.8 1.0 1.5 2.0 2.5 3.0)
previous_steps=(0 3 6 9 13 16 24 32 40)

: "${OPENAI_API_KEY:?OPENAI_API_KEY is required}"
export DYNAMIC_RUBRIC_OPENAI_REQUESTS_PER_MINUTE="${OPENAI_BATCH_REQUESTS_PER_MINUTE}"
export DYNAMIC_RUBRIC_OPENAI_RATE_LIMIT_PATH="${ARTIFACT_ROOT}/audit/openai-rate-limit.lock"
[[ "${JUDGE_WORKERS}" =~ ^[1-9][0-9]*$ ]] || {
  echo "JUDGE_WORKERS must be a positive integer" >&2
  exit 2
}

mkdir -p \
  "${BASE_RUBRIC_ROOT}" "${SCORE_ROOT}" "${IMMEDIATE_SCORE_ROOT}" \
  "${AUX_SCORE_ROOT}" "${PARALLEL_STATE_ROOT}" "${AUDIT_LOG_ROOT}" \
  "${JUDGE_LOCK_ROOT}"

run_marker_root="${ARTIFACT_ROOT}/audit/parallel-runtime-${BASHPID}"
mkdir -p "${run_marker_root}"
child_pids=()

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
  actual=$(wc -l < "${path}")
  [[ "${actual}" -eq "${expected}" ]] || {
    echo "wrong ${label} row count: expected=${expected} actual=${actual} path=${path}" >&2
    return 1
  }
}

cleanup_children() {
  local status=$?
  if [[ "${status}" -ne 0 ]]; then
    local pid
    for pid in "${child_pids[@]:-}"; do
      kill "${pid}" 2>/dev/null || true
    done
  fi
}
trap cleanup_children EXIT INT TERM

grade_with_slot() {
  local slot
  while true; do
    for slot in $(seq 1 "${JUDGE_WORKERS}"); do
      exec 9>"${JUDGE_LOCK_ROOT}/slot-${slot}.lock"
      if flock -n 9; then
        local status=0
        "$@" || status=$?
        flock -u 9
        exec 9>&-
        return "${status}"
      fi
      exec 9>&-
    done
    sleep 2
  done
}

build_checkpoint_rubric() {
  local step=$1
  local output=$2
  local state_root=$3
  if [[ -f "${output}" ]]; then
    require_lines "${output}" "${EXPECTED_PROMPT_COUNT}" "step ${step} rubric"
    return 0
  fi
  local args=(
    build-horizon-rubrics
    --config "${CONFIG_PATH}"
    --run-id "rar-horizon-v1-${DOMAIN}-seed${TRAINING_SEED}-step${step}"
    --prompts "${PROMPTS}"
    --current-pool "${POOL_ROOT}/seed-${TRAINING_SEED}-step-${step}-pool-a.jsonl"
    --control-pool "${POOL_ROOT}/fixed.jsonl"
    --checkpoint-id "step${step}"
    --api-mode batch
    --batch-poll-interval-seconds "${BATCH_POLL_INTERVAL_SECONDS}"
    --rubric-state-root "${state_root}"
    --output "${output}"
  )
  if [[ "${step}" -eq 3 ]]; then
    args+=(--control-rubrics "${RUBRIC_ROOT}/sham.jsonl")
  fi
  run_cli "${args[@]}"
  require_lines "${output}" "${EXPECTED_PROMPT_COUNT}" "step ${step} rubric"
}

grade_current_checkpoint() {
  local step=$1
  local epoch=$2
  local rubric=$3
  local pool_b_output="${IMMEDIATE_SCORE_ROOT}/epoch-${epoch}"
  local pool_a_output="${AUX_SCORE_ROOT}/epoch-${epoch}"
  if [[ ! -f "${pool_a_output}/score_seal.json" ]]; then
    grade_with_slot run_cli grade-horizon \
      --config "${CONFIG_PATH}" \
      --run-id "rar-horizon-v1-${DOMAIN}-seed${TRAINING_SEED}-step${step}-grade-pool-a-aux" \
      --prompts "${PROMPTS}" \
      --pool-a "${POOL_ROOT}/seed-${TRAINING_SEED}-step-${step}-pool-a.jsonl" \
      --rubrics "${rubric}" \
      --checkpoint "${epoch}" \
      --r0-current-only \
      --base-url "${JUDGE_BASE_URL}" \
      --output-dir "${pool_a_output}" || return $?
  fi
  if [[ ! -f "${pool_b_output}/score_seal.json" ]]; then
    grade_with_slot run_cli grade-horizon \
      --config "${CONFIG_PATH}" \
      --run-id "rar-horizon-v1-${DOMAIN}-seed${TRAINING_SEED}-step${step}-grade-current" \
      --prompts "${PROMPTS}" \
      --pool-b "${POOL_ROOT}/seed-${TRAINING_SEED}-step-${step}-pool-b.jsonl" \
      --rubrics "${rubric}" \
      --checkpoint "${epoch}" \
      --r0-current-only \
      --base-url "${JUDGE_BASE_URL}" \
      --output-dir "${pool_b_output}" || return $?
  fi
}

grade_canonical_checkpoint() {
  local step=$1
  local epoch=$2
  local rubric=$3
  local immediate_dir="${IMMEDIATE_SCORE_ROOT}/epoch-${epoch}"
  local output_dir="${SCORE_ROOT}/epoch-${epoch}"
  while [[ ! -f "${immediate_dir}/score_seal.json" ]]; do
    if [[ -f "${run_marker_root}/current-${step}.failed" ]]; then
      echo "immediate grading failed for step ${step}" >&2
      return 1
    fi
    sleep 5
  done
  if [[ -f "${output_dir}/score_seal.json" ]]; then
    return 0
  fi
  grade_with_slot run_cli grade-horizon \
    --config "${CONFIG_PATH}" \
    --run-id "rar-horizon-v1-${DOMAIN}-seed${TRAINING_SEED}-step${step}-grade" \
    --prompts "${PROMPTS}" \
    --pool-b "${POOL_ROOT}/seed-${TRAINING_SEED}-step-${step}-pool-b.jsonl" \
    --rubrics "${rubric}" \
    --checkpoint "${epoch}" \
    --reuse-score-dir "${immediate_dir}" \
    --base-url "${JUDGE_BASE_URL}" \
    --output-dir "${output_dir}"
}

curl -fsS --max-time 10 "${JUDGE_BASE_URL}/health" >/dev/null
require_lines "${PROMPTS}" "${EXPECTED_PROMPT_COUNT}" "final prompts"
require_lines "${RUBRIC_ROOT}/sham.jsonl" "${EXPECTED_PROMPT_COUNT}" "sham rubric"

base_paths=()
build_pids=()
current_started=()
canonical_started=()
current_pids=()
canonical_pids=()

log "launching ${#checkpoint_steps[@]} checkpoint rubric builders concurrently"
for index in "${!checkpoint_steps[@]}"; do
  step=${checkpoint_steps[$index]}
  canonical="${RUBRIC_ROOT}/step-${step}.jsonl"
  if [[ "${step}" -eq 3 || -f "${canonical}" ]]; then
    base="${canonical}"
    state_root="${RUBRIC_STATE_ROOT}/step-${step}"
  else
    base="${BASE_RUBRIC_ROOT}/step-${step}.jsonl"
    state_root="${PARALLEL_STATE_ROOT}/step-${step}"
  fi
  base_paths[$index]="${base}"
  (
    if build_checkpoint_rubric "${step}" "${base}" "${state_root}"; then
      touch "${run_marker_root}/build-${step}.ready"
    else
      touch "${run_marker_root}/build-${step}.failed"
      exit 1
    fi
  ) >"${AUDIT_LOG_ROOT}/rubric-step-${step}.log" 2>&1 &
  build_pids[$index]=$!
  child_pids+=("${build_pids[$index]}")
done

while true; do
  launched_count=0
  canonical_count=0
  for index in "${!checkpoint_steps[@]}"; do
    step=${checkpoint_steps[$index]}
    epoch=${checkpoint_epochs[$index]}
    previous_step=${previous_steps[$index]}
    base=${base_paths[$index]}
    canonical="${RUBRIC_ROOT}/step-${step}.jsonl"

    if [[ -f "${run_marker_root}/build-${step}.failed" ]]; then
      echo "rubric generation failed for step ${step}; see ${AUDIT_LOG_ROOT}/rubric-step-${step}.log" >&2
      exit 1
    fi
    if [[ -f "${base}" ]] && [[ -z "${current_started[$index]:-}" ]]; then
      require_lines "${base}" "${EXPECTED_PROMPT_COUNT}" "step ${step} independent rubric"
      current_started[$index]=1
      (
        if grade_current_checkpoint "${step}" "${epoch}" "${base}"; then
          touch "${run_marker_root}/current-${step}.ready"
        else
          touch "${run_marker_root}/current-${step}.failed"
          exit 1
        fi
      ) >"${AUDIT_LOG_ROOT}/current-eval-step-${step}.log" 2>&1 &
      current_pids[$index]=$!
      child_pids+=("${current_pids[$index]}")
      log "step ${step} rubric ready; launched Pool A then immediate Pool B R0-vs-Rt grading"
    fi

    if [[ ! -f "${canonical}" ]] && [[ -f "${base}" ]]; then
      if [[ "${previous_step}" -eq 0 ]]; then
        control="${RUBRIC_ROOT}/sham.jsonl"
      else
        previous_index=$((index - 1))
        control=${base_paths[$previous_index]}
      fi
      if [[ -f "${control}" ]]; then
        run_cli attach-horizon-controls \
          --current-rubrics "${base}" \
          --control-rubrics "${control}" \
          --output "${canonical}" \
          >"${AUDIT_LOG_ROOT}/attach-control-step-${step}.log" 2>&1
      fi
    fi

    if [[ -f "${canonical}" ]]; then
      require_lines "${canonical}" "${EXPECTED_PROMPT_COUNT}" "step ${step} canonical rubric"
      canonical_count=$((canonical_count + 1))
      if [[ -z "${canonical_started[$index]:-}" ]]; then
        canonical_started[$index]=1
        (
          if grade_canonical_checkpoint "${step}" "${epoch}" "${canonical}"; then
            touch "${run_marker_root}/canonical-${step}.ready"
          else
            touch "${run_marker_root}/canonical-${step}.failed"
            exit 1
          fi
        ) >"${AUDIT_LOG_ROOT}/canonical-eval-step-${step}.log" 2>&1 &
        canonical_pids[$index]=$!
        child_pids+=("${canonical_pids[$index]}")
      fi
    fi
    if [[ -n "${current_started[$index]:-}" ]]; then
      launched_count=$((launched_count + 1))
    fi
  done
  if [[ "${launched_count}" -eq "${#checkpoint_steps[@]}" ]] \
    && [[ "${canonical_count}" -eq "${#checkpoint_steps[@]}" ]]; then
    break
  fi
  sleep 5
done

status=0
for pid in "${build_pids[@]}" "${current_pids[@]}" "${canonical_pids[@]}"; do
  wait "${pid}" || status=1
done
[[ "${status}" -eq 0 ]] || {
  echo "one or more parallel rubric/evaluation workers failed; inspect ${AUDIT_LOG_ROOT}" >&2
  exit 1
}

printf "complete\n" >"${RUBRIC_ROOT}/batch-parallel-rubrics.complete"
printf "complete\n" >"${AUX_SCORE_ROOT}/pool-a-r0-current.complete"
log "parallel rubric generation and completion-driven Pool A/Pool B grading complete"
