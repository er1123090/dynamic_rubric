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
RUBRIC_API_MODE=${RUBRIC_API_MODE:-sync}
RUBRIC_STATE_ROOT=${RUBRIC_STATE_ROOT:-${ARTIFACT_ROOT}/${RUBRIC_API_MODE}/seed-${TRAINING_SEED}}
RUBRIC_SYNC_CONCURRENCY=${RUBRIC_SYNC_CONCURRENCY:-16}
RUBRIC_BATCH_POLL_INTERVAL_SECONDS=${RUBRIC_BATCH_POLL_INTERVAL_SECONDS:-30}

: "${OPENAI_API_KEY:?OPENAI_API_KEY is required}"
case "${RUBRIC_API_MODE}" in
  sync) rubric_api_args=(--api-mode sync --sync-concurrency "${RUBRIC_SYNC_CONCURRENCY}") ;;
  batch) rubric_api_args=(--api-mode batch --batch-poll-interval-seconds "${RUBRIC_BATCH_POLL_INTERVAL_SECONDS}") ;;
  *) echo "unsupported RUBRIC_API_MODE: ${RUBRIC_API_MODE}" >&2; exit 2 ;;
esac
mkdir -p "${RUBRIC_ROOT}" "${RUBRIC_STATE_ROOT}"

run_cli() {
  PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
    uv run --project "${PROJECT_ROOT}" python -m dynamic_rubric "$@"
}

require_lines() {
  local path=$1
  local expected=$2
  [[ -f "${path}" ]] || { echo "missing artifact: ${path}" >&2; return 1; }
  local actual
  actual=$(wc -l < "${path}")
  [[ "${actual}" -eq "${expected}" ]] || {
    echo "wrong row count: expected=${expected} actual=${actual} path=${path}" >&2
    return 1
  }
}

log() {
  printf '%s %s
' "$(date --iso-8601=seconds)" "$*"
}

sham_output="${RUBRIC_ROOT}/sham.jsonl"
if [[ ! -f "${sham_output}" ]]; then
  log "resuming sham rubric dedup"
  run_cli build-horizon-rubrics \
    --config "${CONFIG_PATH}" \
    --run-id "rar-horizon-v1-${DOMAIN}-seed${TRAINING_SEED}-sham" \
    --prompts "${PROMPTS}" \
    --current-pool "${POOL_ROOT}/sham.jsonl" \
    --control-pool "${POOL_ROOT}/fixed.jsonl" \
    --checkpoint-id step0 \
    "${rubric_api_args[@]}" \
    --rubric-state-root "${RUBRIC_STATE_ROOT}/sham" \
    --output "${sham_output}"
fi
require_lines "${sham_output}" 100

prior_rubric=${sham_output}
for step in 3 6 9 13 16 24 32 40 48; do
  output="${RUBRIC_ROOT}/step-${step}.jsonl"
  if [[ ! -f "${output}" ]]; then
    log "building dynamic rubric for step ${step}"
    run_cli build-horizon-rubrics \
      --config "${CONFIG_PATH}" \
      --run-id "rar-horizon-v1-${DOMAIN}-seed${TRAINING_SEED}-step${step}" \
      --prompts "${PROMPTS}" \
      --current-pool "${POOL_ROOT}/seed-${TRAINING_SEED}-step-${step}-pool-a.jsonl" \
      --control-pool "${POOL_ROOT}/fixed.jsonl" \
      --control-rubrics "${prior_rubric}" \
      --checkpoint-id "step${step}" \
      "${rubric_api_args[@]}" \
      --rubric-state-root "${RUBRIC_STATE_ROOT}/step-${step}" \
      --output "${output}"
  fi
  require_lines "${output}" 100
  prior_rubric=${output}
done

printf 'complete
' > "${RUBRIC_ROOT}/${RUBRIC_API_MODE}-rubrics.complete"
log "all dynamic rubrics complete via ${RUBRIC_API_MODE}"
