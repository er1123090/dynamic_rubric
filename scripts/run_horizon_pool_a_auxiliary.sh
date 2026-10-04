#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
DOMAIN=${DOMAIN:?DOMAIN must be medicine or science}
TRAINING_SEED=${TRAINING_SEED:-11}
CONFIG_PATH=${CONFIG_PATH:-${PROJECT_ROOT}/configs/horizon_${DOMAIN}.yaml}
PROMPTS=${PROMPTS:-${PROJECT_ROOT}/data/rar/${DOMAIN}/public/final.jsonl}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts/horizon/${DOMAIN}}
POOL_ROOT=${POOL_ROOT:-${ARTIFACT_ROOT}/pools}
RUBRIC_ROOT=${RUBRIC_ROOT:-${ARTIFACT_ROOT}/rubrics/seed-${TRAINING_SEED}}
AUX_SCORE_ROOT=${AUX_SCORE_ROOT:-${ARTIFACT_ROOT}/scores_pool_a_auxiliary/seed-${TRAINING_SEED}}
JUDGE_BASE_URL=${JUDGE_BASE_URL:-http://127.0.0.1:8102}
TARGET_STEP=${TARGET_STEP:-}

case "${DOMAIN}" in
  medicine|science) ;;
  *) echo "DOMAIN must be medicine or science" >&2; exit 2 ;;
esac

checkpoint_steps=(3 6 9 13 16 24 32 40 48)
checkpoint_epochs=(0.2 0.4 0.6 0.8 1.0 1.5 2.0 2.5 3.0)
expected_prompt_count=100
mkdir -p "${AUX_SCORE_ROOT}"

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
    echo "wrong ${label} row count: expected=${expected}, actual=${actual}, path=${path}" >&2
    return 1
  }
}

curl -fsS --max-time 5 "${JUDGE_BASE_URL}/health" >/dev/null

for index in "${!checkpoint_steps[@]}"; do
  step=${checkpoint_steps[$index]}
  epoch=${checkpoint_epochs[$index]}
  if [[ -n "${TARGET_STEP}" && "${step}" != "${TARGET_STEP}" ]]; then
    continue
  fi
  rubric="${RUBRIC_ROOT}/step-${step}.jsonl"
  pool_a="${POOL_ROOT}/seed-${TRAINING_SEED}-step-${step}-pool-a.jsonl"
  output_dir="${AUX_SCORE_ROOT}/epoch-${epoch}"

  require_lines "${rubric}" "${expected_prompt_count}" "step ${step} rubric"
  require_lines "${pool_a}" "$((expected_prompt_count * 8))" "step ${step} Pool A"
  if [[ -f "${output_dir}/score_seal.json" ]]; then
    echo "reusing Pool A auxiliary score seal: step=${step} epoch=${epoch}"
    continue
  fi

  echo "grading Pool A auxiliary shard: step=${step} epoch=${epoch} responses=800"
  run_cli grade-horizon \
    --config "${CONFIG_PATH}" \
    --run-id "rar-horizon-v1-${DOMAIN}-seed${TRAINING_SEED}-step${step}-grade-pool-a-aux" \
    --prompts "${PROMPTS}" \
    --pool-a "${pool_a}" \
    --rubrics "${rubric}" \
    --checkpoint "${epoch}" \
    --r0-current-only \
    --base-url "${JUDGE_BASE_URL}" \
    --output-dir "${output_dir}"
  [[ -f "${output_dir}/score_seal.json" ]]
done

if [[ -n "${TARGET_STEP}" ]] && [[ ! " ${checkpoint_steps[*]} " =~ " ${TARGET_STEP} " ]]; then
  echo "TARGET_STEP is not a scheduled nonzero checkpoint: ${TARGET_STEP}" >&2
  exit 2
fi

echo "Pool A auxiliary rubric comparison complete: ${AUX_SCORE_ROOT}"
