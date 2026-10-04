#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
CONFIG_PATH=${CONFIG_PATH:-${PROJECT_ROOT}/configs/horizon_medicine.yaml}
PROMPTS=${PROMPTS:-${PROJECT_ROOT}/data/rar/medicine/public/final.jsonl}
POOL_ROOT=${POOL_ROOT:-${PROJECT_ROOT}/artifacts/horizon/medicine/pools_combined}
RUBRIC_ROOT=${RUBRIC_ROOT:-${PROJECT_ROOT}/artifacts/horizon/medicine/rubrics/seed-11}
SCORE_ROOT=${SCORE_ROOT:-${PROJECT_ROOT}/artifacts/horizon/medicine/scores_pool_a_combined/seed-11}
LOG_ROOT=${LOG_ROOT:-${PROJECT_ROOT}/artifacts/logs/pool-a-combined}
PYTHON_BIN=${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}

mkdir -p "${SCORE_ROOT}" "${LOG_ROOT}"

run_one() {
  local base_url=$1
  local step=$2
  local epoch=$3
  local pool="${POOL_ROOT}/seed-11-step-${step}-pool-a-combined.jsonl"
  local rubric="${RUBRIC_ROOT}/step-${step}.jsonl"
  local output_dir="${SCORE_ROOT}/epoch-${epoch}"
  local log="${LOG_ROOT}/step-${step}.log"

  if [[ -f "${output_dir}/score_seal.json" ]]; then
    echo "[$(date -Is)] step=${step} already sealed"
    return
  fi
  [[ $(wc -l < "${pool}") -eq 1600 ]]
  [[ $(wc -l < "${rubric}") -eq 100 ]]
  echo "[$(date -Is)] step=${step} epoch=${epoch} judge=${base_url} start"
  PYTHONPATH="${PROJECT_ROOT}/src" PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -m dynamic_rubric grade-horizon     --config "${CONFIG_PATH}"     --run-id "rar-horizon-v1-medicine-seed11-step${step}-grade-pool-a-combined"     --prompts "${PROMPTS}"     --pool-a-combined "${pool}"     --rubrics "${rubric}"     --checkpoint "${epoch}"     --r0-current-only     --base-url "${base_url}"     --output-dir "${output_dir}" >"${log}" 2>&1
  [[ -f "${output_dir}/score_seal.json" ]]
  echo "[$(date -Is)] step=${step} epoch=${epoch} judge=${base_url} sealed"
}

QUEUE_ROOT="${LOG_ROOT}/queue-$$"
mkdir -p "${QUEUE_ROOT}"
printf '%s\n' 3:0.2 6:0.4 9:0.6 13:0.8 16:1.0 24:1.5 32:2.0 40:2.5 48:3.0 >"${QUEUE_ROOT}/tasks"
printf '0\n' >"${QUEUE_ROOT}/cursor"

claim_task() {
  local index task
  exec 9>"${QUEUE_ROOT}/lock"
  flock 9
  index=$(<"${QUEUE_ROOT}/cursor")
  task=$(sed -n "$((index + 1))p" "${QUEUE_ROOT}/tasks")
  if [[ -n "${task}" ]]; then
    printf '%s\n' "$((index + 1))" >"${QUEUE_ROOT}/cursor"
  fi
  flock -u 9
  printf '%s' "${task}"
}

worker() {
  local base_url=$1
  curl -fsS --max-time 5 "${base_url}/health" >/dev/null
  local item step epoch
  while item=$(claim_task) && [[ -n "${item}" ]]; do
    step=${item%%:*}
    epoch=${item#*:}
    run_one "${base_url}" "${step}" "${epoch}"
  done
}

worker http://127.0.0.1:8108 &
pid_aggregate=$!
wait "${pid_aggregate}"

PYTHONPATH="${PROJECT_ROOT}/src" "${PYTHON_BIN}" - "${SCORE_ROOT}" <<'PYVERIFY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
seals = sorted(root.glob('epoch-*/score_seal.json'))
if len(seals) != 9:
    raise SystemExit(f'expected 9 score seals, found {len(seals)}')
for seal_path in seals:
    seal = json.loads(seal_path.read_text())
    if seal['pool_family'] != 'pool_a_combined' or seal['prompt_count'] != 100 or seal['response_count'] != 1600:
        raise SystemExit(f'invalid seal: {seal_path}')
    summaries = [json.loads(line) for line in (seal_path.parent / 'prompt_summary.jsonl').read_text().splitlines()]
    if any(row['response_count'] != 16 for row in summaries):
        raise SystemExit(f'invalid response count: {seal_path}')
    if any(row['variants']['current']['pairwise']['pair_count'] != 120 for row in summaries):
        raise SystemExit(f'invalid pair count: {seal_path}')
print(f'verified {len(seals)} Pool A-combined score seals')
PYVERIFY
