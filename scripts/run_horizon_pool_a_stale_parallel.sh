#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
CONFIG_PATH=${CONFIG_PATH:-${PROJECT_ROOT}/configs/horizon_medicine.yaml}
PROMPTS=${PROMPTS:-${PROJECT_ROOT}/data/rar/medicine/public/final.jsonl}
POOL_ROOT=${POOL_ROOT:-${PROJECT_ROOT}/artifacts/horizon/medicine/pools_combined}
RUBRIC_ROOT=${RUBRIC_ROOT:-${PROJECT_ROOT}/artifacts/horizon/medicine/rubrics/seed-11}
REUSE_ROOT=${REUSE_ROOT:-${PROJECT_ROOT}/artifacts/horizon/medicine/scores_pool_a_combined/seed-11}
OUTPUT_ROOT=${OUTPUT_ROOT:-${PROJECT_ROOT}/artifacts/horizon/medicine/scores_pool_a_combined_stale/seed-11}
LOG_ROOT=${LOG_ROOT:-${PROJECT_ROOT}/artifacts/logs/pool-a-combined-stale}
PYTHON_BIN=${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-3}

# Step 3 uses a sham rubric and is intentionally excluded from true-stale analysis.
tasks=(6:0.4 9:0.6 13:0.8 16:1.0 24:1.5 32:2.0 40:2.5 48:3.0)
judge_urls=(
  "${JUDGE_URL_1:-http://127.0.0.1:8102}"
  "${JUDGE_URL_2:-http://127.0.0.1:8104}"
  "${JUDGE_URL_3:-http://127.0.0.1:8108}"
)

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"
queue_root=$(mktemp -d "${LOG_ROOT}/queue.XXXXXX")
printf '%s\n' "${tasks[@]}" >"${queue_root}/tasks"
printf '0\n' >"${queue_root}/cursor"

claim_task() {
  local index task
  exec 9>"${queue_root}/lock"
  flock 9
  index=$(<"${queue_root}/cursor")
  task=$(sed -n "$((index + 1))p" "${queue_root}/tasks")
  if [[ -n "${task}" ]]; then
    printf '%s\n' "$((index + 1))" >"${queue_root}/cursor"
  fi
  flock -u 9
  printf '%s' "${task}"
}

grade_one() {
  local base_url=$1
  local step=$2
  local epoch=$3
  local pool="${POOL_ROOT}/seed-11-step-${step}-pool-a-combined.jsonl"
  local rubric="${RUBRIC_ROOT}/step-${step}.jsonl"
  local reuse_dir="${REUSE_ROOT}/epoch-${epoch}"
  local output_dir="${OUTPUT_ROOT}/epoch-${epoch}"
  local attempt=1

  if [[ -f "${output_dir}/score_seal.json" ]]; then
    echo "[$(date -Is)] step=${step} already sealed"
    return
  fi
  [[ $(wc -l <"${pool}") -eq 1600 ]]
  [[ $(wc -l <"${rubric}") -eq 100 ]]
  [[ -f "${reuse_dir}/score_seal.json" ]]

  while (( attempt <= MAX_ATTEMPTS )); do
    echo "[$(date -Is)] step=${step} epoch=${epoch} judge=${base_url} attempt=${attempt}"
    if PYTHONPATH="${PROJECT_ROOT}/src" PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -m dynamic_rubric grade-horizon \
      --config "${CONFIG_PATH}" \
      --run-id "rar-horizon-v1-medicine-seed11-step${step}-grade-pool-a-combined-stale" \
      --prompts "${PROMPTS}" \
      --pool-a-combined "${pool}" \
      --rubrics "${rubric}" \
      --checkpoint "${epoch}" \
      --reuse-score-dir "${reuse_dir}" \
      --base-url "${base_url}" \
      --output-dir "${output_dir}"; then
      [[ -f "${output_dir}/score_seal.json" ]]
      echo "[$(date -Is)] step=${step} sealed"
      return
    fi
    attempt=$((attempt + 1))
  done
  echo "step ${step} failed after ${MAX_ATTEMPTS} attempts" >&2
  return 1
}

worker() {
  local base_url=$1
  local item step epoch
  curl -fsS --max-time 5 "${base_url}/health" >/dev/null
  while item=$(claim_task) && [[ -n "${item}" ]]; do
    step=${item%%:*}
    epoch=${item#*:}
    grade_one "${base_url}" "${step}" "${epoch}"
  done
}

pids=()
for index in "${!judge_urls[@]}"; do
  worker "${judge_urls[$index]}" >"${LOG_ROOT}/worker-${index}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
(( status == 0 )) || exit "${status}"

"${PYTHON_BIN}" - "${OUTPUT_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
seals = sorted(root.glob("epoch-*/score_seal.json"))
if len(seals) != 8:
    raise SystemExit(f"expected 8 true-stale seals, found {len(seals)}")
for path in seals:
    seal = json.loads(path.read_text())
    if seal["comparison_scope"] != "full" or seal["pool_family"] != "pool_a_combined":
        raise SystemExit(f"invalid Pool A stale seal: {path}")
    if seal["response_count"] != 1600 or seal["prompt_count"] != 100:
        raise SystemExit(f"invalid Pool A stale coverage: {path}")
print(f"verified {len(seals)} Pool A combined true-stale score seals")
PY
