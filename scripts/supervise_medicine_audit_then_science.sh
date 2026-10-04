#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
MEDICINE_TRAIN_PID=${MEDICINE_TRAIN_PID:?MEDICINE_TRAIN_PID is required}
MEDICINE_MARKER=${MEDICINE_MARKER:-${PROJECT_ROOT}/artifacts/horizon/medicine/training/seed-11/checkpoints/latest_checkpointed_iteration.txt}
MEDICINE_AUDIT_MARKER=${MEDICINE_AUDIT_MARKER:-${PROJECT_ROOT}/artifacts/horizon/medicine/audit/seed-11.complete}
POLICY_GPU=${POLICY_GPU:-0}

while kill -0 "${MEDICINE_TRAIN_PID}" 2>/dev/null; do
  sleep 60
done

marker=$(<"${MEDICINE_MARKER}")
if [[ "${marker}" != "48" ]]; then
  echo "medicine_not_complete marker=${marker}; audit_and_science_not_started" >&2
  exit 1
fi

for attempt in $(seq 1 120); do
  used=$(nvidia-smi --id="${POLICY_GPU}" --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null | awk '{sum += $1} END {print sum + 0}')
  if [[ "${used}" -lt 2048 ]]; then
    break
  fi
  if [[ "${attempt}" -eq 120 ]]; then
    echo "Trainer GPU ${POLICY_GPU} did not become free after Medicine training" >&2
    exit 1
  fi
  sleep 15
done

env \
  DOMAIN=medicine \
  TRAINING_SEED=11 \
  POLICY_GPU="${POLICY_GPU}" \
  JUDGE_BASE_URL=http://127.0.0.1:8102 \
  bash "${PROJECT_ROOT}/scripts/run_horizon_audit.sh"

if [[ ! -s "${MEDICINE_AUDIT_MARKER}" ]]; then
  echo "Medicine audit completion marker is missing; Science not started" >&2
  exit 1
fi

exec env \
  DOMAIN=science \
  TRAINING_SEED=11 \
  POLICY_GPU="${POLICY_GPU}" \
  RESUME_MODE=disable \
  RESUME_FROM_PATH=null \
  ACTOR_MAX_TOKEN_LEN=12288 \
  ENABLE_GRADIENT_CHECKPOINTING=True \
  ROLLOUT_LOG_PROB_MAX_TOKEN_LEN=49152 \
  REF_LOG_PROB_MAX_TOKEN_LEN=65536 \
  DYNAMIC_RUBRIC_VLLM_URL=http://127.0.0.1:8102 \
  DYNAMIC_RUBRIC_GRADER_MODEL=Qwen/Qwen3-32B \
  DYNAMIC_RUBRIC_GRADER_REVISION=9216db5781bf21249d130ec9da846c4624c16137 \
  DYNAMIC_RUBRIC_GRADER_TOKENIZER_REVISION=9216db5781bf21249d130ec9da846c4624c16137 \
  DYNAMIC_RUBRIC_GRADER_TIMEOUT_SECONDS=900 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  bash "${PROJECT_ROOT}/scripts/run_horizon_static_grpo.sh"
