#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
MEDICINE_TRAIN_PID=${MEDICINE_TRAIN_PID:?MEDICINE_TRAIN_PID is required}
MEDICINE_MARKER=${MEDICINE_MARKER:-${PROJECT_ROOT}/artifacts/horizon/medicine/training/seed-11/checkpoints/latest_checkpointed_iteration.txt}
MEDICINE_AUDIT_MARKER=${MEDICINE_AUDIT_MARKER:-${PROJECT_ROOT}/artifacts/horizon/medicine/audit/seed-11.complete}
POLICY_GPU=${POLICY_GPU:-0}
JUDGE_BASE_URL=${JUDGE_BASE_URL:-http://127.0.0.1:8102}
RUNTIME_WRAPPER=${RUNTIME_WRAPPER:-${PROJECT_ROOT}/scripts/horizon_single_gpu_runtime_wrapper.sh}
REAL_RUNTIME_PYTHON=${REAL_RUNTIME_PYTHON:-${PROJECT_ROOT}/environment/upstream/verl/.venv-runtime/bin/python}
JUDGE_IDLE_GPU_MAX_MIB=${JUDGE_IDLE_GPU_MAX_MIB:-81920}

while kill -0 "${MEDICINE_TRAIN_PID}" 2>/dev/null; do
  sleep 60
done

marker=$(<"${MEDICINE_MARKER}")
if [[ "${marker}" != "48" ]]; then
  echo "medicine_not_complete marker=${marker}; kl_audit_and_science_not_started" >&2
  exit 1
fi

for attempt in $(seq 1 120); do
  used=$(nvidia-smi -i "${POLICY_GPU}" --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null | awk '{sum += $1} END {print sum + 0}')
  if [[ "${used}" -le "${JUDGE_IDLE_GPU_MAX_MIB}" ]]; then
    break
  fi
  if [[ "${attempt}" -eq 120 ]]; then
    echo "Trainer GPU ${POLICY_GPU} retained ${used} MiB after Medicine training; expected only the resident judge" >&2
    exit 1
  fi
  sleep 15
done

curl -fsS --max-time 30 "${JUDGE_BASE_URL}/health" >/dev/null

env \
  DOMAIN=medicine \
  TRAINING_SEED=11 \
  POLICY_GPU="${POLICY_GPU}" \
  JUDGE_BASE_URL="${JUDGE_BASE_URL}" \
  RUNTIME_PYTHON="${RUNTIME_WRAPPER}" \
  REAL_RUNTIME_PYTHON="${REAL_RUNTIME_PYTHON}" \
  bash "${PROJECT_ROOT}/scripts/run_horizon_checkpoint_kl_then_audit.sh"

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
  RUNTIME_PYTHON="${RUNTIME_WRAPPER}" \
  REAL_RUNTIME_PYTHON="${REAL_RUNTIME_PYTHON}" \
  ACTOR_MAX_TOKEN_LEN=12288 \
  ENABLE_GRADIENT_CHECKPOINTING=True \
  ROLLOUT_LOG_PROB_MAX_TOKEN_LEN=49152 \
  REF_LOG_PROB_MAX_TOKEN_LEN=65536 \
  REF_PARAM_OFFLOAD=True \
  ROLLOUT_GPU_MEMORY=0.20 \
  ROLLOUT_MAX_NUM_SEQS=32 \
  ROLLOUT_MAX_NUM_BATCHED_TOKENS=8192 \
  ROLLOUT_AGENT_NUM_WORKERS=4 \
  REWARD_NUM_WORKERS=8 \
  ENFORCE_EAGER=True \
  DYNAMIC_RUBRIC_VLLM_URL="${JUDGE_BASE_URL}" \
  DYNAMIC_RUBRIC_GRADER_MODEL=Qwen/Qwen3-32B \
  DYNAMIC_RUBRIC_GRADER_REVISION=9216db5781bf21249d130ec9da846c4624c16137 \
  DYNAMIC_RUBRIC_GRADER_TOKENIZER_REVISION=9216db5781bf21249d130ec9da846c4624c16137 \
  DYNAMIC_RUBRIC_GRADER_TIMEOUT_SECONDS=1800 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  bash "${PROJECT_ROOT}/scripts/run_horizon_static_grpo.sh"
