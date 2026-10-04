#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
TRAINING_SEED=${TRAINING_SEED:-11}
POLICY_GPU=${POLICY_GPU:-0}
RESUME_STEP=${RESUME_STEP:-40}
RUN_DIR=${RUN_DIR:-${PROJECT_ROOT}/artifacts/horizon/medicine/training/seed-${TRAINING_SEED}}
RESUME_FROM_PATH=${RESUME_FROM_PATH:-${RUN_DIR}/checkpoints/global_step_${RESUME_STEP}}
RUNTIME_WRAPPER=${RUNTIME_WRAPPER:-${PROJECT_ROOT}/scripts/horizon_single_gpu_runtime_wrapper.sh}

[[ -s "${RESUME_FROM_PATH}/actor/model_world_size_1_rank_0.pt" ]] || {
  echo "resume model is missing: ${RESUME_FROM_PATH}" >&2
  exit 1
}
[[ -s "${RESUME_FROM_PATH}/actor/optim_world_size_1_rank_0.pt" ]] || {
  echo "resume optimizer is missing: ${RESUME_FROM_PATH}" >&2
  exit 1
}
curl -fsS --max-time 30 http://127.0.0.1:8102/health >/dev/null

exec env \
  DOMAIN=medicine \
  TRAINING_SEED="${TRAINING_SEED}" \
  POLICY_GPU="${POLICY_GPU}" \
  RUN_DIR="${RUN_DIR}" \
  RUNTIME_PYTHON="${RUNTIME_WRAPPER}" \
  REAL_RUNTIME_PYTHON="${PROJECT_ROOT}/environment/upstream/verl/.venv-runtime/bin/python" \
  RESUME_MODE=resume_path \
  RESUME_FROM_PATH="${RESUME_FROM_PATH}" \
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
  DYNAMIC_RUBRIC_VLLM_URL=http://127.0.0.1:8102 \
  DYNAMIC_RUBRIC_GRADER_MODEL=Qwen/Qwen3-32B \
  DYNAMIC_RUBRIC_GRADER_REVISION=9216db5781bf21249d130ec9da846c4624c16137 \
  DYNAMIC_RUBRIC_GRADER_TOKENIZER_REVISION=9216db5781bf21249d130ec9da846c4624c16137 \
  DYNAMIC_RUBRIC_GRADER_TIMEOUT_SECONDS=1800 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  bash "${PROJECT_ROOT}/scripts/run_horizon_static_grpo.sh"
