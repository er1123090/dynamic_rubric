#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
MODEL_PATH=${MODEL_PATH:-${MODEL_PATH}}
RUNTIME_PYTHON=${RUNTIME_PYTHON:-${PROJECT_ROOT}/environment/upstream/verl/.venv-runtime/bin/python}
DOMAIN=${DOMAIN:?DOMAIN must be medicine or science}
TRAINING_SEED=${TRAINING_SEED:?TRAINING_SEED is required}

case "${DOMAIN}" in
  medicine|science) ;;
  *) echo "DOMAIN must be medicine or science" >&2; exit 2 ;;
esac

export MODEL_PATH
export CONFIG_PATH=${CONFIG_PATH:-${PROJECT_ROOT}/configs/horizon_${DOMAIN}.yaml}
export TRAIN_FILE=${TRAIN_FILE:-${PROJECT_ROOT}/data/rar/${DOMAIN}/verl/train.parquet}
export VAL_FILE=${VAL_FILE:-${PROJECT_ROOT}/data/rar/${DOMAIN}/verl/development.parquet}
export DYNAMIC_RUBRIC_STATIC_RUBRIC_PATH=${DYNAMIC_RUBRIC_STATIC_RUBRIC_PATH:-${PROJECT_ROOT}/data/rar/${DOMAIN}/public/train.jsonl}
export RUN_DIR=${RUN_DIR:-${PROJECT_ROOT}/artifacts/horizon/${DOMAIN}/training/seed-${TRAINING_SEED}}
export TOTAL_STEPS=${TOTAL_STEPS:-48}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-96}
export ROLLOUT_N=${ROLLOUT_N:-16}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-96}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-3584}
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-7680}
export ACTOR_MAX_TOKEN_LEN=${ACTOR_MAX_TOKEN_LEN:-24576}
export ROLLOUT_LOG_PROB_MAX_TOKEN_LEN=${ROLLOUT_LOG_PROB_MAX_TOKEN_LEN:-49152}
export REF_LOG_PROB_MAX_TOKEN_LEN=${REF_LOG_PROB_MAX_TOKEN_LEN:-65536}
export ROLLOUT_GPU_MEMORY=${ROLLOUT_GPU_MEMORY:-0.55}
export ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-128}
export ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-16384}
export ROLLOUT_AGENT_NUM_WORKERS=${ROLLOUT_AGENT_NUM_WORKERS:-16}
export REWARD_NUM_WORKERS=${REWARD_NUM_WORKERS:-32}
export USE_FUSED_KERNELS=${USE_FUSED_KERNELS:-True}
export FUSED_KERNEL_BACKEND=${FUSED_KERNEL_BACKEND:-torch}
export ENABLE_GRADIENT_CHECKPOINTING=${ENABLE_GRADIENT_CHECKPOINTING:-False}
export FULL_DETERMINISM=${FULL_DETERMINISM:-False}
export ENFORCE_EAGER=${ENFORCE_EAGER:-False}
export REF_PARAM_OFFLOAD=${REF_PARAM_OFFLOAD:-False}
export LEARNING_RATE=${LEARNING_RATE:-5e-6}
export WARMUP_RATIO=${WARMUP_RATIO:-0.1}
export KL_COEFFICIENT=${KL_COEFFICIENT:-0.01}
export ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-1.0}
export ROLLOUT_TOP_P=${ROLLOUT_TOP_P:-0.95}
export CHECKPOINT_STEPS=${CHECKPOINT_STEPS:-"[0,3,6,9,13,16,24,32,40,48]"}
export SAVE_FREQ=${SAVE_FREQ:-3}
export PYTHONHASHSEED=${PYTHONHASHSEED:-${TRAINING_SEED}}

checkpoint_root="${RUN_DIR}/checkpoints"
pruner_script="${PROJECT_ROOT}/scripts/prune_horizon_checkpoint_state.py"

"${RUNTIME_PYTHON}" "${pruner_script}" \
  --checkpoint-root "${checkpoint_root}" \
  --watch \
  --while-pid "$$" &
pruner_pid=$!

cleanup_checkpoint_pruner() {
  kill "${pruner_pid}" 2>/dev/null || true
  wait "${pruner_pid}" 2>/dev/null || true
  "${RUNTIME_PYTHON}" "${pruner_script}" --checkpoint-root "${checkpoint_root}"
}
trap cleanup_checkpoint_pruner EXIT

PYTHONPATH="${PROJECT_ROOT}/src" uv run --project "${PROJECT_ROOT}" \
  python -m dynamic_rubric validate-horizon-launch --config "${CONFIG_PATH}" >/dev/null

"${PROJECT_ROOT}/scripts/run_static_grpo.sh"
