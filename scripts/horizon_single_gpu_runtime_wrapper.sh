#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
REAL_RUNTIME_PYTHON=${REAL_RUNTIME_PYTHON:-${PROJECT_ROOT}/environment/upstream/verl/.venv-runtime/bin/python}

if [[ "${1:-}" == "-m" && "${2:-}" == "verl.trainer.main_ppo" ]]; then
  exec "${REAL_RUNTIME_PYTHON}" "$@" \
    actor_rollout_ref.actor.fsdp_config.param_offload="${SINGLE_GPU_ACTOR_PARAM_OFFLOAD:-True}" \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload="${SINGLE_GPU_ACTOR_OPTIMIZER_OFFLOAD:-True}" \
    actor_rollout_ref.ref.fsdp_config.param_offload="${SINGLE_GPU_REF_PARAM_OFFLOAD:-True}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${SINGLE_GPU_ROLLOUT_GPU_MEMORY:-0.20}" \
    actor_rollout_ref.rollout.max_num_seqs="${SINGLE_GPU_ROLLOUT_MAX_NUM_SEQS:-32}" \
    actor_rollout_ref.rollout.max_num_batched_tokens="${SINGLE_GPU_ROLLOUT_MAX_NUM_BATCHED_TOKENS:-8192}" \
    actor_rollout_ref.rollout.agent.num_workers="${SINGLE_GPU_ROLLOUT_AGENT_NUM_WORKERS:-4}" \
    actor_rollout_ref.rollout.enforce_eager="${SINGLE_GPU_ROLLOUT_ENFORCE_EAGER:-True}" \
    reward.num_workers="${SINGLE_GPU_REWARD_NUM_WORKERS:-8}" \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_memory_bytes="${SINGLE_GPU_ROLLOUT_KV_CACHE_MEMORY_BYTES:-12G}"
fi

if [[ "${1:-}" == "-m" && "${2:-}" == "vllm.entrypoints.openai.api_server" ]]; then
  exec "${REAL_RUNTIME_PYTHON}" "$@" \
    --kv-cache-memory-bytes "${SINGLE_GPU_POLICY_SERVER_KV_CACHE_MEMORY_BYTES:-8G}" \
    --gpu-memory-utilization "${SINGLE_GPU_POLICY_SERVER_GPU_MEMORY:-0.20}" \
    --max-num-seqs "${SINGLE_GPU_POLICY_SERVER_MAX_NUM_SEQS:-32}" \
    --max-num-batched-tokens "${SINGLE_GPU_POLICY_SERVER_MAX_NUM_BATCHED_TOKENS:-8192}" \
    --enforce-eager
fi

exec "${REAL_RUNTIME_PYTHON}" "$@"
