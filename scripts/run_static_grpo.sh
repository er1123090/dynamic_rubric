#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
VERL_ROOT=${VERL_ROOT:-${PROJECT_ROOT}/environment/upstream/verl}
RUNTIME_PYTHON=${RUNTIME_PYTHON:-${VERL_ROOT}/.venv-runtime/bin/python}
MODEL_PATH=${MODEL_PATH:-${PROJECT_ROOT}/models/Qwen3-4B-Instruct-2507}
TRAIN_FILE=${TRAIN_FILE:-${PROJECT_ROOT}/artifacts/smoke/static-r0/verl-data/train.parquet}
VAL_FILE=${VAL_FILE:-${PROJECT_ROOT}/artifacts/smoke/static-r0/verl-data/probes.parquet}
STATIC_RUBRIC_PATH=${DYNAMIC_RUBRIC_STATIC_RUBRIC_PATH:-${PROJECT_ROOT}/artifacts/smoke/static-r0/static_rubrics.jsonl}
RUN_DIR=${RUN_DIR:-${PROJECT_ROOT}/artifacts/smoke/verl-run}
ROLLOUT_CACHE_DIR=${ROLLOUT_CACHE_DIR:-${PROJECT_ROOT}/artifacts/provider_cache/policy-rollouts}
TRACKING_PROJECT_NAME=${TRACKING_PROJECT_NAME:-dynamic_rubric_static_grpo}
TRACKING_EXPERIMENT_NAME=${TRACKING_EXPERIMENT_NAME:-static_r0_grpo__$(basename "${RUN_DIR}")}

TOTAL_STEPS=${TOTAL_STEPS:-1}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-4}
ROLLOUT_N=${ROLLOUT_N:-2}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-4}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-256}
ROLLOUT_GPU_MEMORY=${ROLLOUT_GPU_MEMORY:-0.42}
TEST_FREQ=${TEST_FREQ:--1}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}
SAVE_FREQ=${SAVE_FREQ:-${TOTAL_STEPS}}
CHECKPOINT_STEPS=${CHECKPOINT_STEPS:-"[0,1,2,3,5,10,20,30,50,75,100]"}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-0}
ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-sdpa}
USE_REMOVE_PADDING=${USE_REMOVE_PADDING:-False}
USE_FUSED_KERNELS=${USE_FUSED_KERNELS:-False}
FUSED_KERNEL_BACKEND=${FUSED_KERNEL_BACKEND:-torch}
ENABLE_GRADIENT_CHECKPOINTING=${ENABLE_GRADIENT_CHECKPOINTING:-True}
FULL_DETERMINISM=${FULL_DETERMINISM:-True}
ENFORCE_EAGER=${ENFORCE_EAGER:-True}
POLICY_GPU=${POLICY_GPU:-0}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-1}
ROLLOUT_TENSOR_PARALLEL_SIZE=${ROLLOUT_TENSOR_PARALLEL_SIZE:-1}
ROLLOUT_DATA_PARALLEL_SIZE=$((N_GPUS_PER_NODE / ROLLOUT_TENSOR_PARALLEL_SIZE))
RESUME_MODE=${RESUME_MODE:-disable}
RESUME_FROM_PATH=${RESUME_FROM_PATH:-null}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4096}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}
ACTOR_MAX_TOKEN_LEN=${ACTOR_MAX_TOKEN_LEN:-8192}
ROLLOUT_LOG_PROB_MAX_TOKEN_LEN=${ROLLOUT_LOG_PROB_MAX_TOKEN_LEN:-${ACTOR_MAX_TOKEN_LEN}}
REF_LOG_PROB_MAX_TOKEN_LEN=${REF_LOG_PROB_MAX_TOKEN_LEN:-${ACTOR_MAX_TOKEN_LEN}}
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-64}
ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-8192}
ROLLOUT_UPDATE_WEIGHTS_BUCKET_MEGABYTES=${ROLLOUT_UPDATE_WEIGHTS_BUCKET_MEGABYTES:-2048}
[[ "${ROLLOUT_UPDATE_WEIGHTS_BUCKET_MEGABYTES}" =~ ^[1-9][0-9]*$ ]] || {
  echo "ROLLOUT_UPDATE_WEIGHTS_BUCKET_MEGABYTES must be a positive integer" >&2; exit 2;
}
ROLLOUT_KV_ARGS=()
if [[ -n "${ROLLOUT_KV_CACHE_MEMORY_BYTES:-}" ]]; then
  [[ "${ROLLOUT_KV_CACHE_MEMORY_BYTES}" =~ ^[1-9][0-9]*$ ]] || {
    echo "ROLLOUT_KV_CACHE_MEMORY_BYTES must be positive integer bytes" >&2; exit 2;
  }
  ROLLOUT_KV_ARGS+=("++actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_memory_bytes=${ROLLOUT_KV_CACHE_MEMORY_BYTES}")
fi
ROLLOUT_ENABLE_CHUNKED_PREFILL=${ROLLOUT_ENABLE_CHUNKED_PREFILL:-False}
ROLLOUT_AGENT_NUM_WORKERS=${ROLLOUT_AGENT_NUM_WORKERS:-8}
REWARD_NUM_WORKERS=${REWARD_NUM_WORKERS:-8}
REF_PARAM_OFFLOAD=${REF_PARAM_OFFLOAD:-True}
ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-False}
ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD:-False}
LEARNING_RATE=${LEARNING_RATE:-1e-6}
WARMUP_RATIO=${WARMUP_RATIO:-0.1}
KL_COEFFICIENT=${KL_COEFFICIENT:-0.001}
ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-1.0}
ROLLOUT_TOP_P=${ROLLOUT_TOP_P:-0.95}
TRAINING_SEED=${TRAINING_SEED:-11}

: "${DYNAMIC_RUBRIC_VLLM_URL:?DYNAMIC_RUBRIC_VLLM_URL is required}"
: "${DYNAMIC_RUBRIC_GRADER_MODEL:?DYNAMIC_RUBRIC_GRADER_MODEL is required}"
: "${DYNAMIC_RUBRIC_GRADER_REVISION:?DYNAMIC_RUBRIC_GRADER_REVISION is required}"
: "${DYNAMIC_RUBRIC_GRADER_TOKENIZER_REVISION:?DYNAMIC_RUBRIC_GRADER_TOKENIZER_REVISION is required}"

export CUDA_VISIBLE_DEVICES=${POLICY_GPU}
export VLLM_USE_DEEP_GEMM=0
export PYTHONPATH=${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}
export DYNAMIC_RUBRIC_STATIC_RUBRIC_PATH=${STATIC_RUBRIC_PATH}
export DYNAMIC_RUBRIC_ROLLOUT_CACHE_DIR=${ROLLOUT_CACHE_DIR}
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

cd "${VERL_ROOT}"
exec "${RUNTIME_PYTHON}" -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH}" \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  +data.train_drop_last=False \
  data.dataloader_num_workers="${DATALOADER_NUM_WORKERS}" \
  data.seed="${TRAINING_SEED}" \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  +actor_rollout_ref.model.override_config.attn_implementation="${ATTN_IMPLEMENTATION}" \
  actor_rollout_ref.model.use_remove_padding="${USE_REMOVE_PADDING}" \
  actor_rollout_ref.model.use_fused_kernels="${USE_FUSED_KERNELS}" \
  actor_rollout_ref.model.fused_kernel_options.impl_backend="${FUSED_KERNEL_BACKEND}" \
  actor_rollout_ref.model.enable_gradient_checkpointing="${ENABLE_GRADIENT_CHECKPOINTING}" \
  actor_rollout_ref.actor.optim.lr="${LEARNING_RATE}" \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio="${WARMUP_RATIO}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef="${KL_COEFFICIENT}" \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.fsdp_config.param_offload="${ACTOR_PARAM_OFFLOAD}" \
  actor_rollout_ref.actor.fsdp_config.seed="${TRAINING_SEED}" \
  actor_rollout_ref.actor.data_loader_seed="${TRAINING_SEED}" \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload="${ACTOR_OPTIMIZER_OFFLOAD}" \
  actor_rollout_ref.actor.fsdp_config.full_determinism="${FULL_DETERMINISM}" \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${ACTOR_MAX_TOKEN_LEN}" \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.seed="${TRAINING_SEED}" \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.full_determinism="${FULL_DETERMINISM}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TENSOR_PARALLEL_SIZE}" \
  actor_rollout_ref.rollout.data_parallel_size="${ROLLOUT_DATA_PARALLEL_SIZE}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY}" \
  actor_rollout_ref.rollout.max_num_seqs="${ROLLOUT_MAX_NUM_SEQS}" \
  actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}" \
  actor_rollout_ref.rollout.max_num_batched_tokens="${ROLLOUT_MAX_NUM_BATCHED_TOKENS}" \
  actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes="${ROLLOUT_UPDATE_WEIGHTS_BUCKET_MEGABYTES}" \
  actor_rollout_ref.rollout.enable_chunked_prefill="${ROLLOUT_ENABLE_CHUNKED_PREFILL}" \
  actor_rollout_ref.rollout.enable_prefix_caching=True \
  actor_rollout_ref.rollout.enforce_eager="${ENFORCE_EAGER}" \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.temperature="${ROLLOUT_TEMPERATURE}" \
  actor_rollout_ref.rollout.top_p="${ROLLOUT_TOP_P}" \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${ROLLOUT_LOG_PROB_MAX_TOKEN_LEN}" \
  actor_rollout_ref.rollout.agent.num_workers="${ROLLOUT_AGENT_NUM_WORKERS}" \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="${PROJECT_ROOT}/configs/seeded_agent_loops.yaml" \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
  actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${REF_LOG_PROB_MAX_TOKEN_LEN}" \
  actor_rollout_ref.ref.fsdp_config.param_offload="${REF_PARAM_OFFLOAD}" \
  actor_rollout_ref.ref.fsdp_config.seed="${TRAINING_SEED}" \
  actor_rollout_ref.ref.fsdp_config.full_determinism="${FULL_DETERMINISM}" \
  reward.num_workers="${REWARD_NUM_WORKERS}" \
  reward.custom_reward_function.path="${PROJECT_ROOT}/src/dynamic_rubric/training/verl_reward.py" \
  reward.custom_reward_function.name=compute_score \
  trainer.use_v1=False \
  trainer.critic_warmup=0 \
  'trainer.logger=["console"]' \
  trainer.project_name="${TRACKING_PROJECT_NAME}" \
  trainer.experiment_name="${TRACKING_EXPERIMENT_NAME}" \
  trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
  trainer.nnodes=1 \
  trainer.save_freq="${SAVE_FREQ}" \
  +trainer.checkpoint_steps="${CHECKPOINT_STEPS}" \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.val_before_train="${VAL_BEFORE_TRAIN}" \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  trainer.total_training_steps="${TOTAL_STEPS}" \
  trainer.resume_mode="${RESUME_MODE}" \
  trainer.resume_from_path="${RESUME_FROM_PATH}" \
  trainer.default_local_dir="${RUN_DIR}/checkpoints" \
  trainer.rollout_data_dir="${RUN_DIR}/rollouts" \
  trainer.validation_data_dir="${RUN_DIR}/probes" \
  ray_kwargs.ray_init.runtime_env.py_executable=null \
  "${ROLLOUT_KV_ARGS[@]}"
