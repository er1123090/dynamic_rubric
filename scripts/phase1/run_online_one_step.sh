#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
VERL_ROOT=${VERL_ROOT:-${PROJECT_ROOT}/environment/upstream/verl}
RUNTIME_PYTHON=${RUNTIME_PYTHON:-${VERL_ROOT}/.venv-runtime/bin/python}

for required in MODEL_PATH TRAIN_FILE VAL_FILE RUN_DIR ONLINE_STEP_ARTIFACT_ROOT \
  ONLINE_CONTROL_URL ONLINE_CONTROL_CHECKPOINT_HASH ONLINE_CONTROL_LAUNCH_SPEC \
  PHASE1_GPT_OSS_BASE_URL PHASE1_QWEN32B_BASE_URL \
  ONLINE_STEP_HOOK_PATH ONLINE_STEP_RUNTIME_PATH TRACKING_EXPERIMENT_NAME; do
  [[ -n "${!required:-}" ]] || { echo "${required} is required" >&2; exit 2; }
done
[[ "${ONLINE_CONTROL_POLICY:-}" == "pi_ref" ]] || { echo "canary requires frozen pi_ref" >&2; exit 2; }
[[ "${TRAIN_BATCH_SIZE:-}" == "96" ]] || { echo "canary requires TRAIN_BATCH_SIZE=96" >&2; exit 2; }
[[ "${ROLLOUT_N:-}" == "16" ]] || { echo "canary requires ROLLOUT_N=16" >&2; exit 2; }
[[ "${ELICITATION_PAIRS:-}" == "8" ]] || { echo "canary requires ELICITATION_PAIRS=8" >&2; exit 2; }
[[ "${EXPECTED_UPDATES:-}" == "1" ]] || { echo "canary requires EXPECTED_UPDATES=1" >&2; exit 2; }
[[ "${N_GPUS_PER_NODE:-}" == "1" ]] || { echo "canary requires N_GPUS_PER_NODE=1" >&2; exit 2; }
[[ "${CUDA_VISIBLE_DEVICES:-}" == "0" ]] || { echo "canary requires trainer GPU 0" >&2; exit 2; }
[[ "${ONLINE_EXTRACTOR_BASE_URL:-}" == "${PHASE1_GPT_OSS_BASE_URL}" ]] || { echo "extractor endpoint drift" >&2; exit 2; }
[[ "${ONLINE_GRADER_BASE_URL:-}" == "${PHASE1_QWEN32B_BASE_URL}" ]] || { echo "grader endpoint drift" >&2; exit 2; }

export PYTHONPATH=${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_DEEP_GEMM=0
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
cd "${VERL_ROOT}"
exec "${RUNTIME_PYTHON}" -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
  data.train_batch_size=96 \
  data.max_prompt_length="${MAX_PROMPT_LENGTH:-4096}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH:-3584}" \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.dataloader_num_workers=0 \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  +actor_rollout_ref.model.override_config.attn_implementation="${ATTN_IMPLEMENTATION:-sdpa}" \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.model.use_fused_kernels=False \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=5e-6 \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
  actor_rollout_ref.actor.ppo_mini_batch_size=96 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.01 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${ACTOR_MAX_TOKEN_LEN:-24576}" \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  ++actor_rollout_ref.actor.checkpoint.async_save=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.data_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY:-0.55}" \
  actor_rollout_ref.rollout.max_num_seqs="${ROLLOUT_MAX_NUM_SEQS:-128}" \
  actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN:-7680}" \
  actor_rollout_ref.rollout.max_num_batched_tokens="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-16384}" \
  actor_rollout_ref.rollout.enable_prefix_caching=True \
  actor_rollout_ref.rollout.enforce_eager="${ENFORCE_EAGER:-False}" \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=0.95 \
  actor_rollout_ref.rollout.n=16 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${ROLLOUT_LOG_PROB_MAX_TOKEN_LEN:-49152}" \
  actor_rollout_ref.rollout.agent.num_workers="${ROLLOUT_AGENT_NUM_WORKERS:-16}" \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="${PROJECT_ROOT}/configs/seeded_agent_loops.yaml" \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=False \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${REF_LOG_PROB_MAX_TOKEN_LEN:-65536}" \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  ++reward.online_step_hook.enabled=true \
  ++reward.online_step_hook.path="${ONLINE_STEP_HOOK_PATH}" \
  ++reward.online_step_hook.name="${ONLINE_STEP_HOOK_NAME:-prepare_rewards}" \
  ++reward.online_step_hook.commit_name="${ONLINE_STEP_COMMIT_NAME:-commit_step}" \
  ++reward.online_step_runtime.path="${ONLINE_STEP_RUNTIME_PATH}" \
  ++reward.online_step_runtime.name="${ONLINE_STEP_RUNTIME_NAME:-create_online_reward_runtime}" \
  ++reward.online_step_runtime.artifact_root="${ONLINE_STEP_ARTIFACT_ROOT}" \
  ++reward.online_step_runtime.control_policy=pi_ref \
  ++reward.online_step_runtime.frozen_control=true \
  trainer.use_v1=False \
  trainer.critic_warmup=0 \
  'trainer.logger=["console"]' \
  trainer.project_name="${TRACKING_PROJECT_NAME:-phase1_dynamic_evaluator_updates}" \
  trainer.experiment_name="${TRACKING_EXPERIMENT_NAME}" \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.save_freq=1 \
  trainer.test_freq=-1 \
  trainer.val_before_train=False \
  trainer.total_epochs=1 \
  trainer.total_training_steps=1 \
  trainer.resume_mode=disable \
  trainer.resume_from_path=null \
  trainer.default_local_dir="${RUN_DIR}/checkpoints" \
  trainer.rollout_data_dir="${RUN_DIR}/rollouts" \
  trainer.validation_data_dir="${RUN_DIR}/validation" \
  ray_kwargs.ray_init.runtime_env.py_executable=null
