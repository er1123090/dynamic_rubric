#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}
VERL_ROOT=${VERL_ROOT:-${PROJECT_ROOT}/environment/upstream/verl}
RUNTIME_PYTHON=${RUNTIME_PYTHON:-${VERL_ROOT}/.venv-runtime/bin/python}
MODEL_PATH=${MODEL_PATH:?MODEL_PATH is required}
TRAIN_FILE=${TRAIN_FILE:?TRAIN_FILE is required}
VAL_FILE=${VAL_FILE:?VAL_FILE is required}
RUN_DIR=${RUN_DIR:?RUN_DIR is required}
ONLINE_STEP_ARTIFACT_ROOT=${ONLINE_STEP_ARTIFACT_ROOT:?ONLINE_STEP_ARTIFACT_ROOT is required}
ONLINE_CONTROL_POLICY=${ONLINE_CONTROL_POLICY:?ONLINE_CONTROL_POLICY is required}
ONLINE_STEP_HOOK_PATH=${ONLINE_STEP_HOOK_PATH:?ONLINE_STEP_HOOK_PATH is required}
ONLINE_STEP_HOOK_NAME=${ONLINE_STEP_HOOK_NAME:-prepare_rewards}
ONLINE_STEP_COMMIT_NAME=${ONLINE_STEP_COMMIT_NAME:-commit_step}
ONLINE_STEP_RUNTIME_PATH=${ONLINE_STEP_RUNTIME_PATH:?ONLINE_STEP_RUNTIME_PATH is required}
ONLINE_STEP_RUNTIME_NAME=${ONLINE_STEP_RUNTIME_NAME:-create_online_reward_runtime}
TRACKING_PROJECT_NAME=${TRACKING_PROJECT_NAME:-dynamic_rubric_online_rl}
TRACKING_EXPERIMENT_NAME=${TRACKING_EXPERIMENT_NAME:-dynamic_online_rubric_grpo}

TOTAL_EPOCHS=${TOTAL_EPOCHS:-3}
EXPECTED_UPDATES=${EXPECTED_UPDATES:-45}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-96}
ROLLOUT_N=${ROLLOUT_N:-16}
ELICITATION_PAIRS=${ELICITATION_PAIRS:-8}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-96}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-3584}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4096}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}
ACTOR_MAX_TOKEN_LEN=${ACTOR_MAX_TOKEN_LEN:-24576}
ROLLOUT_LOG_PROB_MAX_TOKEN_LEN=${ROLLOUT_LOG_PROB_MAX_TOKEN_LEN:-49152}
REF_LOG_PROB_MAX_TOKEN_LEN=${REF_LOG_PROB_MAX_TOKEN_LEN:-65536}
ROLLOUT_GPU_MEMORY=${ROLLOUT_GPU_MEMORY:-0.55}
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-128}
ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-16384}
ROLLOUT_AGENT_NUM_WORKERS=${ROLLOUT_AGENT_NUM_WORKERS:-16}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-2}
LEARNING_RATE=${LEARNING_RATE:-5e-6}
WARMUP_RATIO=${WARMUP_RATIO:-0.1}
KL_COEFFICIENT=${KL_COEFFICIENT:-0.01}
ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-1.0}
ROLLOUT_TOP_P=${ROLLOUT_TOP_P:-0.95}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-0}
FULL_DETERMINISM=${FULL_DETERMINISM:-False}
ENFORCE_EAGER=${ENFORCE_EAGER:-False}
RESUME_MODE=${RESUME_MODE:-disable}
RESUME_FROM_PATH=${RESUME_FROM_PATH:-null}
CHECKPOINT_INTERVAL_STEPS=${CHECKPOINT_INTERVAL_STEPS:-1}
CHECKPOINT_STEPS=${CHECKPOINT_STEPS:?CHECKPOINT_STEPS is required}

PHASE1_TUNING_MODE=${PHASE1_TUNING_MODE:-paper}
case "${PHASE1_TUNING_MODE}" in
  paper)
    [ "${TOTAL_EPOCHS}" = "3" ] || { echo "paper online training requires TOTAL_EPOCHS=3" >&2; exit 2; }
    [ "${TRAIN_BATCH_SIZE}" = "96" ] || { echo "paper online training requires TRAIN_BATCH_SIZE=96" >&2; exit 2; }
    [ "${LEARNING_RATE}" = "5e-06" ] || [ "${LEARNING_RATE}" = "5e-6" ] || { echo "paper online training requires LEARNING_RATE=5e-6" >&2; exit 2; }
    [ "${WARMUP_RATIO}" = "0.1" ] || { echo "paper online training requires WARMUP_RATIO=0.1" >&2; exit 2; }
    [ "${KL_COEFFICIENT}" = "0.01" ] || { echo "paper online training requires KL_COEFFICIENT=0.01" >&2; exit 2; }
    ;;
  custom) ;;
  *) echo "PHASE1_TUNING_MODE must be paper or custom" >&2; exit 2 ;;
esac
[ "${ROLLOUT_N}" = "16" ] || { echo "Online reward backend requires ROLLOUT_N=16" >&2; exit 2; }
[ "${ELICITATION_PAIRS}" = "8" ] || { echo "Online reward backend requires ELICITATION_PAIRS=8" >&2; exit 2; }
[ "${CHECKPOINT_INTERVAL_STEPS}" = "1" ] || { echo "dense online training requires CHECKPOINT_INTERVAL_STEPS=1" >&2; exit 2; }
case "${ONLINE_CONTROL_POLICY}" in pi_ref|pi_old) ;; *) echo "invalid control policy" >&2; exit 2;; esac

gpt_vllm_urls=${PHASE1_GPT_OSS_BASE_URLS:-${PHASE1_GPT_OSS_BASE_URL:-}}
qwen_vllm_urls=${PHASE1_QWEN32B_BASE_URLS:-${PHASE1_QWEN32B_BASE_URL:-}}
if [[ -n "${gpt_vllm_urls}" || -n "${qwen_vllm_urls}" ]]; then
  [[ -n "${gpt_vllm_urls}" && -n "${qwen_vllm_urls}" ]] || {
    echo "both GPT-OSS and Qwen vLLM endpoint families are required" >&2; exit 2;
  }
else
  : "${OPENAI_API_KEY:?OPENAI_API_KEY is required when vLLM evaluator endpoints are absent}"
fi

if [[ -n "${ONLINE_CONTROL_CACHE:-}" ]]; then
  [[ -f "${ONLINE_CONTROL_CACHE}" ]] || {
    echo "ONLINE_CONTROL_CACHE must point to an immutable manifest file" >&2; exit 2;
  }
else
  : "${ONLINE_CONTROL_URL:?ONLINE_CONTROL_URL is required when immutable pi0 cache is absent}"
fi
: "${ONLINE_CONTROL_CHECKPOINT_HASH:?ONLINE_CONTROL_CHECKPOINT_HASH must bind frozen pi_ref to local A0}"

export PYTHONPATH=${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_DEEP_GEMM=0
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

checkpoint_root="${RUN_DIR}/checkpoints"
pruner_script="${PROJECT_ROOT}/scripts/prune_online_checkpoint_state.py"
"${RUNTIME_PYTHON}" "${pruner_script}" \
  --checkpoint-root "${checkpoint_root}" \
  --watch \
  --while-pid $$ &
pruner_pid=$!
cleanup_checkpoint_pruner() {
  kill "${pruner_pid}" 2>/dev/null || true
  wait "${pruner_pid}" 2>/dev/null || true
  "${RUNTIME_PYTHON}" "${pruner_script}" --checkpoint-root "${checkpoint_root}"
}
trap cleanup_checkpoint_pruner EXIT

cd "${VERL_ROOT}"
"${RUNTIME_PYTHON}" -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH}" \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.dataloader_num_workers="${DATALOADER_NUM_WORKERS}" \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.model.use_fused_kernels=False \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr="${LEARNING_RATE}" \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio="${WARMUP_RATIO}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef="${KL_COEFFICIENT}" \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.actor.fsdp_config.full_determinism="${FULL_DETERMINISM}" \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${ACTOR_MAX_TOKEN_LEN}" \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  ++actor_rollout_ref.actor.checkpoint.async_save=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.full_determinism="${FULL_DETERMINISM}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.data_parallel_size="${N_GPUS_PER_NODE}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY}" \
  actor_rollout_ref.rollout.max_num_seqs="${ROLLOUT_MAX_NUM_SEQS}" \
  actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}" \
  actor_rollout_ref.rollout.max_num_batched_tokens="${ROLLOUT_MAX_NUM_BATCHED_TOKENS}" \
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
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${REF_LOG_PROB_MAX_TOKEN_LEN}" \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  ++reward.online_step_hook.enabled=true \
  ++reward.online_step_hook.path="${ONLINE_STEP_HOOK_PATH}" \
  ++reward.online_step_hook.name="${ONLINE_STEP_HOOK_NAME}" \
  ++reward.online_step_hook.commit_name="${ONLINE_STEP_COMMIT_NAME}" \
  ++reward.online_step_runtime.path="${ONLINE_STEP_RUNTIME_PATH}" \
  ++reward.online_step_runtime.name="${ONLINE_STEP_RUNTIME_NAME}" \
  ++reward.online_step_runtime.artifact_root="${ONLINE_STEP_ARTIFACT_ROOT}" \
  ++reward.online_step_runtime.control_policy="${ONLINE_CONTROL_POLICY}" \
  ++reward.online_step_runtime.frozen_control=true \
  trainer.use_v1=False \
  trainer.critic_warmup=0 \
  'trainer.logger=["console"]' \
  trainer.project_name="${TRACKING_PROJECT_NAME}" \
  trainer.experiment_name="${TRACKING_EXPERIMENT_NAME}" \
  trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
  trainer.nnodes=1 \
  trainer.save_freq="${CHECKPOINT_INTERVAL_STEPS}" \
  +trainer.checkpoint_steps="${CHECKPOINT_STEPS}" \
  trainer.test_freq=-1 \
  trainer.val_before_train=False \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  trainer.resume_mode="${RESUME_MODE}" \
  trainer.resume_from_path="${RESUME_FROM_PATH}" \
  trainer.default_local_dir="${RUN_DIR}/checkpoints" \
  trainer.rollout_data_dir="${RUN_DIR}/rollouts" \
  trainer.validation_data_dir="${RUN_DIR}/validation" \
  ray_kwargs.ray_init.runtime_env.py_executable=null
