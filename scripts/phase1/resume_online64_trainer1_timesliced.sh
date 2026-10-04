#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
RUN_ID=phase1-online-rubrics-medicine-full-dense-20260919-seed11
RUN_ROOT="${PROJECT_ROOT}/outputs/medicine/online_rubrics/seed-11/${RUN_ID}"
CHECKPOINT_ROOT="${RUN_ROOT}/verl-run/checkpoints"
RESUME_CHECKPOINT="${CHECKPOINT_ROOT}/global_step_48"

[[ "$(cat "${CHECKPOINT_ROOT}/latest_checkpointed_iteration.txt")" == 48 ]] || {
  echo "checkpoint tracker is not at step 48" >&2; exit 2;
}
for required in \
  "${RESUME_CHECKPOINT}/actor/model_world_size_1_rank_0.pt" \
  "${RESUME_CHECKPOINT}/actor/optim_world_size_1_rank_0.pt" \
  "${RESUME_CHECKPOINT}/actor/extra_state_world_size_1_rank_0.pt" \
  "${RESUME_CHECKPOINT}/data.pt"; do
  [[ -s "${required}" ]] || {
    echo "incomplete resume checkpoint: ${required}" >&2; exit 2;
  }
done

curl -fsS --max-time 5 http://127.0.0.1:28011/v1/models >/dev/null
curl -fsS --max-time 5 http://127.0.0.1:28014/v1/models >/dev/null

export PROJECT_ROOT
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=1
export CONFIG_PATH="${RUN_ROOT}/config.resolved.json"
export TRAIN_FILE="${RUN_ROOT}/verl-data/train-online-full.parquet"
export VAL_FILE="${RUN_ROOT}/verl-data/validation-unused.parquet"
export MODEL_PATH=${POLICY_MODEL_PATH}
export RUN_DIR="${RUN_ROOT}/verl-run"
export ONLINE_STEP_ARTIFACT_ROOT="${RUN_ROOT}/verl-run/online_steps"
export ONLINE_RUN_ID="${RUN_ID}"
export ONLINE_CONFIG_HASH=453f1871eb19e541d9b422615fb2f28ad7cb372ae77f7a24ce394b0560a080ab
export ONLINE_EXPERIMENT_ARM=phase1_online_rubrics_full_dynamic
export ONLINE_BASELINE_ARM=static_r0_grpo
export ONLINE_CONTROL_POLICY=pi_ref
export ONLINE_CONTROL_CACHE="${PROJECT_ROOT}/outputs/medicine/shared/seed-11/pi0_control_cache/manifest-ad74decb90ce01cfc4a3048e21757e6fca58d6c77095d86c32d40507802d0407.json"
export ONLINE_CONTROL_CHECKPOINT_HASH=0d4e207e73f80935be04a13a57819764d0c4b6913675c2b53243b9ef10527c4c
export ONLINE_REPRODUCTION_CLAIM=phase1_full_dynamic_evaluator_training
export ONLINE_CRITERIA_SCOPE=prompt_step_ephemeral
export ONLINE_FAILURE_POLICY=fail_closed
export ONLINE_EXTRACTOR_MODEL=openai/gpt-oss-120b
export ONLINE_EXTRACTOR_RETURNED_MODEL=openai/gpt-oss-120b
export ONLINE_EXTRACTOR_REASONING_EFFORT=medium
export ONLINE_EXTRACTOR_MAX_OUTPUT_TOKENS=8192
export ONLINE_DEDUP_MAX_OUTPUT_TOKENS=8192
export ONLINE_GRADER_MODEL=Qwen/Qwen3-32B
export ONLINE_GRADER_RETURNED_MODEL=Qwen/Qwen3-32B
export ONLINE_GRADER_MAX_OUTPUT_TOKENS=4096
export ONLINE_ACTOR_MODEL=Qwen/Qwen3-4B-Instruct-2507
export ONLINE_ACTOR_REVISION=cdbee75f17c01a7cc42f958dc650907174af0554
export ONLINE_CONTROL_MODEL=Qwen/Qwen3-4B-Instruct-2507
export ONLINE_CONTROL_REVISION=cdbee75f17c01a7cc42f958dc650907174af0554
export ONLINE_CONTROL_TOKENIZER_REVISION=cdbee75f17c01a7cc42f958dc650907174af0554
export ONLINE_SEED=11
export PHASE1_GPT_OSS_BASE_URLS=http://127.0.0.1:28011
export PHASE1_QWEN32B_BASE_URLS=http://127.0.0.1:28014
export ONLINE_EXTRACTOR_CONCURRENCY=48
export ONLINE_GRADER_CONCURRENCY=64
export PHASE1_VLLM_TIMEOUT_SECONDS=600
export PHASE1_VLLM_MAX_RETRIES=4
export TOTAL_EPOCHS=4
export EXPECTED_UPDATES=64
export TRAIN_BATCH_SIZE=96
export ROLLOUT_N=16
export ELICITATION_PAIRS=8
export PPO_MINI_BATCH_SIZE=96
export N_GPUS_PER_NODE=1
export ATTN_IMPLEMENTATION=sdpa
export TRAINING_SEED=11
export CHECKPOINT_INTERVAL_STEPS=1
export CHECKPOINT_STEPS='[49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64]'
export ONLINE_STEP_HOOK_PATH=pkg://dynamic_rubric.training.online_step
export ONLINE_STEP_HOOK_NAME=prepare_rewards
export ONLINE_STEP_COMMIT_NAME=commit_step
export ONLINE_STEP_RUNTIME_PATH=pkg://dynamic_rubric.training.verl_online_runtime
export ONLINE_STEP_RUNTIME_NAME=create_online_reward_runtime
export TRACKING_PROJECT_NAME=phase1_dynamic_evaluator_updates
export TRACKING_EXPERIMENT_NAME="${RUN_ID}-extend64-timesliced"
export RESUME_MODE=resume_path
export RESUME_FROM_PATH="${RESUME_CHECKPOINT}"
export ONLINE_LOGPROB_PREFETCH=false
export ROLLOUT_GPU_MEMORY=0.55
export ROLLOUT_MAX_NUM_SEQS=128
export ROLLOUT_MAX_NUM_BATCHED_TOKENS=16384

unset OPENAI_API_KEY PHASE1_GPT_OSS_BASE_URL PHASE1_QWEN32B_BASE_URL \
  ONLINE_CONTROL_URL ONLINE_CONTROL_LAUNCH_SPEC

# The paper launcher remains the source of truth. These four substitutions are
# scoped to the requested 48->64 extension and GPU time-slicing.
sed \
  -e 's/\[ "${TOTAL_EPOCHS}" = "3" \]/[ "${TOTAL_EPOCHS}" = "4" ]/' \
  -e 's/\[ "${EXPECTED_UPDATES}" = "48" \]/[ "${EXPECTED_UPDATES}" = "64" ]/' \
  -e 's/actor_rollout_ref.actor.fsdp_config.param_offload=False/actor_rollout_ref.actor.fsdp_config.param_offload=True/' \
  -e 's/actor_rollout_ref.actor.fsdp_config.optimizer_offload=False/actor_rollout_ref.actor.fsdp_config.optimizer_offload=True/' \
  "${PROJECT_ROOT}/scripts/phase1/run_online_full.sh" | exec bash
