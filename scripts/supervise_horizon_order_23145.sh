#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
CURRENT_SCIENCE_PGID=${CURRENT_SCIENCE_PGID:?CURRENT_SCIENCE_PGID is required}
TRAINING_SEED=${TRAINING_SEED:-11}
POLICY_GPU=${POLICY_GPU:-0}
PAUSE_STEP=${PAUSE_STEP:-3}
JUDGE_BASE_URL=${JUDGE_BASE_URL:-http://127.0.0.1:8102}
RUNTIME_WRAPPER=${RUNTIME_WRAPPER:-${PROJECT_ROOT}/scripts/horizon_single_gpu_runtime_wrapper.sh}
REAL_RUNTIME_PYTHON=${REAL_RUNTIME_PYTHON:-${PROJECT_ROOT}/environment/upstream/verl/.venv-runtime/bin/python}
SCIENCE_RUN_ROOT=${SCIENCE_RUN_ROOT:-${PROJECT_ROOT}/artifacts/horizon/science/training/seed-${TRAINING_SEED}}
SCIENCE_CHECKPOINT_ROOT=${SCIENCE_CHECKPOINT_ROOT:-${SCIENCE_RUN_ROOT}/checkpoints}
SCIENCE_MARKER=${SCIENCE_MARKER:-${SCIENCE_CHECKPOINT_ROOT}/latest_checkpointed_iteration.txt}
SCIENCE_RESUME_PATH=${SCIENCE_RESUME_PATH:-${SCIENCE_CHECKPOINT_ROOT}/global_step_${PAUSE_STEP}}
MEDICINE_AUDIT_MARKER=${MEDICINE_AUDIT_MARKER:-${PROJECT_ROOT}/artifacts/horizon/medicine/audit/seed-${TRAINING_SEED}.complete}
SCIENCE_AUDIT_MARKER=${SCIENCE_AUDIT_MARKER:-${PROJECT_ROOT}/artifacts/horizon/science/audit/seed-${TRAINING_SEED}.complete}
SCIENCE_FULL_PROMPTS=${SCIENCE_FULL_PROMPTS:-${PROJECT_ROOT}/data/rar/science/public/final.jsonl}
SCIENCE_EVAL100_PROMPTS=${SCIENCE_EVAL100_PROMPTS:-${PROJECT_ROOT}/data/rar/science/public/final_eval100.jsonl}
SCIENCE_FULL_POOL_ROOT=${SCIENCE_FULL_POOL_ROOT:-${PROJECT_ROOT}/artifacts/horizon/science/pools}
SCIENCE_EVAL100_POOL_ROOT=${SCIENCE_EVAL100_POOL_ROOT:-${PROJECT_ROOT}/artifacts/horizon/science/pools-eval100}
STATE_FILE=${STATE_FILE:-${PROJECT_ROOT}/artifacts/horizon/order-23145.state}
JUDGE_IDLE_GPU_MAX_MIB=${JUDGE_IDLE_GPU_MAX_MIB:-81920}

log() {
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$*"
}

set_stage() {
  local stage=$1
  local temporary="${STATE_FILE}.tmp.$$"
  mkdir -p "$(dirname "${STATE_FILE}")"
  printf '%s\n' "${stage}" >"${temporary}"
  mv "${temporary}" "${STATE_FILE}"
  log "stage=${stage}"
}

marker_value() {
  if [[ -s "${SCIENCE_MARKER}" ]]; then
    tr -d '[:space:]' <"${SCIENCE_MARKER}"
  else
    printf 'missing'
  fi
}

group_alive() {
  kill -0 -- "-${CURRENT_SCIENCE_PGID}" 2>/dev/null
}

require_resume_checkpoint() {
  [[ "$(marker_value)" == "${PAUSE_STEP}" ]] || {
    echo "Science marker is not ${PAUSE_STEP}: $(marker_value)" >&2
    return 1
  }
  [[ -s "${SCIENCE_RESUME_PATH}/actor/model_world_size_1_rank_0.pt" ]] || {
    echo "Science resume model is missing" >&2
    return 1
  }
  [[ -s "${SCIENCE_RESUME_PATH}/actor/optim_world_size_1_rank_0.pt" ]] || {
    echo "Science resume optimizer is missing" >&2
    return 1
  }
  [[ -s "${SCIENCE_RESUME_PATH}/data.pt" ]] || {
    echo "Science resume dataloader state is missing" >&2
    return 1
  }
}

wait_for_pause_checkpoint() {
  set_stage "waiting_science_step_${PAUSE_STEP}_checkpoint"
  while true; do
    if [[ "$(marker_value)" == "${PAUSE_STEP}" ]]; then
      require_resume_checkpoint
      return 0
    fi
    if ! group_alive; then
      echo "Science training exited before checkpoint ${PAUSE_STEP}; marker=$(marker_value)" >&2
      return 1
    fi
    sleep 30
  done
}

stop_current_science() {
  local command
  command=$(ps -o args= -p "${CURRENT_SCIENCE_PGID}" 2>/dev/null || true)
  [[ "${command}" == *run_horizon_static_grpo.sh* ]] || {
    echo "refusing to stop unexpected process group leader: ${command}" >&2
    return 1
  }
  [[ "$(ps -o pgid= -p $$ | tr -d ' ')" != "${CURRENT_SCIENCE_PGID}" ]] || {
    echo "supervisor and Science training share a process group" >&2
    return 1
  }

  set_stage "stopping_science_after_step_${PAUSE_STEP}"
  kill -TERM -- "-${CURRENT_SCIENCE_PGID}"
  for _ in $(seq 1 60); do
    group_alive || break
    sleep 2
  done
  if group_alive; then
    log "Science process group did not stop after TERM; sending KILL"
    kill -KILL -- "-${CURRENT_SCIENCE_PGID}"
  fi
  "${REAL_RUNTIME_PYTHON}" "${PROJECT_ROOT}/scripts/prune_horizon_checkpoint_state.py" \
    --checkpoint-root "${SCIENCE_CHECKPOINT_ROOT}"
  require_resume_checkpoint
}

wait_for_policy_capacity() {
  for attempt in $(seq 1 120); do
    local used
    used=$(nvidia-smi -i "${POLICY_GPU}" --query-compute-apps=used_memory \
      --format=csv,noheader,nounits 2>/dev/null | awk '{sum += $1} END {print sum + 0}')
    if [[ "${used}" -le "${JUDGE_IDLE_GPU_MAX_MIB}" ]]; then
      return 0
    fi
    if [[ "${attempt}" -eq 120 ]]; then
      echo "GPU ${POLICY_GPU} still uses ${used} MiB after Science pause" >&2
      return 1
    fi
    sleep 15
  done
}

preflight() {
  : "${OPENAI_API_KEY:?OPENAI_API_KEY is required for GPT-5-mini Batch extraction}"
  curl -fsS --max-time 30 "${JUDGE_BASE_URL}/health" >/dev/null
  "${REAL_RUNTIME_PYTHON}" - <<'PY'
import json
import os
import urllib.request

request = urllib.request.Request(
    "https://api.openai.com/v1/batches?limit=1",
    headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
)
with urllib.request.urlopen(request, timeout=30) as response:
    if response.status != 200:
        raise RuntimeError(f"OpenAI Batch preflight failed: HTTP {response.status}")
    json.load(response)
PY
}

run_medicine_kl_and_audit() {
  if [[ -s "${MEDICINE_AUDIT_MARKER}" ]]; then
    log "Medicine audit marker already exists; skipping stages 2 and 3"
    return 0
  fi
  set_stage "medicine_checkpoint_kl_and_pool_preparation"
  env \
    DOMAIN=medicine \
    TRAINING_SEED="${TRAINING_SEED}" \
    POLICY_GPU="${POLICY_GPU}" \
    JUDGE_BASE_URL="${JUDGE_BASE_URL}" \
    RUNTIME_PYTHON="${RUNTIME_WRAPPER}" \
    REAL_RUNTIME_PYTHON="${REAL_RUNTIME_PYTHON}" \
    bash "${PROJECT_ROOT}/scripts/run_horizon_checkpoint_kl_then_audit.sh"
  [[ -s "${MEDICINE_AUDIT_MARKER}" ]] || {
    echo "Medicine audit did not produce its completion marker" >&2
    return 1
  }
}

resume_science_training() {
  if [[ "$(marker_value)" == "48" ]]; then
    log "Science training is already complete"
    return 0
  fi
  require_resume_checkpoint
  set_stage "science_training_step_${PAUSE_STEP}_to_48"
  env \
    DOMAIN=science \
    TRAINING_SEED="${TRAINING_SEED}" \
    POLICY_GPU="${POLICY_GPU}" \
    RESUME_MODE=resume_path \
    RESUME_FROM_PATH="${SCIENCE_RESUME_PATH}" \
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
  [[ "$(marker_value)" == "48" ]] || {
    echo "Science training returned without checkpoint 48; marker=$(marker_value)" >&2
    return 1
  }
}

prepare_science_eval100() {
  set_stage "science_eval100_static_pool_preparation"
  "${REAL_RUNTIME_PYTHON}" "${PROJECT_ROOT}/scripts/prepare_horizon_eval_subset.py" \
    --prompts "${SCIENCE_FULL_PROMPTS}" \
    --split-manifest "${PROJECT_ROOT}/data/rar/science/public/split_manifest.json" \
    --source-pool-root "${SCIENCE_FULL_POOL_ROOT}" \
    --output-prompts "${SCIENCE_EVAL100_PROMPTS}" \
    --output-pool-root "${SCIENCE_EVAL100_POOL_ROOT}" \
    --count 100 \
    --training-seed "${TRAINING_SEED}" >/dev/null
}

run_science_kl_and_audit() {
  if [[ -s "${SCIENCE_AUDIT_MARKER}" ]]; then
    log "Science audit marker already exists; skipping stages 4 and 5"
    return 0
  fi
  set_stage "science_checkpoint_kl_and_pool_preparation"
  env \
    DOMAIN=science \
    TRAINING_SEED="${TRAINING_SEED}" \
    POLICY_GPU="${POLICY_GPU}" \
    PROMPTS="${SCIENCE_EVAL100_PROMPTS}" \
    POOL_ROOT="${SCIENCE_EVAL100_POOL_ROOT}" \
    JUDGE_BASE_URL="${JUDGE_BASE_URL}" \
    RUNTIME_PYTHON="${RUNTIME_WRAPPER}" \
    REAL_RUNTIME_PYTHON="${REAL_RUNTIME_PYTHON}" \
    bash "${PROJECT_ROOT}/scripts/run_horizon_checkpoint_kl_then_audit.sh"
  [[ -s "${SCIENCE_AUDIT_MARKER}" ]] || {
    echo "Science audit did not produce its completion marker" >&2
    return 1
  }
}

main() {
  preflight
  prepare_science_eval100
  wait_for_pause_checkpoint
  stop_current_science
  wait_for_policy_capacity
  run_medicine_kl_and_audit
  resume_science_training
  wait_for_policy_capacity
  run_science_kl_and_audit
  set_stage "complete"
}

main "$@"
