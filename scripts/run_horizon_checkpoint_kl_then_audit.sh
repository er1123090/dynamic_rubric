#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
DOMAIN=${DOMAIN:?DOMAIN must be medicine or science}
TRAINING_SEED=${TRAINING_SEED:-11}
POLICY_GPU=${POLICY_GPU:-0}
CONFIG_PATH=${CONFIG_PATH:-${PROJECT_ROOT}/configs/horizon_${DOMAIN}.yaml}
PROMPTS=${PROMPTS:-${PROJECT_ROOT}/data/rar/${DOMAIN}/public/final.jsonl}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts/horizon/${DOMAIN}}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-${ARTIFACT_ROOT}/training/seed-${TRAINING_SEED}/checkpoints}
EXPORT_ROOT=${EXPORT_ROOT:-${ARTIFACT_ROOT}/inference/seed-${TRAINING_SEED}}
POOL_ROOT=${POOL_ROOT:-${ARTIFACT_ROOT}/pools}
AUDIT_LOG_ROOT=${AUDIT_LOG_ROOT:-${ARTIFACT_ROOT}/audit/logs}
KL_ROOT=${KL_ROOT:-${ARTIFACT_ROOT}/checkpoint_kl/seed-${TRAINING_SEED}}
KL_SCORE_ROOT=${KL_SCORE_ROOT:-${KL_ROOT}/scores}
KL_SEAL=${KL_SEAL:-${KL_ROOT}/adjacent_kl_seal.json}
RUNTIME_PYTHON=${RUNTIME_PYTHON:-${PROJECT_ROOT}/environment/upstream/verl/.venv-runtime/bin/python}
VERL_ROOT=${VERL_ROOT:-${PROJECT_ROOT}/environment/upstream/verl}
POLICY_UPSTREAM_PORT=${POLICY_UPSTREAM_PORT:-8200}
POLICY_PROXY_PORT=${POLICY_PROXY_PORT:-8201}
POLICY_CONCURRENCY=${POLICY_CONCURRENCY:-256}
KL_BATCH_SIZE=${KL_BATCH_SIZE:-32}
POLICY_MODEL=${POLICY_MODEL:-Qwen/Qwen3-1.7B}
POLICY_REVISION=${POLICY_REVISION:-70d244cc86ccca08cf5af4e1e306ecf908b1ad5e}

checkpoint_steps=(0 3 6 9 13 16 24 32 40 48)
expected_prompt_count=100
policy_upstream_pid=
policy_proxy_pid=

case "${DOMAIN}" in
  medicine|science) ;;
  *) echo "DOMAIN must be medicine or science" >&2; exit 2 ;;
esac

mkdir -p "${EXPORT_ROOT}" "${POOL_ROOT}" "${AUDIT_LOG_ROOT}" "${KL_SCORE_ROOT}"

log() {
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$*"
}

run_cli() {
  PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
    uv run --project "${PROJECT_ROOT}" python -m dynamic_rubric "$@"
}

wait_url() {
  local url=$1
  local label=$2
  local attempt
  for attempt in $(seq 1 240); do
    if curl -fsS --max-time 3 "${url}" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  echo "${label} did not become ready: ${url}" >&2
  return 1
}

require_lines() {
  local path=$1
  local expected=$2
  local label=$3
  [[ -f "${path}" ]] || { echo "missing ${label}: ${path}" >&2; return 1; }
  local actual
  actual=$(wc -l < "${path}")
  [[ "${actual}" -eq "${expected}" ]] || {
    echo "wrong ${label} row count: expected=${expected}, actual=${actual}, path=${path}" >&2
    return 1
  }
}

checkpoint_hash() {
  sha256sum "$1" | awk '{print $1}'
}

stop_policy_server() {
  if [[ -n "${policy_proxy_pid}" ]]; then
    kill "${policy_proxy_pid}" 2>/dev/null || true
    wait "${policy_proxy_pid}" 2>/dev/null || true
    policy_proxy_pid=
  fi
  if [[ -n "${policy_upstream_pid}" ]]; then
    kill "${policy_upstream_pid}" 2>/dev/null || true
    wait "${policy_upstream_pid}" 2>/dev/null || true
    policy_upstream_pid=
  fi
}

trap stop_policy_server EXIT

export_checkpoint() {
  local step=$1
  local actor_dir="${CHECKPOINT_ROOT}/global_step_${step}/actor"
  local source_model="${actor_dir}/model_world_size_1_rank_0.pt"
  local export_dir="${EXPORT_ROOT}/global_step_${step}"
  local temporary_dir="${EXPORT_ROOT}/.global_step_${step}.tmp.$$"
  local merge_log="${AUDIT_LOG_ROOT}/checkpoint-merge-step-${step}.log"

  [[ -f "${source_model}" ]] || {
    echo "checkpoint model is missing: ${source_model}" >&2
    return 1
  }
  [[ -d "${actor_dir}/huggingface" ]] || {
    echo "checkpoint Hugging Face metadata is missing: ${actor_dir}/huggingface" >&2
    return 1
  }
  if [[ -f "${export_dir}/config.json" ]] && compgen -G "${export_dir}/*.safetensors" >/dev/null; then
    printf '%s\n' "${export_dir}"
    return 0
  fi
  if [[ -e "${export_dir}" ]]; then
    echo "incomplete inference export requires inspection: ${export_dir}" >&2
    return 1
  fi

  mkdir -p "${temporary_dir}"
  if ! (
    cd "${VERL_ROOT}"
    "${RUNTIME_PYTHON}" -m verl.model_merger merge \
      --backend fsdp \
      --use_cpu_initialization \
      --local_dir "${actor_dir}" \
      --target_dir "${temporary_dir}" \
      >"${merge_log}" 2>&1
  ); then
    echo "checkpoint export failed; temporary directory retained: ${temporary_dir}" >&2
    return 1
  fi
  [[ -f "${temporary_dir}/config.json" ]] || {
    echo "checkpoint export omitted config.json: ${temporary_dir}" >&2
    return 1
  }
  compgen -G "${temporary_dir}/*.safetensors" >/dev/null || {
    echo "checkpoint export omitted safetensors: ${temporary_dir}" >&2
    return 1
  }
  mv "${temporary_dir}" "${export_dir}"
  printf '%s\n' "${export_dir}"
}

start_policy_server() {
  local model_path=$1
  local hash=$2
  local step=$3
  local upstream_log="${AUDIT_LOG_ROOT}/policy-kl-vllm-step-${step}.log"
  local proxy_log="${AUDIT_LOG_ROOT}/policy-kl-proxy-step-${step}.log"

  stop_policy_server
  env \
    CUDA_VISIBLE_DEVICES="${POLICY_GPU}" \
    VLLM_USE_DEEP_GEMM=0 \
    VLLM_MOE_USE_DEEP_GEMM=0 \
    VLLM_DEEP_GEMM_WARMUP=skip \
    "${RUNTIME_PYTHON}" -m vllm.entrypoints.openai.api_server \
      --model "${model_path}" \
      --served-model-name "${POLICY_MODEL}" \
      --host 127.0.0.1 \
      --port "${POLICY_UPSTREAM_PORT}" \
      --dtype bfloat16 \
      --gpu-memory-utilization 0.92 \
      --max-model-len 7680 \
      --max-num-seqs 256 \
      --max-num-batched-tokens 65536 \
      --enable-prefix-caching \
      --generation-config vllm \
      >"${upstream_log}" 2>&1 &
  policy_upstream_pid=$!
  wait_url "http://127.0.0.1:${POLICY_UPSTREAM_PORT}/v1/models" "policy vLLM step ${step}"

  PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
    "${RUNTIME_PYTHON}" -m dynamic_rubric.services.vllm_policy_proxy \
      --upstream "http://127.0.0.1:${POLICY_UPSTREAM_PORT}" \
      --model-path "${model_path}" \
      --served-model "${POLICY_MODEL}" \
      --model-revision "${POLICY_REVISION}" \
      --tokenizer-revision "${POLICY_REVISION}" \
      --checkpoint-hash "${hash}" \
      --timeout-seconds 900 \
      --host 127.0.0.1 \
      --port "${POLICY_PROXY_PORT}" \
      >"${proxy_log}" 2>&1 &
  policy_proxy_pid=$!
  wait_url "http://127.0.0.1:${POLICY_PROXY_PORT}/health" "policy identity proxy step ${step}"
}

generate_checkpoint_pools() {
  local step=$1
  local hash=$2
  local pool_a="${POOL_ROOT}/seed-${TRAINING_SEED}-step-${step}-pool-a.jsonl"
  local pool_b="${POOL_ROOT}/seed-${TRAINING_SEED}-step-${step}-pool-b.jsonl"
  local base_url="http://127.0.0.1:${POLICY_PROXY_PORT}"

  if [[ ! -f "${pool_a}" ]]; then
    DYNAMIC_RUBRIC_POLICY_CONCURRENCY="${POLICY_CONCURRENCY}" run_cli \
      generate-horizon-pools \
      --config "${CONFIG_PATH}" \
      --run-id "rar-horizon-v1-${DOMAIN}-seed${TRAINING_SEED}-step${step}-pool-a" \
      --prompts "${PROMPTS}" \
      --pool-family pool_a \
      --count 8 \
      --policy-step "${step}" \
      --training-seed "${TRAINING_SEED}" \
      --checkpoint-hash "${hash}" \
      --base-url "${base_url}" \
      --output "${pool_a}"
  fi
  require_lines "${pool_a}" "$((expected_prompt_count * 8))" "step ${step} Pool A"

  if [[ ! -f "${pool_b}" ]]; then
    DYNAMIC_RUBRIC_POLICY_CONCURRENCY="${POLICY_CONCURRENCY}" run_cli \
      generate-horizon-pools \
      --config "${CONFIG_PATH}" \
      --run-id "rar-horizon-v1-${DOMAIN}-seed${TRAINING_SEED}-step${step}-pool-b" \
      --prompts "${PROMPTS}" \
      --pool-family pool_b \
      --count 16 \
      --policy-step "${step}" \
      --training-seed "${TRAINING_SEED}" \
      --checkpoint-hash "${hash}" \
      --base-url "${base_url}" \
      --output "${pool_b}"
  fi
  require_lines "${pool_b}" "$((expected_prompt_count * 16))" "step ${step} Pool B"
}

score_checkpoint() {
  local step=$1
  local hash=$2
  shift 2
  PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
    "${RUNTIME_PYTHON}" "${PROJECT_ROOT}/scripts/compute_horizon_checkpoint_kl.py" score \
      --prompts "${PROMPTS}" \
      --pool-b "$@" \
      --policy-step "${step}" \
      --checkpoint-hash "${hash}" \
      --identity-base-url "http://127.0.0.1:${POLICY_PROXY_PORT}" \
      --score-base-url "http://127.0.0.1:${POLICY_UPSTREAM_PORT}" \
      --served-model "${POLICY_MODEL}" \
      --model-revision "${POLICY_REVISION}" \
      --tokenizer-revision "${POLICY_REVISION}" \
      --output-dir "${KL_SCORE_ROOT}" \
      --batch-size "${KL_BATCH_SIZE}"
}

if [[ -f "${KL_SEAL}" ]]; then
  log "validating completed adjacent checkpoint KL seal"
else
  log "preparing Pool B and adjacent checkpoint KL before 100-prompt evaluation"
  for index in $(seq 0 9); do
    step=${checkpoint_steps[$index]}
    source_model="${CHECKPOINT_ROOT}/global_step_${step}/actor/model_world_size_1_rank_0.pt"
    hash=$(checkpoint_hash "${source_model}")
    model_path=$(export_checkpoint "${step}")
    log "serving checkpoint step ${step} for Pool B preparation and KL scoring"
    start_policy_server "${model_path}" "${hash}" "${step}"

    if [[ "${step}" -ne 0 ]]; then
      generate_checkpoint_pools "${step}" "${hash}"
    fi
    current_pool="${POOL_ROOT}/seed-${TRAINING_SEED}-step-${step}-pool-b.jsonl"
    require_lines "${current_pool}" "$((expected_prompt_count * 16))" "step ${step} Pool B"
    score_pools=("${current_pool}")
    if [[ "${index}" -gt 0 ]]; then
      previous_step=${checkpoint_steps[$((index - 1))]}
      previous_pool="${POOL_ROOT}/seed-${TRAINING_SEED}-step-${previous_step}-pool-b.jsonl"
      require_lines "${previous_pool}" "$((expected_prompt_count * 16))" "step ${previous_step} Pool B"
      score_pools=("${previous_pool}" "${current_pool}")
    fi
    log "scoring checkpoint step ${step} on ${#score_pools[@]} Pool-B shard(s)"
    score_checkpoint "${step}" "${hash}" "${score_pools[@]}"
    stop_policy_server
  done
fi

log "building sealed adjacent checkpoint KL report"
PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
  "${RUNTIME_PYTHON}" "${PROJECT_ROOT}/scripts/compute_horizon_checkpoint_kl.py" analyze \
    --score-dir "${KL_SCORE_ROOT}" \
    --checkpoint-steps "${checkpoint_steps[@]}" \
    --prompts "${PROMPTS}" \
    --output-dir "${KL_ROOT}" \
    --responses-per-prompt 16
[[ -s "${KL_SEAL}" ]] || { echo "adjacent checkpoint KL seal is missing" >&2; exit 1; }

log "adjacent checkpoint KL complete; starting the existing 100-prompt rubric evaluation"
stop_policy_server
trap - EXIT
exec env \
  DOMAIN="${DOMAIN}" \
  TRAINING_SEED="${TRAINING_SEED}" \
  POLICY_GPU="${POLICY_GPU}" \
  CONFIG_PATH="${CONFIG_PATH}" \
  PROMPTS="${PROMPTS}" \
  ARTIFACT_ROOT="${ARTIFACT_ROOT}" \
  bash "${PROJECT_ROOT}/scripts/run_horizon_audit.sh"
