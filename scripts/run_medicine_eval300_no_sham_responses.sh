#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
CONFIG_PATH=${CONFIG_PATH:-${PROJECT_ROOT}/configs/horizon_medicine_eval300_no_sham.yaml}
PROMPTS=${PROMPTS:-${PROJECT_ROOT}/data/rar/medicine/eval300/final.jsonl}
DELTA_PROMPTS=${DELTA_PROMPTS:-${PROJECT_ROOT}/data/rar/medicine/eval300/final_delta200.jsonl}
SOURCE_POOL_ROOT=${SOURCE_POOL_ROOT:-${PROJECT_ROOT}/artifacts/horizon/medicine/pools}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts/horizon/medicine/eval300-no-sham}
DELTA_POOL_ROOT=${DELTA_POOL_ROOT:-${ARTIFACT_ROOT}/delta200/pools}
POOL_ROOT=${POOL_ROOT:-${ARTIFACT_ROOT}/pools}
EXPORT_ROOT=${EXPORT_ROOT:-${PROJECT_ROOT}/artifacts/horizon/medicine/inference/seed-11}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-${PROJECT_ROOT}/artifacts/horizon/medicine/training/seed-11/checkpoints}
LOG_ROOT=${LOG_ROOT:-${ARTIFACT_ROOT}/logs}
POLICY_GPU=${POLICY_GPU:-0}
POLICY_UPSTREAM_PORT=${POLICY_UPSTREAM_PORT:-8300}
POLICY_PROXY_PORT=${POLICY_PROXY_PORT:-8301}
POLICY_CONCURRENCY=${POLICY_CONCURRENCY:-256}
POLICY_MODEL=${POLICY_MODEL:-Qwen/Qwen3-1.7B}
POLICY_REVISION=${POLICY_REVISION:-70d244cc86ccca08cf5af4e1e306ecf908b1ad5e}
POLICY_BASE_PATH=${POLICY_BASE_PATH:-/models/Qwen3-1.7B}
RUNTIME_PYTHON=${RUNTIME_PYTHON:-${PROJECT_ROOT}/environment/upstream/verl/.venv-runtime/bin/python}

checkpoint_steps=(3 6 9 13 16 24 32 40 48)
policy_upstream_pid=
policy_proxy_pid=

mkdir -p "${DELTA_POOL_ROOT}" "${POOL_ROOT}" "${LOG_ROOT}"

log() {
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$*"
}

run_cli() {
  PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
    uv run --project "${PROJECT_ROOT}" python -m dynamic_rubric "$@"
}

require_lines() {
  local path=$1
  local expected=$2
  local actual
  [[ -f "${path}" ]] || { echo "missing file: ${path}" >&2; return 1; }
  actual=$(wc -l < "${path}")
  [[ "${actual}" -eq "${expected}" ]] || {
    echo "wrong row count: expected=${expected}, actual=${actual}, path=${path}" >&2
    return 1
  }
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

start_policy_server() {
  local model_path=$1
  local checkpoint_hash=$2
  local step=$3
  local upstream_log="${LOG_ROOT}/policy-vllm-step-${step}.log"
  local proxy_log="${LOG_ROOT}/policy-proxy-step-${step}.log"

  stop_policy_server
  if curl -fsS --max-time 2 "http://127.0.0.1:${POLICY_UPSTREAM_PORT}/v1/models" >/dev/null 2>&1; then
    echo "policy upstream port is occupied: ${POLICY_UPSTREAM_PORT}" >&2
    return 1
  fi
  if curl -fsS --max-time 2 "http://127.0.0.1:${POLICY_PROXY_PORT}/health" >/dev/null 2>&1; then
    echo "policy proxy port is occupied: ${POLICY_PROXY_PORT}" >&2
    return 1
  fi

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
      --checkpoint-hash "${checkpoint_hash}" \
      --timeout-seconds 900 \
      --host 127.0.0.1 \
      --port "${POLICY_PROXY_PORT}" \
      >"${proxy_log}" 2>&1 &
  policy_proxy_pid=$!
  wait_url "http://127.0.0.1:${POLICY_PROXY_PORT}/health" "policy proxy step ${step}"
}

generate_delta_pool() {
  local family=$1
  local count=$2
  local step=$3
  local run_id=$4
  local checkpoint_hash=$5
  local output=$6
  local args=(
    generate-horizon-pools
    --config "${CONFIG_PATH}"
    --run-id "${run_id}"
    --prompts "${DELTA_PROMPTS}"
    --pool-family "${family}"
    --count "${count}"
    --policy-step "${step}"
    --checkpoint-hash "${checkpoint_hash}"
    --base-url "http://127.0.0.1:${POLICY_PROXY_PORT}"
    --output "${output}"
  )
  if [[ "${family}" == "pool_a" || "${family}" == "pool_b" ]]; then
    args+=(--training-seed 11)
  fi
  if [[ ! -f "${output}" ]]; then
    DYNAMIC_RUBRIC_POLICY_CONCURRENCY="${POLICY_CONCURRENCY}" run_cli "${args[@]}"
  fi
  require_lines "${output}" "$((200 * count))"
}

merge_pool() {
  local existing=$1
  local delta=$2
  local count=$3
  local output=$4
  PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
    uv run --project "${PROJECT_ROOT}" python \
      "${PROJECT_ROOT}/scripts/merge_horizon_pool_extension.py" \
      --prompts "${PROMPTS}" \
      --prefix-prompts 100 \
      --existing-pool "${existing}" \
      --delta-pool "${delta}" \
      --count "${count}" \
      --output "${output}"
}

require_checkpoint_hash() {
  local existing_pool=$1
  local expected_hash=$2
  local actual_hash
  actual_hash=$(PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
    uv run --project "${PROJECT_ROOT}" python -c \
      'import json,sys; print(json.loads(open(sys.argv[1]).readline())["checkpoint_hash"])' \
      "${existing_pool}")
  [[ "${actual_hash}" == "${expected_hash}" ]] || {
    echo "checkpoint hash drift: expected=${actual_hash}, actual=${expected_hash}" >&2
    return 1
  }
}

require_lines "${PROMPTS}" 300
require_lines "${DELTA_PROMPTS}" 200
run_cli validate-config --config "${CONFIG_PATH}" >/dev/null
gpu_processes=$(nvidia-smi -i "${POLICY_GPU}" --query-compute-apps=pid --format=csv,noheader)
[[ -z "${gpu_processes}" ]] || {
  echo "Trainer GPU ${POLICY_GPU} is already occupied by PIDs: ${gpu_processes}" >&2
  exit 1
}

log "starting base policy on Trainer GPU ${POLICY_GPU}"
start_policy_server "${POLICY_BASE_PATH}" "${POLICY_REVISION}" 0
delta_fixed="${DELTA_POOL_ROOT}/fixed.jsonl"
require_checkpoint_hash "${SOURCE_POOL_ROOT}/fixed.jsonl" "${POLICY_REVISION}"
generate_delta_pool fixed_control 8 0 "rar-horizon-v1-medicine" "${POLICY_REVISION}" "${delta_fixed}"
merge_pool "${SOURCE_POOL_ROOT}/fixed.jsonl" "${delta_fixed}" 8 "${POOL_ROOT}/fixed.jsonl"

delta_step0="${DELTA_POOL_ROOT}/seed-11-step-0-pool-b.jsonl"
require_checkpoint_hash "${SOURCE_POOL_ROOT}/seed-11-step-0-pool-b.jsonl" "${POLICY_REVISION}"
generate_delta_pool pool_b 16 0 "rar-horizon-v1-medicine-seed11-step0" "${POLICY_REVISION}" "${delta_step0}"
merge_pool \
  "${SOURCE_POOL_ROOT}/seed-11-step-0-pool-b.jsonl" \
  "${delta_step0}" 16 "${POOL_ROOT}/seed-11-step-0-pool-b.jsonl"
stop_policy_server

for step in "${checkpoint_steps[@]}"; do
  model_path="${EXPORT_ROOT}/global_step_${step}"
  [[ -f "${model_path}/model.safetensors" ]] || {
    echo "missing exported checkpoint: ${model_path}/model.safetensors" >&2
    exit 1
  }
  source_model="${CHECKPOINT_ROOT}/global_step_${step}/actor/model_world_size_1_rank_0.pt"
  [[ -f "${source_model}" ]] || {
    echo "missing source checkpoint: ${source_model}" >&2
    exit 1
  }
  checkpoint_hash=$(sha256sum "${source_model}" | awk '{print $1}')
  require_checkpoint_hash \
    "${SOURCE_POOL_ROOT}/seed-11-step-${step}-pool-a.jsonl" \
    "${checkpoint_hash}"
  log "starting checkpoint step ${step} on Trainer GPU ${POLICY_GPU}"
  start_policy_server "${model_path}" "${checkpoint_hash}" "${step}"
  for family in pool_a pool_b; do
    count=8
    [[ "${family}" == "pool_b" ]] && count=16
    pool_label=${family//_/-}
    delta_pool="${DELTA_POOL_ROOT}/seed-11-step-${step}-${pool_label}.jsonl"
    merged_pool="${POOL_ROOT}/seed-11-step-${step}-${pool_label}.jsonl"
    generate_delta_pool \
      "${family}" "${count}" "${step}" \
      "rar-horizon-v1-medicine-seed11-step${step}-${pool_label}" \
      "${checkpoint_hash}" "${delta_pool}"
    merge_pool \
      "${SOURCE_POOL_ROOT}/seed-11-step-${step}-${pool_label}.jsonl" \
      "${delta_pool}" "${count}" "${merged_pool}"
  done
  stop_policy_server
done

pool_files=("${POOL_ROOT}/fixed.jsonl")
pool_files+=("${POOL_ROOT}/seed-11-step-0-pool-b.jsonl")
for step in "${checkpoint_steps[@]}"; do
  pool_files+=("${POOL_ROOT}/seed-11-step-${step}-pool-a.jsonl")
  pool_files+=("${POOL_ROOT}/seed-11-step-${step}-pool-b.jsonl")
done
run_cli validate-horizon-inventory \
  --config "${CONFIG_PATH}" \
  --prompts "${PROMPTS}" \
  --pools "${pool_files[@]}" >/dev/null
touch "${ARTIFACT_ROOT}/responses.complete"
log "Medicine eval300 no-sham response generation complete"
