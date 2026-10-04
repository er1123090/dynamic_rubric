#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PIPELINE_PYTHON=${PIPELINE_PYTHON:-${PYTHON_BIN}}
RUN_ID=${RUN_ID:-pilot-static-r0-v1}
CONFIG=${CONFIG:-configs/pilot.yaml}
RESUME_STATIC_ONLY=${RESUME_STATIC_ONLY:-false}

POLICY_MODEL_PATH=${POLICY_MODEL_PATH:-${POLICY_MODEL_PATH}}
GRADER_MODEL_PATH=${GRADER_MODEL_PATH:-${MODEL_PATH}}
EMBEDDING_MODEL_PATH=${EMBEDDING_MODEL_PATH:-${EMBEDDING_MODEL_PATH}}
POLICY_LAUNCH_SPEC=${POLICY_LAUNCH_SPEC:-${PROJECT_ROOT}/environment/policy-stage4-launch.json}

POLICY_URL=${DYNAMIC_RUBRIC_POLICY_URL:-http://127.0.0.1:8001}
GRADER_UPSTREAM_URL=${DYNAMIC_RUBRIC_GRADER_UPSTREAM_URL:-http://127.0.0.1:8002}
GRADER_UPSTREAM_URLS_RAW=${DYNAMIC_RUBRIC_GRADER_UPSTREAM_URLS:-${GRADER_UPSTREAM_URL}}
GRADER_URL=${DYNAMIC_RUBRIC_VLLM_URL:-http://127.0.0.1:8102}
read -r -a grader_upstream_urls <<<"${GRADER_UPSTREAM_URLS_RAW}"
if ((${#grader_upstream_urls[@]} == 0)); then
  echo "At least one grader upstream URL is required" >&2
  exit 1
fi
GRADER_UPSTREAM_URL=${grader_upstream_urls[0]}

policy_pid=""
grader_pid=""
proxy_pid=""

stop_process() {
  local pid=${1:-}
  if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    kill "${pid}"
    wait "${pid}" 2>/dev/null || true
  fi
}

cleanup() {
  stop_process "${policy_pid}"
  stop_process "${proxy_pid}"
  stop_process "${grader_pid}"
}
trap cleanup EXIT INT TERM

wait_url() {
  local url=$1
  local label=$2
  local attempts=${3:-180}
  for ((attempt = 1; attempt <= attempts; attempt++)); do
    if curl --fail --silent --show-error "${url}" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  echo "${label} did not become ready: ${url}" >&2
  return 1
}

wait_gpu0_free() {
  for ((attempt = 1; attempt <= 120; attempt++)); do
    local used
    used=$(nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits)
    if ((used < 1024)); then
      return 0
    fi
    sleep 2
  done
  echo "GPU 0 did not release the Stage 4 policy server allocation" >&2
  return 1
}

start_grader_if_needed() {
  local proxy_ready=false
  if curl --fail --silent "${GRADER_URL}/health" >/dev/null 2>&1; then
    proxy_ready=true
  fi
  if ! curl --fail --silent "${GRADER_UPSTREAM_URL}/v1/models" >/dev/null 2>&1; then
    if [[ "${GRADER_UPSTREAM_URL}" != "http://127.0.0.1:8002" ]]; then
      echo "Configured external grader upstream is unavailable: ${GRADER_UPSTREAM_URL}" >&2
      return 1
    fi
    CUDA_VISIBLE_DEVICES=1 nohup vllm serve "${GRADER_MODEL_PATH}" \
      --served-model-name Qwen/Qwen3-32B \
      --revision 9216db5781bf21249d130ec9da846c4624c16137 \
      --tokenizer-revision 9216db5781bf21249d130ec9da846c4624c16137 \
      --gpu-memory-utilization 0.90 \
      --max-model-len 2048 \
      --max-num-seqs 64 \
      --max-num-batched-tokens 16384 \
      --dtype bfloat16 \
      --enable-prefix-caching \
      --generation-config vllm \
      --host 127.0.0.1 \
      --port 8002 \
      >"/tmp/dynamic-rubric-grader-${RUN_ID}.log" 2>&1 &
    grader_pid=$!
    wait_url "${GRADER_UPSTREAM_URL}/v1/models" "GPU 1 Qwen3-32B grader"
  fi
  local upstream
  for upstream in "${grader_upstream_urls[@]}"; do
    wait_url "${upstream}/v1/models" "Qwen3-32B grader replica"
  done
  if [[ "${proxy_ready}" == true ]]; then
    return 0
  fi
  local proxy_upstream_args=()
  for upstream in "${grader_upstream_urls[@]}"; do
    proxy_upstream_args+=(--upstream "${upstream}")
  done
  PYTHONPATH="${PROJECT_ROOT}/src" nohup "${PIPELINE_PYTHON}" \
    -m dynamic_rubric.services.vllm_score_proxy \
    "${proxy_upstream_args[@]}" \
    --model-path "${GRADER_MODEL_PATH}" \
    --served-model Qwen/Qwen3-32B \
    --model-revision 9216db5781bf21249d130ec9da846c4624c16137 \
    --tokenizer-revision 9216db5781bf21249d130ec9da846c4624c16137 \
    --cache-dir "${PROJECT_ROOT}/artifacts/provider_cache/vllm-score-proxy" \
    --host 127.0.0.1 \
    --port 8102 \
    >"/tmp/dynamic-rubric-score-proxy-${RUN_ID}.log" 2>&1 &
  proxy_pid=$!
  wait_url "${GRADER_URL}/health" "GPU 1 score proxy"
}

start_policy() {
  if curl --fail --silent "${POLICY_URL}/v1/models" >/dev/null 2>&1; then
    return 0
  fi
  CUDA_VISIBLE_DEVICES=0 nohup vllm serve "${POLICY_MODEL_PATH}" \
    --served-model-name Qwen/Qwen3-4B-Instruct-2507 \
    --revision cdbee75f17c01a7cc42f958dc650907174af0554 \
    --tokenizer-revision cdbee75f17c01a7cc42f958dc650907174af0554 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 5632 \
    --max-num-seqs 64 \
    --max-num-batched-tokens 65536 \
    --dtype bfloat16 \
    --enable-prefix-caching \
    --generation-config vllm \
    --host 127.0.0.1 \
    --port 8001 \
    >"/tmp/dynamic-rubric-policy-${RUN_ID}.log" 2>&1 &
  policy_pid=$!
  wait_url "${POLICY_URL}/v1/models" "GPU 0 Qwen3-4B Stage 4 policy"
}

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export DYNAMIC_RUBRIC_POLICY_URL="${POLICY_URL}"
export DYNAMIC_RUBRIC_POLICY_LAUNCH_SPEC="${POLICY_LAUNCH_SPEC}"
export DYNAMIC_RUBRIC_VLLM_URL="${GRADER_URL}"
export DYNAMIC_RUBRIC_EMBEDDING_MODEL_PATH="${EMBEDDING_MODEL_PATH}"
export DYNAMIC_RUBRIC_EMBEDDING_DEVICE=${DYNAMIC_RUBRIC_EMBEDDING_DEVICE:-cuda:1}
export DYNAMIC_RUBRIC_POLICY_CONCURRENCY=${DYNAMIC_RUBRIC_POLICY_CONCURRENCY:-32}
export DYNAMIC_RUBRIC_OPENAI_CONCURRENCY=${DYNAMIC_RUBRIC_OPENAI_CONCURRENCY:-16}
export DYNAMIC_RUBRIC_POLICY_GPU=0
export DYNAMIC_RUBRIC_ROLLOUT_GPU_MEMORY=${DYNAMIC_RUBRIC_ROLLOUT_GPU_MEMORY:-0.42}
export VLLM_USE_DEEP_GEMM=0

start_grader_if_needed

if [[ "${RESUME_STATIC_ONLY}" == true ]]; then
  "${PIPELINE_PYTHON}" -m dynamic_rubric validate-config --config "${CONFIG}"
  "${PIPELINE_PYTHON}" -m dynamic_rubric train-static --config "${CONFIG}" --run-id "${RUN_ID}"
  exit 0
fi

: "${OPENAI_API_KEY:?Export a newly rotated OPENAI_API_KEY before running Stage 4}"
start_policy

"${PIPELINE_PYTHON}" -m dynamic_rubric validate-config --config "${CONFIG}"
"${PIPELINE_PYTHON}" -m dynamic_rubric preflight --config "${CONFIG}" --run-id "${RUN_ID}"
"${PIPELINE_PYTHON}" -m dynamic_rubric generate-static --config "${CONFIG}" --run-id "${RUN_ID}"

stop_process "${policy_pid}"
policy_pid=""
wait_gpu0_free
unset OPENAI_API_KEY

"${PIPELINE_PYTHON}" -m dynamic_rubric train-static --config "${CONFIG}" --run-id "${RUN_ID}"
