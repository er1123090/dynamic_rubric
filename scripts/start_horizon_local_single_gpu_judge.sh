#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
RUNTIME_PYTHON=${RUNTIME_PYTHON:-${PROJECT_ROOT}/environment/upstream/verl/.venv-runtime/bin/python}
PROXY_PYTHON=${PROXY_PYTHON:-${PYTHON_BIN}}
JUDGE_GPU=${JUDGE_GPU:-0}
JUDGE_PORT=${JUDGE_PORT:-8014}
PROXY_PORT=${PROXY_PORT:-8102}
JUDGE_KV_CACHE_MEMORY_BYTES=${JUDGE_KV_CACHE_MEMORY_BYTES:-8G}
JUDGE_MAX_NUM_SEQS=${JUDGE_MAX_NUM_SEQS:-16}
JUDGE_MAX_NUM_BATCHED_TOKENS=${JUDGE_MAX_NUM_BATCHED_TOKENS:-8192}
JUDGE_SESSION=${JUDGE_SESSION:-horizon-qwen-judge-trainer0}
PROXY_SESSION=${PROXY_SESSION:-horizon-qwen-judge-proxy}
MODEL_REVISION=${MODEL_REVISION:-9216db5781bf21249d130ec9da846c4624c16137}
MODEL_PATH=${MODEL_PATH:-/models/Qwen3-32B}
SERVED_MODEL=${SERVED_MODEL:-Qwen/Qwen3-32B}
LOG_ROOT=${LOG_ROOT:-${PROJECT_ROOT}/artifacts/logs}
CACHE_DIR=${CACHE_DIR:-${PROJECT_ROOT}/artifacts/provider_cache/horizon-vllm-score-proxy}

mkdir -p "${LOG_ROOT}" "${CACHE_DIR}"

if tmux has-session -t "${JUDGE_SESSION}" 2>/dev/null; then
  echo "judge session already exists: ${JUDGE_SESSION}" >&2
  exit 1
fi
if tmux has-session -t "${PROXY_SESSION}" 2>/dev/null; then
  echo "proxy session already exists: ${PROXY_SESSION}" >&2
  exit 1
fi

judge_command=$(printf '%q ' \
  env \
  "CUDA_VISIBLE_DEVICES=${JUDGE_GPU}" \
  VLLM_USE_DEEP_GEMM=0 \
  VLLM_MOE_USE_DEEP_GEMM=0 \
  VLLM_DEEP_GEMM_WARMUP=skip \
  "${RUNTIME_PYTHON}" -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL}" \
  --revision "${MODEL_REVISION}" \
  --tokenizer-revision "${MODEL_REVISION}" \
  --tensor-parallel-size 1 \
  --kv-cache-memory-bytes "${JUDGE_KV_CACHE_MEMORY_BYTES}" \
  --max-model-len 7680 \
  --max-num-seqs "${JUDGE_MAX_NUM_SEQS}" \
  --max-num-batched-tokens "${JUDGE_MAX_NUM_BATCHED_TOKENS}" \
  --dtype bfloat16 \
  --enable-prefix-caching \
  --enforce-eager \
  --generation-config vllm \
  --host 127.0.0.1 \
  --port "${JUDGE_PORT}")
tmux new-session -d -s "${JUDGE_SESSION}" \
  "${judge_command} >> '${LOG_ROOT}/horizon-qwen-judge-trainer0-low-memory.log' 2>&1"

ready=false
for _ in $(seq 1 180); do
  if curl -fsS --max-time 3 "http://127.0.0.1:${JUDGE_PORT}/v1/models" >/dev/null 2>&1; then
    ready=true
    break
  fi
  if ! tmux has-session -t "${JUDGE_SESSION}" 2>/dev/null; then
    break
  fi
  sleep 2
done
if [[ "${ready}" != true ]]; then
  echo "local judge failed to become ready" >&2
  tail -n 80 "${LOG_ROOT}/horizon-qwen-judge-trainer0-low-memory.log" >&2 || true
  tmux kill-session -t "${JUDGE_SESSION}" 2>/dev/null || true
  exit 1
fi

proxy_command=$(printf '%q ' \
  env "PYTHONPATH=${PROJECT_ROOT}/src" \
  "${PROXY_PYTHON}" -m dynamic_rubric.services.vllm_score_proxy \
  --upstream "http://127.0.0.1:${JUDGE_PORT}" \
  --upstream-weight 1 \
  --model-path "${MODEL_PATH}" \
  --served-model "${SERVED_MODEL}" \
  --model-revision "${MODEL_REVISION}" \
  --tokenizer-revision "${MODEL_REVISION}" \
  --cache-dir "${CACHE_DIR}" \
  --host 127.0.0.1 \
  --port "${PROXY_PORT}")
tmux new-session -d -s "${PROXY_SESSION}" \
  "${proxy_command} >> '${LOG_ROOT}/horizon-qwen-judge-proxy-local.log' 2>&1"

for _ in $(seq 1 60); do
  if curl -fsS --max-time 3 "http://127.0.0.1:${PROXY_PORT}/health" >/dev/null 2>&1; then
    curl -fsS -H 'Content-Type: application/json' \
      -d '{"rendered_prompts":["Criterion: synthetic\\nResponse: synthetic\\nAnswer:"],"targets":[" YES"," NO"],"temperature":0,"thinking":false}' \
      "http://127.0.0.1:${PROXY_PORT}/dynamic-rubric/score-targets" >/dev/null
    echo "local judge ready on GPU ${JUDGE_GPU}; proxy=http://127.0.0.1:${PROXY_PORT}"
    exit 0
  fi
  sleep 2
done

echo "local score proxy failed to become ready" >&2
tmux kill-session -t "${PROXY_SESSION}" 2>/dev/null || true
tmux kill-session -t "${JUDGE_SESSION}" 2>/dev/null || true
exit 1
