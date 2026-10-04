#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=1
export VLLM_USE_DEEP_GEMM=0

exec ${VLLM_BIN} serve \
  "${GPT_OSS_MODEL_PATH:?Set GPT_OSS_MODEL_PATH to a local model directory}" \
  --served-model-name openai/gpt-oss-120b \
  --tensor-parallel-size 1 \
  --dtype auto \
  --gpu-memory-utilization 0.55 \
  --cpu-offload-gb 0 \
  --max-model-len 32768 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 2048 \
  --enforce-eager \
  --enable-prefix-caching \
  --generation-config vllm \
  --host 127.0.0.1 \
  --port 28013
