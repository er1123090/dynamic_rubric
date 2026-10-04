#!/usr/bin/env bash
set -euo pipefail
exec env CUDA_VISIBLE_DEVICES=1 CPATH=${PYTHON_INCLUDE_DIR} \
  ${VLLM_BIN} serve \
  "${GPT_OSS_MODEL_PATH:?Set GPT_OSS_MODEL_PATH to a local model directory}" \
  --served-model-name openai/gpt-oss-120b \
  --tensor-parallel-size 1 --gpu-memory-utilization 0.70 \
  --max-model-len 32768 --enable-prefix-caching \
  --max-num-batched-tokens 16384 --max-num-seqs 128 \
  --host 127.0.0.1 --port 28011
