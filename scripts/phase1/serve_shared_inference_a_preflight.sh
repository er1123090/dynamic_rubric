#!/usr/bin/env bash
# Execute inside the explicitly configured optional inference container.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=1
case "${1:?gpt or judge required}" in
  gpt)
    export VLLM_USE_DEEP_GEMM=0
    exec "${VLLM_BIN:?Set VLLM_BIN to the vllm executable}" serve \
      "${GPT_OSS_MODEL_PATH:?Set GPT_OSS_MODEL_PATH to a local model directory}" \
      --served-model-name openai/gpt-oss-120b --tensor-parallel-size 2 \
      --gpu-memory-utilization 0.48 --max-model-len 32768 \
      --max-num-seqs 16 --max-num-batched-tokens 2048 --enforce-eager \
      --enable-prefix-caching --generation-config vllm --host 0.0.0.0 --port 8001
    ;;
  judge)
    exec "${VLLM_BIN:?Set VLLM_BIN to the vllm executable}" serve \
      "${MODEL_PATH:?Set MODEL_PATH to a local model directory}" \
      --served-model-name Qwen/Qwen3-32B --tensor-parallel-size 2 --dtype bfloat16 \
      --gpu-memory-utilization 0.46 --max-model-len 32768 \
      --max-num-seqs 16 --max-num-batched-tokens 2048 --enforce-eager \
      --enable-prefix-caching --generation-config vllm --host 0.0.0.0 --port 8004
    ;;
  *) exit 2 ;;
esac
