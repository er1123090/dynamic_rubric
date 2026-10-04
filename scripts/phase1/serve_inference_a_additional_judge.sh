#!/usr/bin/env bash
# Optional Docker helper; execute through stdin on the configured inference host.
# Dedicated replica; never stop or alter the existing GPU 0,1 services.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=2,3
export OMP_NUM_THREADS=1
if ! nvidia-smi -i 2,3 --query-gpu=memory.used --format=csv,noheader,nounits |
  awk 'BEGIN {ok=1; n=0} {n++; if ($1 > 1024) ok=0} END {exit !(ok && n==2)}'; then
  echo 'Refusing launch: Inference A GPU 2 or 3 is already occupied.' >&2
  exit 1
fi
exec ${VLLM_BIN} serve \
  ${MODEL_PATH} \
  --served-model-name Qwen/Qwen3-32B --tensor-parallel-size 2 --dtype bfloat16 \
  --gpu-memory-utilization 0.90 --max-model-len 32768 \
  --max-num-seqs 128 --max-num-batched-tokens 16384 \
  --enable-prefix-caching --generation-config vllm --host 0.0.0.0 --port 8005
