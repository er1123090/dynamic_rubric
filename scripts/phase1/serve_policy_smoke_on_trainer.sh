#!/usr/bin/env bash
set -euo pipefail

PORT="${PHASE1_POLICY_PORT:-8000}"
MODEL="${POLICY_MODEL_PATH}"

if [[ ! -d "${MODEL}" ]]; then
  echo "Pinned Qwen3-4B snapshot is missing: ${MODEL}" >&2
  exit 2
fi

echo "This temporary server occupies trainer GPU 0; stop it before the GRPO optimizer starts." >&2
exec env CUDA_VISIBLE_DEVICES=0 vllm serve "${MODEL}"   --served-model-name Qwen/Qwen3-4B-Instruct-2507   --tensor-parallel-size 1   --host 0.0.0.0   --port "${PORT}"
