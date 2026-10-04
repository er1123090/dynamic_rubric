#!/usr/bin/env bash
# Inference-only readiness test. Stop this server before the GRPO trainer.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=1
exec ${VLLM_BIN} serve \
  ${PROJECT_ROOT}/outputs/medicine/online_rubrics/seed-11/phase1-fixed-probe-regular-through45-20260908/exports/global_step_45 \
  --served-model-name phase1-policy-checkpoint-45 --tensor-parallel-size 1 \
  --dtype bfloat16 --gpu-memory-utilization 0.20 --max-model-len 32768 \
  --max-num-seqs 16 --max-num-batched-tokens 4096 --enforce-eager \
  --generation-config vllm --host 127.0.0.1 --port 28010
