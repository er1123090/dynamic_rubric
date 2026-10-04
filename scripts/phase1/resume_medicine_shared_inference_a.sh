#!/usr/bin/env bash
# Resume the existing scientific run; do not create a new experiment/config.
set -euo pipefail
cd ${PROJECT_ROOT}
run=outputs/medicine/online_rubrics/seed-11/phase1-online-rubrics-medicine-full-20260905-seed11-final
export PYTHONPATH=${PROJECT_ROOT}/src
export CUDA_VISIBLE_DEVICES=1
export ONLINE_CONTROL_CACHE=${PROJECT_ROOT}/outputs/medicine/shared/seed-11/pi0_control_cache/manifest-ad74decb90ce01cfc4a3048e21757e6fca58d6c77095d86c32d40507802d0407.json
export PHASE1_GPT_OSS_BASE_URLS=http://127.0.0.1:28011
export PHASE1_QWEN32B_BASE_URLS=http://127.0.0.1:28002
export PHASE1_QWEN32B_EXPECTED_COUNT=1
export ONLINE_LOGPROB_PREFETCH=true
export PYTHONUNBUFFERED=1
# Allow an explicit serving-memory override for concurrent KL; scientific
# sampling and optimizer configuration are unchanged.
export ROLLOUT_GPU_MEMORY=${ROLLOUT_GPU_MEMORY:-0.55}
export ROLLOUT_MAX_NUM_SEQS=128
export ROLLOUT_MAX_NUM_BATCHED_TOKENS=16384
exec flock -n "${run}/resume-training.lock" \
  ${PYTHON_BIN} -m dynamic_rubric.phase1 resume-online \
  --config "${run}/config.resolved.json" --repo-root . \
  --run-id phase1-online-rubrics-medicine-full-20260905-seed11-final
