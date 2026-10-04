#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SSH_CONFIG="${PHASE1_SSH_CONFIG:-${REPO_ROOT}/configs/phase1/ssh_config}"
SSH_TARGET="${PHASE1_INFERENCE_A_SSH_TARGET:?Set PHASE1_INFERENCE_A_SSH_TARGET to an SSH config alias or destination}"
ACTION="${1:-start}"
PORT="${PHASE1_GPT_OSS_PORT:-8001}"
CONTAINER="${PHASE1_INFERENCE_A_CONTAINER:?Set PHASE1_INFERENCE_A_CONTAINER for this optional Docker helper}"
VLLM_BIN="${PHASE1_INFERENCE_A_VLLM_BIN:?Set PHASE1_INFERENCE_A_VLLM_BIN to the vllm executable inside the container}"
MODEL="${PHASE1_GPT_OSS_MODEL:-openai/gpt-oss-120b}"
LOG_FILE="${PHASE1_GPT_OSS_LOG:?Set PHASE1_GPT_OSS_LOG to a writable path inside the container}"
GPUS="${PHASE1_INFERENCE_A_GPUS:-0,1}"
GPU_MEMORY_UTILIZATION="${PHASE1_GPT_OSS_GPU_MEMORY_UTILIZATION:-0.72}"
MAX_MODEL_LEN="${PHASE1_GPT_OSS_MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${PHASE1_GPT_OSS_MAX_NUM_SEQS:-40}"
MAX_NUM_BATCHED_TOKENS="${PHASE1_GPT_OSS_MAX_NUM_BATCHED_TOKENS:-8192}"

if [[ ! -r "${SSH_CONFIG}" ]]; then
  echo "SSH config not readable: ${SSH_CONFIG}" >&2
  exit 2
fi
if [[ "${ACTION}" != "start" && "${ACTION}" != "status" ]]; then
  echo "Usage: $0 [start|status]" >&2
  exit 2
fi

exec ssh -F "${SSH_CONFIG}" -t "${SSH_TARGET}" bash -s -- \
  "${ACTION}" "${PORT}" "${CONTAINER}" "${VLLM_BIN}" "${MODEL}" "${LOG_FILE}" "${GPUS}" \
  "${GPU_MEMORY_UTILIZATION}" "${MAX_MODEL_LEN}" "${MAX_NUM_SEQS}" "${MAX_NUM_BATCHED_TOKENS}" <<'REMOTE'
set -euo pipefail
ACTION="$1"
PORT="$2"
CONTAINER="$3"
VLLM_BIN="$4"
MODEL="$5"
LOG_FILE="$6"
GPUS="$7"
GPU_MEMORY_UTILIZATION="$8"
MAX_MODEL_LEN="$9"
MAX_NUM_SEQS="${10}"
MAX_NUM_BATCHED_TOKENS="${11}"

if [[ "${GPUS}" != "0,1" ]]; then
  echo "This retraining requires inference_a GPUs 0,1; got: ${GPUS}" >&2; exit 2
fi

endpoint_ready() {
  docker exec "${CONTAINER}" sh -lc \
    "curl -fsS --max-time 3 http://127.0.0.1:${PORT}/v1/models >/dev/null" 2>/dev/null
}
process_running() {
  docker exec "${CONTAINER}" sh -lc \
    "ps -eo args | grep -F 'serve' | grep -F 'gpt-oss-120b' | grep -v grep >/dev/null" 2>/dev/null
}

if ! docker inspect "${CONTAINER}" >/dev/null 2>&1; then
  echo "inference_a container does not exist: ${CONTAINER}" >&2
  exit 3
fi
if [[ "$(docker inspect -f '{{.State.Running}}' "${CONTAINER}")" != "true" ]]; then
  echo "inference_a container is not running: ${CONTAINER}" >&2
  exit 3
fi

if endpoint_ready; then
  echo "READY inference_a gpt-oss-120b http://127.0.0.1:${PORT}/v1"
  exit 0
fi
if process_running; then
  echo "STARTING inference_a gpt-oss-120b; log=${LOG_FILE}"
  exit 0
fi
if [[ "${ACTION}" == "status" ]]; then
  echo "STOPPED inference_a gpt-oss-120b"
  exit 1
fi
if docker exec "${CONTAINER}" sh -lc \
  "command -v ss >/dev/null && ss -ltn | grep -q ':${PORT} '"; then
  echo "Port ${PORT} is occupied by another service; refusing to relaunch." >&2
  exit 4
fi
docker exec "${CONTAINER}" test -x "${VLLM_BIN}"
VLLM_VERSION="$(docker exec "${CONTAINER}" "${VLLM_BIN}" --version)"
if [[ "${VLLM_VERSION}" != *"0.20.1"* ]]; then
  echo "Expected vLLM 0.20.1, got: ${VLLM_VERSION}" >&2
  exit 5
fi
docker exec "${CONTAINER}" mkdir -p "$(dirname "${LOG_FILE}")"
docker exec -d \
  -e CUDA_VISIBLE_DEVICES="${GPUS}" \
  -e VLLM_USE_DEEP_GEMM=0 \
  "${CONTAINER}" sh -lc \
  "exec '${VLLM_BIN}' serve '${MODEL}' --served-model-name 'openai/gpt-oss-120b' --tensor-parallel-size 2 --dtype auto --gpu-memory-utilization '${GPU_MEMORY_UTILIZATION}' --max-model-len '${MAX_MODEL_LEN}' --max-num-seqs '${MAX_NUM_SEQS}' --max-num-batched-tokens '${MAX_NUM_BATCHED_TOKENS}' --enforce-eager --enable-prefix-caching --generation-config vllm --host 0.0.0.0 --port '${PORT}' >>'${LOG_FILE}' 2>&1"
echo "LAUNCHED inference_a gpt-oss-120b; gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}; max_model_len=${MAX_MODEL_LEN}; max_num_seqs=${MAX_NUM_SEQS}; max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS}; log=${LOG_FILE}"
REMOTE
