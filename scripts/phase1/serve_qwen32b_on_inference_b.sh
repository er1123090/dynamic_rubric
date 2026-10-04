#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SSH_CONFIG="${PHASE1_SSH_CONFIG:-${REPO_ROOT}/configs/phase1/ssh_config}"
SSH_TARGET="${PHASE1_INFERENCE_B_SSH_TARGET:?Set PHASE1_INFERENCE_B_SSH_TARGET to an SSH config alias or destination}"
ACTION="${1:-start}"
PORT="${PHASE1_QWEN32B_PORT:-8002}"
CONTAINER="${PHASE1_INFERENCE_B_CONTAINER:?Set PHASE1_INFERENCE_B_CONTAINER for this optional Docker helper}"
IMAGE="${PHASE1_INFERENCE_B_IMAGE:?Set PHASE1_INFERENCE_B_IMAGE for this optional Docker helper}"
VLLM_BIN="${PHASE1_INFERENCE_B_VLLM_BIN:?Set PHASE1_INFERENCE_B_VLLM_BIN to the vllm executable inside the container}"
MODEL="${PHASE1_QWEN32B_MODEL_PATH:?Set PHASE1_QWEN32B_MODEL_PATH to the model path inside the container}"
LOG_FILE="${PHASE1_QWEN32B_LOG:?Set PHASE1_QWEN32B_LOG to a writable path inside the container}"
MOUNT="${PHASE1_INFERENCE_B_MOUNT:?Set PHASE1_INFERENCE_B_MOUNT to host_path:container_path}"

if [[ ! -r "${SSH_CONFIG}" ]]; then
  echo "SSH config not readable: ${SSH_CONFIG}" >&2
  exit 2
fi
if [[ "${ACTION}" != "start" && "${ACTION}" != "status" ]]; then
  echo "Usage: $0 [start|status]" >&2
  exit 2
fi

exec ssh -F "${SSH_CONFIG}" -t "${SSH_TARGET}" bash -s -- \
  "${ACTION}" "${PORT}" "${CONTAINER}" "${IMAGE}" "${VLLM_BIN}" "${MODEL}" "${LOG_FILE}" "${MOUNT}" <<'REMOTE'
set -euo pipefail
ACTION="$1"
PORT="$2"
CONTAINER="$3"
IMAGE="$4"
VLLM_BIN="$5"
MODEL="$6"
LOG_FILE="$7"
MOUNT="$8"

container_exists() { docker inspect "${CONTAINER}" >/dev/null 2>&1; }
container_running() {
  container_exists && [[ "$(docker inspect -f '{{.State.Running}}' "${CONTAINER}")" == "true" ]]
}
endpoint_ready() {
  container_running && docker exec "${CONTAINER}" sh -lc \
    "curl -fsS --max-time 3 http://127.0.0.1:${PORT}/v1/models >/dev/null" 2>/dev/null
}
process_running() {
  container_running && docker exec "${CONTAINER}" sh -lc \
    "ps -eo args | grep -F 'serve' | grep -F 'Qwen3-32B' | grep -v grep >/dev/null" 2>/dev/null
}

if endpoint_ready; then
  echo "READY inference_b Qwen3-32B http://127.0.0.1:${PORT}/v1"
  exit 0
fi
if process_running; then
  echo "STARTING inference_b Qwen3-32B; log=${LOG_FILE}"
  exit 0
fi
if [[ "${ACTION}" == "status" ]]; then
  echo "STOPPED inference_b Qwen3-32B"
  exit 1
fi

if ! container_exists; then
  docker run -d \
    --name "${CONTAINER}" \
    --network host \
    --gpus '"device=0,1"' \
    -e NVIDIA_VISIBLE_DEVICES=0,1 \
    -v "${MOUNT}" \
    "${IMAGE}" sleep infinity >/dev/null
elif ! container_running; then
  docker start "${CONTAINER}" >/dev/null
fi

if docker exec "${CONTAINER}" sh -lc \
  "command -v ss >/dev/null && ss -ltn | grep -q ':${PORT} '"; then
  echo "Port ${PORT} is occupied by another service; refusing to relaunch." >&2
  exit 4
fi
docker exec "${CONTAINER}" test -x "${VLLM_BIN}"
docker exec "${CONTAINER}" test -d "${MODEL}"
VLLM_VERSION="$(docker exec "${CONTAINER}" "${VLLM_BIN}" --version)"
if [[ "${VLLM_VERSION}" != *"0.19.1"* ]]; then
  echo "Expected vLLM 0.19.1, got: ${VLLM_VERSION}" >&2
  exit 5
fi
docker exec "${CONTAINER}" mkdir -p "$(dirname "${LOG_FILE}")"
docker exec -d \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e NVIDIA_VISIBLE_DEVICES=0,1 \
  -e VLLM_USE_DEEP_GEMM=0 \
  "${CONTAINER}" sh -lc \
  "exec '${VLLM_BIN}' serve '${MODEL}' --served-model-name 'Qwen/Qwen3-32B' --revision 9216db5781bf21249d130ec9da846c4624c16137 --tokenizer-revision 9216db5781bf21249d130ec9da846c4624c16137 --tensor-parallel-size 2 --dtype bfloat16 --gpu-memory-utilization 0.90 --max-model-len 8192 --enable-prefix-caching --generation-config vllm --host 0.0.0.0 --port '${PORT}' >>'${LOG_FILE}' 2>&1"
echo "LAUNCHED inference_b Qwen3-32B; log=${LOG_FILE}"
REMOTE
