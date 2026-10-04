#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
MODEL_PATH=${MODEL_PATH:?Set MODEL_PATH to the local Qwen3-32B directory}
PYTHON_BIN=${PYTHON_BIN:?Set PYTHON_BIN to the local Python executable}
VLLM_BIN=${VLLM_BIN:?Set VLLM_BIN to the local vllm executable}
INFERENCE_A_ROOT=${INFERENCE_A_ROOT:?Set INFERENCE_A_ROOT to the working directory inside the remote container}
INFERENCE_A_VLLM_BIN=${INFERENCE_A_VLLM_BIN:?Set INFERENCE_A_VLLM_BIN to the remote vllm executable}
SSH_TARGET=${SSH_TARGET:?Set SSH_TARGET to user@inference-host}
SSH_PORT=${SSH_PORT:-14233}
SSH_SOCKET=${SSH_SOCKET:-/tmp/dynamic-rubric-inference_a-paper-judge.sock}
INFERENCE_A_CONTAINER=${INFERENCE_A_CONTAINER:?Set INFERENCE_A_CONTAINER for this optional Docker helper}
ACTION=${1:-start}

TRAINER_PORT=8004
INFERENCE_A0_PORT=8006
INFERENCE_A1_PORT=8007
INFERENCE_A0_TUNNEL=18047
INFERENCE_A1_TUNNEL=18057
PROXY_PORT=8104

wait_url() {
  local url=$1
  local label=$2
  for _ in $(seq 1 180); do
    if curl -fsS --max-time 3 "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  echo "$label did not become ready: $url" >&2
  return 1
}

remote() {
  ssh -S "$SSH_SOCKET" -p "$SSH_PORT" "$SSH_TARGET" "$@"
}

status() {
  curl -fsS --max-time 3 "http://127.0.0.1:${TRAINER_PORT}/v1/models" || true
  curl -fsS --max-time 3 "http://127.0.0.1:${INFERENCE_A0_TUNNEL}/v1/models" || true
  curl -fsS --max-time 3 "http://127.0.0.1:${INFERENCE_A1_TUNNEL}/v1/models" || true
  curl -fsS --max-time 3 "http://127.0.0.1:${PROXY_PORT}/dynamic-rubric/routing" || true
}

stop() {
  tmux kill-session -t paper-qwen-trainer0 2>/dev/null || true
  tmux kill-session -t paper-qwen-proxy 2>/dev/null || true
  tmux kill-session -t paper-qwen-inference_a-tunnels 2>/dev/null || true
  if ssh -S "$SSH_SOCKET" -O check -p "$SSH_PORT" "$SSH_TARGET" >/dev/null 2>&1; then
    remote "docker exec ${INFERENCE_A_CONTAINER} sh -lc 'tmux kill-session -t paper-qwen-inference_a0 2>/dev/null || true; tmux kill-session -t paper-qwen-inference_a1 2>/dev/null || true'" || true
  fi
}

if [[ "$ACTION" == "status" ]]; then
  status
  exit 0
fi
if [[ "$ACTION" == "stop" ]]; then
  stop
  exit 0
fi
if [[ "$ACTION" != "start" ]]; then
  echo "usage: $0 [start|status|stop]" >&2
  exit 2
fi

if ! ssh -S "$SSH_SOCKET" -O check -p "$SSH_PORT" "$SSH_TARGET" >/dev/null 2>&1; then
  echo "Missing authenticated inference_a SSH master socket: $SSH_SOCKET" >&2
  echo "Create it with: ssh -M -S $SSH_SOCKET -fnNT -p $SSH_PORT $SSH_TARGET" >&2
  exit 1
fi

trainer_used=$(nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits)
if ((trainer_used > 1024)); then
  echo "trainer GPU 0 is not free (${trainer_used} MiB used)" >&2
  exit 1
fi
remote_used=$(remote "docker exec ${INFERENCE_A_CONTAINER} nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits")
if awk -F, '$1 ~ /^[[:space:]]*[01][[:space:]]*$/ && $2 + 0 > 1024 {exit 1}' <<<"$remote_used"; then
  :
else
  echo "inference_a GPU 0 or 1 is not free" >&2
  exit 1
fi

mkdir -p "$PROJECT_ROOT/artifacts/logs"
tmux new-session -d -s paper-qwen-trainer0 \
  "cd $PROJECT_ROOT && CUDA_VISIBLE_DEVICES=0 VLLM_USE_DEEP_GEMM=0 $VLLM_BIN serve $MODEL_PATH --served-model-name Qwen/Qwen3-32B --revision 9216db5781bf21249d130ec9da846c4624c16137 --tokenizer-revision 9216db5781bf21249d130ec9da846c4624c16137 --gpu-memory-utilization 0.90 --max-model-len 2048 --max-num-seqs 48 --max-num-batched-tokens 12288 --dtype bfloat16 --enable-prefix-caching --generation-config vllm --host 127.0.0.1 --port $TRAINER_PORT > artifacts/logs/paper-qwen-trainer0.log 2>&1"

remote "docker exec ${INFERENCE_A_CONTAINER} sh -lc \"tmux new-session -d -s paper-qwen-inference_a0 'cd $INFERENCE_A_ROOT && CUDA_VISIBLE_DEVICES=0 VLLM_USE_DEEP_GEMM=0 $INFERENCE_A_VLLM_BIN serve $MODEL_PATH --served-model-name Qwen/Qwen3-32B --revision 9216db5781bf21249d130ec9da846c4624c16137 --tokenizer-revision 9216db5781bf21249d130ec9da846c4624c16137 --gpu-memory-utilization 0.90 --max-model-len 2048 --max-num-seqs 48 --max-num-batched-tokens 12288 --dtype bfloat16 --enable-prefix-caching --generation-config vllm --host 127.0.0.1 --port $INFERENCE_A0_PORT > /tmp/paper-qwen-inference_a0.log 2>&1'\""
remote "docker exec ${INFERENCE_A_CONTAINER} sh -lc \"tmux new-session -d -s paper-qwen-inference_a1 'cd $INFERENCE_A_ROOT && CUDA_VISIBLE_DEVICES=1 VLLM_USE_DEEP_GEMM=0 $INFERENCE_A_VLLM_BIN serve $MODEL_PATH --served-model-name Qwen/Qwen3-32B --revision 9216db5781bf21249d130ec9da846c4624c16137 --tokenizer-revision 9216db5781bf21249d130ec9da846c4624c16137 --gpu-memory-utilization 0.90 --max-model-len 2048 --max-num-seqs 48 --max-num-batched-tokens 12288 --dtype bfloat16 --enable-prefix-caching --generation-config vllm --host 127.0.0.1 --port $INFERENCE_A1_PORT > /tmp/paper-qwen-inference_a1.log 2>&1'\""

tmux new-session -d -s paper-qwen-inference_a-tunnels \
  "ssh -S $SSH_SOCKET -N -p $SSH_PORT -L ${INFERENCE_A0_TUNNEL}:127.0.0.1:${INFERENCE_A0_PORT} -L ${INFERENCE_A1_TUNNEL}:127.0.0.1:${INFERENCE_A1_PORT} $SSH_TARGET"

wait_url "http://127.0.0.1:${TRAINER_PORT}/v1/models" "trainer GPU 0 Qwen judge"
wait_url "http://127.0.0.1:${INFERENCE_A0_TUNNEL}/v1/models" "inference_a GPU 0 Qwen judge"
wait_url "http://127.0.0.1:${INFERENCE_A1_TUNNEL}/v1/models" "inference_a GPU 1 Qwen judge"

tmux new-session -d -s paper-qwen-proxy \
  "cd $PROJECT_ROOT && PYTHONPATH=$PROJECT_ROOT/src $PYTHON_BIN -m dynamic_rubric.services.vllm_score_proxy --upstream http://127.0.0.1:$TRAINER_PORT --upstream-weight 5 --upstream http://127.0.0.1:$INFERENCE_A0_TUNNEL --upstream-weight 2 --upstream http://127.0.0.1:$INFERENCE_A1_TUNNEL --upstream-weight 2 --model-path $MODEL_PATH --served-model Qwen/Qwen3-32B --model-revision 9216db5781bf21249d130ec9da846c4624c16137 --tokenizer-revision 9216db5781bf21249d130ec9da846c4624c16137 --cache-dir $PROJECT_ROOT/artifacts/provider_cache/vllm-score-proxy-paper-v1 --host 127.0.0.1 --port $PROXY_PORT > artifacts/logs/paper-qwen-proxy.log 2>&1"
wait_url "http://127.0.0.1:${PROXY_PORT}/health" "paper judge score proxy"
status
