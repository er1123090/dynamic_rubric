#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
TRAINING_PGID=${TRAINING_PGID:-2916874}
ACTIVE_PROXY_PORT=${ACTIVE_PROXY_PORT:-8102}
CANDIDATE_PROXY_PORT=${CANDIDATE_PROXY_PORT:-8103}
LOCAL_JUDGE_PORT=${LOCAL_JUDGE_PORT:-8014}
INFERENCE_A_TUNNEL_PORT=${INFERENCE_A_TUNNEL_PORT:-18002}
MODEL_PATH=${MODEL_PATH:-${MODEL_PATH}}
MODEL_NAME=${MODEL_NAME:-Qwen/Qwen3-32B}
MODEL_REVISION=${MODEL_REVISION:-9216db5781bf21249d130ec9da846c4624c16137}
PYTHON_BIN=${PYTHON_BIN:-${PYTHON_BIN}}
CACHE_DIR=${CACHE_DIR:-${PROJECT_ROOT}/artifacts/provider_cache/horizon-vllm-score-proxy}
ACTIVE_PROXY_SESSION=${ACTIVE_PROXY_SESSION:-horizon-qwen-judge-proxy}
CANDIDATE_PROXY_SESSION=${CANDIDATE_PROXY_SESSION:-horizon-qwen-judge-proxy-inference_a-candidate}
LOCAL_JUDGE_SESSION=${LOCAL_JUDGE_SESSION:-horizon-qwen-judge-trainer0}
LOG_PATH=${LOG_PATH:-${PROJECT_ROOT}/artifacts/logs/horizon-qwen-judge-proxy-inference_a.log}

paused=false
switched=false

wait_health() {
  local port=$1
  for _ in $(seq 1 120); do
    if curl -fsS --max-time 3 "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.25
  done
  return 1
}

start_proxy() {
  local upstream=$1
  local log_path=$2
  tmux new-session -d -s "${ACTIVE_PROXY_SESSION}" \
    "cd '${PROJECT_ROOT}' && PYTHONPATH='${PROJECT_ROOT}/src' '${PYTHON_BIN}' -m dynamic_rubric.services.vllm_score_proxy --upstream '${upstream}' --upstream-weight 1 --model-path '${MODEL_PATH}' --served-model '${MODEL_NAME}' --model-revision '${MODEL_REVISION}' --tokenizer-revision '${MODEL_REVISION}' --cache-dir '${CACHE_DIR}' --host 127.0.0.1 --port '${ACTIVE_PROXY_PORT}' >> '${log_path}' 2>&1"
}

resume_training() {
  if [[ "${paused}" == true ]]; then
    kill -CONT -- "-${TRAINING_PGID}" 2>/dev/null || true
    paused=false
  fi
}

recover_local_proxy() {
  tmux kill-session -t "${ACTIVE_PROXY_SESSION}" 2>/dev/null || true
  for _ in $(seq 1 40); do
    if ! lsof -nP -iTCP:"${ACTIVE_PROXY_PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
      break
    fi
    sleep 0.1
  done
  if tmux has-session -t "${LOCAL_JUDGE_SESSION}" 2>/dev/null; then
    start_proxy "http://127.0.0.1:${LOCAL_JUDGE_PORT}" \
      "${PROJECT_ROOT}/artifacts/logs/horizon-qwen-judge-proxy-local-recovery.log" || true
    wait_health "${ACTIVE_PROXY_PORT}" || true
  fi
}

on_exit() {
  status=$?
  if [[ "${status}" -ne 0 && "${switched}" != true ]]; then
    recover_local_proxy
  fi
  resume_training
  exit "${status}"
}
trap on_exit EXIT

curl -fsS --max-time 3 "http://127.0.0.1:${CANDIDATE_PROXY_PORT}/health" >/dev/null
curl -fsS --max-time 3 "http://127.0.0.1:${INFERENCE_A_TUNNEL_PORT}/v1/models" >/dev/null
kill -0 "${TRAINING_PGID}" 2>/dev/null

for _ in $(seq 1 1200); do
  if lsof -nP -iTCP:"${ACTIVE_PROXY_PORT}" -sTCP:ESTABLISHED >/dev/null 2>&1; then
    sleep 0.1
    continue
  fi
  kill -STOP -- "-${TRAINING_PGID}"
  paused=true
  sleep 0.25
  if ! lsof -nP -iTCP:"${ACTIVE_PROXY_PORT}" -sTCP:ESTABLISHED >/dev/null 2>&1; then
    break
  fi
  resume_training
  sleep 0.1
done

if [[ "${paused}" != true ]]; then
  echo "could not acquire an idle score-proxy handoff window" >&2
  exit 1
fi

tmux kill-session -t "${ACTIVE_PROXY_SESSION}"
for _ in $(seq 1 80); do
  if ! lsof -nP -iTCP:"${ACTIVE_PROXY_PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
    break
  fi
  sleep 0.1
done
if lsof -nP -iTCP:"${ACTIVE_PROXY_PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "old score proxy did not release port ${ACTIVE_PROXY_PORT}" >&2
  exit 1
fi

start_proxy "http://127.0.0.1:${INFERENCE_A_TUNNEL_PORT}" "${LOG_PATH}"
wait_health "${ACTIVE_PROXY_PORT}"
curl -fsS --max-time 3 "http://127.0.0.1:${ACTIVE_PROXY_PORT}/dynamic-rubric/routing"

switched=true
tmux kill-session -t "${CANDIDATE_PROXY_SESSION}" 2>/dev/null || true
tmux kill-session -t "${LOCAL_JUDGE_SESSION}" 2>/dev/null || true
resume_training
trap - EXIT

echo "active judge switched to Inference A GPUs 0,1; Trainer local judge released"
