#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SSH_CONFIG="${PHASE1_SSH_CONFIG:-${REPO_ROOT}/configs/phase1/ssh_config}"
ACTION="${1:-start}"

if [[ ! -r "${SSH_CONFIG}" ]]; then
  echo "SSH config not readable: ${SSH_CONFIG}" >&2
  exit 2
fi
if [[ "${ACTION}" != "start" && "${ACTION}" != "status" && "${ACTION}" != "stop" ]]; then
  echo "Usage: $0 [start|status|stop]" >&2
  exit 2
fi

manage_one() {
  local name="$1" target="$2" local_port="$3" remote_port="$4"
  local socket="/tmp/phase1-${name}-tunnel-${UID}.sock"
  local ssh_args=(-F "${SSH_CONFIG}" -S "${socket}")

  if ssh "${ssh_args[@]}" -O check "${target}" >/dev/null 2>&1; then
    if [[ "${ACTION}" == "stop" ]]; then
      ssh "${ssh_args[@]}" -O exit "${target}" >/dev/null
      echo "STOPPED ${name} tunnel"
    else
      echo "READY ${name} tunnel http://127.0.0.1:${local_port}/v1"
    fi
    return
  fi

  if [[ "${ACTION}" == "status" ]]; then
    echo "STOPPED ${name} tunnel"
    return 1
  fi
  if [[ "${ACTION}" == "stop" ]]; then
    echo "STOPPED ${name} tunnel"
    return
  fi
  if command -v ss >/dev/null && ss -ltn | awk '{print $4}' | grep -Eq "(^|:)${local_port}$"; then
    echo "Local port ${local_port} is already occupied; refusing to replace it." >&2
    return 4
  fi
  ssh -F "${SSH_CONFIG}" \
    -M -S "${socket}" -o ExitOnForwardFailure=yes -fNT \
    -L "127.0.0.1:${local_port}:127.0.0.1:${remote_port}" "${target}"
  echo "LAUNCHED ${name} tunnel http://127.0.0.1:${local_port}/v1"
}

rc=0
manage_one gpt-oss "${PHASE1_INFERENCE_A_SSH_TARGET:-inference_a}" \
  "${PHASE1_GPT_OSS_TUNNEL_PORT:-18001}" "${PHASE1_GPT_OSS_PORT:-8001}" || rc=$?
manage_one qwen32b "${PHASE1_INFERENCE_B_SSH_TARGET:-inference_b}" \
  "${PHASE1_QWEN32B_TUNNEL_PORT:-18002}" "${PHASE1_QWEN32B_PORT:-8002}" || rc=$?
exit "${rc}"
