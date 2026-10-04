#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
rc=0
"${SCRIPT_DIR}/serve_gpt_oss_on_inference_a.sh" status || rc=1
"${SCRIPT_DIR}/serve_qwen32b_on_inference_b.sh" status || rc=1
"${SCRIPT_DIR}/manage_vllm_tunnels.sh" status || rc=1

for item in \
  "gpt-oss:${PHASE1_GPT_OSS_BASE_URL:-http://127.0.0.1:18001}" \
  "qwen32b:${PHASE1_QWEN32B_BASE_URL:-http://127.0.0.1:18002}"; do
  name="${item%%:*}"
  url="${item#*:}"
  if curl -fsS --max-time 3 "${url%/}/v1/models" >/dev/null; then
    echo "REACHABLE ${name} ${url%/}/v1"
  else
    echo "UNREACHABLE ${name} ${url%/}/v1"
    rc=1
  fi
done
exit "${rc}"
