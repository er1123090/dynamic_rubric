#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DOMAIN="${1:-medicine}"
METHOD="${2:-online_rubrics}"

: "${PHASE1_GPT_OSS_BASE_URL:?set the inference_a gpt-oss-120b vLLM base URL}"
: "${PHASE1_QWEN32B_BASE_URL:?set the inference_b Qwen3-32B vLLM base URL}"

cd "${REPO_ROOT}"
PYTHONPATH=src .venv/bin/python -m dynamic_rubric.phase1 preflight \
  --config "configs/phase1/${DOMAIN}_${METHOD}.yaml" \
  --repo-root . \
  --require-endpoints
