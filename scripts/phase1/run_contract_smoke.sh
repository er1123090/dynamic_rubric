#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DOMAIN="${1:-medicine}"
METHOD="${2:-online_rubrics}"
RUN_ID="${3:-contract-smoke-v1}"

case "${DOMAIN}:${METHOD}" in
  medicine:online_rubrics|medicine:evorubrics|science:online_rubrics|science:evorubrics) ;;
  *)
    echo "usage: $0 {medicine|science} {online_rubrics|evorubrics} [run_id]" >&2
    exit 2
    ;;
esac

cd "${REPO_ROOT}"
PYTHONPATH=src .venv/bin/python -m dynamic_rubric.phase1 smoke   --config "configs/phase1/${DOMAIN}_${METHOD}.yaml"   --repo-root .   --run-id "${RUN_ID}"
