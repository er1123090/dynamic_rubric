#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

cd "${REPO_ROOT}"
for domain in medicine science; do
  PYTHONPATH=src .venv/bin/python -m dynamic_rubric.phase1 prepare-probe     --config "configs/phase1/${domain}_online_rubrics.yaml"     --repo-root .
  PYTHONPATH=src .venv/bin/python -m dynamic_rubric.phase1 prepare-probe     --config "configs/phase1/${domain}_evorubrics.yaml"     --repo-root .
done
