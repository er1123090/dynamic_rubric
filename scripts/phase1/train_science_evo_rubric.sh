#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
exec "${LAUNCH_PYTHON:-${root}/.venv/bin/python}" "${root}/scripts/phase1/launch_training.py" \
  --config "${root}/configs/launch/science_evorubric.yaml" "$@"
