#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
project_root=$(cd "${script_dir}/../.." && pwd)
exec "${LAUNCH_PYTHON:-${project_root}/.venv/bin/python}" \
  "${script_dir}/precompute_pi0.py" \
  --config "${project_root}/configs/launch/science_online_rubric.yaml" \
  "$@"
