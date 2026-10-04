#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export EVORUBRICS_DOMAIN=science
exec bash "${root}/scripts/phase1/run_evorubrics.sh" "$@"
