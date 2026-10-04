#!/usr/bin/env bash
# Compatibility entry point; the actual runner is domain-neutral.
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export EVORUBRICS_DOMAIN=${EVORUBRICS_DOMAIN:-medicine}
exec bash "${root}/scripts/phase1/run_evorubrics.sh" "$@"
