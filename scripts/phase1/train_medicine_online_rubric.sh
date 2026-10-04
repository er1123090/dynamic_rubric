#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
exec bash "${root}/scripts/phase1/train_online_rubric.sh" "$@"
