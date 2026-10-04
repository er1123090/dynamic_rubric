#!/usr/bin/env bash
set -euo pipefail

project_root=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
runtime_python=${RUNTIME_PYTHON:-${PYTHON_BIN}}
run_id=${SCIENCE_RUN_ID:-phase1-online-rubrics-science-full-dense-20260924-seed11}
cache_root=${project_root}/outputs/science/shared/seed-11/pi0_control_cache

export PYTHONPATH=${project_root}/src${PYTHONPATH:+:${PYTHONPATH}}
export PHASE1_GPT_OSS_BASE_URLS=${PHASE1_GPT_OSS_BASE_URLS:-http://127.0.0.1:28011}
export PHASE1_QWEN32B_BASE_URLS=${PHASE1_QWEN32B_BASE_URLS:-http://127.0.0.1:28015}
export PHASE1_QWEN32B_EXPECTED_COUNT=${PHASE1_QWEN32B_EXPECTED_COUNT:-1}
export ONLINE_EXTRACTOR_CONCURRENCY=${ONLINE_EXTRACTOR_CONCURRENCY:-48}
export ONLINE_GRADER_CONCURRENCY=${ONLINE_GRADER_CONCURRENCY:-64}
export ONLINE_LOGPROB_PREFETCH=true
export PYTHONUNBUFFERED=1

if [[ -z "${ONLINE_CONTROL_CACHE:-}" ]]; then
  [[ -d "${cache_root}" ]] || {
    echo "science pi0 cache is absent; run precompute_science_pi0.sh first" >&2
    exit 2
  }
  mapfile -t manifests < <(find "${cache_root}" -maxdepth 1 -type f -name 'manifest-*.json' | sort)
  [[ "${#manifests[@]}" = 1 ]] || {
    echo "expected exactly one science pi0 manifest; run precompute_science_pi0.sh first" >&2
    exit 2
  }
  export ONLINE_CONTROL_CACHE=${manifests[0]}
fi
[[ -f "${ONLINE_CONTROL_CACHE}" ]] || {
  echo "science pi0 manifest is missing: ${ONLINE_CONTROL_CACHE}" >&2; exit 2;
}

cd "${project_root}"
exec "${runtime_python}" -m dynamic_rubric.phase1 train-online \
  --config configs/phase1/science_online_rubrics.yaml \
  --repo-root . \
  --run-id "${run_id}"
