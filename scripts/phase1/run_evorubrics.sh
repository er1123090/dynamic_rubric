#!/usr/bin/env bash
set -euo pipefail

action=${1:-status}
project_root=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
runtime_python=${EVORUBRICS_PYTHON:-${project_root}/.venvs/evorubrics/bin/python}
domain=${EVORUBRICS_DOMAIN:-medicine}
[[ ${domain} == medicine || ${domain} == science ]] || { echo "invalid EVORUBRICS_DOMAIN: ${domain}" >&2; exit 2; }
config=${EVORUBRICS_CONFIG:-configs/phase1/${domain}_evorubrics.yaml}
judge_url=${EVORUBRICS_JUDGE_URL:-http://127.0.0.1:28011/v1}
trainer_gpu=${EVORUBRICS_TRAINER_GPU:-1}
smoke_run_id=${EVORUBRICS_SMOKE_RUN_ID:-phase1-evo-${domain}-live-smoke-dense-20260923-v5}
full_run_id=${EVORUBRICS_FULL_RUN_ID:-phase1-evo-${domain}-full-dense-20260923-seed11}
run_base=${EVORUBRICS_RUN_BASE:-${project_root}/outputs/${domain}/evorubrics/seed-11}
smoke_root=${run_base}/${smoke_run_id}
full_root=${run_base}/${full_run_id}

export PYTHONPATH=${project_root}/src${PYTHONPATH:+:${PYTHONPATH}}
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false

require_file() {
  local path=$1
  [[ -f "${path}" ]] || {
    echo "missing required file: ${path}" >&2
    exit 2
  }
}

require_runtime() {
  [[ -x "${runtime_python}" ]] || {
    echo "missing EvoRubrics runtime: ${runtime_python}" >&2
    exit 2
  }
  require_file "${project_root}/${config}"
  require_file "${project_root}/environment/evorubrics-runtime-lock.txt"
  require_file "${project_root}/docs/EvoRubrics-2155.zip"
}

require_gpu_idle() {
  local pids
  pids=$(nvidia-smi -i "${trainer_gpu}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)
  [[ -z "${pids}" ]] || {
    echo "GPU ${trainer_gpu} is busy (PIDs: ${pids//$'\n'/,}); refusing to overlap training" >&2
    exit 3
  }
}

prepare_run() {
  local run_id=$1
  local mode=${2:-full}
  local run_root=${run_base}/${run_id}
  if [[ -f "${run_root}/launch_spec.json" ]]; then
    echo "already prepared: ${run_root}"
    return
  fi
  local args=(
    -m dynamic_rubric.phase1.evorubrics_run prepare
    --config "${config}"
    --repo-root "${project_root}"
    --run-id "${run_id}"
  )
  if [[ "${mode}" == smoke ]]; then
    args+=(--smoke)
  fi
  CUDA_VISIBLE_DEVICES= "${runtime_python}" "${args[@]}"
}

prepare_all() {
  prepare_run "${smoke_run_id}" smoke
  prepare_run "${full_run_id}" full
}

run_smoke_train() {
  require_gpu_idle
  require_file "${smoke_root}/launch_spec.json"
  if [[ ! -f "${smoke_root}/judge-preflight.json" ]]; then
    "${runtime_python}" scripts/phase1/preflight_evorubrics_judge.py \
      --run-root "${smoke_root}" --judge-base-url "${judge_url}"
  fi
  if [[ -f "${smoke_root}/training_complete.json" ]]; then
    echo "smoke training already complete: ${smoke_root}"
    return
  fi
  "${runtime_python}" -m dynamic_rubric.phase1.evorubrics_run launch \
    --repo-root "${project_root}" --run-root "${smoke_root}" \
    --runtime-python "${runtime_python}" --judge-base-url "${judge_url}" \
    --gpu "${trainer_gpu}"
}

run_smoke_probe() {
  require_file "${smoke_root}/training_complete.json"
  if [[ ! -f "${smoke_root}/audit/fixed_probe/analysis.json" ]]; then
    "${runtime_python}" -m dynamic_rubric.phase1.evorubrics_probe \
      --run-root "${smoke_root}" --pairs all --max-prompts 1 --generate
    "${runtime_python}" -m dynamic_rubric.phase1.evorubrics_probe \
      --run-root "${smoke_root}" --pairs all --max-prompts 1 \
      --score --judge-base-url "${judge_url}"
  fi
  if [[ ! -f "${smoke_root}/audit/fixed_probe/policy_kl/theta_000000_to_000001.json" ]]; then
    "${runtime_python}" -m dynamic_rubric.phase1.evorubrics_probe \
      --run-root "${smoke_root}" --pairs all --max-prompts 1 --kl
  fi
}

run_smoke_resume() {
  require_gpu_idle
  require_file "${smoke_root}/training_complete.json"
  require_file "${smoke_root}/audit/fixed_probe/analysis.json"
  require_file "${smoke_root}/audit/fixed_probe/policy_kl/theta_000000_to_000001.json"
  if [[ -f "${smoke_root}/resume_verified.json" ]]; then
    echo "smoke resume already verified: ${smoke_root}"
    return
  fi
  "${runtime_python}" -m dynamic_rubric.phase1.evorubrics_run launch \
    --repo-root "${project_root}" --run-root "${smoke_root}" \
    --runtime-python "${runtime_python}" --judge-base-url "${judge_url}" \
    --gpu "${trainer_gpu}" --resume-step 1
}

run_smoke_validate() {
  require_file "${smoke_root}/resume_verified.json"
  if [[ -f "${smoke_root}/smoke_complete.json" ]]; then
    echo "smoke already validated: ${smoke_root}"
    return
  fi
  "${runtime_python}" -m dynamic_rubric.phase1.evorubrics_probe \
    --run-root "${smoke_root}" --max-prompts 1 --validate-smoke
}

run_smoke() {
  run_smoke_train
  run_smoke_probe
  run_smoke_resume
  run_smoke_validate
}

run_full() {
  require_gpu_idle
  require_file "${full_root}/launch_spec.json"
  require_file "${smoke_root}/smoke_complete.json"
  "${runtime_python}" -m dynamic_rubric.phase1.evorubrics_run launch \
    --repo-root "${project_root}" --run-root "${full_root}" \
    --runtime-python "${runtime_python}" --judge-base-url "${judge_url}" \
    --gpu "${trainer_gpu}" --smoke-proof "${smoke_root}/smoke_complete.json"
}

show_status() {
  echo "smoke run: ${smoke_root}"
  [[ -f "${smoke_root}/launch_spec.json" ]] && echo "  prepared: yes" || echo "  prepared: no"
  [[ -f "${smoke_root}/smoke_complete.json" ]] && echo "  validated: yes" || echo "  validated: no"
  echo "full run: ${full_root}"
  [[ -f "${full_root}/launch_spec.json" ]] && echo "  prepared: yes" || echo "  prepared: no"
  [[ -f "${full_root}/training_started.json" ]] && echo "  started: yes" || echo "  started: no"
  nvidia-smi -i "${trainer_gpu}" --query-gpu=index,memory.used,utilization.gpu \
    --format=csv,noheader,nounits
}

cd "${project_root}"
require_runtime
case "${action}" in
  prepare)
    prepare_all
    ;;
  smoke)
    prepare_all
    run_smoke
    ;;
  smoke-train)
    prepare_all
    run_smoke_train
    ;;
  smoke-probe)
    prepare_all
    run_smoke_probe
    ;;
  smoke-resume)
    prepare_all
    run_smoke_resume
    ;;
  smoke-validate)
    prepare_all
    run_smoke_validate
    ;;
  full)
    prepare_all
    run_full
    ;;
  status)
    show_status
    ;;
  *)
    echo "usage: $0 {prepare|smoke|smoke-train|smoke-probe|smoke-resume|smoke-validate|full|status}" >&2
    exit 2
    ;;
esac
