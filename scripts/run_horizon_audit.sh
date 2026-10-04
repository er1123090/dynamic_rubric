#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
DOMAIN=${DOMAIN:?DOMAIN must be medicine or science}
TRAINING_SEED=${TRAINING_SEED:-11}
POLICY_GPU=${POLICY_GPU:-0}
CONFIG_PATH=${CONFIG_PATH:-${PROJECT_ROOT}/configs/horizon_${DOMAIN}.yaml}
PROMPTS=${PROMPTS:-${PROJECT_ROOT}/data/rar/${DOMAIN}/public/final.jsonl}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts/horizon/${DOMAIN}}
RESULT_ROOT=${RESULT_ROOT:-${PROJECT_ROOT}/results/horizon/${DOMAIN}}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-${ARTIFACT_ROOT}/training/seed-${TRAINING_SEED}/checkpoints}
EXPORT_ROOT=${EXPORT_ROOT:-${ARTIFACT_ROOT}/inference/seed-${TRAINING_SEED}}
POOL_ROOT=${POOL_ROOT:-${ARTIFACT_ROOT}/pools}
RUBRIC_ROOT=${RUBRIC_ROOT:-${ARTIFACT_ROOT}/rubrics/seed-${TRAINING_SEED}}
SCORE_ROOT=${SCORE_ROOT:-${ARTIFACT_ROOT}/scores/seed-${TRAINING_SEED}}
RUBRIC_API_MODE=${RUBRIC_API_MODE:-sync}
RUBRIC_SYNC_CONCURRENCY=${RUBRIC_SYNC_CONCURRENCY:-16}
RUBRIC_STATE_ROOT=${RUBRIC_STATE_ROOT:-${ARTIFACT_ROOT}/${RUBRIC_API_MODE}/seed-${TRAINING_SEED}}
AUDIT_LOG_ROOT=${AUDIT_LOG_ROOT:-${ARTIFACT_ROOT}/audit/logs}
AUDIT_COMPLETE_MARKER=${AUDIT_COMPLETE_MARKER:-${ARTIFACT_ROOT}/audit/seed-${TRAINING_SEED}.complete}
RUNTIME_PYTHON=${RUNTIME_PYTHON:-${PROJECT_ROOT}/environment/upstream/verl/.venv-runtime/bin/python}
VERL_ROOT=${VERL_ROOT:-${PROJECT_ROOT}/environment/upstream/verl}
JUDGE_BASE_URL=${JUDGE_BASE_URL:-http://127.0.0.1:8102}
POLICY_UPSTREAM_PORT=${POLICY_UPSTREAM_PORT:-8200}
POLICY_PROXY_PORT=${POLICY_PROXY_PORT:-8201}
POLICY_CONCURRENCY=${POLICY_CONCURRENCY:-256}
POLICY_MODEL=${POLICY_MODEL:-Qwen/Qwen3-1.7B}
POLICY_REVISION=${POLICY_REVISION:-70d244cc86ccca08cf5af4e1e306ecf908b1ad5e}
POLICY_BASE_PATH=${POLICY_BASE_PATH:-/models/Qwen3-1.7B}
BATCH_POLL_INTERVAL_SECONDS=${BATCH_POLL_INTERVAL_SECONDS:-60}
RUN_POOL_A_AUXILIARY=${RUN_POOL_A_AUXILIARY:-1}

case "${DOMAIN}" in
  medicine|science) ;;
  *) echo "DOMAIN must be medicine or science" >&2; exit 2 ;;
esac


checkpoint_steps=(0 3 6 9 13 16 24 32 40 48)
checkpoint_epochs=(0.0 0.2 0.4 0.6 0.8 1.0 1.5 2.0 2.5 3.0)
expected_prompt_count=100
policy_upstream_pid=
policy_proxy_pid=
grader_pid=
sham_pid=

mkdir -p \
  "${EXPORT_ROOT}" "${POOL_ROOT}" "${RUBRIC_ROOT}" "${SCORE_ROOT}" \
  "${RUBRIC_STATE_ROOT}" "${AUDIT_LOG_ROOT}" "${RESULT_ROOT}"

log() {
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$*"
}

run_cli() {
  PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
    uv run --project "${PROJECT_ROOT}" python -m dynamic_rubric "$@"
}

wait_url() {
  local url=$1
  local label=$2
  local attempt
  for attempt in $(seq 1 240); do
    if curl -fsS --max-time 3 "${url}" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  echo "${label} did not become ready: ${url}" >&2
  return 1
}

stop_policy_server() {
  if [[ -n "${policy_proxy_pid}" ]]; then
    kill "${policy_proxy_pid}" 2>/dev/null || true
    wait "${policy_proxy_pid}" 2>/dev/null || true
    policy_proxy_pid=
  fi
  if [[ -n "${policy_upstream_pid}" ]]; then
    kill "${policy_upstream_pid}" 2>/dev/null || true
    wait "${policy_upstream_pid}" 2>/dev/null || true
    policy_upstream_pid=
  fi
}

cleanup_runtime() {
  stop_policy_server
  if [[ -n "${grader_pid}" ]]; then
    kill "${grader_pid}" 2>/dev/null || true
    wait "${grader_pid}" 2>/dev/null || true
  fi
  if [[ -n "${sham_pid}" ]]; then
    kill "${sham_pid}" 2>/dev/null || true
    wait "${sham_pid}" 2>/dev/null || true
  fi
}

trap cleanup_runtime EXIT

require_lines() {
  local path=$1
  local expected=$2
  local label=$3
  [[ -f "${path}" ]] || { echo "missing ${label}: ${path}" >&2; return 1; }
  local actual
  actual=$(wc -l < "${path}")
  [[ "${actual}" -eq "${expected}" ]] || {
    echo "wrong ${label} row count: expected=${expected}, actual=${actual}, path=${path}" >&2
    return 1
  }
}

validate_static_inputs() {
  require_lines "${PROMPTS}" "${expected_prompt_count}" "final prompts"
  require_lines "${POOL_ROOT}/fixed.jsonl" "$((expected_prompt_count * 8))" "fixed control"
  require_lines "${POOL_ROOT}/sham.jsonl" "$((expected_prompt_count * 8))" "sham control"
  require_lines \
    "${POOL_ROOT}/seed-${TRAINING_SEED}-step-0-pool-b.jsonl" \
    "$((expected_prompt_count * 16))" \
    "checkpoint-zero Pool B"
  curl -fsS --max-time 30 "${JUDGE_BASE_URL}/health" >/dev/null
  run_cli validate-config --config "${CONFIG_PATH}" >/dev/null
}

checkpoint_hash() {
  sha256sum "$1" | awk '{print $1}'
}

export_checkpoint() {
  local step=$1
  local actor_dir="${CHECKPOINT_ROOT}/global_step_${step}/actor"
  local source_model="${actor_dir}/model_world_size_1_rank_0.pt"
  local export_dir="${EXPORT_ROOT}/global_step_${step}"
  local temporary_dir="${EXPORT_ROOT}/.global_step_${step}.tmp.$$"
  local merge_log="${AUDIT_LOG_ROOT}/checkpoint-merge-step-${step}.log"

  [[ -f "${source_model}" ]] || {
    echo "checkpoint model is missing: ${source_model}" >&2
    return 1
  }
  [[ -d "${actor_dir}/huggingface" ]] || {
    echo "checkpoint Hugging Face metadata is missing: ${actor_dir}/huggingface" >&2
    return 1
  }
  if [[ -f "${export_dir}/config.json" ]] && compgen -G "${export_dir}/*.safetensors" >/dev/null; then
    printf '%s\n' "${export_dir}"
    return 0
  fi
  if [[ -e "${export_dir}" ]]; then
    echo "incomplete inference export requires inspection: ${export_dir}" >&2
    return 1
  fi

  mkdir -p "${temporary_dir}"
  if ! (
    cd "${VERL_ROOT}"
    "${RUNTIME_PYTHON}" -m verl.model_merger merge \
      --backend fsdp \
      --use_cpu_initialization \
      --local_dir "${actor_dir}" \
      --target_dir "${temporary_dir}" \
      >"${merge_log}" 2>&1
  ); then
    echo "checkpoint export failed; temporary directory retained: ${temporary_dir}" >&2
    return 1
  fi
  [[ -f "${temporary_dir}/config.json" ]] || {
    echo "checkpoint export omitted config.json: ${temporary_dir}" >&2
    return 1
  }
  compgen -G "${temporary_dir}/*.safetensors" >/dev/null || {
    echo "checkpoint export omitted safetensors: ${temporary_dir}" >&2
    return 1
  }
  mv "${temporary_dir}" "${export_dir}"
  printf '%s\n' "${export_dir}"
}

start_policy_server() {
  local model_path=$1
  local hash=$2
  local step=$3
  local upstream_log="${AUDIT_LOG_ROOT}/policy-vllm-step-${step}.log"
  local proxy_log="${AUDIT_LOG_ROOT}/policy-proxy-step-${step}.log"

  stop_policy_server
  if curl -fsS --max-time 2 "http://127.0.0.1:${POLICY_UPSTREAM_PORT}/v1/models" >/dev/null 2>&1; then
    echo "policy upstream port is already occupied: ${POLICY_UPSTREAM_PORT}" >&2
    return 1
  fi
  if curl -fsS --max-time 2 "http://127.0.0.1:${POLICY_PROXY_PORT}/health" >/dev/null 2>&1; then
    echo "policy proxy port is already occupied: ${POLICY_PROXY_PORT}" >&2
    return 1
  fi

  env \
    CUDA_VISIBLE_DEVICES="${POLICY_GPU}" \
    VLLM_USE_DEEP_GEMM=0 \
    VLLM_MOE_USE_DEEP_GEMM=0 \
    VLLM_DEEP_GEMM_WARMUP=skip \
    "${RUNTIME_PYTHON}" -m vllm.entrypoints.openai.api_server \
      --model "${model_path}" \
      --served-model-name "${POLICY_MODEL}" \
      --host 127.0.0.1 \
      --port "${POLICY_UPSTREAM_PORT}" \
      --dtype bfloat16 \
      --gpu-memory-utilization 0.92 \
      --max-model-len 7680 \
      --max-num-seqs 256 \
      --max-num-batched-tokens 65536 \
      --enable-prefix-caching \
      --generation-config vllm \
      >"${upstream_log}" 2>&1 &
  policy_upstream_pid=$!
  wait_url "http://127.0.0.1:${POLICY_UPSTREAM_PORT}/v1/models" "policy vLLM step ${step}"

  PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
    "${RUNTIME_PYTHON}" -m dynamic_rubric.services.vllm_policy_proxy \
      --upstream "http://127.0.0.1:${POLICY_UPSTREAM_PORT}" \
      --model-path "${model_path}" \
      --served-model "${POLICY_MODEL}" \
      --model-revision "${POLICY_REVISION}" \
      --tokenizer-revision "${POLICY_REVISION}" \
      --checkpoint-hash "${hash}" \
      --timeout-seconds 900 \
      --host 127.0.0.1 \
      --port "${POLICY_PROXY_PORT}" \
      >"${proxy_log}" 2>&1 &
  policy_proxy_pid=$!
  wait_url "http://127.0.0.1:${POLICY_PROXY_PORT}/health" "policy identity proxy step ${step}"
}

generate_checkpoint_pools() {
  local step=$1
  local hash=$2
  local pool_a="${POOL_ROOT}/seed-${TRAINING_SEED}-step-${step}-pool-a.jsonl"
  local pool_b="${POOL_ROOT}/seed-${TRAINING_SEED}-step-${step}-pool-b.jsonl"
  local base_url="http://127.0.0.1:${POLICY_PROXY_PORT}"

  if [[ ! -f "${pool_a}" ]]; then
    DYNAMIC_RUBRIC_POLICY_CONCURRENCY="${POLICY_CONCURRENCY}" run_cli \
      generate-horizon-pools \
      --config "${CONFIG_PATH}" \
      --run-id "rar-horizon-v1-${DOMAIN}-seed${TRAINING_SEED}-step${step}-pool-a" \
      --prompts "${PROMPTS}" \
      --pool-family pool_a \
      --count 8 \
      --policy-step "${step}" \
      --training-seed "${TRAINING_SEED}" \
      --checkpoint-hash "${hash}" \
      --base-url "${base_url}" \
      --output "${pool_a}"
  fi
  require_lines "${pool_a}" "$((expected_prompt_count * 8))" "step ${step} Pool A"

  if [[ ! -f "${pool_b}" ]]; then
    DYNAMIC_RUBRIC_POLICY_CONCURRENCY="${POLICY_CONCURRENCY}" run_cli \
      generate-horizon-pools \
      --config "${CONFIG_PATH}" \
      --run-id "rar-horizon-v1-${DOMAIN}-seed${TRAINING_SEED}-step${step}-pool-b" \
      --prompts "${PROMPTS}" \
      --pool-family pool_b \
      --count 16 \
      --policy-step "${step}" \
      --training-seed "${TRAINING_SEED}" \
      --checkpoint-hash "${hash}" \
      --base-url "${base_url}" \
      --output "${pool_b}"
  fi
  require_lines "${pool_b}" "$((expected_prompt_count * 16))" "step ${step} Pool B"
}

build_rubric() {
  local step=$1
  local control_rubric=$2
  local output="${RUBRIC_ROOT}/step-${step}.jsonl"
  if [[ ! -f "${output}" ]]; then
    run_cli build-horizon-rubrics \
      --config "${CONFIG_PATH}" \
      --run-id "rar-horizon-v1-${DOMAIN}-seed${TRAINING_SEED}-step${step}" \
      --prompts "${PROMPTS}" \
      --current-pool "${POOL_ROOT}/seed-${TRAINING_SEED}-step-${step}-pool-a.jsonl" \
      --control-pool "${POOL_ROOT}/fixed.jsonl" \
      --control-rubrics "${control_rubric}" \
      --checkpoint-id "step${step}" \
      --batch-poll-interval-seconds "${BATCH_POLL_INTERVAL_SECONDS}" \
      --api-mode "${RUBRIC_API_MODE}" \
      --sync-concurrency "${RUBRIC_SYNC_CONCURRENCY}" \
      --rubric-state-root "${RUBRIC_STATE_ROOT}/step-${step}" \
      --output "${output}" >&2
  fi
  require_lines "${output}" "${expected_prompt_count}" "step ${step} rubric"
  printf '%s\n' "${output}"
}

grade_checkpoint() {
  local step=$1
  local epoch=$2
  local rubric=${3:-}
  local output_dir="${SCORE_ROOT}/epoch-${epoch}"
  if [[ -f "${output_dir}/score_seal.json" ]]; then
    return 0
  fi
  local args=(
    grade-horizon
    --config "${CONFIG_PATH}"
    --run-id "rar-horizon-v1-${DOMAIN}-seed${TRAINING_SEED}-step${step}-grade"
    --prompts "${PROMPTS}"
    --pool-b "${POOL_ROOT}/seed-${TRAINING_SEED}-step-${step}-pool-b.jsonl"
    --checkpoint "${epoch}"
    --base-url "${JUDGE_BASE_URL}"
    --output-dir "${output_dir}"
  )
  if [[ -n "${rubric}" ]]; then
    args+=(--rubrics "${rubric}")
  fi
  run_cli "${args[@]}"
  [[ -f "${output_dir}/score_seal.json" ]]
}

build_sham_rubric() {
  local output="${RUBRIC_ROOT}/sham.jsonl"
  if [[ ! -f "${output}" ]]; then
    run_cli build-horizon-rubrics \
      --config "${CONFIG_PATH}" \
      --run-id "rar-horizon-v1-${DOMAIN}-seed${TRAINING_SEED}-sham" \
      --prompts "${PROMPTS}" \
      --current-pool "${POOL_ROOT}/sham.jsonl" \
      --control-pool "${POOL_ROOT}/fixed.jsonl" \
      --checkpoint-id step0 \
      --batch-poll-interval-seconds "${BATCH_POLL_INTERVAL_SECONDS}" \
      --api-mode "${RUBRIC_API_MODE}" \
      --sync-concurrency "${RUBRIC_SYNC_CONCURRENCY}" \
      --rubric-state-root "${RUBRIC_STATE_ROOT}/sham" \
      --output "${output}" >&2
  fi
  require_lines "${output}" "${expected_prompt_count}" "sham rubric"
  printf '%s\n' "${output}"
}

validate_full_inventory() {
  local pools=("${POOL_ROOT}/fixed.jsonl" "${POOL_ROOT}/sham.jsonl")
  local step
  for step in "${checkpoint_steps[@]}"; do
    pools+=("${POOL_ROOT}/seed-${TRAINING_SEED}-step-${step}-pool-b.jsonl")
    if [[ "${step}" -ne 0 ]]; then
      pools+=("${POOL_ROOT}/seed-${TRAINING_SEED}-step-${step}-pool-a.jsonl")
    fi
  done
  run_cli validate-horizon-inventory \
    --config "${CONFIG_PATH}" \
    --prompts "${PROMPTS}" \
    --pools "${pools[@]}" >/dev/null
}

finalize_audit() {
  local summaries=()
  local epoch
  for epoch in "${checkpoint_epochs[@]}"; do
    summaries+=("${SCORE_ROOT}/epoch-${epoch}/prompt_summary.jsonl")
  done
  local observations="${ARTIFACT_ROOT}/observations-seed-${TRAINING_SEED}.jsonl"
  local report="${RESULT_ROOT}/horizon_report_seed-${TRAINING_SEED}.json"
  run_cli build-horizon-observations \
    --config "${CONFIG_PATH}" \
    --prompts "${PROMPTS}" \
    --summaries "${summaries[@]}" \
    --output "${observations}" >/dev/null
  run_cli analyze-horizon \
    --config "${CONFIG_PATH}" \
    --observations "${observations}" \
    --output "${report}" >/dev/null
  [[ -s "${report}" ]] || { echo "horizon report is missing: ${report}" >&2; return 1; }
  printf '%s  %s\n' "$(sha256sum "${report}" | awk '{print $1}')" "${report}" \
    > "${AUDIT_COMPLETE_MARKER}.tmp.$$"
  mv "${AUDIT_COMPLETE_MARKER}.tmp.$$" "${AUDIT_COMPLETE_MARKER}"
}

run_pool_a_auxiliary() {
  [[ "${RUN_POOL_A_AUXILIARY}" == 1 ]] || return 0
  DOMAIN="${DOMAIN}" \
    TRAINING_SEED="${TRAINING_SEED}" \
    CONFIG_PATH="${CONFIG_PATH}" \
    PROMPTS="${PROMPTS}" \
    ARTIFACT_ROOT="${ARTIFACT_ROOT}" \
    POOL_ROOT="${POOL_ROOT}" \
    RUBRIC_ROOT="${RUBRIC_ROOT}" \
    SCORE_ROOT="${SCORE_ROOT}" \
    JUDGE_BASE_URL="${JUDGE_BASE_URL}" \
    bash "${PROJECT_ROOT}/scripts/run_horizon_pool_a_auxiliary.sh"
}

if [[ -f "${AUDIT_COMPLETE_MARKER}" ]]; then
  log "canonical Pool B audit already complete: ${AUDIT_COMPLETE_MARKER}"
  run_pool_a_auxiliary
  exit 0
fi

: "${OPENAI_API_KEY:?OPENAI_API_KEY is required for GPT-5-mini rubric extraction}"

log "validating ${DOMAIN} seed ${TRAINING_SEED} 100-prompt audit inputs"
validate_static_inputs

log "starting checkpoint-zero grading on Inference A judge replicas"
grade_checkpoint 0 0.0 &
grader_pid=$!

log "starting checkpoint-zero sham criterion control with GPT-5-mini ${RUBRIC_API_MODE}"
build_sham_rubric >/dev/null &
sham_pid=$!

log "generating all nonzero checkpoint pools on Trainer GPU ${POLICY_GPU}"
for index in $(seq 1 9); do
  step=${checkpoint_steps[$index]}
  pool_a="${POOL_ROOT}/seed-${TRAINING_SEED}-step-${step}-pool-a.jsonl"
  pool_b="${POOL_ROOT}/seed-${TRAINING_SEED}-step-${step}-pool-b.jsonl"
  if [[ -f "${pool_a}" && -f "${pool_b}" ]]; then
    require_lines "${pool_a}" "$((expected_prompt_count * 8))" "step ${step} Pool A"
    require_lines "${pool_b}" "$((expected_prompt_count * 16))" "step ${step} Pool B"
    log "reusing complete response pools for checkpoint step ${step}"
    continue
  fi

  source_model="${CHECKPOINT_ROOT}/global_step_${step}/actor/model_world_size_1_rank_0.pt"
  hash=$(checkpoint_hash "${source_model}")
  log "exporting and serving ${DOMAIN} checkpoint step ${step} on Trainer GPU ${POLICY_GPU}"
  model_path=$(export_checkpoint "${step}")
  start_policy_server "${model_path}" "${hash}" "${step}"
  generate_checkpoint_pools "${step}" "${hash}"
  stop_policy_server
done

wait "${sham_pid}"
sham_pid=
prior_rubric="${RUBRIC_ROOT}/sham.jsonl"

log "pipelining GPT rubric refresh with checkpoint grading"
for index in $(seq 1 9); do
  step=${checkpoint_steps[$index]}
  epoch=${checkpoint_epochs[$index]}
  log "building current rubric for checkpoint step ${step} with GPT-5-mini ${RUBRIC_API_MODE}"
  current_rubric=$(build_rubric "${step}" "${prior_rubric}")

  wait "${grader_pid}"
  grader_pid=
  log "grading checkpoint step ${step} on Inference A judge replicas"
  grade_checkpoint "${step}" "${epoch}" "${current_rubric}" &
  grader_pid=$!
  prior_rubric=${current_rubric}
done
wait "${grader_pid}"
grader_pid=

log "validating complete 100-prompt checkpoint inventory"
validate_full_inventory
log "building sealed observations and final horizon report"
finalize_audit
log "canonical Pool B audit complete: ${AUDIT_COMPLETE_MARKER}"
run_pool_a_auxiliary
