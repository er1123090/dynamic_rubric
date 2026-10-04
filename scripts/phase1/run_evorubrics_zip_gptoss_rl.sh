#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
SOURCE_ROOT="${PROJECT_ROOT}/environment/upstream/EvoRubrics-2155-clean"
MODEL_PATH="${PROJECT_ROOT}/outputs/medicine/evorubrics_sft/seed-11/healthbench4k-gptoss120b-sft-zip-settings-20260924/sft_export_fp32/merged_model"
RUN_ID="${RUN_ID:-healthbench4k-gptoss120b-zip-clean-20260924}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/outputs/medicine/evorubrics_rl/seed-11/${RUN_ID}}"
JUDGE_BASE_URL="${JUDGE_BASE_URL:-http://127.0.0.1:28014/v1}"
RUNTIME_VENV="${RUNTIME_VENV:-${PROJECT_ROOT}/.venvs/evorubrics}"
RAY_TMPDIR="${RAY_TMPDIR:-/tmp/evorl_zip_clean}"

export PATH="${RUNTIME_VENV}/bin:${PATH}"
export LLM_EVALUATOR_TIMEOUT_SECONDS="${LLM_EVALUATOR_TIMEOUT_SECONDS:-1200}"
export LLM_EVALUATOR_MAX_COMPLETION_TOKENS="${LLM_EVALUATOR_MAX_COMPLETION_TOKENS:-4096}"

TRAIN_DATA_PATH="${SOURCE_ROOT}/data/healthbench_train_easy_4k.json"
EVAL_TEST_FILE="${SOURCE_ROOT}/data/healthbench_official_hard_1k.json"
mkdir -p "${RUN_ROOT}/checkpoints" "${RAY_TMPDIR}"
exec > >(tee -a "${RUN_ROOT}/launcher.log") 2>&1

write_state() {
    local state="$1"
    local detail="$2"
    printf '{\n  "state": "%s",\n  "detail": "%s",\n  "updated_at": "%s"\n}\n' \
        "${state}" "${detail}" "$(date --iso-8601=seconds)" > "${RUN_ROOT}/run_state.json"
}

on_exit() {
    local code=$?
    printf '%s\n' "${code}" > "${RUN_ROOT}/exit_code.txt"
    if [ "${code}" -eq 0 ]; then
        write_state completed "clean ZIP EvoRubrics run completed"
    else
        write_state failed "clean ZIP EvoRubrics run exited with code ${code}"
    fi
}
trap on_exit EXIT

for required_path in \
    "${RUNTIME_VENV}/bin/python" \
    "${MODEL_PATH}/config.json" \
    "${MODEL_PATH}/model.safetensors.index.json" \
    "${TRAIN_DATA_PATH}" \
    "${EVAL_TEST_FILE}" \
    "${SOURCE_ROOT}/scripts/run_unified_golden.sh"; do
    if [ ! -e "${required_path}" ]; then
        write_state failed "missing required path: ${required_path}"
        exit 2
    fi
done

for endpoint in \
    "http://127.0.0.1:28003/v1/models" \
    "http://127.0.0.1:28004/v1/models" \
    "${JUDGE_BASE_URL}/models"; do
    if ! curl -fsS --max-time 15 "${endpoint}" >/dev/null; then
        write_state failed "judge preflight failed: ${endpoint}"
        exit 3
    fi
done

write_state starting "Trainer GPU 1 GRPO; Inference B GPU 0-3 GPT-OSS judges; pristine ZIP source plus two compatibility patches"

CUDA_VISIBLE_DEVICES="1" \
RAY_TMPDIR="${RAY_TMPDIR}" \
MODEL_PATH="${MODEL_PATH}" \
TRAIN_DATA_PATH="${TRAIN_DATA_PATH}" \
EVAL_TEST_FILE="${EVAL_TEST_FILE}" \
EXP_NAME="${RUN_ID}" \
RESUME_ENABLED="false" \
N_GPUS_PER_NODE="1" \
BATCH_SIZE="4" \
GRAD_ACCUM_STEPS="1" \
NUM_RUBRICS="4" \
NUM_ANSWERS="4" \
TOTAL_EPOCHS="1" \
LORA_R="32" \
LORA_ALPHA="64" \
POLICY_LLM_LR="2e-5" \
RUBRICS_GENERATOR_LR="5e-6" \
SAVE_LORA_FREQ="1" \
SAVE_FREQ="1" \
MAX_CONCURRENT_API_CALLS="64" \
EVAL_ENABLED="true" \
EVAL_NUM_SAMPLES="1000" \
EVAL_BEFORE_TRAIN="true" \
EVAL_FREQ="100" \
LLM_EVALUATOR_TYPE="deepseek" \
DEEPSEEK_BASE_URL="${JUDGE_BASE_URL}" \
DEEPSEEK_API_KEY="EMPTY" \
DEEPSEEK_MODEL="openai/gpt-oss-120b" \
LLM_EVALUATOR_TIMEOUT_SECONDS="${LLM_EVALUATOR_TIMEOUT_SECONDS}" \
LLM_EVALUATOR_MAX_COMPLETION_TOKENS="${LLM_EVALUATOR_MAX_COMPLETION_TOKENS}" \
bash "${SOURCE_ROOT}/scripts/run_unified_golden.sh" \
    "trainer.default_local_dir=${RUN_ROOT}/checkpoints"
