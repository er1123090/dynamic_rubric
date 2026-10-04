#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
SOURCE_ROOT="${PROJECT_ROOT}/environment/upstream/EvoRubrics-2155-clean"
MODEL_PATH="${PROJECT_ROOT}/outputs/medicine/evorubrics_sft/seed-11/healthbench4k-gptoss120b-sft-zip-settings-20260924/sft_export_fp32/merged_model"
RUN_ID="healthbench4k-gptoss120b-zip-clean-20260924"
RUN_ROOT="${PROJECT_ROOT}/outputs/medicine/evorubrics_rl/seed-11/${RUN_ID}"
RESUME_STEP_NUMBER="${PHASE1_RESUME_STEP:-19}"
POLICY_LORA="${RUN_ROOT}/checkpoints/policy_llm/step_${RESUME_STEP_NUMBER}/lora_adapter_policy_llm"
RUBRICS_LORA="${RUN_ROOT}/checkpoints/rubrics_generator/step_${RESUME_STEP_NUMBER}/lora_adapter_rubrics_generator"
JUDGE_BASE_URL="http://127.0.0.1:28014/v1"
RUNTIME_VENV="${RUNTIME_VENV:-${PROJECT_ROOT}/.venvs/evorubrics}"
RAY_TMPDIR="/tmp/evorl_zip_clean_resume${RESUME_STEP_NUMBER}_trainer1"
LOG_FILE="${RUN_ROOT}/resume_step${RESUME_STEP_NUMBER}_trainer1_inference_a.log"

mkdir -p "${RAY_TMPDIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "$(date -Is) waiting for Trainer GPU 1 to become free"
stable_free_samples=0
while (( stable_free_samples < 6 )); do
    used_mib="$(nvidia-smi --id=1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
    if [[ "${used_mib}" =~ ^[0-9]+$ ]] && (( used_mib < 4096 )); then
        stable_free_samples=$((stable_free_samples + 1))
    else
        stable_free_samples=0
    fi
    echo "$(date -Is) Trainer GPU 1 memory.used=${used_mib} MiB; stable_free=${stable_free_samples}/6"
    if (( stable_free_samples < 6 )); then
        sleep 10
    fi
done

for required_path in \
    "${RUNTIME_VENV}/bin/python" \
    "${MODEL_PATH}/config.json" \
    "${POLICY_LORA}/adapter_model.safetensors" \
    "${RUBRICS_LORA}/adapter_model.safetensors" \
    "${RUN_ROOT}/checkpoints/policy_llm/step_${RESUME_STEP_NUMBER}/optimizer_policy_llm/optimizer_state.pt" \
    "${RUN_ROOT}/checkpoints/rubrics_generator/step_${RESUME_STEP_NUMBER}/optimizer_rubrics_generator/optimizer_state.pt"; do
    test -e "${required_path}"
done

curl -fsS --max-time 15 "${JUDGE_BASE_URL}/models" >/dev/null
echo "$(date -Is) preflight passed; resuming EvoRubrics RL from step ${RESUME_STEP_NUMBER} on Trainer GPU 1"

export PATH="${RUNTIME_VENV}/bin:${PATH}"
export LLM_EVALUATOR_TIMEOUT_SECONDS=1200
export LLM_EVALUATOR_MAX_COMPLETION_TOKENS=4096
export RAY_memory_usage_threshold=0.98

exec env \
    CUDA_VISIBLE_DEVICES=1 \
    RAY_TMPDIR="${RAY_TMPDIR}" \
    MODEL_PATH="${MODEL_PATH}" \
    TRAIN_DATA_PATH="${SOURCE_ROOT}/data/healthbench_train_easy_4k.json" \
    EVAL_TEST_FILE="${SOURCE_ROOT}/data/healthbench_official_hard_1k.json" \
    EXP_NAME="${RUN_ID}" \
    RESUME_ENABLED=true \
    RESUME_STEP="${RESUME_STEP_NUMBER}" \
    RESUME_POLICY_LORA="${POLICY_LORA}" \
    RESUME_RUBRICS_LORA="${RUBRICS_LORA}" \
    N_GPUS_PER_NODE=1 \
    BATCH_SIZE=4 \
    GRAD_ACCUM_STEPS=1 \
    NUM_RUBRICS=4 \
    NUM_ANSWERS=4 \
    TOTAL_EPOCHS=1 \
    LORA_R=32 \
    LORA_ALPHA=64 \
    POLICY_LLM_LR=2e-5 \
    RUBRICS_GENERATOR_LR=5e-6 \
    SAVE_LORA_FREQ=1 \
    SAVE_FREQ=1 \
    MAX_CONCURRENT_API_CALLS=64 \
    EVAL_ENABLED=true \
    EVAL_NUM_SAMPLES=1000 \
    EVAL_BEFORE_TRAIN=false \
    EVAL_FREQ=100 \
    LLM_EVALUATOR_TYPE=deepseek \
    DEEPSEEK_BASE_URL="${JUDGE_BASE_URL}" \
    DEEPSEEK_API_KEY=EMPTY \
    DEEPSEEK_MODEL=openai/gpt-oss-120b \
    bash "${SOURCE_ROOT}/scripts/run_unified_golden.sh" \
        "trainer.default_local_dir=${RUN_ROOT}/checkpoints"
