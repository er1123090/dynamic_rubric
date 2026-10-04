#!/usr/bin/env bash
set -euo pipefail


RUN_ROOT="${1:?usage: transition_healthbench_sft_warmup.sh RUN_ROOT}"
FULL_SESSION="${FULL_SESSION:-evorubrics-sft-hb4k-full}"
TEACHER_SESSION="${TEACHER_SESSION:-evorubrics-sft-hb4k-teacher}"
PYTHON="${EVORUBRICS_PYTHON}"
TRANSITION_LOG="${RUN_ROOT}/logs/transition.log"

mkdir -p "${RUN_ROOT}/logs"
echo "$(date -Is) waiting_for_teacher_generation" >> "${TRANSITION_LOG}"
while tmux has-session -t "${FULL_SESSION}" 2>/dev/null; do
  sleep 30
done

TEACHER_DATA_ROOT="${RUN_ROOT}/teacher"
if ! PYTHONPATH=src "${PYTHON}" -c \
  'import json,sys; d=json.load(open(sys.argv[1])); assert d["status"] == "complete" and d["count"] == 4000' \
  "${TEACHER_DATA_ROOT}/manifest.json" 2>/dev/null; then
  echo "$(date -Is) starting_teacher_repair" >> "${TRANSITION_LOG}"
  set +e
  env PYTHONPATH=src PYTHONUNBUFFERED=1 "${PYTHON}" \
    scripts/phase1/repair_evorubrics_sft_teacher.py \
    --train "${RUN_ROOT}/source/train.jsonl" \
    --heldout "${RUN_ROOT}/source/heldout.jsonl" \
    --run-root "${RUN_ROOT}" --base-url http://127.0.0.1:28011/v1 \
    --concurrency 64 --expected-train 4000 --expected-heldout 1000 \
    >> "${RUN_ROOT}/logs/teacher-repair.log" 2>&1
  repair_status=$?
  set -e
  if [ "${repair_status}" -ne 0 ]; then
    echo "$(date -Is) teacher_repair_failed status=${repair_status}" >> "${TRANSITION_LOG}"
    exit "${repair_status}"
  fi
  TEACHER_DATA_ROOT="${RUN_ROOT}/teacher_final"
fi

PYTHONPATH=src "${PYTHON}" -c \
  'import json,sys; d=json.load(open(sys.argv[1])); assert d["status"] == "complete" and d["count"] == 4000' \
  "${TEACHER_DATA_ROOT}/manifest.json"

echo "$(date -Is) teacher_complete_stopping_server" >> "${TRANSITION_LOG}"
if tmux has-session -t "${TEACHER_SESSION}" 2>/dev/null; then
  tmux send-keys -t "${TEACHER_SESSION}" C-c
  for _ in $(seq 1 60); do
    tmux has-session -t "${TEACHER_SESSION}" 2>/dev/null || break
    sleep 2
  done
  if tmux has-session -t "${TEACHER_SESSION}" 2>/dev/null; then
    tmux kill-session -t "${TEACHER_SESSION}"
  fi
fi

for _ in $(seq 1 60); do
  if ! nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
    break
  fi
  sleep 2
done
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
  echo "$(date -Is) gpu_release_timeout" >> "${TRANSITION_LOG}"
  exit 3
fi

echo "$(date -Is) starting_sft" >> "${TRANSITION_LOG}"
set +e
env CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  PYTHONPATH=src PYTHONUNBUFFERED=1 "${PYTHON}" \
  scripts/phase1/train_evorubrics_sft.py \
  --data "${TEACHER_DATA_ROOT}/train.jsonl" \
  --run-root "${RUN_ROOT}/sft" \
  --epochs 1 --batch-size 4 --learning-rate 2e-5 \
  --gpu 0 --max-length 4608 --seed 11 --expected-examples 4000 \
  >> "${RUN_ROOT}/logs/sft-train.log" 2>&1
sft_status=$?
set -e

if [ "${sft_status}" -eq 0 ]; then
  echo "$(date -Is) sft_complete" >> "${TRANSITION_LOG}"
  exit 0
fi
if [ ! -f "${RUN_ROOT}/sft/checkpoint/adapter/adapter_model.safetensors" ]; then
  echo "$(date -Is) sft_failed_before_adapter status=${sft_status}" >> "${TRANSITION_LOG}"
  exit "${sft_status}"
fi

echo "$(date -Is) starting_fp32_export status=${sft_status}" >> "${TRANSITION_LOG}"
exec env CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  PYTHONPATH=src PYTHONUNBUFFERED=1 "${PYTHON}" \
  scripts/phase1/export_evorubrics_sft_checkpoint.py \
  --source-run "${RUN_ROOT}/sft" --training-log "${RUN_ROOT}/logs/sft-train.log" \
  --run-root "${RUN_ROOT}/sft_export_fp32" --gpu 0 \
  --expected-steps 1000 --expected-exposures 4000 \
  >> "${RUN_ROOT}/logs/sft-export-fp32.log" 2>&1
