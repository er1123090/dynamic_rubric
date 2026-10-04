#!/usr/bin/env bash
set -euo pipefail

project_root=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
run_id=phase1-online-rubrics-medicine-full-dense-20260919-seed11
run_root="${project_root}/outputs/medicine/online_rubrics/seed-11/${run_id}"
checkpoint_root="${run_root}/verl-run/checkpoints"
checkpoint_tracker="${checkpoint_root}/latest_checkpointed_iteration.txt"
target_step=7
supervisor_session=phase1-dense-supervisor
runtime_python="${project_root}/environment/upstream/verl/.venv-runtime/bin/python"

echo "[$(date -Is)] armed: waiting for sealed checkpoint ${target_step}"
while true; do
  sealed_step=$(cat "${checkpoint_tracker}" 2>/dev/null || true)
  if [[ "${sealed_step}" =~ ^[0-9]+$ ]] && ((sealed_step >= target_step)); then
    break
  fi
  if ! tmux has-session -t "${supervisor_session}" 2>/dev/null; then
    echo "[$(date -Is)] error: training supervisor disappeared before checkpoint ${target_step}" >&2
    exit 1
  fi
  sleep 5
done

checkpoint="${checkpoint_root}/global_step_${target_step}"
required_files=(
  "${checkpoint}/actor/model_world_size_1_rank_0.pt"
  "${checkpoint}/actor/optim_world_size_1_rank_0.pt"
  "${checkpoint}/actor/extra_state_world_size_1_rank_0.pt"
  "${checkpoint}/data.pt"
)
for path in "${required_files[@]}"; do
  if [[ ! -s "${path}" ]]; then
    echo "[$(date -Is)] error: sealed checkpoint file is absent or empty: ${path}" >&2
    exit 1
  fi
done

curl -fsS --max-time 5 http://127.0.0.1:28011/v1/models >/dev/null
curl -fsS --max-time 5 http://127.0.0.1:28014/v1/models >/dev/null
echo "[$(date -Is)] checkpoint ${target_step} sealed; stopping the old trainer"

mapfile -t trainer_pids < <(
  pgrep -f "verl.trainer.main_ppo.*trainer.experiment_name=${run_id}" || true
)
if ((${#trainer_pids[@]} != 1)); then
  echo "[$(date -Is)] error: expected one trainer, found ${#trainer_pids[@]}" >&2
  exit 1
fi
old_trainer_pid=${trainer_pids[0]}
kill -TERM "${old_trainer_pid}"

for _ in $(seq 1 120); do
  if ! kill -0 "${old_trainer_pid}" 2>/dev/null; then
    break
  fi
  sleep 1
done
if kill -0 "${old_trainer_pid}" 2>/dev/null; then
  echo "[$(date -Is)] error: trainer did not stop within 120 seconds" >&2
  exit 1
fi

supervisor_command="cd ${project_root} && trap ':' TERM HUP && while true; do env -u PHASE1_GPT_OSS_BASE_URL -u PHASE1_QWEN32B_BASE_URL PYTHONPATH=${project_root}/src PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=1 ONLINE_CONTROL_CACHE=${project_root}/outputs/medicine/shared/seed-11/pi0_control_cache/manifest-ad74decb90ce01cfc4a3048e21757e6fca58d6c77095d86c32d40507802d0407.json PHASE1_GPT_OSS_BASE_URLS=http://127.0.0.1:28011 PHASE1_QWEN32B_BASE_URLS=http://127.0.0.1:28014 PHASE1_QWEN32B_EXPECTED_COUNT=1 ONLINE_EXTRACTOR_CONCURRENCY=48 ONLINE_GRADER_CONCURRENCY=64 ONLINE_LOGPROB_PREFETCH=true ROLLOUT_GPU_MEMORY=0.55 ROLLOUT_MAX_NUM_SEQS=128 ROLLOUT_MAX_NUM_BATCHED_TOKENS=16384 ${runtime_python} -m dynamic_rubric.phase1 resume-online --config ${run_root}/config.resolved.json --repo-root ${project_root} --run-id ${run_id}; phase1_rc=\$?; date -Is; if [ \$phase1_rc -eq 0 ]; then echo TRAINING_COMPLETE; exit 0; fi; echo TRAINING_EXIT=\$phase1_rc RETRY_IN=30s; sleep 30; done"

tmux respawn-pane -k -t "${supervisor_session}:0.0" "${supervisor_command}"
echo "[$(date -Is)] supervisor replaced; waiting for concurrency=48 trainer"

for _ in $(seq 1 180); do
  mapfile -t resumed_pids < <(
    pgrep -f "verl.trainer.main_ppo.*trainer.experiment_name=${run_id}" || true
  )
  for pid in "${resumed_pids[@]}"; do
    if tr '\0' '\n' <"/proc/${pid}/environ" 2>/dev/null \
      | grep -qx 'ONLINE_EXTRACTOR_CONCURRENCY=48'; then
      echo "[$(date -Is)] switched: trainer_pid=${pid} extractor_concurrency=48"
      exit 0
    fi
  done
  sleep 2
done

echo "[$(date -Is)] error: no concurrency=48 trainer appeared within 360 seconds" >&2
exit 1
