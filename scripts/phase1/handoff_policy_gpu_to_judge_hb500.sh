#!/usr/bin/env bash
set -euo pipefail
run_root=${PROJECT_ROOT}/outputs/policy_eval/medicine_checkpoint_trajectory_hb500_20260912/full-f0cdda051c001185
status_file="$run_root/status/generation.json"
config=${PROJECT_ROOT}/configs/evaluation/medicine_checkpoint_trajectory_hb500_20260912.yaml
while true; do
  state=$(python -c 'import json,sys; print(json.load(open(sys.argv[1])).get("state", ""))' "$status_file" 2>/dev/null || true)
  if [ "$state" = complete ]; then break; fi
  sleep 30
done
tmux kill-session -t medicine-policy-hb500-grade 2>/dev/null || true
tmux kill-session -t medicine-policy-hb500-trainer-judge 2>/dev/null || true
tmux new-session -d -s medicine-policy-hb500-trainer-judge "CUDA_VISIBLE_DEVICES=0 exec ${VLLM_BIN} serve ${MODEL_PATH} --served-model-name Qwen/Qwen3-32B --revision 9216db5781bf21249d130ec9da846c4624c16137 --tokenizer-revision 9216db5781bf21249d130ec9da846c4624c16137 --tensor-parallel-size 1 --dtype bfloat16 --gpu-memory-utilization 0.90 --max-model-len 32768 --max-num-seqs 64 --enable-prefix-caching --generation-config vllm --host 127.0.0.1 --port 28136 >> '$run_root/launcher_logs/qwen32b_trainer0.log' 2>&1"
for _ in $(seq 1 300); do
  if curl -fsS --max-time 3 http://127.0.0.1:28136/v1/models >/dev/null 2>&1; then break; fi
  sleep 3
done
curl -fsS --max-time 5 http://127.0.0.1:28136/v1/models >/dev/null
tmux new-session -d -s medicine-policy-hb500-grade "cd ${PROJECT_ROOT} && until uv run --project ${PROJECT_ROOT} python scripts/phase1/run_policy_checkpoint_trajectory.py --config '$config' --mode grade-watch --dataset rar_medicine_test --poll-seconds 20 --judge-base-url http://127.0.0.1:28134/v1 --judge-base-url http://127.0.0.1:28135/v1 --judge-base-url http://127.0.0.1:28136/v1 --judge-base-url http://127.0.0.1:28138/v1 >> '$run_root/launcher_logs/grade_watch.log' 2>&1; do sleep 20; done; until uv run --project ${PROJECT_ROOT} python scripts/phase1/run_policy_checkpoint_trajectory.py --config '$config' --mode grade-watch --dataset healthbench --poll-seconds 20 --judge-base-url http://127.0.0.1:28134/v1 --judge-base-url http://127.0.0.1:28135/v1 --judge-base-url http://127.0.0.1:28136/v1 --judge-base-url http://127.0.0.1:28138/v1 >> '$run_root/launcher_logs/grade_watch.log' 2>&1; do sleep 20; done; uv run --project ${PROJECT_ROOT} python scripts/phase1/plot_policy_checkpoint_trajectory.py '$run_root' >> '$run_root/launcher_logs/plot.log' 2>&1"
while tmux has-session -t medicine-policy-hb500-grade 2>/dev/null; do sleep 30; done
tmux kill-session -t medicine-policy-hb500-trainer-judge 2>/dev/null || true
