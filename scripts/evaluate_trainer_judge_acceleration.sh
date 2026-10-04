#!/usr/bin/env bash
set -euo pipefail

run_root="${TRAINER_JUDGE_RUN_ROOT:-${PROJECT_ROOT}/outputs/medicine/static_r0_matched/seed-11/phase1-static-r0-medicine-qwen3-4b-matched-dense-20260928-seed11}"
evidence_path="${TRAINER_JUDGE_EVIDENCE_PATH:-${run_root}/logs/trainer-judge-acceleration-evidence.json}"

python3 - "$evidence_path" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path


evidence_path = Path(sys.argv[1])
if not evidence_path.is_file():
    raise SystemExit(f"FAIL: missing evaluator evidence: {evidence_path}")

evidence = json.loads(evidence_path.read_text())
baseline = float(evidence["baseline_score_rate_per_second"])
observed = float(evidence["observed_score_rate_per_second"])
window = float(evidence["measurement_window_seconds"])
counts = [int(value) for value in evidence["upstream_request_counts"]]
activation_checkpoint = int(evidence["activation_checkpoint"])
verified_checkpoint = int(evidence["verified_checkpoint"])
sleeping = bool(evidence["trainer_is_sleeping_before_update"])
sleep_memory_mib = int(evidence["trainer_gpu_memory_used_mib_after_sleep"])
sleep_memory_limit_mib = int(evidence["trainer_gpu_memory_limit_mib_after_sleep"])
errors = list(evidence.get("errors_after_activation", []))

failures: list[str] = []
if baseline != 4.70:
    failures.append(f"baseline drifted: {baseline} != 4.70")
if window < 60:
    failures.append(f"measurement window too short: {window:.1f}s < 60s")
if observed < baseline * 1.15:
    failures.append(
        f"throughput improvement too small: {observed:.3f} < {baseline * 1.15:.3f} scores/s"
    )
if len(counts) != 3 or any(count <= 0 for count in counts):
    failures.append(f"all three replicas must serve requests: {counts}")
if not sleeping:
    failures.append("Trainer judge was not sleeping before training compute resumed")
if sleep_memory_mib > sleep_memory_limit_mib:
    failures.append(
        "Trainer GPU memory remained too high after sleep: "
        f"{sleep_memory_mib} MiB > {sleep_memory_limit_mib} MiB"
    )
if verified_checkpoint <= activation_checkpoint:
    failures.append(
        "no post-activation checkpoint completed: "
        f"activation={activation_checkpoint}, verified={verified_checkpoint}"
    )
if errors:
    failures.append(f"runtime errors after activation: {errors}")

if failures:
    print("FAIL")
    for failure in failures:
        print(f"- {failure}")
    raise SystemExit(1)

print("PASS")
print(f"- throughput: {observed:.3f} scores/s vs {baseline:.3f} baseline")
print(f"- 3-replica request counts: {counts}")
print(f"- Trainer post-sleep GPU memory: {sleep_memory_mib} MiB")
print(f"- post-activation checkpoint: {verified_checkpoint}")
PY
