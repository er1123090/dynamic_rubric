#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_RUNNER="${SCRIPT_DIR}/evaluate_final_policies.py"

usage() {
  cat <<'EOF'
Usage:
  run_final_eval_grade_queue.sh \
    --config CONFIG.yaml \
    --run-dir FULL_RUN_DIRECTORY \
    --expected-grades-per-model 60091 \
    [--tmux-session rq2_static_base_grade] \
    [--judge-url http://127.0.0.1:28132/v1] \
    [--judge-model Qwen/Qwen3-32B] \
    [--python .venv/bin/python] \
    [--runner scripts/phase1/evaluate_final_policies.py] \
    [--evaluation-output-dir OUTPUT_ROOT] \
    [--poll-seconds 30] \
    [--dry-run]

Waits for the existing static_base tmux job, verifies its exact immutable grade
count, then resumes static_final, online_base, and online_final grading in order.
No credentials are accepted or embedded.
EOF
}

CONFIG=""
RUN_DIR=""
EXPECTED_GRADES=""
TMUX_SESSION="rq2_static_base_grade"
JUDGE_URL="http://127.0.0.1:28132/v1"
JUDGE_MODEL="Qwen/Qwen3-32B"
PYTHON_BIN=".venv/bin/python"
RUNNER="${DEFAULT_RUNNER}"
EVALUATION_OUTPUT_DIR=""
POLL_SECONDS=30
DRY_RUN=0

while (( $# )); do
  case "$1" in
    --config) CONFIG="${2:?missing --config value}"; shift 2 ;;
    --run-dir) RUN_DIR="${2:?missing --run-dir value}"; shift 2 ;;
    --expected-grades-per-model)
      EXPECTED_GRADES="${2:?missing --expected-grades-per-model value}"
      shift 2
      ;;
    --tmux-session) TMUX_SESSION="${2:?missing --tmux-session value}"; shift 2 ;;
    --judge-url) JUDGE_URL="${2:?missing --judge-url value}"; shift 2 ;;
    --judge-model) JUDGE_MODEL="${2:?missing --judge-model value}"; shift 2 ;;
    --python) PYTHON_BIN="${2:?missing --python value}"; shift 2 ;;
    --runner) RUNNER="${2:?missing --runner value}"; shift 2 ;;
    --evaluation-output-dir)
      EVALUATION_OUTPUT_DIR="${2:?missing --evaluation-output-dir value}"
      shift 2
      ;;
    --poll-seconds) POLL_SECONDS="${2:?missing --poll-seconds value}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "${CONFIG}" ]] || { echo "--config is required" >&2; exit 2; }
[[ -n "${RUN_DIR}" ]] || { echo "--run-dir is required" >&2; exit 2; }
[[ "${EXPECTED_GRADES}" =~ ^[1-9][0-9]*$ ]] || {
  echo "--expected-grades-per-model must be a positive integer" >&2
  exit 2
}
[[ "${POLL_SECONDS}" =~ ^[1-9][0-9]*$ ]] || {
  echo "--poll-seconds must be a positive integer" >&2
  exit 2
}

grade_command() {
  local model="$1"
  GRADE_COMMAND=(
    "${PYTHON_BIN}" "${RUNNER}"
    --config "${CONFIG}"
    --stage grade
    --model "${model}"
  )
  if [[ -n "${EVALUATION_OUTPUT_DIR}" ]]; then
    GRADE_COMMAND+=(--output-dir "${EVALUATION_OUTPUT_DIR}")
  fi
}

summarize_command() {
  SUMMARY_COMMAND=(
    "${PYTHON_BIN}" "${RUNNER}"
    --config "${CONFIG}"
    --stage summarize
  )
  if [[ -n "${EVALUATION_OUTPUT_DIR}" ]]; then
    SUMMARY_COMMAND+=(--output-dir "${EVALUATION_OUTPUT_DIR}")
  fi
}

if (( DRY_RUN )); then
  printf 'dry_run=true\n'
  printf 'wait_tmux_session=%s\n' "${TMUX_SESSION}"
  printf 'run_dir=%s\n' "${RUN_DIR}"
  printf 'expected_grades_per_model=%s\n' "${EXPECTED_GRADES}"
  printf 'judge_models_endpoint=%s/models\n' "${JUDGE_URL%/}"
  printf 'judge_expected_model=%s\n' "${JUDGE_MODEL}"
  for model in static_final online_base online_final; do
    grade_command "${model}"
    printf 'grade_command='
    printf '%q ' "${GRADE_COMMAND[@]}"
    printf '\n'
  done
  summarize_command
  printf 'summarize_command='
  printf '%q ' "${SUMMARY_COMMAND[@]}"
  printf '\n'
  exit 0
fi

[[ -f "${CONFIG}" ]] || { echo "config does not exist: ${CONFIG}" >&2; exit 2; }
[[ -f "${RUNNER}" ]] || { echo "runner does not exist: ${RUNNER}" >&2; exit 2; }
[[ -x "${PYTHON_BIN}" ]] || { echo "python is not executable: ${PYTHON_BIN}" >&2; exit 2; }
[[ -d "${RUN_DIR}" ]] || { echo "full run directory does not exist: ${RUN_DIR}" >&2; exit 2; }

check_judge_endpoint() {
  "${PYTHON_BIN}" - "${JUDGE_URL%/}/models" "${JUDGE_MODEL}" <<'PY'
import json
import sys
import urllib.request

url, expected = sys.argv[1:]
with urllib.request.urlopen(url, timeout=20) as response:
    payload = json.load(response)
models = {
    str(item.get("id", ""))
    for item in payload.get("data", [])
    if isinstance(item, dict)
}
if expected not in models:
    raise SystemExit(f"judge served-model mismatch: expected {expected!r}, got {sorted(models)!r}")
print(f"judge_endpoint_ok={url} model={expected}")
PY
}

require_expected_count() {
  local model="$1"
  "${PYTHON_BIN}" - "${RUN_DIR}" "${model}" "${EXPECTED_GRADES}" "${JUDGE_MODEL}" <<'PY'
import json
from pathlib import Path
import sys

run_dir, model, expected_text, judge_model = sys.argv[1:]
root = Path(run_dir)
expected_count = int(expected_text)
prepared_path = root / "prepared" / "prompts.jsonl"
if not prepared_path.is_file():
    raise SystemExit(f"missing prepared prompt manifest: {prepared_path}")

prompts = [json.loads(line) for line in prepared_path.read_text().splitlines() if line.strip()]
expected = set()
for prompt in prompts:
    dataset = str(prompt["dataset"])
    prompt_id = str(prompt["prompt_id"])
    criteria = prompt.get("criteria")
    if not isinstance(criteria, list) or not criteria:
        raise SystemExit(f"invalid prepared criteria: {dataset}/{prompt_id}")
    for criterion in criteria:
        expected.add((dataset, prompt_id, str(criterion["criterion_id"])))

responses = {}
for path in (root / "responses" / model).rglob("*.json"):
    record = json.loads(path.read_text())
    required = ("schema_version", "dataset", "prompt_id", "model_name", "response_id", "text")
    if any(key not in record for key in required):
        raise SystemExit(f"response schema mismatch: {path}")
    if record["schema_version"] != 1 or record["model_name"] != model:
        raise SystemExit(f"response identity mismatch: {path}")
    if not isinstance(record["text"], str) or not record["text"].strip():
        raise SystemExit(f"response text missing: {path}")
    key = (str(record["dataset"]), str(record["prompt_id"]))
    if key in responses:
        raise SystemExit(f"duplicate response identity: {key}")
    responses[key] = str(record["response_id"])

seen = set()
grade_paths = list((root / "grades" / model).rglob("*.json"))
for path in grade_paths:
    record = json.loads(path.read_text())
    required = (
        "schema_version", "dataset", "model_name", "prompt_id", "response_id",
        "criterion_id", "criterion", "points", "tags", "criteria_met", "explanation",
        "requested_model", "returned_model",
    )
    if any(key not in record for key in required):
        raise SystemExit(f"grade schema mismatch: {path}")
    if record["schema_version"] != 1 or record["model_name"] != model:
        raise SystemExit(f"grade model identity mismatch: {path}")
    if type(record["criteria_met"]) is not bool or not isinstance(record["explanation"], str):
        raise SystemExit(f"grade value schema mismatch: {path}")
    if record["requested_model"] != judge_model or record["returned_model"] != judge_model:
        raise SystemExit(f"grade judge identity mismatch: {path}")
    key = (str(record["dataset"]), str(record["prompt_id"]), str(record["criterion_id"]))
    if key not in expected:
        raise SystemExit(f"unexpected grade identity: {key}")
    response_key = key[:2]
    if responses.get(response_key) != str(record["response_id"]):
        raise SystemExit(f"grade response identity mismatch: {path}")
    if key in seen:
        raise SystemExit(f"duplicate grade identity: {key}")
    seen.add(key)

if seen != expected:
    missing = len(expected - seen)
    extra = len(seen - expected)
    raise SystemExit(f"grade inventory mismatch: missing={missing}, extra={extra}")
if len(grade_paths) != expected_count or len(seen) != expected_count:
    raise SystemExit(
        f"{model} grade count mismatch: expected {expected_count}, "
        f"files={len(grade_paths)}, identities={len(seen)}"
    )
print(f"{model}_grades_semantically_verified={len(seen)}")
PY
}

wait_for_static_base() {
  if ! tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
    echo "tmux_session_not_running=${TMUX_SESSION}; validating existing static_base artifacts"
    return
  fi
  echo "waiting_for_tmux_session=${TMUX_SESSION}"
  while tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; do
    local dead_statuses
    dead_statuses="$(tmux list-panes -t "${TMUX_SESSION}" -F '#{pane_dead}:#{pane_dead_status}')"
    if [[ "${dead_statuses}" == *"1:"* ]]; then
      if [[ "${dead_statuses}" != *"1:0"* ]]; then
        echo "tmux grading pane exited unsuccessfully: ${dead_statuses}" >&2
        exit 1
      fi
      break
    fi
    sleep "${POLL_SECONDS}"
  done
}

echo "queue_start static_base -> static_final -> online_base -> online_final -> summarize"
check_judge_endpoint
wait_for_static_base
check_judge_endpoint
require_expected_count static_base

for model in static_final online_base online_final; do
  check_judge_endpoint
  grade_command "${model}"
  echo "grading_start=${model} existing_immutable_grades_resume=true"
  "${GRADE_COMMAND[@]}"
  check_judge_endpoint
  require_expected_count "${model}"
done

summarize_command
echo "summarize_start=true"
"${SUMMARY_COMMAND[@]}"
echo "queue_complete=true"
