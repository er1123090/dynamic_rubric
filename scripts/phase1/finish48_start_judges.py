#!/usr/bin/env python3
"""Turn final-probe services into Trainer/Inference A judges after verified checkpoint 48."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

from dynamic_rubric.artifacts import artifact_record, read_json, write_json_atomic
from dynamic_rubric.phase1.audit_policy import load_run_contract
from dynamic_rubric.phase1.probe_adjacent_scoring import (
    endpoint_identity,
    load_cell_plan,
    load_evaluator_rubrics,
    load_pool_b,
)


MODEL = "Qwen/Qwen3-32B"
REVISION = "9216db5781bf21249d130ec9da846c4624c16137"
ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = Path(os.environ.get("JUDGE_MODEL_PATH") or ROOT / "models/Qwen3-32B")
VLLM = os.environ.get("JUDGE_VLLM_BIN") or str(ROOT / ".venvs/judge/bin/vllm")


def required_setting(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise Final48TransitionError(
            f"Set {name} for your inference machine before running this optional helper"
        )
    return value


class Final48TransitionError(RuntimeError):
    """Raised when a service cannot be transitioned without crossing ownership."""


def process_identity(pid: int) -> dict[str, Any]:
    root = Path("/proc") / str(pid)
    stat = (root / "stat").read_text().rsplit(")", 1)[1].split()
    return {
        "pid": pid,
        "starttime": stat[19],
        "state": stat[0],
        "pgid": os.getpgid(pid),
        "command": [item for item in (root / "cmdline").read_bytes().decode().split("\0") if item],
    }


def alive(identity: Mapping[str, Any]) -> bool:
    try:
        current = process_identity(int(identity["pid"]))
    except (FileNotFoundError, ProcessLookupError):
        return False
    return current["starttime"] == str(identity["starttime"]) and current["state"] not in {"Z", "X"}


def validate_owned_local(identity: Mapping[str, Any], expected_command: Sequence[str]) -> None:
    current = process_identity(int(identity["pid"]))
    if current["starttime"] != str(identity["starttime"]):
        raise Final48TransitionError("local service PID was reused")
    if current["command"] != list(expected_command):
        raise Final48TransitionError("local service command differs from its launch journal")
    if current["pgid"] != current["pid"]:
        raise Final48TransitionError("local service does not own its process group")


def stop_owned_local(identity: Mapping[str, Any]) -> None:
    if alive(identity):
        os.killpg(int(identity["pid"]), signal.SIGTERM)


def wait_dead(identity: Mapping[str, Any], timeout: float = 120) -> None:
    deadline = time.monotonic() + timeout
    while alive(identity):
        if time.monotonic() > deadline:
            raise Final48TransitionError(f"owned service did not exit: {identity['pid']}")
        time.sleep(2)


def ssh_command(host: str, remote_code: str) -> list[str]:
    root = required_setting("PHASE1_REMOTE_PROJECT_ROOT")
    python = required_setting("PHASE1_REMOTE_PYTHON")
    container = os.environ.get("PHASE1_REMOTE_CONTAINER", "").strip()
    command = [python, "-c", remote_code]
    if container:
        remote = shlex.join(["docker", "exec", "-w", root, container, *command])
    else:
        remote = "cd " + shlex.quote(root) + " && " + shlex.join(command)
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        "-p",
        os.environ.get("PHASE1_REMOTE_SSH_PORT") or "22",
        host,
        remote,
    ]


def remote_stop_code(services: Sequence[Mapping[str, Any]]) -> str:
    specs = json.dumps(list(services))
    return f"""
import json, os, signal, time
specs=json.loads({specs!r})
for spec in specs:
 p=int(spec['pid']); root=f'/proc/{{p}}'
 stat=open(root+'/stat').read().rsplit(')',1)[1].split()
 cmd=[x for x in open(root+'/cmdline','rb').read().decode().split('\\0') if x]
 env=open(root+'/environ','rb').read().split(b'\\0')
 if stat[19] != str(spec['starttime']): raise RuntimeError('remote PID reused')
 if not all(part in cmd for part in spec['required_parts']): raise RuntimeError('remote command mismatch')
 if b'CUDA_VISIBLE_DEVICES=0,1' not in env: raise RuntimeError('remote service GPU scope mismatch')
 if os.getpgid(p) != p: raise RuntimeError('remote service does not own process group')
for spec in specs: os.killpg(int(spec['pid']), signal.SIGTERM)
deadline=time.monotonic()+120
while time.monotonic() < deadline:
 alive=[]
 for spec in specs:
  try:
   stat=open(f\"/proc/{{spec['pid']}}/stat\").read().rsplit(')',1)[1].split()
   if stat[19] == str(spec['starttime']) and stat[0] not in ('Z','X'): alive.append(spec['pid'])
  except FileNotFoundError: pass
 if not alive: print(json.dumps({{'state':'stopped','pids':[x['pid'] for x in specs]}})); break
 time.sleep(2)
else: raise RuntimeError(f'remote services did not exit: {{alive}}')
import subprocess
deadline=time.monotonic()+180
while time.monotonic() < deadline:
 used=[int(x.strip()) for x in subprocess.check_output(['nvidia-smi','-i','0,1','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines()]
 if max(used) < 512: break
 time.sleep(5)
else: raise RuntimeError(f'Inference A GPU0,1 did not become free: {{used}}')
"""


def remote_launch_code(log_path: str) -> str:
    command = judge_command(port=8004, tensor_parallel_size=2)
    command[0] = required_setting("PHASE1_REMOTE_VLLM_BIN")
    command[2] = required_setting("PHASE1_REMOTE_MODEL_PATH")
    return f"""
import json, os, subprocess
command=json.loads({json.dumps(json.dumps(command))})
env=dict(os.environ, CUDA_VISIBLE_DEVICES='0,1', VLLM_USE_DEEP_GEMM='0')
os.makedirs(os.path.dirname({log_path!r}), exist_ok=True)
log=open({log_path!r},'ab',buffering=0)
p=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env,start_new_session=True)
stat=open(f'/proc/{{p.pid}}/stat').read().rsplit(')',1)[1].split()
print(json.dumps({{'pid':p.pid,'starttime':stat[19],'command':command}}))
"""


def judge_command(*, port: int, tensor_parallel_size: int) -> list[str]:
    return [
        VLLM,
        "serve",
        str(MODEL_PATH),
        "--served-model-name",
        MODEL,
        "--revision",
        REVISION,
        "--tokenizer-revision",
        REVISION,
        "--tensor-parallel-size",
        str(tensor_parallel_size),
        "--dtype",
        "bfloat16",
        "--gpu-memory-utilization",
        "0.90",
        "--max-model-len",
        "32768",
        "--max-num-seqs",
        "128",
        "--max-num-batched-tokens",
        "16384",
        "--enable-prefix-caching",
        "--generation-config",
        "vllm",
        "--host",
        "0.0.0.0" if tensor_parallel_size == 2 else "127.0.0.1",
        "--port",
        str(port),
    ]


def scorer_command(
    run_dir: Path,
    artifact_root: Path,
    output_root: Path,
    plan: Path,
    url: str,
    *,
    concurrency: int,
) -> list[str]:
    return [
        sys.executable,
        "scripts/phase1/score_regular_probe_adjacent.py",
        "--run-dir",
        str(run_dir),
        "--artifact-root",
        str(artifact_root),
        "--output-root",
        str(output_root),
        "--cell-plan",
        str(plan),
        "--judge-urls",
        url,
        "--concurrency",
        str(concurrency),
        "--bounded-grading-whitespace",
        "--max-client-restarts",
        "8",
        "--restart-delay",
        "30",
        "--wait-timeout",
        "604800",
    ]


def spawn(command: Sequence[str], log_path: Path, *, env: Mapping[str, str] | None = None):
    with log_path.open("ab") as stream:
        return subprocess.Popen(
            list(command),
            stdout=stream,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=dict(env) if env else None,
            start_new_session=True,
        )


def wait_endpoint(url: str, process=None, timeout: float = 900) -> Mapping[str, Any]:
    deadline = time.monotonic() + timeout
    while True:
        if process is not None and process.poll() is not None:
            raise Final48TransitionError(f"judge exited during startup: {url}")
        try:
            return endpoint_identity(url, MODEL, REVISION)
        except Exception:
            if time.monotonic() > deadline:
                raise Final48TransitionError(f"judge readiness timed out: {url}")
            time.sleep(5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--score-root", required=True, type=Path)
    parser.add_argument("--service-journal", required=True, type=Path)
    parser.add_argument("--helper-status", required=True, type=Path)
    parser.add_argument("--helper-pid", required=True, type=int)
    parser.add_argument("--helper-starttime", required=True)
    parser.add_argument("--trainer-plan", required=True, type=Path)
    parser.add_argument("--inference_a01-plan", required=True, type=Path)
    parser.add_argument("--inference_a-gpt-pid", required=True, type=int)
    parser.add_argument("--inference_a-gpt-starttime", required=True)
    parser.add_argument("--inference_a-judge-pid", required=True, type=int)
    parser.add_argument("--inference_a-judge-starttime", required=True)
    parser.add_argument("--inference_a-host", required=True, help="SSH target: user@inference-host")
    parser.add_argument("--trainer-port", type=int, default=28012)
    parser.add_argument("--inference_a-tunnel-url", default="http://127.0.0.1:28002")
    args = parser.parse_args()
    for name in (
        "PHASE1_REMOTE_PROJECT_ROOT",
        "PHASE1_REMOTE_PYTHON",
        "PHASE1_REMOTE_VLLM_BIN",
        "PHASE1_REMOTE_MODEL_PATH",
    ):
        required_setting(name)
    if not MODEL_PATH.is_dir() or not os.access(VLLM, os.X_OK):
        parser.error("Prepare JUDGE_MODEL_PATH and JUDGE_VLLM_BIN before transitioning services")

    run = args.run_dir.resolve()
    root = args.artifact_root.resolve()
    score_root = args.score_root.resolve()
    logs = root / "logs/final48-judge-transition-20260909"
    logs.mkdir(parents=True, exist_ok=True)
    with (logs / "transition.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        status_path = logs / "status.json"
        state: dict[str, Any] = {"state": "preflight", "started_at": time.time()}

        def publish(phase: str, **fields: Any) -> None:
            state.update(state=phase, updated_at=time.time(), **fields)
            write_json_atomic(status_path, state, immutable=False)

        try:
            expected_host = os.environ.get("EXPECTED_TRAINER_HOST", "").strip()
            if expected_host and socket.gethostname().split(".")[0] != expected_host:
                raise Final48TransitionError("transition host differs from EXPECTED_TRAINER_HOST")
            journal = read_json(args.service_journal)
            local_services = []
            for name in ("policy", "proxy"):
                spec = journal[name]
                validate_owned_local(spec, spec["command"])
                local_services.append(spec)
            helper = {"pid": args.helper_pid, "starttime": args.helper_starttime}
            if not alive(helper):
                raise Final48TransitionError("checkpoint-48 helper identity is not alive")
            for plan in (args.trainer_plan, args.inference_a01_plan):
                load_cell_plan(plan)
            publish("waiting_for_final48", helper=helper)
            deadline = time.monotonic() + 14400
            while True:
                if (
                    args.helper_status.is_file()
                    and read_json(args.helper_status).get("state") == "complete"
                ):
                    break
                if not alive(helper):
                    raise Final48TransitionError(
                        "checkpoint-48 helper exited before complete status"
                    )
                if time.monotonic() > deadline:
                    raise Final48TransitionError("checkpoint-48 preparation timed out")
                time.sleep(10)
            wait_dead(helper)
            contract = load_run_contract(run)
            responses, response_manifest = load_pool_b(contract, root, 48)
            rubrics, rubric_manifest = load_evaluator_rubrics(contract, root, 48)
            if len(responses) != 1600 or len(rubrics) != 100:
                raise Final48TransitionError(
                    "checkpoint-48 Pool B or fresh rubric inventory is incomplete"
                )
            publish(
                "final48_verified",
                pool_b_manifest=response_manifest,
                fresh_rubric_manifest=rubric_manifest,
            )

            for spec in reversed(local_services):
                stop_owned_local(spec)
            for spec in local_services:
                wait_dead(spec)
            remote_specs = [
                {
                    "pid": args.inference_a_gpt_pid,
                    "starttime": args.inference_a_gpt_starttime,
                    "required_parts": ["openai/gpt-oss-120b", "--port", "8001"],
                },
                {
                    "pid": args.inference_a_judge_pid,
                    "starttime": args.inference_a_judge_starttime,
                    "required_parts": [MODEL, "--port", "8004"],
                },
            ]
            stopped = subprocess.run(
                ssh_command(args.inference_a_host, remote_stop_code(remote_specs)),
                check=True,
                text=True,
                capture_output=True,
            )
            publish("generation_services_stopped", inference_a_stop=stopped.stdout.strip())

            env = {
                **os.environ,
                "CUDA_VISIBLE_DEVICES": "1",
                "VLLM_USE_DEEP_GEMM": "0",
            }
            trainer_command = judge_command(port=args.trainer_port, tensor_parallel_size=1)
            trainer_server = spawn(trainer_command, logs / "trainer-qwen32b.log", env=env)
            remote_log = str(
                Path(required_setting("PHASE1_REMOTE_PROJECT_ROOT"))
                / "artifacts/logs/final48-inference-a-qwen32b.log"
            )
            launched = subprocess.run(
                ssh_command(args.inference_a_host, remote_launch_code(remote_log)),
                check=True,
                text=True,
                capture_output=True,
            )
            remote_server = json.loads(launched.stdout.strip().splitlines()[-1])
            publish(
                "judges_starting",
                trainer_judge=process_identity(trainer_server.pid),
                inference_a_judge=remote_server,
            )
            trainer_url = f"http://127.0.0.1:{args.trainer_port}"
            trainer_identity = wait_endpoint(trainer_url, trainer_server)
            inference_a_identity = wait_endpoint(args.inference_a_tunnel_url)
            if trainer_identity["version"] != inference_a_identity["version"]:
                raise Final48TransitionError("judge replicas use different vLLM versions")
            publish(
                "judges_ready",
                trainer_identity=trainer_identity,
                inference_a_identity=inference_a_identity,
            )

            trainer_scorer_command = scorer_command(
                run,
                root,
                score_root / "trainer",
                args.trainer_plan.resolve(),
                trainer_url,
                concurrency=64,
            )
            inference_a_scorer_command = scorer_command(
                run,
                root,
                score_root / "inference_a01",
                args.inference_a01_plan.resolve(),
                args.inference_a_tunnel_url,
                concurrency=48,
            )
            trainer_scorer = spawn(trainer_scorer_command, logs / "trainer-scoring.log")
            inference_a_scorer = spawn(
                inference_a_scorer_command, logs / "inference_a01-scoring.log"
            )
            publish(
                "scorers_started",
                trainer_plan=artifact_record(args.trainer_plan),
                inference_a01_plan=artifact_record(args.inference_a01_plan),
                trainer_scorer=process_identity(trainer_scorer.pid),
                inference_a01_scorer=process_identity(inference_a_scorer.pid),
            )
        except BaseException as error:
            publish("failed", failed_phase=state["state"], error=repr(error))
            raise


if __name__ == "__main__":
    main()
