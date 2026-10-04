"""After verified probe generation, repurpose Trainer GPU1 for column-sharded judging."""

from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

from dynamic_rubric.artifacts import (
    artifact_record,
    read_json,
    validate_artifact_record,
    write_json_atomic,
)
from dynamic_rubric.phase1.audit_policy import load_run_contract
from dynamic_rubric.phase1.audit_scoring import AuditScoreConfig, score_pool, write_score_receipts
from dynamic_rubric.phase1.probe_adjacent_scoring import (
    endpoint_identity,
    load_cell_plan,
    load_evaluator_rubrics,
    load_pool_b,
)
from dynamic_rubric.providers.vllm_chat import VLLMChatAdapter

MODEL = "Qwen/Qwen3-32B"
REVISION = "9216db5781bf21249d130ec9da846c4624c16137"
ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = Path(os.environ.get("JUDGE_MODEL_PATH") or ROOT / "models/Qwen3-32B")
VLLM = os.environ.get("JUDGE_VLLM_BIN") or str(ROOT / ".venvs/judge/bin/vllm")
REQUIRED_FRESH = set(range(3, 46, 3)) | {13, 16, 32, 34, 40}


def process_identity(pid):
    p = Path("/proc") / str(pid)
    fields = (p / "stat").read_text().rsplit(")", 1)[1].split()
    return {
        "pid": pid,
        "starttime": fields[19],
        "state": fields[0],
        "ppid": int(fields[1]),
        "command": [x for x in (p / "cmdline").read_bytes().decode().split("\0") if x],
    }


def alive(identity):
    try:
        current = process_identity(identity["pid"])
    except FileNotFoundError:
        return False
    return current["starttime"] == identity["starttime"] and current["state"] not in ("Z", "X")


def stop_exact(identity):
    if alive(identity):
        os.kill(identity["pid"], signal.SIGTERM)


def generation_complete(root):
    """Neither a successful subset nor a stale completion flag unlocks shutdown."""
    fresh = read_json(root / "logs/fresh_queue_status.json")
    pools = read_json(root / "logs/pool_a_status.json")
    if fresh.get("state") != "complete" or pools.get("state") != "complete":
        return False
    if not REQUIRED_FRESH <= set(fresh.get("completed_checkpoints", [])):
        raise RuntimeError("fresh completion status omits required checkpoints")
    for step in sorted({0} | REQUIRED_FRESH):
        provenance = read_json(root / "responses" / f"checkpoint-{step:06d}" / "provenance.json")
        required = {"probe_B": 1600} if step == 0 else {"probe_A": 800, "probe_B": 1600}
        if provenance.get("global_step") != step or any(
            provenance.get("pool_counts", {}).get(k) != v for k, v in required.items()
        ):
            raise RuntimeError(f"probe inventory incomplete at checkpoint {step}")
        for record in provenance["artifacts"]:
            validate_artifact_record(record)
        if step:
            directory = root / "rubrics" / f"checkpoint-{step:06d}"
            status = read_json(directory / "status.json")
            rubric = artifact_record(directory / "fresh_rubrics.jsonl")
            if (
                status.get("state") != "complete"
                or status.get("prompt_count") != 100
                or status.get("fresh_rubrics_sha256") != rubric["sha256"]
            ):
                raise RuntimeError(f"fresh rubric inventory incomplete at checkpoint {step}")
    return True


def verify_replica(primary, secondary):
    for key in ("version",):
        if primary[key] != secondary[key]:
            raise RuntimeError("judge replicas must use the same vLLM version")
    for key in ("id", "max_model_len"):
        if primary["model"][key] != secondary["model"][key]:
            raise RuntimeError(f"judge replica identity differs: {key}")


def smoke_judge(run, root, output, url):
    """Real schema/grade smoke on saved Pool B; isolated from production scores."""
    contract = load_run_contract(run)
    responses, _ = load_pool_b(contract, root, 21)
    prompt_id = responses[0]["prompt_id"]
    group = [r for r in responses if r["prompt_id"] == prompt_id]
    if len(group) != 16:
        raise RuntimeError("smoke requires one complete 16-response group")
    judge = read_json(contract.config_path)["models"]["judge"]
    config = AuditScoreConfig(
        domain=contract.domain,
        method=contract.method,
        seed=contract.seed,
        judge_model=judge["model"],
        judge_revision=judge["revision"],
        max_output_tokens=int(judge["max_output_tokens"]),
        concurrency=16,
    )
    grader = VLLMChatAdapter(
        url, MODEL, output / "provider_cache", timeout_seconds=600, max_retries=4
    )
    evidence = []
    for step in (9, 21):
        rubric, _ = load_evaluator_rubrics(contract, root, step)
        records = score_pool(
            group,
            rubric,
            evaluator_checkpoint=str(step),
            policy_checkpoint="21",
            config=config,
            grader=grader,
            cache_dir=output / f"e{step}-cache",
        )
        if {r["response_id"] for r in records} != {r["response_id"] for r in group}:
            raise RuntimeError("smoke response identity changed")
        path = output / f"e{step}-scores.jsonl"
        write_score_receipts(path, records)
        evidence.append(artifact_record(path))
    write_json_atomic(
        output / "receipt.json",
        {
            "state": "passed",
            "grade_count": 32,
            "same_pool_b": True,
            "production_data": False,
            "bitwise_equivalence_claimed": False,
            "artifacts": evidence,
        },
    )


def spawn(command, log, environment=None):
    with log.open("ab") as stream:
        return subprocess.Popen(
            command,
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=environment,
            start_new_session=True,
        )


def scorer_command(run, root, output, plan, url):
    return [
        sys.executable,
        "scripts/phase1/score_regular_probe_adjacent.py",
        "--run-dir",
        str(run),
        "--artifact-root",
        str(root),
        "--output-root",
        str(output),
        "--cell-plan",
        str(plan),
        "--judge-urls",
        url,
        "--concurrency",
        "32",
        "--wait-timeout",
        "604800",
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--cell-plan", required=True, type=Path)
    parser.add_argument("--gpt-pid", required=True, type=int)
    parser.add_argument("--fresh-pid", required=True, type=int)
    parser.add_argument("--scorer-pid", required=True, type=int)
    parser.add_argument("--inference_b-url", default="http://127.0.0.1:28002")
    parser.add_argument("--port", type=int, default=28012)
    args = parser.parse_args()
    expected_host = os.environ.get("EXPECTED_TRAINER_HOST", "").strip()
    if expected_host and socket.gethostname().split(".")[0] != expected_host:
        parser.error("transition host differs from EXPECTED_TRAINER_HOST")
    if not MODEL_PATH.is_dir() or not os.access(VLLM, os.X_OK):
        parser.error("Prepare JUDGE_MODEL_PATH and JUDGE_VLLM_BIN before transitioning services")
    run, root = args.run_dir.resolve(), args.artifact_root.resolve()
    logs = root / "logs/trainer-judge-transition"
    logs.mkdir(parents=True, exist_ok=True)
    status_path = logs / "status.json"
    with (logs / "transition.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = {"state": "preflight", "created_at": time.time(), "training_restart": False}

        def publish(state, **fields):
            status.update(state=state, updated_at=time.time(), **fields)
            write_json_atomic(status_path, status, immutable=False)

        try:
            load_cell_plan(args.cell_plan)
            index = read_json(MODEL_PATH / "model.safetensors.index.json")
            if not all((MODEL_PATH / name).is_file() for name in set(index["weight_map"].values())):
                raise RuntimeError("cached judge model is incomplete")
            inference_b_identity = endpoint_identity(args.inference_b_url, MODEL, REVISION)
            gpt, fresh, old_scorer = [
                process_identity(pid) for pid in (args.gpt_pid, args.fresh_pid, args.scorer_pid)
            ]
            if (
                "--port" not in gpt["command"]
                or gpt["command"][gpt["command"].index("--port") + 1] != "28011"
                or not any("models--openai--gpt-oss-120b/snapshots/" in x for x in gpt["command"])
            ):
                raise RuntimeError("GPT PID is not the dedicated audit extractor")
            environment = (Path("/proc") / str(args.gpt_pid) / "environ").read_bytes().split(b"\0")
            if b"CUDA_VISIBLE_DEVICES=1" not in environment:
                raise RuntimeError("extractor PID is not scoped to GPU1")
            if (
                "scripts/phase1/run_regular_fresh_rubrics.py" not in fresh["command"]
                or "scripts/phase1/score_regular_probe_adjacent.py" not in old_scorer["command"]
                or not any(root.name in x for x in fresh["command"])
                or not any(root.name in x for x in old_scorer["command"])
            ):
                raise RuntimeError("producer/scorer PID does not belong to this audit")
            owned = [gpt]
            for p in Path("/proc").iterdir():
                if not p.name.isdigit():
                    continue
                try:
                    identity = process_identity(int(p.name))
                    if identity["ppid"] == gpt["pid"]:
                        owned.append(identity)
                except FileNotFoundError:
                    pass
            publish(
                "waiting_for_generation",
                owned_extractor_processes=owned,
                fresh_process=fresh,
                inference_b_scorer=old_scorer,
                plan=artifact_record(args.cell_plan),
                inference_b_identity=inference_b_identity,
            )
            while not generation_complete(root):
                if not alive(fresh):
                    raise RuntimeError(
                        "fresh producer exited before verified completion; extractor retained"
                    )
                time.sleep(15)
            while alive(fresh):
                time.sleep(2)
            publish("validating_completed_generation_provenance")
            contract = load_run_contract(run)
            for step in sorted({0} | REQUIRED_FRESH):
                load_pool_b(contract, root, step)
                if step:
                    load_evaluator_rubrics(contract, root, step)
            publish("stopping_completed_generation_models")
            stop_exact(gpt)
            deadline = time.monotonic() + 60
            while any(alive(x) for x in owned) and time.monotonic() < deadline:
                time.sleep(2)
            for identity in owned:
                stop_exact(identity)
            deadline = time.monotonic() + 120
            while True:
                used = int(
                    subprocess.check_output(
                        [
                            "nvidia-smi",
                            "-i",
                            "1",
                            "--query-gpu=memory.used",
                            "--format=csv,noheader,nounits",
                        ],
                        text=True,
                    ).strip()
                )
                if used < 512:
                    break
                if time.monotonic() > deadline:
                    raise RuntimeError(
                        f"GPU1 still has {used} MiB in use; no unknown process will be killed"
                    )
                time.sleep(5)
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": "1", "VLLM_USE_DEEP_GEMM": "0"}
            command = [
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
                "1",
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
                "127.0.0.1",
                "--port",
                str(args.port),
            ]
            with socket.socket() as port_check:
                port_check.bind(("127.0.0.1", args.port))
            server = spawn(command, logs / "qwen32b-server.log", env)
            publish(
                "starting_trainer_judge",
                judge_process=process_identity(server.pid),
                server_command=command,
            )
            url = f"http://127.0.0.1:{args.port}"
            deadline = time.monotonic() + 900
            while True:
                if server.poll() is not None:
                    raise RuntimeError(
                        "Trainer judge exited during startup; Inference B left untouched"
                    )
                try:
                    secondary = endpoint_identity(url, MODEL, REVISION)
                    break
                except Exception:
                    if time.monotonic() > deadline:
                        raise RuntimeError(
                            "Trainer judge readiness timed out; Inference B left untouched"
                        )
                    time.sleep(5)
            verify_replica(endpoint_identity(args.inference_b_url, MODEL, REVISION), secondary)
            publish("smoke_testing_trainer_judge", trainer_identity=secondary)
            smoke = logs / f"smoke-{server.pid}-{status['judge_process']['starttime']}"
            smoke.mkdir(exist_ok=False)
            smoke_judge(run, root, smoke, url)
            validate_artifact_record(status["plan"])
            publish("partitioning_columns")
            # Freeze Inference B before inventorying so no newly started column can be assigned to Trainer.
            if not alive(old_scorer):
                raise RuntimeError(
                    "original Inference B scorer is no longer active; refuse implicit replacement"
                )
            stop_exact(old_scorer)
            deadline = time.monotonic() + 60
            while alive(old_scorer):
                if time.monotonic() > deadline:
                    raise RuntimeError("old Inference B scorer failed to exit")
                time.sleep(1)
            inference_b_worker = trainer_worker = None
            try:
                from dynamic_rubric.phase1.judge_partition import partition_cells

                original = read_json(args.cell_plan)
                inference_b_root = root / "scores-adjacent"
                started, completed = set(), set()
                for cell in original["cells"]:
                    e, t = cell["evaluator_step"], cell["policy_step"]
                    directory = inference_b_root / f"policy-{t:06d}" / f"evaluator-{e:06d}"
                    if (directory / "manifest.json").is_file():
                        completed.add((e, t))
                        started.add(t)
                    if any((directory / "grade_cache").glob("*.json")):
                        started.add(t)
                partition = partition_cells(original, started, completed_cells=completed)
                inference_b_plan, trainer_plan = partition["inference_b"], partition["trainer"]
                if not trainer_plan["cells"]:
                    raise RuntimeError("no untouched policy columns remain for Trainer")
                inference_b_path, trainer_path = (
                    logs / "inference_b-plan.json",
                    logs / "trainer-plan.json",
                )
                write_json_atomic(inference_b_path, inference_b_plan)
                write_json_atomic(trainer_path, trainer_plan)
                inference_b_command = scorer_command(
                    run, root, inference_b_root, inference_b_path, args.inference_b_url
                )
                trainer_command = scorer_command(
                    run, root, root / "scores-trainer", trainer_path, url
                )
                write_json_atomic(
                    logs / "partition.json",
                    {
                        "schema_version": 1,
                        "original_plan": artifact_record(args.cell_plan),
                        "inference_b_plan": artifact_record(inference_b_path),
                        "trainer_plan": artifact_record(trainer_path),
                        "inference_b_output_root": str(inference_b_root),
                        "trainer_output_root": str(root / "scores-trainer"),
                        "unit": "whole_policy_column",
                        "bitwise_equivalence_claimed": False,
                        "summary": partition["summary"],
                        "smoke_receipt": artifact_record(smoke / "receipt.json"),
                        "inference_b_command": inference_b_command,
                        "trainer_command": trainer_command,
                    },
                )
                inference_b_worker = spawn(inference_b_command, logs / "inference_b-scoring.log")
                trainer_worker = spawn(trainer_command, logs / "trainer-scoring.log")
            except Exception:
                if inference_b_worker is not None:
                    stop_exact(process_identity(inference_b_worker.pid))
                    inference_b_worker.wait(timeout=60)
                # If no replacement took ownership, resume the exact original Inference B command.
                with (root / "scores-adjacent/scorer.lock").open("a") as score_lock:
                    try:
                        fcntl.flock(score_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        pass
                    else:
                        fcntl.flock(score_lock.fileno(), fcntl.LOCK_UN)
                        spawn(old_scorer["command"], logs / "inference_b-rollback.log")
                raise
            publish(
                "parallel_workers_started",
                inference_b_worker=process_identity(inference_b_worker.pid),
                trainer_worker=process_identity(trainer_worker.pid),
                inference_b_cell_count=len(inference_b_plan["cells"]),
                trainer_cell_count=len(trainer_plan["cells"]),
            )
            deadline = time.monotonic() + 900
            while not any(
                (root / "scores-trainer").glob("policy-*/evaluator-*/grade_cache/*.json")
            ):
                if inference_b_worker.poll() is not None or trainer_worker.poll() is not None:
                    raise RuntimeError(
                        "a scoring worker exited; inspect shard logs and preserve existing receipts"
                    )
                if time.monotonic() > deadline:
                    raise RuntimeError("Trainer scorer produced no grades within 15 minutes")
                time.sleep(5)
            publish("parallel_scoring_verified")
        except Exception as error:
            publish("failed", failed_phase=status["state"], error=repr(error))
            raise


if __name__ == "__main__":
    main()
