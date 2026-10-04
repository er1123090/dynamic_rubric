#!/usr/bin/env python3
"""Queue a three-seed HealthBench generation-variability audit on Trainer GPU 1."""

from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from typing import Any, Callable, Mapping, Sequence

from dynamic_rubric.artifacts import read_jsonl, write_json_atomic
from scripts.phase1 import evaluate_final_policies as evaluation
from scripts.phase1 import run_policy_checkpoint_trajectory as trajectory


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    ROOT
    / "configs/evaluation/medicine_online5_healthbench500_generation_variability_trainer_gpu1_20260928.yaml"
)
QWEN_REVISION = "9216db5781bf21249d130ec9da846c4624c16137"
QWEN_MODEL = "Qwen/Qwen3-32B"
QWEN_PATH = ROOT / "models/Qwen3-32B"


class QueueError(RuntimeError):
    """Raised when the queued audit cannot safely continue."""


def _log(message: str) -> None:
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} {message}", flush=True)


def _publish(path: Path, state: str, **fields: Any) -> None:
    write_json_atomic(
        path,
        {
            "schema_version": 1,
            "state": state,
            "updated_at_unix": time.time(),
            "launcher_pid": os.getpid(),
            **fields,
        },
        immutable=False,
    )


def _gpu_memory_used_mib(gpu: int) -> int:
    result = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(gpu),
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(result.stdout.strip())


def _matching_processes(fragment: str) -> list[dict[str, Any]]:
    matches = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = [
                part
                for part in (entry / "cmdline").read_bytes().decode(errors="replace").split("\0")
                if part
            ]
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if fragment in " ".join(command):
            matches.append({"pid": int(entry.name), "command": command})
    return sorted(matches, key=lambda row: row["pid"])


def _wait_for_gpu(
    runtime: Mapping[str, Any], status_path: Path, *, phase: str
) -> None:
    gpu = int(runtime["gpu"])
    fragment = str(runtime["blocking_command_fragment"])
    threshold = int(runtime.get("free_memory_threshold_mib", 4096))
    poll_seconds = float(runtime.get("poll_seconds", 60))
    while True:
        blockers = _matching_processes(fragment)
        used = _gpu_memory_used_mib(gpu)
        if not blockers and used <= threshold:
            _publish(
                status_path,
                "gpu_ready",
                phase=phase,
                gpu=gpu,
                memory_used_mib=used,
                threshold_mib=threshold,
            )
            return
        _publish(
            status_path,
            "waiting_for_gpu",
            phase=phase,
            gpu=gpu,
            memory_used_mib=used,
            threshold_mib=threshold,
            blockers=blockers,
        )
        _log(
            f"waiting phase={phase} gpu={gpu} used_mib={used} "
            f"blocking_pids={[row['pid'] for row in blockers]}"
        )
        time.sleep(poll_seconds)


def _assert_port_free(port: int) -> None:
    with socket.socket() as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except OSError as error:
            raise QueueError(f"local port {port} is already occupied") from error


def _wait_server(
    base_url: str,
    served_model: str,
    process: subprocess.Popen,
    timeout_seconds: float,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    endpoint = f"{base_url.rstrip('/')}/models"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise QueueError(
                f"vLLM exited before ready: model={served_model} code={process.returncode}"
            )
        try:
            with urllib.request.urlopen(endpoint, timeout=5) as response:
                payload = json.load(response)
            returned = {str(item.get("id")) for item in payload.get("data", [])}
            if returned == {served_model}:
                return
        except Exception:
            pass
        time.sleep(3)
    raise QueueError(f"vLLM readiness timed out: {served_model}")


def _stop_owned(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=120)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=30)


def _run_server(
    *,
    command: Sequence[str],
    environment: Mapping[str, str],
    base_url: str,
    served_model: str,
    log_path: Path,
    startup_timeout_seconds: float,
    task: Callable[[], None],
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab", buffering=0) as stream:
        process = subprocess.Popen(
            list(command),
            cwd=ROOT,
            env=dict(environment),
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    try:
        _wait_server(base_url, served_model, process, startup_timeout_seconds)
        task()
    finally:
        _stop_owned(process)


def _repeat_configs(config_path: Path) -> list[dict[str, Any]]:
    base = evaluation.load_config(config_path.resolve(), limit=None)
    experiment = base.get("experiment")
    if not isinstance(experiment, Mapping):
        raise QueueError("experiment mapping is required")
    seeds = experiment.get("generation_seeds")
    if not isinstance(seeds, list) or len(seeds) < 2:
        raise QueueError("at least two generation seeds are required")
    configs = []
    base_output = Path(str(base["output_root"]))
    for raw_seed in seeds:
        seed = int(raw_seed)
        config = copy.deepcopy(base)
        config["seed"] = seed
        config["generation"]["seed"] = seed
        config["output_root"] = str((base_output / f"seed-{seed}").resolve())
        configs.append(config)
    return configs


def _prepare(configs: Sequence[Mapping[str, Any]]) -> list[Path]:
    outputs = [evaluation.prepare(config) for config in configs]
    identities = []
    for output in outputs:
        rows = read_jsonl(output / "prepared" / "prompts.jsonl")
        identities.append([(row["dataset"], row["prompt_id"]) for row in rows])
    if not identities or any(identity != identities[0] for identity in identities[1:]):
        raise QueueError("generation repeats do not share identical ordered prompt IDs")
    if len(identities[0]) != 500:
        raise QueueError(f"expected 500 prompts, found {len(identities[0])}")
    return outputs


def _responses_complete(
    config: Mapping[str, Any], model_name: str
) -> bool:
    root, _, prompts = evaluation._load_run(config)
    return trajectory._response_complete(root, model_name, prompts)


def _grades_complete(config: Mapping[str, Any]) -> bool:
    root, _, prompts = evaluation._load_run(config)
    return all(
        trajectory._grade_complete(root, model_name, prompts)
        for model_name in evaluation.ordered_model_names(config)
    )


def _policy_command(
    config: Mapping[str, Any], model_name: str
) -> tuple[list[str], str, int]:
    runtime = config["runtime"]
    model = config["models"][model_name]
    model_path = Path(str(model["local_path"]))
    if not (model_path / "config.json").is_file() or not list(model_path.glob("*.safetensors")):
        raise QueueError(f"incomplete local checkpoint export: {model_path}")
    port = int(runtime["policy_port"])
    command = [
        str(runtime["vllm"]),
        "serve",
        str(model_path),
        "--served-model-name",
        str(model["served_model"]),
        "--tensor-parallel-size",
        "1",
        "--dtype",
        "bfloat16",
        "--gpu-memory-utilization",
        str(runtime["policy_gpu_memory_utilization"]),
        "--max-model-len",
        str(runtime["max_model_len"]),
        "--max-num-seqs",
        str(config["generation"].get("max_in_flight", 64)),
        "--max-num-batched-tokens",
        "16384",
        "--enable-prefix-caching",
        "--generation-config",
        "vllm",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    return command, str(model["served_model"]), port


def _judge_command(config: Mapping[str, Any]) -> tuple[list[str], int]:
    runtime = config["runtime"]
    if not (QWEN_PATH / "config.json").is_file() or not list(QWEN_PATH.glob("*.safetensors")):
        raise QueueError(f"incomplete local judge snapshot: {QWEN_PATH}")
    port = int(runtime["judge_port"])
    command = [
        str(runtime["vllm"]),
        "serve",
        str(QWEN_PATH),
        "--served-model-name",
        QWEN_MODEL,
        "--revision",
        QWEN_REVISION,
        "--tokenizer-revision",
        QWEN_REVISION,
        "--tensor-parallel-size",
        "1",
        "--dtype",
        "bfloat16",
        "--gpu-memory-utilization",
        str(runtime["judge_gpu_memory_utilization"]),
        "--max-model-len",
        str(runtime["max_model_len"]),
        "--max-num-seqs",
        str(config["grading"].get("max_in_flight", 64)),
        "--max-num-batched-tokens",
        "16384",
        "--enable-prefix-caching",
        "--generation-config",
        "vllm",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    return command, port


def _generate_all(
    configs: Sequence[Mapping[str, Any]], output_root: Path, status_path: Path
) -> None:
    runtime = configs[0]["runtime"]
    environment = dict(os.environ)
    environment.update(
        CUDA_VISIBLE_DEVICES=str(runtime["gpu"]),
        OMP_NUM_THREADS="1",
        PYTHONUNBUFFERED="1",
    )
    models = evaluation.ordered_model_names(configs[0])
    for index, model_name in enumerate(models):
        if all(_responses_complete(config, model_name) for config in configs):
            _publish(
                status_path,
                "generation_model_already_complete",
                model=model_name,
                completed_models=index + 1,
                total_models=len(models),
            )
            continue
        _wait_for_gpu(runtime, status_path, phase=f"before_generation:{model_name}")
        command, served_model, port = _policy_command(configs[0], model_name)
        _assert_port_free(port)
        base_url = str(configs[0]["generation"]["base_url"])
        _publish(
            status_path,
            "starting_policy_server",
            model=model_name,
            completed_models=index,
            total_models=len(models),
        )

        def task() -> None:
            for config in configs:
                seed = int(config["generation"]["seed"])
                _publish(
                    status_path,
                    "generating",
                    model=model_name,
                    generation_seed=seed,
                    completed_models=index,
                    total_models=len(models),
                )
                evaluation.generate(config, model_name)
                if not _responses_complete(config, model_name):
                    raise QueueError(f"response inventory incomplete: {model_name}/seed-{seed}")

        _run_server(
            command=command,
            environment=environment,
            base_url=base_url,
            served_model=served_model,
            log_path=output_root / "launcher_logs" / f"policy-{model_name}.log",
            startup_timeout_seconds=float(runtime["startup_timeout_seconds"]),
            task=task,
        )
        _publish(
            status_path,
            "generation_model_complete",
            model=model_name,
            completed_models=index + 1,
            total_models=len(models),
        )
    _publish(status_path, "generation_complete", completed_models=len(models))


def _grade_all(
    configs: Sequence[Mapping[str, Any]], output_root: Path, status_path: Path
) -> None:
    runtime = configs[0]["runtime"]
    if all(_grades_complete(config) for config in configs):
        for config in configs:
            evaluation.summarize(config)
        _publish(status_path, "grading_already_complete")
        return
    _wait_for_gpu(runtime, status_path, phase="before_grading")
    command, port = _judge_command(configs[0])
    _assert_port_free(port)
    base_url = f"http://127.0.0.1:{port}/v1"
    environment = dict(os.environ)
    environment.update(
        CUDA_VISIBLE_DEVICES=str(runtime["gpu"]),
        OMP_NUM_THREADS="1",
        PYTHONUNBUFFERED="1",
    )
    _publish(status_path, "starting_judge_server")

    def task() -> None:
        for index, config in enumerate(configs):
            seed = int(config["generation"]["seed"])
            _publish(
                status_path,
                "grading",
                generation_seed=seed,
                completed_repeats=index,
                total_repeats=len(configs),
            )
            evaluation.grade(config, runtime_base_urls=[base_url])
            if not _grades_complete(config):
                raise QueueError(f"grade inventory incomplete: seed-{seed}")
            evaluation.summarize(config)
        _publish(status_path, "grading_complete", completed_repeats=len(configs))

    _run_server(
        command=command,
        environment=environment,
        base_url=base_url,
        served_model=QWEN_MODEL,
        log_path=output_root / "launcher_logs" / "qwen3-32b-judge.log",
        startup_timeout_seconds=float(runtime["startup_timeout_seconds"]),
        task=task,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve()
    configs = _repeat_configs(config_path)
    runtime = configs[0]["runtime"]
    if socket.gethostname().split(".")[0] != str(runtime["hostname"]):
        raise QueueError("this audit is restricted to Trainer")
    if not Path("/.dockerenv").is_file():
        raise QueueError("this audit must run inside the existing container")
    if int(runtime["gpu"]) != 1:
        raise QueueError("this audit is restricted to Trainer GPU 1")

    output_root = Path(str(evaluation.load_config(config_path, None)["output_root"]))
    output_root.mkdir(parents=True, exist_ok=True)
    status_path = output_root / "queue_status.json"
    lock_path = output_root / "launcher.lock"
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise QueueError("another variability launcher already holds the lock") from error
        try:
            _publish(status_path, "preparing", config=str(config_path))
            outputs = _prepare(configs)
            _publish(
                status_path,
                "prepared",
                run_directories=[str(path) for path in outputs],
                generation_seeds=[int(config["generation"]["seed"]) for config in configs],
            )
            _generate_all(configs, output_root, status_path)
            _grade_all(configs, output_root, status_path)
            analysis_script = ROOT / "scripts/phase1/analyze_healthbench_generation_variability.py"
            _publish(status_path, "analyzing")
            subprocess.run(
                [sys.executable, str(analysis_script), "--config", str(config_path)],
                cwd=ROOT,
                check=True,
            )
            _publish(status_path, "complete")
            _log("generation-variability audit complete")
        except BaseException as error:
            _publish(status_path, "failed", error=repr(error))
            raise


if __name__ == "__main__":
    main()
