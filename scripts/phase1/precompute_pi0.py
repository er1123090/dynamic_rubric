#!/usr/bin/env python3
"""Build the immutable OnlineRubrics pi0 cache from one launch YAML."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dynamic_rubric.phase1.config import Phase1ConfigError, load_yaml_config  # noqa: E402
from dynamic_rubric.training.live_online import _directory_tree_hash  # noqa: E402


class PrecomputeError(RuntimeError):
    pass


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PrecomputeError(f"{name} must be a mapping")
    return value


def _positive_int(value: object, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PrecomputeError(f"{name} must be an integer")
    if value < minimum:
        raise PrecomputeError(f"{name} must be >= {minimum}")
    return value


def _fraction(value: object, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise PrecomputeError(f"{name} must be numeric") from error
    if not 0 < number <= 1:
        raise PrecomputeError(f"{name} must be in (0, 1]")
    return number


def _root_path(value: object, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise PrecomputeError(f"{name} must be a non-empty path")
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


@dataclass(frozen=True)
class Plan:
    config: str
    domain: str
    model_name: str
    model_revision: str
    model_path: str
    train_path: str
    prompt_count: int
    seed: int
    output_dir: str
    gpus: tuple[int, ...]
    tensor_parallel_size: int
    gpu_memory_utilization: float
    max_model_len: int
    max_num_seqs: int
    max_num_batched_tokens: int
    upstream_port: int
    proxy_port: int
    concurrency: int
    progress_every: int
    free_memory_threshold_mib: int
    vllm_bin: str
    runtime_python: str


def load_plan(config_path: Path) -> Plan:
    raw = load_yaml_config(config_path)
    if str(raw.get("method", "")) != "online_rubrics":
        raise PrecomputeError("pi0 precompute requires method=online_rubrics")
    domain = str(raw.get("domain", ""))
    if not domain:
        raise PrecomputeError("domain must be recorded")
    data = _mapping(raw.get("data"), "data")
    policy = _mapping(_mapping(raw.get("models"), "models").get("policy"), "models.policy")
    infrastructure = _mapping(raw.get("infrastructure"), "infrastructure")
    pi0 = _mapping(infrastructure.get("pi0_control"), "infrastructure.pi0_control")
    launch = _mapping(raw.get("launch"), "launch")
    launch_environment = _mapping(launch.get("environment", {}), "launch.environment")

    raw_gpus = pi0.get("gpus")
    if not isinstance(raw_gpus, Sequence) or isinstance(raw_gpus, (str, bytes)):
        raise PrecomputeError("infrastructure.pi0_control.gpus must be a sequence")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in raw_gpus):
        raise PrecomputeError("pi0 GPUs must be integer indices")
    gpus = tuple(raw_gpus)
    if not gpus or any(value < 0 for value in gpus) or len(set(gpus)) != len(gpus):
        raise PrecomputeError("pi0 GPUs must be unique non-negative indices")
    tensor_parallel_size = _positive_int(
        pi0.get("tensor_parallel_size", len(gpus)), "pi0 tensor_parallel_size"
    )
    if tensor_parallel_size != len(gpus):
        raise PrecomputeError("pi0 tensor_parallel_size must equal the number of GPUs")
    upstream_port = _positive_int(pi0.get("upstream_port", 28101), "pi0 upstream_port")
    proxy_port = _positive_int(pi0.get("proxy_port", 28102), "pi0 proxy_port")
    if upstream_port > 65535 or proxy_port > 65535 or upstream_port == proxy_port:
        raise PrecomputeError("pi0 ports must be distinct integers in the range 1..65535")

    cache_value = launch_environment.get("ONLINE_CONTROL_CACHE_DIR")
    if cache_value is None:
        cache_value = f"outputs/{domain}/shared/seed-{int(raw.get('seed', -1))}/pi0_control_cache"
    return Plan(
        config=str(config_path.resolve()),
        domain=domain,
        model_name=str(policy.get("model", "")),
        model_revision=str(policy.get("revision", "")),
        model_path=str(_root_path(policy.get("local_snapshot"), "models.policy.local_snapshot")),
        train_path=str(_root_path(data.get("train_path"), "data.train_path")),
        prompt_count=_positive_int(data.get("train_prompt_count"), "data.train_prompt_count"),
        seed=_positive_int(raw.get("seed"), "seed", minimum=0),
        output_dir=str(_root_path(cache_value, "ONLINE_CONTROL_CACHE_DIR")),
        gpus=gpus,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=_fraction(
            pi0.get("gpu_memory_utilization", 0.45), "pi0 gpu_memory_utilization"
        ),
        max_model_len=_positive_int(pi0.get("max_model_len", 7680), "pi0 max_model_len"),
        max_num_seqs=_positive_int(pi0.get("max_num_seqs", 64), "pi0 max_num_seqs"),
        max_num_batched_tokens=_positive_int(
            pi0.get("max_num_batched_tokens", 16384), "pi0 max_num_batched_tokens"
        ),
        upstream_port=upstream_port,
        proxy_port=proxy_port,
        concurrency=_positive_int(pi0.get("concurrency", 16), "pi0 concurrency"),
        progress_every=_positive_int(pi0.get("progress_every", 25), "pi0 progress_every"),
        free_memory_threshold_mib=_positive_int(
            pi0.get("free_memory_threshold_mib", 1024), "pi0 free_memory_threshold_mib", minimum=0
        ),
        vllm_bin=str(_root_path(pi0.get("vllm_bin", ".venvs/judge/bin/vllm"), "pi0 vllm_bin")),
        runtime_python=str(
            _root_path(
                pi0.get("runtime_python", launch.get("entry_python", ".venv/bin/python")),
                "pi0 runtime_python",
            )
        ),
    )


def validate_files(plan: Plan) -> None:
    required = {
        "policy model snapshot": Path(plan.model_path),
        "training data": Path(plan.train_path),
        "vLLM executable": Path(plan.vllm_bin),
        "runtime Python": Path(plan.runtime_python),
    }
    missing = [f"{label}: {path}" for label, path in required.items() if not path.exists()]
    for label in ("vLLM executable", "runtime Python"):
        path = required[label]
        if path.exists() and (not path.is_file() or not os.access(path, os.X_OK)):
            missing.append(f"{label} is not executable: {path}")
    if missing:
        raise PrecomputeError("missing/invalid required paths:\n  " + "\n  ".join(missing))
    with Path(plan.train_path).open(encoding="utf-8") as handle:
        count = sum(1 for line in handle if line.strip())
    if count != plan.prompt_count:
        raise PrecomputeError(
            f"training data has {count} non-empty rows; expected {plan.prompt_count}"
        )


def _assert_gpus_free(plan: Plan) -> None:
    for gpu in plan.gpus:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={gpu}",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        used = int(result.stdout.strip())
        if used >= plan.free_memory_threshold_mib:
            raise PrecomputeError(
                f"GPU {gpu} is already using {used} MiB (limit {plan.free_memory_threshold_mib} MiB); "
                "no processes were stopped"
            )


def _wait_http(url: str, process: subprocess.Popen[Any], log_path: Path, attempts: int) -> None:
    for _ in range(attempts):
        if process.poll() is not None:
            tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-100:]
            raise PrecomputeError(
                f"service exited early ({process.returncode}):\n" + "\n".join(tail)
            )
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if 200 <= response.status < 300:
                    return
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(1)
    raise PrecomputeError(f"service did not become ready: {url}")


def _stop_owned(process: subprocess.Popen[Any] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def run(plan: Plan) -> None:
    validate_files(plan)
    _assert_gpus_free(plan)
    output_dir = Path(plan.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_hash = _directory_tree_hash(Path(plan.model_path))
    child_env = os.environ.copy()
    child_env["CUDA_VISIBLE_DEVICES"] = ",".join(str(value) for value in plan.gpus)
    child_env["PYTHONPATH"] = str(SRC) + (
        os.pathsep + child_env["PYTHONPATH"] if child_env.get("PYTHONPATH") else ""
    )
    vllm_log = output_dir / "policy-vllm.log"
    proxy_log = output_dir / "policy-proxy.log"
    precompute_log = output_dir / "precompute.log"
    vllm_process: subprocess.Popen[Any] | None = None
    proxy_process: subprocess.Popen[Any] | None = None
    try:
        with vllm_log.open("w", encoding="utf-8") as log:
            vllm_process = subprocess.Popen(
                [
                    plan.vllm_bin,
                    "serve",
                    plan.model_path,
                    "--served-model-name",
                    plan.model_name,
                    "--tensor-parallel-size",
                    str(plan.tensor_parallel_size),
                    "--dtype",
                    "auto",
                    "--gpu-memory-utilization",
                    str(plan.gpu_memory_utilization),
                    "--max-model-len",
                    str(plan.max_model_len),
                    "--max-num-seqs",
                    str(plan.max_num_seqs),
                    "--max-num-batched-tokens",
                    str(plan.max_num_batched_tokens),
                    "--enable-prefix-caching",
                    "--generation-config",
                    "vllm",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(plan.upstream_port),
                ],
                env=child_env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        _wait_http(f"http://127.0.0.1:{plan.upstream_port}/v1/models", vllm_process, vllm_log, 180)
        with proxy_log.open("w", encoding="utf-8") as log:
            proxy_process = subprocess.Popen(
                [
                    plan.runtime_python,
                    "-m",
                    "dynamic_rubric.services.vllm_policy_proxy",
                    "--upstream",
                    f"http://127.0.0.1:{plan.upstream_port}",
                    "--model-path",
                    plan.model_path,
                    "--served-model",
                    plan.model_name,
                    "--model-revision",
                    plan.model_revision,
                    "--tokenizer-revision",
                    plan.model_revision,
                    "--checkpoint-hash",
                    checkpoint_hash,
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(plan.proxy_port),
                ],
                env=child_env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        _wait_http(
            f"http://127.0.0.1:{plan.proxy_port}/dynamic-rubric/identity",
            proxy_process,
            proxy_log,
            60,
        )
        command = [
            plan.runtime_python,
            "-m",
            "dynamic_rubric.phase1.pi0_cache",
            "--train-jsonl",
            plan.train_path,
            "--output-dir",
            plan.output_dir,
            "--base-url",
            f"http://127.0.0.1:{plan.proxy_port}",
            "--model",
            plan.model_name,
            "--revision",
            plan.model_revision,
            "--tokenizer-revision",
            plan.model_revision,
            "--checkpoint-hash",
            checkpoint_hash,
            "--seed",
            str(plan.seed),
            "--prompt-count",
            str(plan.prompt_count),
            "--concurrency",
            str(plan.concurrency),
            "--progress-every",
            str(plan.progress_every),
        ]
        with precompute_log.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command, env=child_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="")
                log.write(line)
            code = process.wait()
        if code:
            raise PrecomputeError(f"pi0 cache generation exited with code {code}")
    finally:
        _stop_owned(proxy_process)
        _stop_owned(vllm_process)

    manifests = sorted(output_dir.glob("manifest-*.json"))
    if len(manifests) != 1:
        raise PrecomputeError(f"expected exactly one sealed pi0 manifest, found {len(manifests)}")
    from dynamic_rubric.phase1.pi0_cache import ImmutablePi0Cache

    cache = ImmutablePi0Cache(manifests[0], expected_prompt_count=plan.prompt_count)
    print(cache.manifest_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--check", action="store_true", help="validate and print the plan only")
    args = parser.parse_args()
    try:
        plan = load_plan(args.config)
        if args.check:
            validate_files(plan)
            print(json.dumps(asdict(plan), indent=2, sort_keys=True))
        else:
            run(plan)
    except (OSError, Phase1ConfigError, PrecomputeError, subprocess.SubprocessError) as error:
        print(f"pi0 precompute error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
