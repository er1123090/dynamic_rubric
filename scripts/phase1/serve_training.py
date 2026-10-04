"""Start one YAML-selected inference instance locally; --check never starts a server.

Run this on the inference machine. Host labels in YAML do not initiate SSH.
Static scoring includes its required identity/score proxy; Online/Evo use chat APIs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from dynamic_rubric.phase1.config import load_yaml_config  # noqa: E402


def _path(value: object, root: Path, *, label: str = "path") -> Path:
    if value is None or not str(value).strip():
        raise ValueError(f"{label} must be configured with a non-empty path")
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def _integer(value: object, name: str, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return value


def prepare_service(config: Path, service: str, instance: int = 0, *, root: Path = ROOT) -> dict:
    raw = load_yaml_config(config)
    method = raw["method"]
    if method not in {"static_r0_matched", "online_rubrics", "evorubrics"}:
        raise ValueError(f"unsupported method: {method}")
    if service not in {"judge", "extractor"} or (
        service == "extractor" and method != "online_rubrics"
    ):
        raise ValueError("extractor is only used by OnlineRubrics")
    model = raw["models"][service]
    model_path = _path(model.get("local_snapshot"), root, label=f"models.{service}.local_snapshot")
    if not model_path.is_dir():
        raise ValueError(f"missing inference model: {model_path}")
    key = "qwen3_32b" if model["model"] == "Qwen/Qwen3-32B" else "gpt_oss_120b"
    instances = raw["infrastructure"]["services"][key]["instances"]
    if isinstance(instance, bool) or instance < 0 or instance >= len(instances):
        raise ValueError(f"instance must be between 0 and {len(instances) - 1}")
    selected = instances[instance]
    gpus = selected["gpus"]
    if (
        not isinstance(gpus, list)
        or not gpus
        or any(isinstance(g, bool) or not isinstance(g, int) or g < 0 for g in gpus)
        or len(set(gpus)) != len(gpus)
    ):
        raise ValueError("instance.gpus must be unique nonnegative GPU IDs")
    tp = _integer(selected["tensor_parallel_size"], "tensor_parallel_size")
    if tp != len(gpus):
        raise ValueError("tensor_parallel_size must match the instance GPU count")
    binary = _path(selected.get("vllm_bin", ".venvs/judge/bin/vllm"), root)
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise ValueError(f"missing executable vllm_bin: {binary}")
    port = _integer(selected["port"], "port", 65535)
    bind = str(selected.get("bind_host", "127.0.0.1"))
    memory = float(selected.get("gpu_memory_utilization", 0.45))
    if not math.isfinite(memory) or not 0 < memory <= 1:
        raise ValueError("gpu_memory_utilization must be in (0, 1]")
    command = [
        str(binary),
        "serve",
        str(model_path),
        "--served-model-name",
        model["model"],
        "--tensor-parallel-size",
        str(tp),
        "--dtype",
        str(selected.get("dtype", "auto")),
        "--gpu-memory-utilization",
        str(memory),
        "--host",
        bind,
        "--port",
        str(port),
    ]
    for key, default in (
        ("max_model_len", 32768),
        ("max_num_seqs", 16),
        ("max_num_batched_tokens", 2048),
    ):
        command.extend(
            ["--" + key.replace("_", "-"), str(_integer(selected.get(key, default), key))]
        )
    if model.get("revision"):
        command.extend(
            [
                "--revision",
                str(model["revision"]),
                "--tokenizer-revision",
                str(model.get("tokenizer_revision", model["revision"])),
            ]
        )
    command.extend(["--enable-prefix-caching", "--generation-config", "vllm", "--enforce-eager"])
    local_host = "127.0.0.1" if bind == "0.0.0.0" else bind
    local_url = f"http://{local_host}:{port}"
    commands = [command]
    ports = [port]
    if method == "static_r0_matched":
        proxy_port = _integer(selected["proxy_port"], "proxy_port", 65535)
        if proxy_port == port:
            raise ValueError("proxy_port must differ from the raw vLLM port")
        python = _path(selected.get("python", ".venvs/judge/bin/python"), root)
        if not python.is_file() or not os.access(python, os.X_OK):
            raise ValueError(f"missing executable proxy Python: {python}")
        commands.append(
            [
                str(python),
                "-m",
                "dynamic_rubric.services.vllm_score_proxy",
                "--upstream",
                local_url,
                "--model-path",
                str(model_path),
                "--served-model",
                model["model"],
                "--model-revision",
                model["revision"],
                "--tokenizer-revision",
                model["tokenizer_revision"],
                "--host",
                bind,
                "--port",
                str(proxy_port),
                "--cache-dir",
                str(
                    _path(
                        selected.get(
                            "cache_dir", f"artifacts/provider_cache/static-judge-{instance}"
                        ),
                        root,
                    )
                ),
            ]
        )
        ports.append(proxy_port)
    return {
        "method": method,
        "service": service,
        "instance": instance,
        "host_label": selected.get("host", "local"),
        "bind_host": bind,
        "gpus": gpus,
        "ports": ports,
        "commands": commands,
        "upstream_health": local_url + "/v1/models",
        "client_url": f"http://{local_host}:{ports[-1]}",
    }


def _wait_ready(url: str, process: subprocess.Popen, timeout: float = 600) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"vLLM exited before becoming ready (code {process.returncode})")
        try:
            with urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except OSError:
            pass
        time.sleep(1)
    raise RuntimeError(f"service startup timed out: {url}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--service", choices=("judge", "extractor"), required=True)
    parser.add_argument("--instance", type=int, default=0)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    try:
        plan = prepare_service(args.config, args.service, args.instance)
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.exit(2, f"service preflight failed: {error}\n")
    print(json.dumps(plan, indent=2), flush=True)
    if args.check:
        return 0
    # Refuse occupied listening ports; never reuse/stop another process.
    for port in plan["ports"]:
        with socket.socket() as sock:
            sock.bind((plan["bind_host"], port))
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, plan["gpus"]))
    environment["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + environment.get("PYTHONPATH", "")
    environment.setdefault("VLLM_USE_DEEP_GEMM", "0")
    processes = []

    def stop(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    try:
        for index, command in enumerate(plan["commands"]):
            if index:
                _wait_ready(plan["upstream_health"], processes[0])
            processes.append(
                subprocess.Popen(command, cwd=ROOT, env=environment, start_new_session=True)
            )
        while all(process.poll() is None for process in processes):
            time.sleep(1)
        return next((p.returncode or 0 for p in processes if p.poll() is not None), 1)
    except KeyboardInterrupt:
        return 130
    finally:
        for process in processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process in processes:
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


if __name__ == "__main__":
    raise SystemExit(main())
