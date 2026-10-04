#!/usr/bin/env python3
"""Run one-checkpoint-at-a-time policy generation and concurrent resumable grading."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed

from dynamic_rubric.artifacts import read_jsonl, write_json_atomic
try:
    from scripts.phase1 import evaluate_final_policies as evaluation
except ModuleNotFoundError:
    import evaluate_final_policies as evaluation


def _write_status(root: Path, lane: str, value: Mapping[str, Any]) -> None:
    write_json_atomic(
        root / "status" / f"{lane}.json",
        {"schema_version": 1, "updated_at_unix": time.time(), **value},
        immutable=False,
    )


def _response_complete(root: Path, model_name: str, prompts: Sequence[Mapping[str, Any]]) -> bool:
    return all(
        evaluation._response_path(root, model_name, row["dataset"], row["prompt_id"]).is_file()
        for row in prompts
    )


def _grade_complete(root: Path, model_name: str, prompts: Sequence[Mapping[str, Any]]) -> bool:
    return all(
        evaluation._criterion_path(
            root, model_name, row["dataset"], row["prompt_id"], criterion["criterion_id"]
        ).is_file()
        for row in prompts
        for criterion in row["criteria"]
    )


def _download_model(model_name: str, spec: Mapping[str, Any], staging_root: Path) -> tuple[Path, bool]:
    configured = spec.get("local_path")
    if configured and Path(str(configured)).is_dir():
        return Path(str(configured)).resolve(), False
    repo, revision = str(spec["repo_id"]), str(spec["revision"])
    destination = staging_root / model_name
    sentinel = destination / ".trajectory_checkpoint.json"
    identity = {"repo_id": repo, "revision": revision, "artifact": spec["artifact"]}
    if sentinel.is_file() and json.loads(sentinel.read_text()) == identity:
        return destination, True
    destination.mkdir(parents=True, exist_ok=True)
    command = [
        "hf", "download", repo, "--revision", revision, "--local-dir", str(destination),
        "--exclude", "original_checkpoint/**", "--quiet",
    ]
    subprocess.run(command, check=True)
    required = [destination / "config.json", destination / "tokenizer_config.json"]
    if not all(path.is_file() for path in required) or not list(destination.glob("*.safetensors")):
        raise evaluation.EvaluationError(f"incomplete model download: {model_name}")
    subprocess.run(
        ["hf", "cache", "verify", repo, "--revision", revision, "--local-dir", str(destination)],
        check=True,
    )
    write_json_atomic(sentinel, identity, immutable=False)
    return destination, True


def _wait_server(base_url: str, model_name: str, process: subprocess.Popen, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    endpoint = f"{base_url.rstrip('/')}/models"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise evaluation.EvaluationError(
                f"vLLM exited before becoming ready for {model_name}: code={process.returncode}"
            )
        try:
            with urllib.request.urlopen(endpoint, timeout=5) as response:
                payload = json.load(response)
            names = {item.get("id") for item in payload.get("data", [])}
            if model_name in names:
                return
        except Exception:
            pass
        time.sleep(3)
    raise evaluation.EvaluationError(f"timed out waiting for vLLM model {model_name}")


def _stop_server(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=90)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=30)


def download_all(config: Mapping[str, Any], workers: int | None = None) -> None:
    root, _, prompts = evaluation._load_run(config)
    runtime = config["runtime"]
    staging_root = Path(str(runtime["temporary_download_root"]))
    staging_root.mkdir(parents=True, exist_ok=True)
    models = [
        name
        for name in evaluation.ordered_model_names(config)
        if not _response_complete(root, name, prompts)
    ]
    worker_count = int(workers or runtime.get("download_workers", 8))
    if worker_count <= 0:
        raise evaluation.EvaluationError("download workers must be positive")
    _write_status(
        root,
        "prefetch",
        {
            "state": "downloading",
            "completed_models": 0,
            "total_models": len(models),
            "download_workers": min(worker_count, len(models)) if models else 0,
        },
    )

    def fetch(model_name: str) -> tuple[str, str]:
        path, _ = _download_model(model_name, config["models"][model_name], staging_root)
        return model_name, str(path)

    completed = 0
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {executor.submit(fetch, name): name for name in models}
        for future in as_completed(futures):
            model_name, model_path = future.result()
            completed += 1
            _write_status(
                root,
                "prefetch",
                {
                    "state": "downloading",
                    "last_completed_model": model_name,
                    "last_completed_path": model_path,
                    "completed_models": completed,
                    "total_models": len(models),
                    "download_workers": min(worker_count, len(models)),
                },
            )
    _write_status(
        root,
        "prefetch",
        {
            "state": "complete",
            "completed_models": len(models),
            "total_models": len(models),
            "download_workers": min(worker_count, len(models)) if models else 0,
        },
    )


def generate_all(config: Mapping[str, Any]) -> None:
    root, _, prompts = evaluation._load_run(config)
    runtime = config["runtime"]
    staging_root = Path(str(runtime["temporary_download_root"]))
    staging_root.mkdir(parents=True, exist_ok=True)
    base_url = str(config["generation"]["base_url"])
    port = int(runtime.get("policy_port", 28131))
    gpu = str(runtime.get("policy_gpu", 0))
    models = evaluation.ordered_model_names(config)
    for index, model_name in enumerate(models):
        if _response_complete(root, model_name, prompts):
            _write_status(root, "generation", {"state": "skipped_complete", "model": model_name,
                                                "completed_models": index + 1, "total_models": len(models)})
            continue
        spec = config["models"][model_name]
        _write_status(root, "generation", {"state": "downloading", "model": model_name,
                                            "completed_models": index, "total_models": len(models)})
        model_path, temporary = _download_model(model_name, spec, staging_root)
        log_path = root / "logs" / "policy_servers" / f"{model_name}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            str(runtime["vllm"]), "serve", str(model_path),
            "--served-model-name", model_name,
            "--tensor-parallel-size", "1",
            "--dtype", "bfloat16",
            "--gpu-memory-utilization", str(runtime.get("gpu_memory_utilization", 0.9)),
            "--max-model-len", str(runtime.get("max_model_len", 32768)),
            "--max-num-seqs", str(config["generation"].get("max_in_flight", 64)),
            "--enable-prefix-caching", "--generation-config", "vllm",
            "--host", "127.0.0.1", "--port", str(port),
        ]
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = gpu
        with log_path.open("ab") as log:
            process = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, env=environment,
                start_new_session=True,
            )
        try:
            _write_status(root, "generation", {"state": "starting_server", "model": model_name,
                                                "pid": process.pid, "completed_models": index,
                                                "total_models": len(models)})
            _wait_server(base_url, model_name, process, float(runtime.get("startup_timeout_seconds", 900)))
            _write_status(root, "generation", {"state": "generating", "model": model_name,
                                                "pid": process.pid, "completed_models": index,
                                                "total_models": len(models)})
            evaluation.generate(config, model_name)
            if not _response_complete(root, model_name, prompts):
                raise evaluation.EvaluationError(f"response inventory incomplete: {model_name}")
        finally:
            _stop_server(process)
        if temporary and bool(runtime.get("delete_download_after_generation", True)):
            sentinel = model_path / ".trajectory_checkpoint.json"
            if sentinel.is_file() and json.loads(sentinel.read_text()) == {
                "repo_id": spec["repo_id"], "revision": spec["revision"], "artifact": spec["artifact"]
            }:
                shutil.rmtree(model_path)
        _write_status(root, "generation", {"state": "model_complete", "model": model_name,
                                            "completed_models": index + 1, "total_models": len(models)})
    _write_status(root, "generation", {"state": "complete", "completed_models": len(models),
                                        "total_models": len(models)})


def grade_watch(
    config: Mapping[str, Any],
    poll_seconds: float,
    runtime_base_urls: Sequence[str] | None = None,
    dataset_name: str | None = None,
) -> None:
    root, _, all_prompts = evaluation._load_run(config)
    if dataset_name is not None and dataset_name not in config["datasets"]:
        raise evaluation.EvaluationError(f"unknown dataset {dataset_name}")
    prompts = [row for row in all_prompts if dataset_name is None or row["dataset"] == dataset_name]
    lane = "grading" if dataset_name is None else f"grading_{dataset_name}"
    models = evaluation.ordered_model_names(config)
    completed: set[str] = set()
    while len(completed) < len(models):
        progress = False
        for model_name in models:
            if model_name in completed:
                continue
            if _grade_complete(root, model_name, prompts):
                completed.add(model_name)
                progress = True
                continue
            if not _response_complete(root, model_name, prompts):
                continue
            attempt = 0
            while True:
                attempt += 1
                _write_status(root, lane, {"state": "grading", "model": model_name,
                                                "attempt": attempt, "completed_models": len(completed),
                                                "total_models": len(models)})
                try:
                    evaluation.grade(
                        config, model_name, runtime_base_urls=runtime_base_urls, dataset_name=dataset_name
                    )
                    break
                except Exception as error:
                    _write_status(root, lane, {"state": "retrying", "model": model_name,
                                                    "attempt": attempt, "error": repr(error),
                                                    "completed_models": len(completed),
                                                    "total_models": len(models)})
                    time.sleep(min(60.0, poll_seconds * max(1, attempt)))
            if not _grade_complete(root, model_name, prompts):
                raise evaluation.EvaluationError(f"grade inventory incomplete: {model_name}")
            completed.add(model_name)
            progress = True
        if len(completed) < len(models) and not progress:
            _write_status(root, lane, {"state": "waiting_for_responses",
                                            "completed_models": len(completed),
                                            "total_models": len(models)})
            time.sleep(poll_seconds)
    _write_status(root, lane, {"state": "complete", "completed_models": len(models),
                                    "total_models": len(models)})
    if all(_grade_complete(root, model_name, all_prompts) for model_name in models):
        evaluation.summarize(config)
        _write_status(root, "analysis", {"state": "summary_complete"})


def status(config: Mapping[str, Any]) -> None:
    root, _, prompts = evaluation._load_run(config)
    required_responses = len(prompts)
    required_grades = sum(len(row["criteria"]) for row in prompts)
    rows = []
    for model_name in evaluation.ordered_model_names(config):
        responses = sum(
            evaluation._response_path(root, model_name, row["dataset"], row["prompt_id"]).is_file()
            for row in prompts
        )
        grades = sum(
            evaluation._criterion_path(root, model_name, row["dataset"], row["prompt_id"], criterion["criterion_id"]).is_file()
            for row in prompts for criterion in row["criteria"]
        )
        rows.append({"model": model_name, "responses": responses,
                     "responses_required": required_responses, "grades": grades,
                     "grades_required": required_grades})
    print(json.dumps({"run_root": str(root), "models": rows}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("prepare", "download-all", "generate-all", "grade-watch", "status", "summarize"), required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    parser.add_argument("--download-workers", type=int)
    parser.add_argument("--judge-base-url", action="append")
    parser.add_argument("--dataset")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = evaluation.load_config(args.config.resolve(), args.limit)
    if args.mode == "prepare":
        print(evaluation.prepare(config))
    elif args.mode == "download-all":
        download_all(config, args.download_workers)
    elif args.mode == "generate-all":
        generate_all(config)
    elif args.mode == "grade-watch":
        grade_watch(config, args.poll_seconds, args.judge_base_url, args.dataset)
    elif args.mode == "status":
        status(config)
    else:
        print(json.dumps(evaluation.summarize(config), indent=2))


if __name__ == "__main__":
    main()
