#!/usr/bin/env python3
"""Wait for the exact evaluator topology, smoke it, then launch Phase-1 training."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from dynamic_rubric.providers.base import GenerationRequest
from dynamic_rubric.providers.vllm_chat import VLLMChatAdapter


ENDPOINTS = (
    ("gpt-oss-inference_a-01", "http://127.0.0.1:28011", "openai/gpt-oss-120b", "low"),
    ("qwen3-32b-inference_a-01", "http://127.0.0.1:28014", "Qwen/Qwen3-32B", None),
)


def _log(message: str) -> None:
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} {message}", flush=True)


def _served_models(base_url: str, timeout_seconds: float = 5.0) -> tuple[str, ...]:
    request = urllib.request.Request(
        f"{base_url}/v1/models",
        headers={"Authorization": "Bearer EMPTY"},
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        payload = json.loads(response.read())
    return tuple(str(item["id"]) for item in payload.get("data", ()))


def _wait_for_model(label: str, base_url: str, expected_model: str, poll_seconds: float) -> None:
    while True:
        try:
            models = _served_models(base_url)
        except (OSError, ValueError, json.JSONDecodeError, urllib.error.URLError) as error:
            _log(f"waiting-for-endpoint label={label} error={type(error).__name__}")
        else:
            if expected_model in models:
                _log(f"endpoint-ready label={label} model={expected_model}")
                return
            _log(f"waiting-for-model label={label} returned={models!r}")
        time.sleep(poll_seconds)


def _smoke_endpoint(
    label: str,
    base_url: str,
    expected_model: str,
    reasoning_effort: str | None,
    cache_root: Path,
) -> None:
    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
        "additionalProperties": False,
    }
    adapter = VLLMChatAdapter(
        base_url,
        expected_model,
        cache_root / label,
        timeout_seconds=600,
        max_retries=4,
    )
    result = adapter.generate(
        GenerationRequest(
            prompt_id=f"phase1-launch-smoke-{label}",
            messages=(
                {
                    "role": "system",
                    "content": "Return only JSON matching the supplied schema.",
                },
                {"role": "user", "content": "Confirm readiness with ok=true."},
            ),
            family="phase1_launch_smoke",
            seed=11,
            max_output_tokens=64,
            temperature=0.0,
            top_p=1.0,
            json_schema=schema,
            schema_name="phase1_launch_smoke_v1",
            reasoning_effort=reasoning_effort,
        )
    )
    payload = json.loads(result.text)
    if payload != {"ok": True}:
        raise RuntimeError(f"structured smoke failed for {label}: {payload!r}")
    if result.returned_model != expected_model:
        raise RuntimeError(
            f"model identity mismatch for {label}: {result.returned_model!r}"
        )
    _log(f"structured-smoke-passed label={label} request_id={result.request_id}")


def _gpu_memory_used_mib(index: int) -> int:
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={index}",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(result.stdout.strip())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")

    root = args.repo_root.resolve()
    cache_root = (
        root
        / "outputs/medicine/online_rubrics/seed-11/supervisor/smoke-cache"
        / f"attempt-{time.time_ns()}-{os.getpid()}"
    )
    for label, base_url, expected_model, reasoning_effort in ENDPOINTS:
        _wait_for_model(label, base_url, expected_model, args.poll_seconds)
        _smoke_endpoint(label, base_url, expected_model, reasoning_effort, cache_root)

    manifests = tuple(
        (root / "outputs/medicine/shared/seed-11/pi0_control_cache").glob("manifest-*.json")
    )
    if len(manifests) != 1:
        raise RuntimeError(f"expected exactly one immutable pi0 manifest, found {len(manifests)}")
    used_mib = _gpu_memory_used_mib(1)
    if used_mib >= 1024:
        raise RuntimeError(f"trainer GPU1 is not free: {used_mib} MiB used")

    environment = dict(os.environ)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "1",
            "ONLINE_CONTROL_CACHE": str(manifests[0].resolve()),
            "ONLINE_LOGPROB_PREFETCH": "true",
            "PHASE1_GPT_OSS_BASE_URLS": "http://127.0.0.1:28011",
            "PHASE1_QWEN32B_BASE_URLS": "http://127.0.0.1:28014",
            "PHASE1_QWEN32B_EXPECTED_COUNT": "1",
            "PYTHONPATH": str(root / "src"),
            "PYTHONUNBUFFERED": "1",
        }
    )
    command = [
        sys.executable,
        "-m",
        "dynamic_rubric.phase1",
        "train-online",
        "--config",
        "configs/phase1/medicine_online_rubrics.yaml",
        "--repo-root",
        ".",
        "--run-id",
        args.run_id,
    ]
    _log(f"launching-full-run run_id={args.run_id}")
    os.chdir(root)
    os.execvpe(command[0], command, environment)


if __name__ == "__main__":
    raise SystemExit(main())
