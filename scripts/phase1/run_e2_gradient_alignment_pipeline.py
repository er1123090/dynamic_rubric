#!/usr/bin/env python3
"""Prepare the fixed validation20 data needed by Notion experiment E2.

The pipeline is deliberately analysis-only.  It exports local FSDP actor weights,
serves one policy checkpoint at a time on Trainer GPU1, constructs step-local
rubrics with GPT-OSS, and grades the same Pool-B responses with Qwen.  It never
loads optimizer state and never performs an optimizer update.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

from dynamic_rubric.artifacts import read_jsonl, write_json_atomic, write_jsonl_atomic
from dynamic_rubric.hashing import canonical_json_bytes, sha256_file
from dynamic_rubric.phase1.audit_policy import export_checkpoint, load_run_contract


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = (
    ROOT
    / "outputs/medicine/online_rubrics/seed-11"
    / "phase1-online-rubrics-medicine-full-dense-20260919-seed11"
)
OUTPUT_ROOT = ROOT / "outputs/analysis/medicine_online_e2_gradient_alignment_20260921"
EXPORT_ROOT = OUTPUT_ROOT / "policy_exports"
SOURCE_VALIDATION = (
    ROOT
    / "outputs/analysis/medicine_static_online_heldout_validation100_full_20260910_v4"
    / "manifests/validation_prompts.jsonl"
)
AUDIT_SCRIPT = ROOT / "scripts/phase1/run_heldout_validation_audit.py"
VERL_ROOT = ROOT / "environment/upstream/verl"
VERL_PYTHON = VERL_ROOT / ".venv-runtime/bin/python"
VLLM = ROOT / ".venvs/judge/bin/vllm"
BASE_MODEL = ROOT / "models/Qwen3-4B-Instruct-2507"

METHOD = "online"
ALL_STEPS = (0, 5, 6, 20, 21, 35, 36)
EXPORT_STEPS = (5, 6, 20, 21, 35, 36)
TARGETS = {6: (0, 5, 6), 21: (0, 20, 21), 36: (0, 35, 36)}
SELECTION_SALT = "notion-e2-validation20-v1"
SERVED_PREFIX = "e2-online"


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def load_audit_module():
    spec = importlib.util.spec_from_file_location("e2_heldout_audit", AUDIT_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import audit module: {AUDIT_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.METHODS = (METHOD,)
    module.ONLINE_STEPS = ALL_STEPS
    module.EXPECTED_STEPS_BY_METHOD = {METHOD: ALL_STEPS}
    module.APPROVED_RUNTIME_JUDGE_BASE_URLS = frozenset(
        (*module.APPROVED_RUNTIME_JUDGE_BASE_URLS, "http://127.0.0.1:28014/v1")
    )
    return module


def config() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "data_role": "heldout_validation",
        "output_root": str(OUTPUT_ROOT),
        "seed": 11,
        "training_enabled": False,
        "optimizer_required": False,
        "checkpoint_steps": {METHOD: list(ALL_STEPS)},
        "methods": {
            METHOD: {
                "training_rubric": "online_rubrics",
                "audit_rubrics": "posthoc_online_style_step_local",
                "base_repo": str(BASE_MODEL),
                "base_revision": BASE_MODEL.name,
                "checkpoint_repo_template": str(
                    RUN_ROOT / "verl-run/checkpoints/global_step_{step}/actor"
                ),
                "checkpoint_revisions": {step: f"local-dense-step-{step}" for step in EXPORT_STEPS},
            }
        },
        "policy_generation": {
            "base_url": "http://127.0.0.1:28131/v1",
            "temperature": 1.0,
            "top_p": 0.95,
            "max_output_tokens": 3584,
            "timeout_seconds": 900,
            "max_retries": 4,
            "workers": 32,
            "max_in_flight": 32,
            "thinking": False,
        },
        "rubric_generation": {
            "base_url": "http://127.0.0.1:28011/v1",
            "model": "openai/gpt-oss-120b",
            "revision": "b5c939de8f754692c1647ca79fbf85e8c1e70f8a",
            "workers": 48,
            "max_in_flight": 48,
            "timeout_seconds": 900,
            "max_retries": 4,
            "extractor_max_output_tokens": 8192,
            "dedup_max_output_tokens": 8192,
            "thinking": False,
        },
        "grading": {
            "base_urls": ["http://127.0.0.1:28014/v1"],
            "model": "Qwen/Qwen3-32B",
            "revision": "9216db5781bf21249d130ec9da846c4624c16137",
            "workers": 64,
            "max_in_flight": 64,
            "timeout_seconds": 900,
            "max_retries": 4,
            "max_output_tokens": 4096,
            "thinking": False,
        },
        "metrics": {"pairwise_tie_epsilon": 0.01},
    }


def selected_prompts() -> list[dict[str, Any]]:
    rows = read_jsonl(SOURCE_VALIDATION)
    if len(rows) != 100:
        raise RuntimeError("sealed source validation manifest must contain 100 prompts")
    ranked = sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"{SELECTION_SALT}:{row['prompt_id']}".encode("utf-8")
        ).hexdigest(),
    )
    return [dict(row) for row in ranked[:20]]


def prepare() -> Path:
    prompts = selected_prompts()
    prompt_path = OUTPUT_ROOT / "manifests/validation_prompts.jsonl"
    expected = [
        {
            **row,
            "selection_protocol": "sha256_rank_from_sealed_validation100",
            "selection_salt": SELECTION_SALT,
        }
        for row in prompts
    ]
    if prompt_path.is_file() and read_jsonl(prompt_path) != expected:
        raise RuntimeError("frozen E2 validation20 manifest drift")
    write_jsonl_atomic(prompt_path, expected)

    work_plan = []
    for step in ALL_STEPS:
        source = BASE_MODEL if step == 0 else RUN_ROOT / f"verl-run/checkpoints/global_step_{step}/actor"
        work_plan.append(
            {
                "schema_version": 1,
                "data_role": "heldout_validation",
                "method": METHOD,
                "policy_step": step,
                "hf_repo_id": str(source),
                "hf_revision_requested": BASE_MODEL.name if step == 0 else f"local-dense-step-{step}",
                "inference_parameters_only": True,
                "optimizer_required": False,
                "pool_a_per_prompt": 8,
                "pool_b_per_prompt": 16,
                "fresh_rubric_mode": "r0" if step == 0 else "step_local_r0_union_new_criteria",
            }
        )
    write_jsonl_atomic(OUTPUT_ROOT / "manifests/checkpoint_work_plan.jsonl", work_plan)
    manifest = {
        "schema_version": 1,
        "experiment": "notion_e2_gradient_alignment",
        "analysis_only": True,
        "training_enabled": False,
        "optimizer_required": False,
        "source_run": str(RUN_ROOT),
        "source_validation_manifest": str(SOURCE_VALIDATION),
        "source_validation_manifest_sha256": sha256_file(SOURCE_VALIDATION),
        "validation_count": 20,
        "validation_prompt_ids_sha256": digest(sorted(row["prompt_id"] for row in expected)),
        "selection_protocol": "sha256_rank_from_sealed_validation100",
        "selection_salt": SELECTION_SALT,
        "target_checkpoints": sorted(TARGETS),
        "conditions": {str(step): list(evaluators) for step, evaluators in TARGETS.items()},
        "pool_b_per_prompt": 16,
        "same_responses_across_rubric_conditions": True,
        "gradient_update": False,
        "resolved_config": config(),
    }
    write_json_atomic(OUTPUT_ROOT / "manifest.json", manifest, immutable=False)
    return OUTPUT_ROOT


def patch_prepare(audit):
    root = prepare()

    def frozen_prepare(_config: Mapping[str, Any]) -> Path:
        current = selected_prompts()
        frozen = read_jsonl(root / "manifests/validation_prompts.jsonl")
        if [row["prompt_id"] for row in current] != [row["prompt_id"] for row in frozen]:
            raise RuntimeError("E2 validation20 selection changed")
        return root

    audit.prepare = frozen_prepare


def export_step(step: int) -> Path:
    if step not in EXPORT_STEPS:
        raise ValueError(f"step must be one of {EXPORT_STEPS}")
    prepare()
    contract = load_run_contract(RUN_ROOT)
    return export_checkpoint(
        contract,
        step=step,
        export_root=EXPORT_ROOT,
        merger_python=str(VERL_PYTHON),
        verl_root=VERL_ROOT,
    )


def wait_for_model(base_url: str, served_model: str, process: subprocess.Popen, timeout: int = 900) -> None:
    import urllib.request

    deadline = time.monotonic() + timeout
    url = base_url.rstrip("/") + "/models"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"vLLM exited before health check: rc={process.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                payload = json.load(response)
            if served_model in {str(row.get("id")) for row in payload.get("data", [])}:
                return
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError(f"timed out waiting for {served_model} at {url}")


def stop_process_group(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=30)


def serve_and_generate(step: int, *, control_only: bool = False) -> Path:
    prepare()
    audit = load_audit_module()
    patch_prepare(audit)
    model_path = BASE_MODEL if step == 0 else export_step(step)
    served_model = f"{SERVED_PREFIX}-step-{step:03d}"
    log_path = OUTPUT_ROOT / "logs" / f"policy-step-{step:03d}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(VLLM),
        "serve",
        str(model_path),
        "--served-model-name",
        served_model,
        "--host",
        "127.0.0.1",
        "--port",
        "28131",
        "--tensor-parallel-size",
        "1",
        "--gpu-memory-utilization",
        "0.85",
        "--max-model-len",
        "7680",
        "--enable-prefix-caching",
    ]
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = "1"
    with log_path.open("ab") as log:
        process = subprocess.Popen(
            command,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            wait_for_model(config()["policy_generation"]["base_url"], served_model, process)
            if step == 0:
                return audit.generate_control(config(), METHOD, served_model)
            if control_only:
                raise ValueError("control-only is valid only for step 0")
            return audit.generate(config(), METHOD, step, served_model)
        finally:
            stop_process_group(process)


def build_rubrics(steps: tuple[int, ...]) -> list[str]:
    audit = load_audit_module()
    patch_prepare(audit)
    paths = []
    for step in steps:
        paths.append(
            str(
                audit.build_rubrics(
                    config(),
                    METHOD,
                    step,
                    "openai/gpt-oss-120b",
                    runtime_base_url="http://127.0.0.1:28011/v1",
                )
            )
        )
    return paths


def score(
    policy_steps: tuple[int, ...] = tuple(TARGETS),
    *,
    runtime_base_url: str = "http://127.0.0.1:28014/v1",
) -> list[str]:
    audit = load_audit_module()
    patch_prepare(audit)
    paths = []
    audit.APPROVED_RUNTIME_JUDGE_BASE_URLS = frozenset(
        (*audit.APPROVED_RUNTIME_JUDGE_BASE_URLS, runtime_base_url.rstrip("/"))
    )
    for policy_step in policy_steps:
        if policy_step not in TARGETS:
            raise ValueError(f"policy steps must be drawn from {tuple(TARGETS)}")
        evaluator_steps = TARGETS[policy_step]
        for evaluator_step in evaluator_steps:
            paths.append(
                str(
                    audit.score_cell(
                        config(),
                        METHOD,
                        policy_step,
                        evaluator_step,
                        "Qwen/Qwen3-32B",
                        runtime_base_urls=[runtime_base_url],
                        runtime_workers=64,
                    )
                )
            )
    return paths


def parse_steps(raw: str) -> tuple[int, ...]:
    steps = tuple(int(value) for value in raw.split(",") if value.strip())
    if any(step not in ALL_STEPS for step in steps):
        raise ValueError(f"steps must be drawn from {ALL_STEPS}")
    return steps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    export = sub.add_parser("export")
    export.add_argument("--step", type=int, required=True)
    generate = sub.add_parser("generate")
    generate.add_argument("--step", type=int, required=True)
    rubrics = sub.add_parser("rubrics")
    rubrics.add_argument("--steps", default=",".join(map(str, ALL_STEPS)))
    score_parser = sub.add_parser("score")
    score_parser.add_argument("--policy-steps", default=",".join(map(str, TARGETS)))
    score_parser.add_argument(
        "--runtime-base-url", default="http://127.0.0.1:28014/v1"
    )
    args = parser.parse_args()

    if args.command == "prepare":
        result: Any = {"output_root": str(prepare())}
    elif args.command == "export":
        result = {"export": str(export_step(args.step))}
    elif args.command == "generate":
        result = {"responses": str(serve_and_generate(args.step))}
    elif args.command == "rubrics":
        result = {"rubrics": build_rubrics(parse_steps(args.steps))}
    else:
        result = {
            "grades": score(
                parse_steps(args.policy_steps), runtime_base_url=args.runtime_base_url
            )
        }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
