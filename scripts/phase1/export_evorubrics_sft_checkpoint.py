#!/usr/bin/env python3
"""Recover a mathematically valid FP32 merged model from a completed SFT adapter."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = Path(__file__).with_name("train_evorubrics_sft.py")
PINNED_TRAIN_CODE_SHA256 = "090e7f5adecb6eb044522396ac1d46b457fea4b7e68e3ac49173edcb59e673eb"
EXPECTED_STEPS = 47
EXPECTED_EXPOSURES = 1500

if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from dynamic_rubric.artifacts import ArtifactError, read_json, write_json_atomic
from dynamic_rubric.hashing import sha256_file, sha256_json


class ExportError(RuntimeError):
    pass


def _load_training_module() -> Any:
    spec = importlib.util.spec_from_file_location("evorubrics_sft_training_frozen", TRAIN_SCRIPT)
    if spec is None or spec.loader is None:
        raise ExportError("cannot load frozen SFT training helpers")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def parse_optimizer_log(path: Path) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("{"):
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if value.get("event") == "optimizer_step":
            steps.append(value)
    if [item.get("step") for item in steps] != list(range(1, EXPECTED_STEPS + 1)):
        raise ExportError(
            f"training log does not contain {EXPECTED_STEPS} contiguous optimizer steps"
        )
    required_finite = ("token_mean_loss", "gradient_norm_before_clip")
    if any(
        not isinstance(item.get(key), (int, float)) or not math.isfinite(float(item[key]))
        for item in steps
        for key in required_finite
    ):
        raise ExportError("training log contains non-finite loss or gradient evidence")
    if sum(int(item.get("examples", 0)) for item in steps) != EXPECTED_EXPOSURES:
        raise ExportError(
            f"training log does not prove exactly {EXPECTED_EXPOSURES} exposures"
        )
    if steps[-1].get("cumulative_exposures") != EXPECTED_EXPOSURES:
        raise ExportError(f"training log final cumulative exposure is not {EXPECTED_EXPOSURES}")
    return steps


def validate_optimizer_checkpoint(path: Path) -> dict[str, Any]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("steps") != EXPECTED_STEPS or payload.get("exposures") != EXPECTED_EXPOSURES:
        raise ExportError("optimizer checkpoint metadata does not prove completed SFT")
    states = payload.get("optimizer", {}).get("state", {})
    if not states:
        raise ExportError("optimizer checkpoint has no parameter state")
    optimizer_steps = []
    for state in states.values():
        step = state.get("step")
        optimizer_steps.append(int(step.item() if hasattr(step, "item") else step))
    if min(optimizer_steps) != EXPECTED_STEPS or max(optimizer_steps) != EXPECTED_STEPS:
        raise ExportError(
            f"not every trained adapter parameter reached optimizer step {EXPECTED_STEPS}"
        )
    return {
        "metadata_steps": payload["steps"],
        "metadata_exposures": payload["exposures"],
        "parameter_states": len(states),
        "parameter_step_min": min(optimizer_steps),
        "parameter_step_max": max(optimizer_steps),
    }


def validate_adapter(adapter_dir: Path, base_model: str) -> dict[str, Any]:
    from safetensors.torch import load_file

    config = read_json(adapter_dir / "adapter_config.json")
    if (
        config.get("base_model_name_or_path") != base_model
        or config.get("r") != 32
        or config.get("lora_alpha") != 64
        or float(config.get("lora_dropout", -1)) != 0.0
    ):
        raise ExportError("adapter config does not match the frozen SFT contract")
    state = load_file(adapter_dir / "adapter_model.safetensors", device="cpu")
    b_tensors = {name: value for name, value in state.items() if ".lora_B." in name}
    if not b_tensors or any(not value.isfinite().all() for value in state.values()):
        raise ExportError("adapter tensors are missing or non-finite")
    nonzero_b = sum(int(value.count_nonzero()) for value in b_tensors.values())
    if nonzero_b == 0:
        raise ExportError("all LoRA-B tensors remain zero; trained change is unproven")
    return {
        "tensor_count": len(state),
        "lora_b_tensor_count": len(b_tensors),
        "lora_b_nonzero_elements": nonzero_b,
        "adapter_model_sha256": sha256_file(adapter_dir / "adapter_model.safetensors"),
        "adapter_config_sha256": sha256_file(adapter_dir / "adapter_config.json"),
    }


def _tensor_sha256(tensor: Any) -> str:
    import torch

    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def capture_expected_fp32_merge(model: Any) -> tuple[dict[str, str], dict[str, Any]]:
    import torch
    from peft.tuners.lora.layer import LoraLayer

    expected: dict[str, str] = {}
    merged_parameters: dict[str, Any] = {}
    for name, module in model.named_modules():
        if not isinstance(module, LoraLayer):
            continue
        adapters = list(module.active_adapters)
        if len(adapters) != 1:
            raise ExportError(f"expected one active adapter for {name}, got {adapters}")
        base_weight = module.get_base_layer().weight
        if base_weight.dtype != torch.float32:
            raise ExportError(f"base weight is not FP32 before merge: {name}")
        expected[name] = _tensor_sha256(base_weight + module.get_delta_weight(adapters[0]))
        merged_parameters[name] = base_weight
    if not expected:
        raise ExportError("no active LoRA layers found for algebraic merge proof")
    return expected, merged_parameters


def verify_expected_fp32_merge(
    expected: dict[str, str], merged_parameters: dict[str, Any]
) -> dict[str, Any]:
    actual = {name: _tensor_sha256(value) for name, value in merged_parameters.items()}
    mismatches = sorted(name for name in expected if actual.get(name) != expected[name])
    if mismatches:
        raise ExportError(f"FP32 merged parameters differ from W + scaled(B): {mismatches[:3]}")
    return {
        "layer_count": len(expected),
        "expected_hashes_sha256": sha256_json(expected),
    }


def validate_source(source_run: Path, training_log: Path) -> dict[str, Any]:
    training = _load_training_module()
    started = read_json(source_run / "training_started.json")
    progress = read_json(source_run / "progress.json")
    failed = read_json(source_run / "training_failed.json")
    if sha256_file(TRAIN_SCRIPT) != PINNED_TRAIN_CODE_SHA256:
        raise ExportError("frozen training script hash changed")
    if started.get("code_sha256") != PINNED_TRAIN_CODE_SHA256:
        raise ExportError("source run was not produced by the frozen training script")
    if (
        progress.get("completed_optimizer_steps") != EXPECTED_STEPS
        or progress.get("expected_optimizer_steps") != EXPECTED_STEPS
        or progress.get("cumulative_exposures") != EXPECTED_EXPOSURES
    ):
        raise ExportError("source progress does not prove completed SFT")
    if "adapter/merged logits exceed BF16 tolerance" not in str(failed.get("error")):
        raise ExportError("source failure was not isolated to the BF16 merge check")
    copied_data = source_run / "input" / "teacher_sft.jsonl"
    if sha256_file(copied_data) != started.get("data_sha256"):
        raise ExportError("source run teacher data hash changed")
    teacher_proof = started.get("teacher_manifest", {})
    teacher_manifest = Path(teacher_proof.get("path", ""))
    if not teacher_manifest.is_file() or sha256_file(teacher_manifest) != teacher_proof.get(
        "sha256"
    ):
        raise ExportError("source teacher manifest hash changed")
    training.validate_teacher_manifest(Path(started["data_path"]), EXPECTED_EXPOSURES)
    log_steps = parse_optimizer_log(training_log)
    optimizer_path = source_run / "checkpoint" / "optimizer.pt"
    adapter_dir = source_run / "checkpoint" / "adapter"
    paths = {
        "training_started": source_run / "training_started.json",
        "progress": source_run / "progress.json",
        "training_failed": source_run / "training_failed.json",
        "teacher_data": copied_data,
        "teacher_manifest": teacher_manifest,
        "adapter_model": adapter_dir / "adapter_model.safetensors",
        "adapter_config": adapter_dir / "adapter_config.json",
        "optimizer": optimizer_path,
        "training_log": training_log,
    }
    return {
        "base_model": started["base_model_requested"],
        "source_run": str(source_run),
        "files": {
            name: {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for name, path in paths.items()
        },
        "log_steps_sha256": sha256_json(log_steps),
        "first_step": log_steps[0],
        "last_step": log_steps[-1],
    }


def _reserve(run_root: Path, source: dict[str, Any], gpu: str) -> None:
    run_root.mkdir(parents=True, exist_ok=True)
    if any(run_root.iterdir()):
        raise ExportError(f"export run root must be empty: {run_root}")
    write_json_atomic(
        run_root / "export_started.json",
        {
            "schema_version": 1,
            "status": "started",
            "started_at": _utc_now(),
            "export_code_sha256": sha256_file(__file__),
            "source_evidence_sha256": sha256_json(source),
            "source_evidence": source,
            "gpu": gpu,
            "artifact_dtype": "float32",
        },
    )


def _state_exact(left: Any, right: Any) -> bool:
    left_items = dict(left.named_parameters())
    right_items = dict(right.named_parameters())
    return left_items.keys() == right_items.keys() and all(
        __import__("torch").equal(left_items[name], right_items[name]) for name in left_items
    )


def export(args: argparse.Namespace) -> dict[str, Any]:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    source_run = args.source_run.resolve()
    training_log = args.training_log.resolve()
    run_root = args.run_root.resolve()
    source = validate_source(source_run, training_log)
    training = _load_training_module()
    if source["base_model"] != training.BASE_MODEL_SNAPSHOT:
        raise ExportError("source run base model is not the pinned Qwen snapshot")
    _reserve(run_root, source, str(args.gpu))
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        if not torch.cuda.is_available():
            raise ExportError("CUDA is required for FP32 export verification")
        base_model = source["base_model"]
        adapter_dir = source_run / "checkpoint" / "adapter"
        optimizer_evidence = validate_optimizer_checkpoint(
            source_run / "checkpoint" / "optimizer.pt"
        )
        adapter_evidence = validate_adapter(adapter_dir, base_model)
        tokenizer = AutoTokenizer.from_pretrained(base_model, local_files_only=True)
        records = training._load_jsonl(source_run / "input" / "teacher_sft.jsonl")
        indices = (0, len(records) // 2, len(records) - 1)
        examples = [
            training.tokenize_teacher_record(
                records[index], tokenizer, row_number=index + 1, max_length=4608
            )
            for index in indices
        ]
        base = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch.float32, attn_implementation="sdpa", local_files_only=True
        ).to("cuda:0")
        model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=False)
        model.eval()
        probe_ids = [
            torch.tensor([item.input_ids[-min(256, len(item.input_ids)) :]], device="cuda:0")
            for item in examples
        ]
        with torch.inference_mode():
            adapter_logits = [model(input_ids=value).logits[:, -1].cpu() for value in probe_ids]
        expected_merge, merged_parameters = capture_expected_fp32_merge(model)
        merged = model.merge_and_unload(safe_merge=True)
        algebra = verify_expected_fp32_merge(expected_merge, merged_parameters)
        merged.eval()
        with torch.inference_mode():
            merged_logits = [merged(input_ids=value).logits[:, -1].cpu() for value in probe_ids]
        if not all(torch.isfinite(value).all() for value in [*adapter_logits, *merged_logits]):
            raise ExportError("FP32 adapter or merged logits are non-finite")
        diffs = [(left - right).abs() for left, right in zip(adapter_logits, merged_logits)]
        close = [
            bool(torch.allclose(left, right, rtol=1e-4, atol=1e-3))
            for left, right in zip(adapter_logits, merged_logits)
        ]
        if not all(close):
            raise ExportError("FP32 adapter and merged logits exceed rtol=1e-4, atol=1e-3")
        model_dir = run_root / "merged_model"
        merged.save_pretrained(model_dir, safe_serialization=True, max_shard_size="4GB")
        tokenizer.save_pretrained(model_dir)
        reloaded = AutoModelForCausalLM.from_pretrained(
            model_dir, torch_dtype=torch.float32, attn_implementation="sdpa", local_files_only=True
        ).to("cuda:0")
        reloaded.eval()
        if not _state_exact(merged, reloaded):
            raise ExportError("fresh reload parameters differ from the saved FP32 model")
        generation_prompt = torch.tensor(
            [examples[0].input_ids[: examples[0].prompt_tokens]], device="cuda:0"
        )
        with torch.inference_mode():
            reload_logits = [reloaded(input_ids=value).logits[:, -1].cpu() for value in probe_ids]
            generated = reloaded.generate(
                input_ids=generation_prompt,
                do_sample=False,
                max_new_tokens=8,
                pad_token_id=tokenizer.eos_token_id,
            )
        if not all(torch.isfinite(value).all() for value in reload_logits):
            raise ExportError("fresh reload FP32 logits are non-finite")
        if not all(torch.equal(a, b) for a, b in zip(merged_logits, reload_logits)):
            raise ExportError("fresh reload FP32 logits are not exactly reproducible")
        generated_count = int(generated.shape[1] - generation_prompt.shape[1])
        if generated_count < 1:
            raise ExportError("fresh reload did not generate any tokens")
        model_files = [path for path in model_dir.rglob("*") if path.is_file()]
        file_manifest = [
            {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in sorted(model_files)
        ]
        runtime = {
            "python": sys.version.split()[0],
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("torch", "transformers", "peft", "safetensors")
            },
            "gpu_name": torch.cuda.get_device_name(0),
            "cuda_runtime": torch.version.cuda,
            "tf32": False,
            "attention": "sdpa",
        }
        metrics = {
            "probe_indices": list(indices),
            "rtol": 1e-4,
            "atol": 1e-3,
            "per_probe": [
                {
                    "allclose": passed,
                    "max_abs": float(diff.max()),
                    "mean_abs": float(diff.mean()),
                    "p99_abs": float(torch.quantile(diff, 0.99)),
                }
                for passed, diff in zip(close, diffs)
            ],
            "fresh_reload_parameter_exact": True,
            "fresh_reload_logits_exact": True,
            "generated_token_count": generated_count,
            "algebraic_merge": algebra,
            "optimizer": optimizer_evidence,
            "adapter": adapter_evidence,
        }
        write_json_atomic(run_root / "verification_metrics.json", metrics)
        write_json_atomic(run_root / "merged_model_files.json", file_manifest)
        completed = {
            "schema_version": 1,
            "status": "export_passed",
            "actual_training": True,
            "examples": EXPECTED_EXPOSURES,
            "epochs": 1,
            "optimizer_steps": EXPECTED_STEPS,
            "exposures": EXPECTED_EXPOSURES,
            "data_sha256": source["files"]["teacher_data"]["sha256"],
            "training_reused_without_optimizer_steps": True,
            "completed_at": _utc_now(),
            "artifact_dtype": "float32",
            "future_rq2_runtime_dtype": "bfloat16",
            "source_evidence_sha256": sha256_json(source),
            "export_code_sha256": sha256_file(__file__),
            "merged_model_path": str(model_dir),
            "merged_model_files": file_manifest,
            "merged_model_files_sha256": sha256_json(file_manifest),
            "runtime": runtime,
            "verification": metrics,
            "limitation": "Future BF16 loading rounds FP32 merged weights and is not claimed identical to the live adapter path.",
        }
        write_json_atomic(run_root / "training_complete.json", completed)
        return completed
    except BaseException as error:
        try:
            write_json_atomic(
                run_root / "export_failed.json",
                {"status": "failed", "failed_at": _utc_now(), "error": repr(error)},
            )
        except (OSError, ArtifactError):
            pass
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--training-log", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--expected-steps", type=int, default=EXPECTED_STEPS)
    parser.add_argument("--expected-exposures", type=int, default=EXPECTED_EXPOSURES)
    return parser


def main() -> None:
    global EXPECTED_STEPS, EXPECTED_EXPOSURES
    args = build_parser().parse_args()
    if args.expected_steps < 1 or args.expected_exposures < 1:
        raise ExportError("expected steps and exposures must be positive")
    EXPECTED_STEPS = args.expected_steps
    EXPECTED_EXPOSURES = args.expected_exposures
    result = export(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
