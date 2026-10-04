#!/usr/bin/env python3
"""Train and merge the GPT-OSS teacher warm-start used before EvoRubrics RL."""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import random
import shutil
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
TEACHER_MODEL = "openai/gpt-oss-120b"
TEACHER_REVISION = "b5c939de8f754692c1647ca79fbf85e8c1e70f8a"
BASE_MODEL_SNAPSHOT = str(REPO_ROOT / "models/Qwen3-4B-Instruct-2507")
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dynamic_rubric.artifacts import ArtifactError, write_json_atomic
from dynamic_rubric.hashing import sha256_file, sha256_json


class SFTError(RuntimeError):
    """Raised when the SFT artifact cannot be created without weakening its contract."""


@dataclass(frozen=True)
class TokenizedExample:
    prompt_id: str
    prompt_hash: str
    input_ids: tuple[int, ...]
    prompt_tokens: int
    target_tokens: int

    @property
    def labels(self) -> tuple[int, ...]:
        return (-100,) * self.prompt_tokens + self.input_ids[self.prompt_tokens :]


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _required_text(record: dict[str, Any], name: str, row_number: int) -> str:
    value = record.get(name)
    if not isinstance(value, str) or not value.strip():
        raise SFTError(f"row {row_number}: {name} must be a non-empty string")
    return value


def validate_teacher_record(record: Any, row_number: int) -> tuple[str, str, list[dict[str, str]]]:
    if not isinstance(record, dict):
        raise SFTError(f"row {row_number}: expected a JSON object")
    prompt_id = _required_text(record, "prompt_id", row_number)
    prompt_hash = _required_text(record, "prompt_hash", row_number)
    raw_messages = record.get("messages")
    if not isinstance(raw_messages, list) or len(raw_messages) < 2:
        raise SFTError(f"row {row_number}: messages must contain a prompt and teacher answer")
    messages: list[dict[str, str]] = []
    for message_index, message in enumerate(raw_messages):
        if not isinstance(message, dict):
            raise SFTError(f"row {row_number}: message {message_index} must be an object")
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant", "tool"}:
            raise SFTError(f"row {row_number}: invalid role at message {message_index}")
        if not isinstance(content, str) or not content.strip():
            raise SFTError(f"row {row_number}: empty content at message {message_index}")
        messages.append({"role": role, "content": content})
    if messages[-1]["role"] != "assistant":
        raise SFTError(f"row {row_number}: final message must be the GPT-OSS teacher answer")
    if not any(message["role"] == "user" for message in messages[:-1]):
        raise SFTError(f"row {row_number}: prompt has no user message")
    calculated_hash = sha256_json(messages[:-1])
    if prompt_hash != calculated_hash:
        raise SFTError(f"row {row_number}: prompt_hash does not match messages[:-1]")
    return prompt_id, prompt_hash, messages


def tokenize_teacher_record(
    record: dict[str, Any], tokenizer: Any, *, row_number: int, max_length: int
) -> TokenizedExample:
    prompt_id, prompt_hash, messages = validate_teacher_record(record, row_number)
    template_options = {"tokenize": False, "enable_thinking": False}
    prompt_text = tokenizer.apply_chat_template(
        messages[:-1], add_generation_prompt=True, **template_options
    )
    complete_text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=False, **template_options
    )
    prompt_ids = tuple(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
    complete_ids = tuple(tokenizer(complete_text, add_special_tokens=False)["input_ids"])
    if complete_ids[: len(prompt_ids)] != prompt_ids:
        raise SFTError(
            f"row {row_number}: complete chat tokens do not preserve the generation-prompt prefix"
        )
    target_ids = complete_ids[len(prompt_ids) :]
    if not prompt_ids or not target_ids:
        raise SFTError(f"row {row_number}: empty prompt or assistant target token sequence")
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None or eos_token_id not in target_ids:
        raise SFTError(f"row {row_number}: assistant target does not contain the chat EOS token")
    if len(complete_ids) > max_length:
        raise SFTError(
            f"row {row_number}: {len(complete_ids)} tokens exceeds max_length={max_length}; "
            "refusing silent truncation"
        )
    return TokenizedExample(
        prompt_id=prompt_id,
        prompt_hash=prompt_hash,
        input_ids=complete_ids,
        prompt_tokens=len(prompt_ids),
        target_tokens=len(target_ids),
    )


def token_normalization_weights(target_token_counts: Sequence[int]) -> tuple[float, ...]:
    if not target_token_counts or any(count <= 0 for count in target_token_counts):
        raise ValueError("target token counts must be non-empty and positive")
    total = sum(target_token_counts)
    return tuple(count / total for count in target_token_counts)


def effective_batch_indices(
    example_count: int, batch_size: int, epochs: int, seed: int
) -> Iterable[tuple[int, tuple[int, ...]]]:
    if min(example_count, batch_size, epochs) < 1:
        raise ValueError("example_count, batch_size, and epochs must be positive")
    for epoch in range(epochs):
        indices = list(range(example_count))
        random.Random(seed + epoch).shuffle(indices)
        for offset in range(0, example_count, batch_size):
            yield epoch, tuple(indices[offset : offset + batch_size])


def _percentile(values: Sequence[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)]


def _tensor_state_sha256(state: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.view(dtype=__import__("torch").uint8).numpy().tobytes())
    return digest.hexdigest()


def _base_model_files(base_model: str, commit_hash: str | None) -> list[Path]:
    candidate = Path(base_model)
    if not candidate.is_dir():
        try:
            from huggingface_hub import snapshot_download

            candidate = Path(
                snapshot_download(repo_id=base_model, revision=commit_hash, local_files_only=True)
            )
        except Exception as error:
            raise SFTError(f"cannot resolve cached base model {base_model!r}: {error}") from error
    files = [
        path
        for path in candidate.rglob("*")
        if path.is_file()
        and (
            path.suffix == ".safetensors"
            or path.name
            in {
                "config.json",
                "generation_config.json",
                "model.safetensors.index.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
            }
        )
    ]
    if not any(path.suffix == ".safetensors" for path in files):
        raise SFTError(f"no cached safetensors weights found under {candidate}")
    return sorted(files)


def _file_manifest(paths: Sequence[Path]) -> list[dict[str, Any]]:
    return [
        {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in paths
    ]


def persist_output_file_manifests(
    run_root: Path, adapter_dir: Path, merged_dir: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    adapter_files = _file_manifest(
        sorted(path for path in adapter_dir.rglob("*") if path.is_file())
    )
    merged_files = _file_manifest(sorted(path for path in merged_dir.rglob("*") if path.is_file()))
    write_json_atomic(run_root / "adapter_files.json", adapter_files)
    write_json_atomic(run_root / "merged_model_files.json", merged_files)
    return adapter_files, merged_files


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for row_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise SFTError(f"row {row_number}: blank JSONL rows are forbidden")
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise SFTError(f"row {row_number}: invalid JSON: {error}") from error
    return records


def validate_teacher_manifest(data_path: Path, expected_examples: int) -> dict[str, Any]:
    manifest_path = data_path.parent / "manifest.json"
    if not manifest_path.is_file():
        raise SFTError(f"missing teacher manifest: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SFTError(f"invalid teacher manifest: {error}") from error
    train_info = manifest.get("train_jsonl")
    teacher = manifest.get("cache_identity", {}).get("teacher", {})
    expected_sha256 = sha256_file(data_path)
    if manifest.get("status") != "complete" or manifest.get("count") != expected_examples:
        raise SFTError("teacher manifest must prove a complete exact-size dataset")
    if teacher.get("model") != TEACHER_MODEL or teacher.get("revision") != TEACHER_REVISION:
        raise SFTError("teacher manifest does not prove the pinned GPT-OSS-120B revision")
    if not isinstance(train_info, dict) or train_info.get("sha256") != expected_sha256:
        raise SFTError("teacher manifest train_jsonl hash does not match --data")
    if train_info.get("bytes") != data_path.stat().st_size:
        raise SFTError("teacher manifest train_jsonl byte count does not match --data")
    return {
        "path": str(manifest_path.resolve()),
        "sha256": sha256_file(manifest_path),
        "teacher_model": teacher["model"],
        "teacher_revision": teacher["revision"],
        "train_jsonl_sha256": expected_sha256,
        "count": expected_examples,
    }


def _reserve_run(run_root: Path, started: dict[str, Any]) -> None:
    run_root.mkdir(parents=True, exist_ok=True)
    marker = run_root / "training_started.json"
    if marker.exists():
        raise SFTError(f"run already started; use a new --run-root: {run_root}")
    forbidden = [
        run_root / name for name in ("checkpoint", "merged_model", "training_complete.json")
    ]
    if any(path.exists() for path in forbidden):
        raise SFTError(f"run root contains prior training artifacts: {run_root}")
    write_json_atomic(marker, started)


def _write_failure(run_root: Path, error: BaseException) -> None:
    try:
        write_json_atomic(
            run_root / "training_failed.json",
            {"status": "failed", "failed_at": utc_now(), "error": repr(error)},
        )
    except (OSError, ArtifactError):
        return


def train(args: argparse.Namespace) -> dict[str, Any]:
    if args.epochs < 1 or args.batch_size < 1 or args.max_length < 2:
        raise SFTError("epochs, batch-size, and max-length must be positive")
    if args.learning_rate <= 0:
        raise SFTError("learning-rate must be positive")
    data_path = args.data.resolve()
    run_root = args.run_root.resolve()
    if not data_path.is_file():
        raise SFTError(f"teacher data does not exist: {data_path}")
    teacher_manifest = validate_teacher_manifest(data_path, args.expected_examples)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import numpy as np
    import torch
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if not torch.cuda.is_available():
        raise SFTError("CUDA is required for the real Qwen3-4B SFT run")
    torch.backends.cuda.matmul.allow_tf32 = True
    runtime = {
        "python": sys.version.split()[0],
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "peft", "accelerate", "safetensors")
        },
        "cuda_runtime": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_capability": list(torch.cuda.get_device_capability(0)),
    }
    optimizer_config = {
        "name": "AdamW",
        "learning_rate": args.learning_rate,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": 0.01,
        "max_gradient_norm": 1.0,
    }
    lora_settings = {
        "rank": 32,
        "alpha": 64,
        "dropout": 0.0,
        "target_modules": "all-linear",
        "bias": "none",
    }
    started = {
        "schema_version": 1,
        "status": "started",
        "started_at": utc_now(),
        "data_path": str(data_path),
        "data_sha256": sha256_file(data_path),
        "teacher_manifest": teacher_manifest,
        "code_path": str(Path(__file__).resolve()),
        "code_sha256": sha256_file(__file__),
        "base_model_requested": args.base_model,
        "epochs": args.epochs,
        "effective_batch_size": args.batch_size,
        "micro_batch_size": 1,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "seed": args.seed,
        "gpu": str(args.gpu),
        "runtime": runtime,
        "optimizer": optimizer_config,
        "lora": lora_settings,
    }
    _reserve_run(run_root, started)
    try:
        input_dir = run_root / "input"
        input_dir.mkdir()
        copied_data = input_dir / "teacher_sft.jsonl"
        shutil.copyfile(data_path, copied_data)
        if sha256_file(copied_data) != started["data_sha256"]:
            raise SFTError("teacher data changed while copying into the immutable run")

        tokenizer = AutoTokenizer.from_pretrained(
            args.base_model, trust_remote_code=True, local_files_only=True
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        records = _load_jsonl(copied_data)
        if len(records) != args.expected_examples:
            raise SFTError(
                f"expected exactly {args.expected_examples} teacher examples, found {len(records)}"
            )
        examples = [
            tokenize_teacher_record(record, tokenizer, row_number=index, max_length=args.max_length)
            for index, record in enumerate(records, start=1)
        ]
        ids = [example.prompt_id for example in examples]
        hashes = [example.prompt_hash for example in examples]
        if len(set(ids)) != len(ids) or len(set(hashes)) != len(hashes):
            raise SFTError("prompt_id and prompt_hash must each be unique across all examples")
        target_counts = [example.target_tokens for example in examples]
        complete_counts = [len(example.input_ids) for example in examples]

        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
            local_files_only=True,
        ).to("cuda:0")
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        commit_hash = getattr(model.config, "_commit_hash", None)
        base_files = _base_model_files(args.base_model, commit_hash)
        base_manifest = _file_manifest(base_files)
        base_identity = {
            "requested": args.base_model,
            "resolved_name_or_path": str(model.config._name_or_path),
            "commit_hash": commit_hash,
            "config_sha256": sha256_json(model.config.to_dict()),
            "files": base_manifest,
            "files_sha256": sha256_json(base_manifest),
        }
        write_json_atomic(run_root / "base_model_identity.json", base_identity)

        lora_config = LoraConfig(
            r=32,
            lora_alpha=64,
            lora_dropout=0.0,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not trainable:
            raise SFTError("PEFT produced no trainable adapter parameters")
        initial_adapter_hash = _tensor_state_sha256(get_peft_model_state_dict(model))
        optimizer = torch.optim.AdamW(
            trainable,
            lr=optimizer_config["learning_rate"],
            betas=tuple(optimizer_config["betas"]),
            eps=optimizer_config["eps"],
            weight_decay=optimizer_config["weight_decay"],
        )
        model.train()
        steps: list[dict[str, Any]] = []
        exposure = 0
        for epoch, batch_indices in effective_batch_indices(
            len(examples), args.batch_size, args.epochs, args.seed
        ):
            optimizer.zero_grad(set_to_none=True)
            batch_target_counts = [examples[index].target_tokens for index in batch_indices]
            weights = token_normalization_weights(batch_target_counts)
            weighted_loss = 0.0
            for example_index, weight in zip(batch_indices, weights):
                example = examples[example_index]
                input_ids = torch.tensor([example.input_ids], dtype=torch.long, device="cuda:0")
                labels = torch.tensor([example.labels], dtype=torch.long, device="cuda:0")
                attention_mask = torch.ones_like(input_ids)
                output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                if not torch.isfinite(output.loss):
                    raise SFTError(f"non-finite loss for prompt_id={example.prompt_id}")
                (output.loss * weight).backward()
                weighted_loss += float(output.loss.detach().cpu()) * weight
                exposure += 1
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable, max_norm=optimizer_config["max_gradient_norm"]
            )
            if not torch.isfinite(grad_norm):
                raise SFTError("non-finite adapter gradient norm")
            optimizer.step()
            step_record = {
                "step": len(steps) + 1,
                "epoch": epoch + 1,
                "examples": len(batch_indices),
                "target_tokens": sum(batch_target_counts),
                "token_mean_loss": weighted_loss,
                "gradient_norm_before_clip": float(grad_norm.detach().cpu()),
                "cumulative_exposures": exposure,
            }
            steps.append(step_record)
            write_json_atomic(
                run_root / "progress.json",
                {
                    "schema_version": 1,
                    "status": "training",
                    "updated_at": utc_now(),
                    "completed_optimizer_steps": len(steps),
                    "expected_optimizer_steps": args.epochs
                    * math.ceil(len(examples) / args.batch_size),
                    "cumulative_exposures": exposure,
                    "last_step": step_record,
                },
                immutable=False,
            )
            print(json.dumps({"event": "optimizer_step", **step_record}), flush=True)

        final_adapter_state = get_peft_model_state_dict(model)
        final_adapter_hash = _tensor_state_sha256(final_adapter_state)
        if final_adapter_hash == initial_adapter_hash:
            raise SFTError("adapter parameters did not change during SFT")
        checkpoint_dir = run_root / "checkpoint"
        adapter_dir = checkpoint_dir / "adapter"
        model.save_pretrained(adapter_dir, safe_serialization=True)
        tokenizer.save_pretrained(adapter_dir)
        torch.save(
            {
                "optimizer": optimizer.state_dict(),
                "epochs": args.epochs,
                "steps": len(steps),
                "exposures": exposure,
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
            },
            checkpoint_dir / "optimizer.pt",
        )

        verification_ids = torch.tensor(
            [examples[0].input_ids[-min(256, len(examples[0].input_ids)) :]],
            dtype=torch.long,
            device="cuda:0",
        )
        generation_prompt_ids = torch.tensor(
            [examples[0].input_ids[: examples[0].prompt_tokens]],
            dtype=torch.long,
            device="cuda:0",
        )
        model.eval()
        with torch.inference_mode():
            adapter_logits = model(input_ids=verification_ids).logits.float().cpu()
        merged = model.merge_and_unload(safe_merge=True)
        merged.eval()
        with torch.inference_mode():
            merged_logits = merged(input_ids=verification_ids).logits.float().cpu()
        adapter_merge_max_abs = float((adapter_logits - merged_logits).abs().max())
        adapter_merge_close = bool(
            torch.allclose(adapter_logits, merged_logits, rtol=0.02, atol=0.25)
        )
        if not adapter_merge_close:
            raise SFTError(
                f"adapter/merged logits exceed BF16 tolerance (max_abs={adapter_merge_max_abs})"
            )
        merged.config.use_cache = True
        merged_dir = run_root / "merged_model"
        merged.save_pretrained(merged_dir, safe_serialization=True, max_shard_size="4GB")
        tokenizer.save_pretrained(merged_dir)
        del model, merged, optimizer, trainable, final_adapter_state
        del output, input_ids, labels, attention_mask
        gc.collect()
        torch.cuda.empty_cache()

        reloaded_tokenizer = AutoTokenizer.from_pretrained(
            merged_dir, trust_remote_code=True, local_files_only=True
        )
        reloaded = AutoModelForCausalLM.from_pretrained(
            merged_dir,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
            local_files_only=True,
        ).to("cuda:0")
        reloaded.eval()
        with torch.inference_mode():
            reloaded_logits = reloaded(input_ids=verification_ids).logits.float().cpu()
            generated = reloaded.generate(
                input_ids=generation_prompt_ids,
                do_sample=False,
                max_new_tokens=8,
                pad_token_id=reloaded_tokenizer.eos_token_id,
            )
        reload_finite = bool(torch.isfinite(reloaded_logits).all())
        reload_close = bool(torch.allclose(merged_logits, reloaded_logits, rtol=0.02, atol=0.25))
        reload_max_abs = float((merged_logits - reloaded_logits).abs().max())
        generated_tokens = generated[0, generation_prompt_ids.shape[1] :].tolist()
        if not reload_finite or not reload_close or not generated_tokens:
            raise SFTError(
                "fresh merged-model reload failed finite forward, numerical, or generation checks"
            )

        adapter_files, merged_files = persist_output_file_manifests(
            run_root, adapter_dir, merged_dir
        )
        metrics = {
            "schema_version": 1,
            "examples": len(examples),
            "epochs": args.epochs,
            "optimizer_steps": len(steps),
            "expected_optimizer_steps": args.epochs * math.ceil(len(examples) / args.batch_size),
            "exposures": exposure,
            "effective_batch_size": args.batch_size,
            "micro_batch_size": 1,
            "runtime": runtime,
            "optimizer": optimizer_config,
            "lora": lora_settings,
            "prompt_tokens": {
                "min": min(example.prompt_tokens for example in examples),
                "p50": _percentile([example.prompt_tokens for example in examples], 0.50),
                "p95": _percentile([example.prompt_tokens for example in examples], 0.95),
                "max": max(example.prompt_tokens for example in examples),
                "total": sum(example.prompt_tokens for example in examples),
            },
            "target_tokens": {
                "min": min(target_counts),
                "p50": _percentile(target_counts, 0.50),
                "p95": _percentile(target_counts, 0.95),
                "max": max(target_counts),
                "total": sum(target_counts),
            },
            "complete_tokens": {
                "min": min(complete_counts),
                "p50": _percentile(complete_counts, 0.50),
                "p95": _percentile(complete_counts, 0.95),
                "max": max(complete_counts),
            },
            "steps": steps,
        }
        write_json_atomic(run_root / "training_metrics.json", metrics)
        completed = {
            "schema_version": 1,
            "status": "training_passed",
            "actual_training": True,
            "completed_at": utc_now(),
            "data_path": str(copied_data),
            "data_sha256": sha256_file(copied_data),
            "code_sha256": started["code_sha256"],
            "base_model_identity_sha256": sha256_json(base_identity),
            "base_model_commit_hash": commit_hash,
            "examples": len(examples),
            "epochs": args.epochs,
            "optimizer_steps": len(steps),
            "exposures": exposure,
            "assistant_only_loss": True,
            "silent_truncation": False,
            "initial_adapter_sha256": initial_adapter_hash,
            "final_adapter_sha256": final_adapter_hash,
            "adapter_changed": True,
            "adapter_files": adapter_files,
            "adapter_files_sha256": sha256_json(adapter_files),
            "optimizer_sha256": sha256_file(checkpoint_dir / "optimizer.pt"),
            "merged_model_path": str(merged_dir),
            "merged_model_files": merged_files,
            "merged_model_files_sha256": sha256_json(merged_files),
            "verification": {
                "adapter_vs_merged_allclose": adapter_merge_close,
                "adapter_vs_merged_max_abs": adapter_merge_max_abs,
                "fresh_reload_finite_forward": reload_finite,
                "merged_vs_reload_allclose": reload_close,
                "merged_vs_reload_max_abs": reload_max_abs,
                "fresh_reload_generated_token_count": len(generated_tokens),
                "rtol": 0.02,
                "atol": 0.25,
            },
        }
        write_json_atomic(run_root / "training_complete.json", completed)
        return completed
    except BaseException as error:
        _write_failure(run_root, error)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="GPT-OSS teacher JSONL")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--base-model", default=BASE_MODEL_SNAPSHOT)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32, help="effective example batch")
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--gpu", default="1", help="physical CUDA device exposed to this process")
    parser.add_argument("--max-length", type=int, default=4608)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--expected-examples", type=int, default=1500)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = train(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
