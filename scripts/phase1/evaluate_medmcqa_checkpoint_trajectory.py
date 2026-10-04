#!/usr/bin/env python3
"""Evaluate the dense OnlineRubrics checkpoint trajectory on MedMCQA.

This runner reproduces the zero-shot multiple-choice contract used by the
EleutherAI lm-evaluation-harness MedMCQA task:

    Question: ...
    Choices:
    A. ...
    B. ...
    C. ...
    D. ...
    Answer:

Each candidate is the single continuation token `` A`` through `` D``.  The
candidate with the greatest conditional log likelihood is selected.  A fixed
Qwen chat template is applied to every checkpoint because the evaluated models
are instruction-tuned policies.  No sampling or free-form answer parsing is
involved.

Results are prompt-level and resumable: a completed checkpoint CSV is validated
and skipped on subsequent runs.  The summary and manifest are rebuilt after
every completed checkpoint.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/evaluation/medicine_dense_all48_healthbench500_20260926.yaml"
DEFAULT_DATASET = ROOT / "data/source/medmcqa/data/validation-00000-of-00001.parquet"
DEFAULT_OUTPUT = ROOT / "outputs/policy_eval/medicine_dense_all48_medmcqa_20260928"
DATASET_REPO = "openlifescienceai/medmcqa"
DATASET_REVISION = "91c6572"
DATASET_SHA256 = "b768a1ea34afc9f80d3106d9b21f80fa8a00ec450a1f6cd641af72ca9e591021"
CHOICES = ("A", "B", "C", "D")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def build_prompt(row: pd.Series) -> str:
    """Mirror lm-evaluation-harness lm_eval/tasks/medmcqa/utils_medmcqa.py."""

    return (
        f"Question: {row['question']}\n"
        "Choices:\n"
        f"A. {row['opa']}\n"
        f"B. {row['opb']}\n"
        f"C. {row['opc']}\n"
        f"D. {row['opd']}\n"
        "Answer:"
    )


def load_dataset(path: Path, limit: int | None) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    observed_hash = sha256_file(path)
    if observed_hash != DATASET_SHA256:
        raise RuntimeError(
            f"MedMCQA dataset hash mismatch: expected {DATASET_SHA256}, found {observed_hash}"
        )
    frame = pd.read_parquet(path)
    required = {"id", "question", "opa", "opb", "opc", "opd", "cop"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RuntimeError(f"MedMCQA dataset is missing columns: {missing}")
    if len(frame) != 4_183:
        raise RuntimeError(f"expected 4,183 validation examples, found {len(frame)}")
    if frame["id"].duplicated().any():
        raise RuntimeError("MedMCQA validation IDs are not unique")
    if not frame["cop"].isin(range(4)).all():
        raise RuntimeError("MedMCQA correct-option indices must be in [0, 3]")
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be positive")
        frame = frame.iloc[:limit]
    return frame.reset_index(drop=True)


def load_models(config_path: Path, start_step: int, end_step: int) -> list[dict[str, Any]]:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    models: list[dict[str, Any]] = []
    for name, spec in raw["models"].items():
        step = int(spec["step"])
        if not start_step <= step <= end_step:
            continue
        local_path = Path(str(spec["local_path"]))
        if not local_path.is_dir():
            raise FileNotFoundError(f"checkpoint {step} is missing: {local_path}")
        required = (local_path / "config.json", local_path / "tokenizer_config.json")
        if not all(path.is_file() for path in required) or not list(local_path.glob("*.safetensors")):
            raise RuntimeError(f"checkpoint {step} is not a complete HF export: {local_path}")
        models.append(
            {
                "name": str(name),
                "step": step,
                "artifact": str(spec["artifact"]),
                "path": local_path.resolve(),
            }
        )
    models.sort(key=lambda value: value["step"])
    expected = list(range(start_step, end_step + 1))
    observed = [value["step"] for value in models]
    if observed != expected:
        raise RuntimeError(f"checkpoint range is incomplete: expected {expected}, found {observed}")
    return models


def batched(values: list[str], size: int) -> Iterable[tuple[int, list[str]]]:
    for start in range(0, len(values), size):
        yield start, values[start : start + size]


def validate_complete(path: Path, expected_ids: list[str], step: int) -> bool:
    if not path.is_file():
        return False
    frame = pd.read_csv(path)
    required = {
        "checkpoint",
        "prompt_id",
        "gold_index",
        "prediction_index",
        "correct",
        "logprob_A",
        "logprob_B",
        "logprob_C",
        "logprob_D",
    }
    if not required.issubset(frame.columns) or len(frame) != len(expected_ids):
        raise RuntimeError(f"incomplete or incompatible cached result: {path}")
    if frame["checkpoint"].nunique() != 1 or int(frame["checkpoint"].iloc[0]) != step:
        raise RuntimeError(f"cached checkpoint mismatch: {path}")
    if frame["prompt_id"].astype(str).tolist() != expected_ids:
        raise RuntimeError(f"cached prompt order mismatch: {path}")
    return True


def evaluate_checkpoint(
    model_spec: dict[str, Any],
    dataset: pd.DataFrame,
    device: str,
    batch_size: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    started = time.perf_counter()
    model_path = model_spec["path"]
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    target_texts = [f" {choice}" for choice in CHOICES]
    target_token_lists = [
        tokenizer.encode(value, add_special_tokens=False) for value in target_texts
    ]
    if any(len(tokens) != 1 for tokens in target_token_lists):
        raise RuntimeError(f"choice continuations are not single tokens: {target_token_lists}")
    target_token_ids = [tokens[0] for tokens in target_token_lists]
    if len(set(target_token_ids)) != 4:
        raise RuntimeError(f"choice continuation tokens are not unique: {target_token_ids}")

    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": build_prompt(row)}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for _, row in dataset.iterrows()
    ]
    token_lengths = np.asarray(
        [len(tokenizer.encode(prompt, add_special_tokens=False)) for prompt in prompts], dtype=int
    )

    load_started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map={"": device},
        low_cpu_mem_usage=True,
    )
    model.eval()
    load_seconds = time.perf_counter() - load_started

    score_blocks: list[np.ndarray] = []
    inference_started = time.perf_counter()
    with torch.inference_mode():
        for _, prompt_batch in batched(prompts, batch_size):
            encoded = tokenizer(
                prompt_batch,
                padding=True,
                add_special_tokens=False,
                return_tensors="pt",
            )
            encoded = {key: value.to(device, non_blocking=True) for key, value in encoded.items()}
            last_token_logits = model(**encoded).logits[:, -1, :]
            log_probs = torch.log_softmax(last_token_logits.float(), dim=-1)[
                :, target_token_ids
            ]
            score_blocks.append(log_probs.cpu().numpy())
    torch.cuda.synchronize(torch.device(device))
    inference_seconds = time.perf_counter() - inference_started

    scores = np.concatenate(score_blocks, axis=0)
    predictions = scores.argmax(axis=1).astype(int)
    gold = dataset["cop"].to_numpy(dtype=int)
    correct = predictions == gold
    sorted_scores = np.sort(scores, axis=1)
    margins = sorted_scores[:, -1] - sorted_scores[:, -2]
    result = pd.DataFrame(
        {
            "checkpoint": model_spec["step"],
            "prompt_id": dataset["id"].astype(str),
            "gold_index": gold,
            "gold_choice": [CHOICES[index] for index in gold],
            "prediction_index": predictions,
            "prediction_choice": [CHOICES[index] for index in predictions],
            "correct": correct.astype(int),
            "logprob_A": scores[:, 0],
            "logprob_B": scores[:, 1],
            "logprob_C": scores[:, 2],
            "logprob_D": scores[:, 3],
            "top2_margin": margins,
            "subject_name": dataset["subject_name"].astype(str),
            "choice_type": dataset["choice_type"].astype(str),
        }
    )
    metadata = {
        "checkpoint": model_spec["step"],
        "model_name": model_spec["name"],
        "model_artifact": model_spec["artifact"],
        "model_path": str(model_path),
        "accuracy": float(correct.mean()),
        "correct": int(correct.sum()),
        "examples": int(len(result)),
        "load_seconds": load_seconds,
        "inference_seconds": inference_seconds,
        "total_seconds": time.perf_counter() - started,
        "choice_continuations": target_texts,
        "choice_token_ids": target_token_ids,
        "min_prompt_tokens": int(token_lengths.min()),
        "median_prompt_tokens": float(np.median(token_lengths)),
        "max_prompt_tokens": int(token_lengths.max()),
        "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
        "config_sha256": sha256_file(model_path / "config.json"),
    }

    del model, tokenizer, encoded, last_token_logits, log_probs
    gc.collect()
    torch.cuda.empty_cache()
    return result, metadata


def rebuild_summary(output_root: Path, expected_ids: list[str]) -> pd.DataFrame:
    rows = []
    for path in sorted((output_root / "prompt_scores").glob("checkpoint_*.csv")):
        frame = pd.read_csv(path)
        if len(frame) != len(expected_ids) or frame["prompt_id"].astype(str).tolist() != expected_ids:
            raise RuntimeError(f"invalid prompt-level result while rebuilding summary: {path}")
        rows.append(
            {
                "checkpoint": int(frame["checkpoint"].iloc[0]),
                "accuracy": float(frame["correct"].mean()),
                "correct": int(frame["correct"].sum()),
                "examples": int(len(frame)),
            }
        )
    summary = pd.DataFrame(rows).sort_values("checkpoint").reset_index(drop=True)
    atomic_csv(output_root / "checkpoint_accuracy.csv", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--start-step", type=int, default=1)
    parser.add_argument("--end-step", type=int, default=48)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    dataset = load_dataset(args.dataset.resolve(), args.limit)
    models = load_models(args.config.resolve(), args.start_step, args.end_step)
    output_root = args.output_root.resolve()
    if args.limit is not None:
        output_root = output_root / f"smoke_{args.limit}"
    prompt_ids = dataset["id"].astype(str).tolist()

    manifest = {
        "schema_version": 1,
        "benchmark": "MedMCQA",
        "dataset_repo": DATASET_REPO,
        "dataset_revision": DATASET_REVISION,
        "dataset_split": "validation",
        "dataset_path": str(args.dataset.resolve()),
        "dataset_sha256": sha256_file(args.dataset.resolve()),
        "examples": len(dataset),
        "config_path": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config.resolve()),
        "checkpoint_range": [args.start_step, args.end_step],
        "scoring": "zero-shot next-token conditional log-likelihood accuracy",
        "prompt_source": "EleutherAI/lm-evaluation-harness lm_eval/tasks/medmcqa",
        "target_delimiter": " ",
        "choices": list(CHOICES),
        "chat_template": True,
        "sampling": False,
        "device": args.device,
        "batch_size": args.batch_size,
        "torch_version": torch.__version__,
    }
    atomic_json(output_root / "manifest.json", manifest)

    for position, model_spec in enumerate(models, start=1):
        step = int(model_spec["step"])
        destination = output_root / "prompt_scores" / f"checkpoint_{step:03d}.csv"
        metadata_path = output_root / "checkpoint_metadata" / f"checkpoint_{step:03d}.json"
        if validate_complete(destination, prompt_ids, step):
            print(f"[{position}/{len(models)}] checkpoint {step}: already complete", flush=True)
            continue
        print(f"[{position}/{len(models)}] checkpoint {step}: evaluating", flush=True)
        result, metadata = evaluate_checkpoint(
            model_spec=model_spec,
            dataset=dataset,
            device=args.device,
            batch_size=args.batch_size,
        )
        atomic_csv(destination, result)
        atomic_json(metadata_path, metadata)
        summary = rebuild_summary(output_root, prompt_ids)
        print(
            f"[{position}/{len(models)}] checkpoint {step}: "
            f"accuracy={metadata['accuracy']:.6f}, "
            f"total={metadata['total_seconds']:.1f}s, completed={len(summary)}",
            flush=True,
        )
    summary = rebuild_summary(output_root, prompt_ids)
    manifest["completed_checkpoints"] = summary["checkpoint"].astype(int).tolist()
    manifest["complete"] = manifest["completed_checkpoints"] == [
        value["step"] for value in models
    ]
    atomic_json(output_root / "manifest.json", manifest)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
