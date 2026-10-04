#!/usr/bin/env python3
"""Evaluate MedMCQA with label/position-bias controlled multiple choice scoring.

For each MedMCQA example, this runner creates four cyclic option permutations so
that every answer content appears once under each label (A, B, C, and D).  The
Qwen chat template ends at the start of the assistant response, so the scored
continuations are the no-leading-space single tokens ``A`` through ``D``.  The
log probabilities are mapped back to the original option contents and averaged
over the four permutations before taking the argmax.

The identity-permutation prediction is retained as a diagnostic.  It shows how
much of the result changes from fixing the continuation delimiter alone, while
the permutation-averaged prediction is the primary metric.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluate_medmcqa_checkpoint_trajectory import (
    CHOICES,
    DEFAULT_CONFIG,
    DEFAULT_DATASET,
    DATASET_REPO,
    DATASET_REVISION,
    atomic_csv,
    atomic_json,
    load_dataset,
    load_models,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "outputs/policy_eval/medicine_dense_all48_medmcqa_perm4_20260928"
ANSWER_INSTRUCTION = "Respond with only the single letter A, B, C, or D."
OPTION_COLUMNS = ("opa", "opb", "opc", "opd")
PERMUTATIONS = tuple(tuple((position + shift) % 4 for position in range(4)) for shift in range(4))


def build_permuted_prompt(row: pd.Series, permutation: tuple[int, ...]) -> str:
    options = [str(row[column]) for column in OPTION_COLUMNS]
    displayed = [options[original_index] for original_index in permutation]
    return (
        f"Question: {row['question']}\n"
        "Choices:\n"
        f"A. {displayed[0]}\n"
        f"B. {displayed[1]}\n"
        f"C. {displayed[2]}\n"
        f"D. {displayed[3]}\n"
        f"{ANSWER_INSTRUCTION}\n"
        "Answer:"
    )


def batched(values: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def score_prompts(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    target_token_ids: list[int],
    device: str,
    batch_size: int,
) -> np.ndarray:
    score_blocks: list[np.ndarray] = []
    token_lengths = np.asarray(
        [len(tokenizer.encode(prompt, add_special_tokens=False)) for prompt in prompts],
        dtype=int,
    )
    order = np.argsort(token_lengths, kind="stable")
    sorted_prompts = [prompts[index] for index in order]
    with torch.inference_mode():
        for prompt_batch in batched(sorted_prompts, batch_size):
            encoded = tokenizer(
                prompt_batch,
                padding=True,
                add_special_tokens=False,
                return_tensors="pt",
            )
            encoded = {key: value.to(device, non_blocking=True) for key, value in encoded.items()}
            last_token_logits = model(**encoded, logits_to_keep=1).logits[:, -1, :]
            log_probs = torch.log_softmax(last_token_logits.float(), dim=-1)[:, target_token_ids]
            score_blocks.append(log_probs.cpu().numpy())
    sorted_scores = np.concatenate(score_blocks, axis=0)
    scores = np.empty_like(sorted_scores)
    scores[order] = sorted_scores
    return scores


def validate_complete(path: Path, expected_ids: list[str], step: int) -> bool:
    if not path.is_file():
        return False
    frame = pd.read_csv(path)
    required = {
        "checkpoint",
        "prompt_id",
        "gold_index",
        "identity_prediction_index",
        "prediction_index",
        "correct",
        "avg_logprob_choice_0",
        "avg_logprob_choice_1",
        "avg_logprob_choice_2",
        "avg_logprob_choice_3",
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

    target_texts = list(CHOICES)
    target_token_lists = [
        tokenizer.encode(value, add_special_tokens=False) for value in target_texts
    ]
    if any(len(tokens) != 1 for tokens in target_token_lists):
        raise RuntimeError(f"choice continuations are not single tokens: {target_token_lists}")
    target_token_ids = [tokens[0] for tokens in target_token_lists]
    if len(set(target_token_ids)) != 4:
        raise RuntimeError(f"choice continuation tokens are not unique: {target_token_ids}")

    chat_tail_probe = tokenizer.apply_chat_template(
        [{"role": "user", "content": "probe"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    if not chat_tail_probe.endswith("assistant\n"):
        raise RuntimeError(f"unexpected chat-template generation tail: {chat_tail_probe[-80:]!r}")

    load_started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map={"": device},
        low_cpu_mem_usage=True,
    )
    model.eval()
    load_seconds = time.perf_counter() - load_started

    example_count = len(dataset)
    mapped_scores = np.empty((len(PERMUTATIONS), example_count, 4), dtype=np.float32)
    permutation_label_predictions = np.empty((len(PERMUTATIONS), example_count), dtype=np.int8)
    permutation_content_predictions = np.empty((len(PERMUTATIONS), example_count), dtype=np.int8)
    prompt_token_lengths: list[np.ndarray] = []

    inference_started = time.perf_counter()
    for permutation_index, permutation in enumerate(PERMUTATIONS):
        prompts = [
            tokenizer.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": build_permuted_prompt(row, permutation),
                    }
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            for _, row in dataset.iterrows()
        ]
        prompt_token_lengths.append(
            np.asarray(
                [len(tokenizer.encode(prompt, add_special_tokens=False)) for prompt in prompts],
                dtype=int,
            )
        )
        label_scores = score_prompts(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            target_token_ids=target_token_ids,
            device=device,
            batch_size=batch_size,
        )
        label_predictions = label_scores.argmax(axis=1).astype(np.int8)
        content_predictions = np.asarray(permutation, dtype=np.int8)[label_predictions]
        permutation_label_predictions[permutation_index] = label_predictions
        permutation_content_predictions[permutation_index] = content_predictions
        for label_index, original_index in enumerate(permutation):
            mapped_scores[permutation_index, :, original_index] = label_scores[:, label_index]

    torch.cuda.synchronize(torch.device(device))
    inference_seconds = time.perf_counter() - inference_started

    averaged_scores = mapped_scores.mean(axis=0)
    predictions = averaged_scores.argmax(axis=1).astype(int)
    identity_predictions = permutation_content_predictions[0].astype(int)
    gold = dataset["cop"].to_numpy(dtype=int)
    correct = predictions == gold
    identity_correct = identity_predictions == gold
    sorted_scores = np.sort(averaged_scores, axis=1)
    margins = sorted_scores[:, -1] - sorted_scores[:, -2]

    result_values: dict[str, Any] = {
        "checkpoint": model_spec["step"],
        "prompt_id": dataset["id"].astype(str),
        "gold_index": gold,
        "gold_choice": [CHOICES[index] for index in gold],
        "identity_prediction_index": identity_predictions,
        "identity_prediction_choice": [CHOICES[index] for index in identity_predictions],
        "identity_correct": identity_correct.astype(int),
        "prediction_index": predictions,
        "prediction_choice": [CHOICES[index] for index in predictions],
        "correct": correct.astype(int),
        "top2_margin": margins,
        "subject_name": dataset["subject_name"].astype(str),
        "choice_type": dataset["choice_type"].astype(str),
    }
    for original_index in range(4):
        result_values[f"avg_logprob_choice_{original_index}"] = averaged_scores[:, original_index]
    for permutation_index in range(len(PERMUTATIONS)):
        result_values[f"perm_{permutation_index}_prediction_label"] = [
            CHOICES[index] for index in permutation_label_predictions[permutation_index]
        ]
        result_values[f"perm_{permutation_index}_prediction_index"] = (
            permutation_content_predictions[permutation_index].astype(int)
        )
    result = pd.DataFrame(result_values)

    all_token_lengths = np.concatenate(prompt_token_lengths)
    label_distributions = []
    for permutation_index in range(len(PERMUTATIONS)):
        counts = np.bincount(
            permutation_label_predictions[permutation_index].astype(int), minlength=4
        )
        label_distributions.append(
            {CHOICES[index]: float(counts[index] / example_count) for index in range(4)}
        )
    metadata = {
        "checkpoint": model_spec["step"],
        "model_name": model_spec["name"],
        "model_artifact": model_spec["artifact"],
        "model_path": str(model_path),
        "accuracy": float(correct.mean()),
        "identity_accuracy": float(identity_correct.mean()),
        "correct": int(correct.sum()),
        "examples": int(len(result)),
        "load_seconds": load_seconds,
        "inference_seconds": inference_seconds,
        "total_seconds": time.perf_counter() - started,
        "choice_continuations": target_texts,
        "choice_token_ids": target_token_ids,
        "answer_instruction": ANSWER_INSTRUCTION,
        "permutations": [list(permutation) for permutation in PERMUTATIONS],
        "permutation_label_distributions": label_distributions,
        "min_prompt_tokens": int(all_token_lengths.min()),
        "median_prompt_tokens": float(np.median(all_token_lengths)),
        "max_prompt_tokens": int(all_token_lengths.max()),
        "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
        "config_sha256": sha256_file(model_path / "config.json"),
    }

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return result, metadata


def rebuild_summary(output_root: Path, expected_ids: list[str]) -> pd.DataFrame:
    rows = []
    for path in sorted((output_root / "prompt_scores").glob("checkpoint_*.csv")):
        frame = pd.read_csv(path)
        if (
            len(frame) != len(expected_ids)
            or frame["prompt_id"].astype(str).tolist() != expected_ids
        ):
            raise RuntimeError(f"invalid prompt-level result while rebuilding summary: {path}")
        rows.append(
            {
                "checkpoint": int(frame["checkpoint"].iloc[0]),
                "accuracy": float(frame["correct"].mean()),
                "identity_accuracy": float(frame["identity_correct"].mean()),
                "correct": int(frame["correct"].sum()),
                "identity_correct": int(frame["identity_correct"].sum()),
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
        "protocol": "chat_no_space_label_perm4_mean_logprob",
        "dataset_repo": DATASET_REPO,
        "dataset_revision": DATASET_REVISION,
        "dataset_split": "validation",
        "dataset_path": str(args.dataset.resolve()),
        "dataset_sha256": sha256_file(args.dataset.resolve()),
        "examples": len(dataset),
        "config_path": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config.resolve()),
        "checkpoint_range": [args.start_step, args.end_step],
        "scoring": "four cyclic permutations; mapped content log probabilities averaged before argmax",
        "prompt_source": "MedMCQA question and choices with explicit single-letter response instruction",
        "target_delimiter": "",
        "choices": list(CHOICES),
        "answer_instruction": ANSWER_INSTRUCTION,
        "permutations": [list(permutation) for permutation in PERMUTATIONS],
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
            f"identity_accuracy={metadata['identity_accuracy']:.6f}, "
            f"total={metadata['total_seconds']:.1f}s, completed={len(summary)}",
            flush=True,
        )
    summary = rebuild_summary(output_root, prompt_ids)
    manifest["completed_checkpoints"] = summary["checkpoint"].astype(int).tolist()
    manifest["complete"] = manifest["completed_checkpoints"] == [value["step"] for value in models]
    atomic_json(output_root / "manifest.json", manifest)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
