#!/usr/bin/env python3
"""Measure Notion E2 reward-surrogate gradient alignment at steps 6/21/36.

For each checkpoint, the script keeps the policy, prompts, generated responses,
response tokens, and response mask fixed.  Only the rubric-derived scalar GRPO
advantages change (R0, immediately previous rubric, current rubric).  The loss is
the veRL on-policy token-mean reward surrogate; KL and optimizer updates are
intentionally excluded.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from dynamic_rubric.artifacts import read_jsonl, write_json_atomic
from dynamic_rubric.hashing import canonical_json_bytes, sha256_file


ROOT = Path(__file__).resolve().parents[2]
E2_ROOT = ROOT / "outputs/analysis/medicine_online_e2_gradient_alignment_20260921"
TARGETS = {6: {"r0": 0, "previous": 5, "current": 6}, 21: {"r0": 0, "previous": 20, "current": 21}, 36: {"r0": 0, "previous": 35, "current": 36}}
EPSILON = 1e-6
CLIP_NORM = 1.0
MAX_BATCH_TOKENS = 8192
MAX_BATCH_SIZE = 8


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def advantages(rows: list[dict[str, Any]]) -> dict[str, float]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["prompt_id"])].append(row)
    output: dict[str, float] = {}
    for prompt_id, group in grouped.items():
        if len(group) != 16:
            raise RuntimeError(f"{prompt_id}: expected 16 Pool-B responses, got {len(group)}")
        rewards = torch.tensor([float(row["reward"]) for row in group], dtype=torch.float64)
        std = rewards.std(unbiased=True)
        mean = rewards.mean()
        values = (rewards - mean) / (std + EPSILON)
        for row, value in zip(group, values.tolist()):
            output[str(row["response_id"])] = float(value)
    return output


def load_step(step: int) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]], dict[str, Any]]:
    response_path = E2_ROOT / f"responses/online/step-{step:03d}/probe_B.jsonl"
    if not response_path.is_file():
        raise FileNotFoundError(response_path)
    responses = read_jsonl(response_path)
    if len(responses) != 20 * 16:
        raise RuntimeError(f"step {step}: expected 320 Pool-B responses")
    response_ids = [str(row["response_id"]) for row in responses]
    if len(response_ids) != len(set(response_ids)):
        raise RuntimeError(f"step {step}: duplicate response IDs")
    condition_advantages: dict[str, dict[str, float]] = {}
    grade_files = {}
    for condition, evaluator_step in TARGETS[step].items():
        grade_path = E2_ROOT / (
            f"grades/cells/online/policy-{step:03d}/evaluator-{evaluator_step:03d}.jsonl"
        )
        grades = read_jsonl(grade_path)
        if {str(row["response_id"]) for row in grades} != set(response_ids):
            raise RuntimeError(f"step {step} {condition}: grade/response inventory drift")
        condition_advantages[condition] = advantages(grades)
        grade_files[condition] = {
            "path": str(grade_path),
            "sha256": sha256_file(grade_path),
            "evaluator_step": evaluator_step,
        }
    provenance = {
        "response_path": str(response_path),
        "response_sha256": sha256_file(response_path),
        "grade_files": grade_files,
    }
    return responses, condition_advantages, provenance


def prompt_tokens(tokenizer, messages: list[dict[str, Any]]) -> list[int]:
    tokens = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    if isinstance(tokens, Mapping):
        tokens = tokens["input_ids"]
    if isinstance(tokens, torch.Tensor):
        tokens = tokens.tolist()
    if tokens and isinstance(tokens[0], list):
        tokens = tokens[0]
    return [int(value) for value in tokens]


def tokenize_rows(tokenizer, responses: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prompts = {
        str(row["prompt_id"]): row
        for row in read_jsonl(E2_ROOT / "manifests/validation_prompts.jsonl")
    }
    prompt_cache = {
        prompt_id: prompt_tokens(tokenizer, list(row["messages"]))
        for prompt_id, row in prompts.items()
    }
    samples = []
    token_hash = hashlib.sha256()
    for response in responses:
        prompt_id = str(response["prompt_id"])
        response_ids = tokenizer.encode(str(response["text"]), add_special_tokens=False)
        if not response_ids:
            raise RuntimeError(f"empty response tokenization: {response['response_id']}")
        prompt_ids = prompt_cache[prompt_id]
        input_ids = prompt_ids + [int(value) for value in response_ids]
        if len(input_ids) > 7680:
            raise RuntimeError(f"sequence exceeds served context: {response['response_id']}")
        record = {
            "response_id": str(response["response_id"]),
            "prompt_id": prompt_id,
            "input_ids": input_ids,
            "prompt_length": len(prompt_ids),
            "response_length": len(response_ids),
        }
        token_hash.update(canonical_json_bytes(record))
        samples.append(record)
    samples.sort(key=lambda row: (len(row["input_ids"]), row["response_id"]))
    lengths = [int(row["response_length"]) for row in samples]
    sequence_lengths = [len(row["input_ids"]) for row in samples]
    manifest = {
        "tokenized_response_inventory_sha256": token_hash.hexdigest(),
        "response_count": len(samples),
        "response_tokens": sum(lengths),
        "response_length_min": min(lengths),
        "response_length_max": max(lengths),
        "response_length_mean": sum(lengths) / len(lengths),
        "sequence_length_max": max(sequence_lengths),
        "chat_template_sha256": hashlib.sha256(
            str(tokenizer.chat_template).encode("utf-8")
        ).hexdigest(),
        "eos_appended": False,
        "mask": "response_text_tokens_only",
    }
    return samples, manifest


def batches(samples: list[dict[str, Any]]) -> Iterable[list[dict[str, Any]]]:
    current: list[dict[str, Any]] = []
    current_max = 0
    for sample in samples:
        length = len(sample["input_ids"])
        proposed_max = max(current_max, length)
        if current and (
            len(current) >= MAX_BATCH_SIZE
            or proposed_max * (len(current) + 1) > MAX_BATCH_TOKENS
        ):
            yield current
            current = []
            current_max = 0
        current.append(sample)
        current_max = max(current_max, length)
    if current:
        yield current


def parameter_group(name: str) -> str:
    marker = ".layers."
    if marker in name:
        suffix = name.split(marker, 1)[1]
        return f"layer_{int(suffix.split('.', 1)[0]):02d}"
    if "embed_tokens" in name:
        return "embedding"
    if name.endswith("lm_head.weight"):
        return "lm_head"
    return "other"


def gradient_norm(gradient: Mapping[str, torch.Tensor]) -> float:
    total = 0.0
    for tensor in gradient.values():
        total += float(torch.sum(tensor * tensor, dtype=torch.float64).item())
    return math.sqrt(total)


def compare_gradients(
    left: Mapping[str, torch.Tensor],
    right: Mapping[str, torch.Tensor],
    left_name: str,
    right_name: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if set(left) != set(right):
        raise RuntimeError("gradient parameter inventories differ")
    totals: dict[str, dict[str, float]] = defaultdict(lambda: {"dot": 0.0, "left2": 0.0, "right2": 0.0, "diff2": 0.0})
    for name in left:
        a = left[name]
        b = right[name]
        difference = a - b
        values = totals[parameter_group(name)]
        values["dot"] += float(torch.sum(a * b, dtype=torch.float64).item())
        values["left2"] += float(torch.sum(a * a, dtype=torch.float64).item())
        values["right2"] += float(torch.sum(b * b, dtype=torch.float64).item())
        values["diff2"] += float(
            torch.sum(difference * difference, dtype=torch.float64).item()
        )
    global_values = {key: sum(group[key] for group in totals.values()) for key in ("dot", "left2", "right2", "diff2")}

    def metrics(values: Mapping[str, float]) -> dict[str, float]:
        left_norm = math.sqrt(values["left2"])
        right_norm = math.sqrt(values["right2"])
        denominator = left_norm * right_norm
        cosine = values["dot"] / denominator if denominator else float("nan")
        diff_norm = math.sqrt(values["diff2"])
        symmetric_base = 0.5 * (left_norm + right_norm)
        left_scale = min(1.0, CLIP_NORM / left_norm) if left_norm else 1.0
        right_scale = min(1.0, CLIP_NORM / right_norm) if right_norm else 1.0
        clipped_diff2 = (
            left_scale * left_scale * values["left2"]
            + right_scale * right_scale * values["right2"]
            - 2.0 * left_scale * right_scale * values["dot"]
        )
        return {
            "cosine": cosine,
            "left_norm": left_norm,
            "right_norm": right_norm,
            "norm_ratio_right_over_left": right_norm / left_norm if left_norm else float("nan"),
            "difference_norm": diff_norm,
            "relative_difference_to_left": diff_norm / left_norm if left_norm else float("nan"),
            "symmetric_relative_difference": diff_norm / symmetric_base if symmetric_base else float("nan"),
            "left_clip_scale": left_scale,
            "right_clip_scale": right_scale,
            "clipped_difference_norm": math.sqrt(max(0.0, clipped_diff2)),
        }

    global_metrics = {
        "left": left_name,
        "right": right_name,
        **metrics(global_values),
    }
    layer_rows = [
        {"left": left_name, "right": right_name, "parameter_group": group, **metrics(values)}
        for group, values in sorted(totals.items())
    ]
    return global_metrics, layer_rows


def compute_gradient(
    model,
    samples: list[dict[str, Any]],
    condition_advantages: Mapping[str, float],
    pad_token_id: int,
    condition: str,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    device = next(model.parameters()).device
    total_response_tokens = sum(int(sample["response_length"]) for sample in samples)
    accumulators: dict[str, torch.Tensor] = {}
    handles = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        def accumulate(param: torch.Tensor, *, parameter_name: str = name) -> None:
            if param.grad is None:
                return
            target = accumulators.get(parameter_name)
            if target is None:
                target = torch.zeros_like(param, dtype=torch.float32, device=device)
                accumulators[parameter_name] = target
            target.add_(param.grad.float())
            param.grad.zero_()

        handles.append(parameter.register_post_accumulate_grad_hook(accumulate))

    model.zero_grad(set_to_none=True)
    loss_sum = 0.0
    batch_count = 0
    for batch in batches(samples):
        batch_count += 1
        max_length = max(len(sample["input_ids"]) for sample in batch)
        input_ids = torch.full(
            (len(batch), max_length), pad_token_id, dtype=torch.long, device=device
        )
        attention_mask = torch.zeros_like(input_ids)
        labels = torch.full((len(batch), max_length - 1), -100, dtype=torch.long, device=device)
        weights = torch.zeros((len(batch), max_length - 1), dtype=torch.float32, device=device)
        for index, sample in enumerate(batch):
            ids = torch.tensor(sample["input_ids"], dtype=torch.long, device=device)
            input_ids[index, : len(ids)] = ids
            attention_mask[index, : len(ids)] = 1
            begin = int(sample["prompt_length"]) - 1
            finish = begin + int(sample["response_length"])
            labels[index, begin:finish] = ids[begin + 1 : finish + 1]
            weights[index, begin:finish] = float(
                condition_advantages[str(sample["response_id"])]
            )
        output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits = output.logits[:, :-1, :]
        token_nll = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).reshape_as(labels)
        loss = torch.sum(token_nll * weights) / total_response_tokens
        loss_sum += float(loss.detach().cpu())
        loss.backward()
        del output, logits, token_nll, loss, input_ids, attention_mask, labels, weights

    for handle in handles:
        handle.remove()
    model.zero_grad(set_to_none=True)
    cpu_gradient = {name: tensor.cpu() for name, tensor in accumulators.items()}
    norm = gradient_norm(cpu_gradient)
    stats = {
        "condition": condition,
        "surrogate_loss": loss_sum,
        "gradient_norm": norm,
        "global_clip_norm": CLIP_NORM,
        "clip_scale": min(1.0, CLIP_NORM / norm) if norm else 1.0,
        "batch_count": batch_count,
        "max_batch_tokens": MAX_BATCH_TOKENS,
        "max_batch_size": MAX_BATCH_SIZE,
    }
    del accumulators
    torch.cuda.empty_cache()
    return cpu_gradient, stats


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_step(step: int) -> dict[str, Any]:
    if step not in TARGETS:
        raise ValueError(f"step must be one of {tuple(TARGETS)}")
    export = E2_ROOT / f"policy_exports/global_step_{step}"
    if not (export / "audit_export_manifest.json").is_file():
        raise FileNotFoundError(f"missing sealed policy export: {export}")
    responses, condition_advantages, input_provenance = load_step(step)
    tokenizer = AutoTokenizer.from_pretrained(export, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    samples, token_manifest = tokenize_rows(tokenizer, responses)
    token_manifest["tokenizer_path"] = str(export)
    token_manifest["tokenizer_config_sha256"] = sha256_file(export / "tokenizer_config.json")

    torch.manual_seed(11)
    torch.cuda.manual_seed_all(11)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = AutoModelForCausalLM.from_pretrained(
        export,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    ).cuda()
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.eval()

    stored: dict[str, dict[str, torch.Tensor]] = {}
    condition_stats = []
    pair_metrics = []
    layer_metrics = []
    for condition in ("r0", "previous", "current"):
        gradient, stats = compute_gradient(
            model,
            samples,
            condition_advantages[condition],
            int(tokenizer.pad_token_id),
            condition,
        )
        condition_stats.append(stats)
        for prior_name, prior_gradient in stored.items():
            global_row, layer_rows = compare_gradients(
                prior_gradient, gradient, prior_name, condition
            )
            pair_metrics.append(global_row)
            layer_metrics.extend(layer_rows)
        if condition != "current":
            stored[condition] = gradient
        else:
            del gradient
        gc.collect()

    del model, stored
    gc.collect()
    torch.cuda.empty_cache()
    result = {
        "schema_version": 1,
        "experiment": "notion_e2_gradient_alignment",
        "checkpoint_step": step,
        "policy_export": str(export),
        "policy_export_manifest_sha256": sha256_file(export / "audit_export_manifest.json"),
        "inputs": input_provenance,
        "tokenization": token_manifest,
        "conditions": {name: evaluator for name, evaluator in TARGETS[step].items()},
        "condition_stats": condition_stats,
        "pair_metrics": pair_metrics,
        "loss_contract": {
            "advantage": "GRPO group scalar, sample std (torch unbiased=True), epsilon=1e-6",
            "surrogate": "negative advantage times response-token log probability",
            "aggregation": "token-mean across the full 20x16 fixed response pool",
            "ppo_ratio": "on-policy evaluation point, ratio=1",
            "kl_included": False,
            "optimizer_step": False,
            "global_clip_norm_reported_not_applied": CLIP_NORM,
            "parameter_dtype": "bfloat16",
            "gradient_accumulation_dtype": "float32",
        },
    }
    step_root = E2_ROOT / f"gradients/step-{step:03d}"
    write_json_atomic(step_root / "metrics.json", result, immutable=False)
    write_csv(step_root / "pair_metrics.csv", pair_metrics)
    write_csv(step_root / "layer_pair_metrics.csv", layer_metrics)
    return result


def summarize(results: list[dict[str, Any]]) -> None:
    rows = []
    for result in results:
        by_pair = {
            f"{row['left']}_vs_{row['right']}": row for row in result["pair_metrics"]
        }
        for pair_name, row in by_pair.items():
            rows.append({"checkpoint_step": result["checkpoint_step"], "pair": pair_name, **{key: value for key, value in row.items() if key not in {"left", "right"}}})
    write_csv(E2_ROOT / "gradients/summary.csv", rows)
    write_json_atomic(
        E2_ROOT / "gradients/summary.json",
        {
            "schema_version": 1,
            "experiment": "notion_e2_gradient_alignment",
            "checkpoints": [result["checkpoint_step"] for result in results],
            "pair_metrics": rows,
        },
        immutable=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", default="6,21,36")
    args = parser.parse_args()
    steps = [int(value) for value in args.steps.split(",") if value.strip()]
    results = [run_step(step) for step in steps]
    summarize(results)
    print(json.dumps({"completed_steps": steps, "output_root": str(E2_ROOT / "gradients")}, sort_keys=True))


if __name__ == "__main__":
    main()
