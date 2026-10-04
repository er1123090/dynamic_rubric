"""RQ2 artifacts emitted by opt-in hooks in the public EvoRubrics trainer.

The observer never supplies training rewards. Step t denotes t completed
co-evolution iterations; training responses at iteration t use input state t-1.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from dynamic_rubric.artifacts import write_bytes_atomic, write_json_atomic
from dynamic_rubric.hashing import sha256_bytes, sha256_file
from dynamic_rubric.phase1.provenance import response_id


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if hasattr(value, "detach"):
        return _plain(value.detach().cpu().tolist())
    if hasattr(value, "tolist"):
        return _plain(value.tolist())
    if hasattr(value, "item"):
        return _plain(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _settings(trainer: Any) -> tuple[Path, str, int]:
    cfg = trainer.config.rq2
    return Path(cfg.run_root), str(cfg.domain), int(cfg.seed)


def _write(trainer: Any, relative: str, value: Any) -> None:
    root, _, _ = _settings(trainer)
    write_json_atomic(root / relative, _plain(value))


def on_iteration(trainer: Any, payload: dict[str, Any]) -> None:
    """Persist the actual training pool and the evaluator before optimization."""
    _, domain, seed = _settings(trainer)
    data = _plain(payload)
    input_step = int(data["input_step"])
    update_step = int(data["update_step"])
    ids = data.get("prompt_ids", data.get("sample_ids"))
    if ids is None or len(ids) != len(data["questions"]):
        raise ValueError("Evo training audit requires original prompt IDs")
    if any(not isinstance(x, str) or not x for x in ids) or len(set(ids)) != len(ids):
        raise ValueError("Evo training prompt IDs must be unique nonempty strings")
    records = []
    for prompt_id, answers in zip(ids, data["answers_per_query"]):
        if len(answers) != 4:
            raise ValueError("Training group must contain M=4 answers")
        for index, answer in enumerate(answers):
            records.append(
                {
                    "domain": domain,
                    "method": "evorubrics",
                    "seed": seed,
                    "global_step": update_step,
                    "checkpoint_id": input_step,
                    "prompt_id": prompt_id,
                    "response_id": response_id(
                        domain=domain,
                        method="evorubrics",
                        seed=seed,
                        prompt_id=prompt_id,
                        pool="train_batch",
                        policy_checkpoint=str(input_step),
                        sample_index=index,
                    ),
                    "pool": "train_batch",
                    "policy_checkpoint": input_step,
                    "evaluator_checkpoint": input_step,
                    "fresh_or_stale": "fresh",
                    "used_for_gradient": True,
                    "sample_index": index,
                    "text": answer,
                }
            )
    data.update(
        {
            "schema_version": 1,
            "domain": domain,
            "method": "evorubrics",
            "seed": seed,
            "pool": "train_batch",
            "prompt_ids": ids,
            "responses": records,
            "input_policy_step": input_step,
            "input_evaluator_step": input_step,
            "completed_update_step": update_step,
            "initial_rubric_is_ground_truth": False,
        }
    )
    trainer._rq2_iteration = data
    _write(trainer, f"audit/train_batch/step_{update_step:06d}.json", data)


def on_advantages(trainer: Any, adapter: str, batch: Any) -> None:
    """Save exactly the tensors used for each adapter's GRPO update."""
    import torch
    from safetensors.torch import save

    step = int(trainer.global_steps) + 1
    tensors = {}
    for key in (
        "input_ids",
        "position_ids",
        "responses",
        "response_mask",
        "attention_mask",
        "token_level_scores",
        "token_level_rewards",
        "advantages",
        "old_log_probs",
        "ref_log_prob",
    ):
        if key in batch.batch:
            value = batch.batch[key]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"advantage field {key} is not a tensor")
            tensors[key] = value.detach().cpu().contiguous().clone()
    required = {"responses", "response_mask", "advantages", "old_log_probs", "ref_log_prob"}
    if missing := required.difference(tensors):
        raise ValueError(f"advantage audit is missing tensors: {sorted(missing)}")
    root, _, _ = _settings(trainer)
    stem = f"audit/advantages/step_{step:06d}_{adapter}"
    artifact_relative = f"{stem}.safetensors"
    artifact = save(tensors)
    write_bytes_atomic(root / artifact_relative, artifact)
    tensor_manifest = {}
    for key, value in sorted(tensors.items()):
        raw = value.view(torch.uint8).numpy().tobytes()
        tensor_manifest[key] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": sha256_bytes(raw),
        }
    uid = _plain(batch.non_tensor_batch.get("uid", []))
    _write(
        trainer,
        f"{stem}.json",
        {
            "schema_version": 1,
            "global_step": step,
            "input_step": step - 1,
            "adapter": adapter,
            "row_alignment": "training DataProto order; uid identifies GRPO groups",
            "artifact": {
                "path": artifact_relative,
                "sha256": sha256_bytes(artifact),
                "bytes": len(artifact),
                "format": "safetensors",
            },
            "tensors": tensor_manifest,
            "tensor_keys": sorted(tensors),
            "uid": uid,
            "uid_count": len(uid),
            "actual_training": True,
        },
    )


def on_step(trainer: Any, metrics: dict[str, Any], batch_dict: Any) -> None:
    root, domain, seed = _settings(trainer)
    step = int(trainer.global_steps)
    payload = trainer._rq2_iteration
    previous = root / f"metrics/step_{step - 1:06d}.json"
    prior = json.loads(previous.read_text()) if previous.exists() else {}
    if step > 1 and not previous.exists():
        raise ValueError("Previous step metrics missing; cannot reconstruct exposure clock")
    completions = len(payload["responses"])
    rubric_completions = sum(map(len, payload["rubrics_per_query"]))
    token_counts = {
        role: payload.get(f"{role}_response_token_counts", [])
        for role in ("policy", "rubric", "reflect")
    }
    generated_tokens = sum(sum(v) for v in token_counts.values())
    generated_completions = sum(len(v) for v in token_counts.values())
    policy_lengths = token_counts["policy"]
    record = {
        "schema_version": 1,
        "global_step": step,
        "domain": domain,
        "method": "evorubrics",
        "seed": seed,
        "batch_prompt_count": len(payload["prompt_ids"]),
        "cumulative_prompt_exposures": prior.get("cumulative_prompt_exposures", 0)
        + len(payload["prompt_ids"]),
        "policy_completions": completions,
        "cumulative_policy_completions": prior.get("cumulative_policy_completions", 0)
        + completions,
        "rubric_completions": rubric_completions,
        "token_counts": token_counts,
        "cumulative_completions": prior.get("cumulative_completions", 0) + generated_completions,
        "cumulative_response_tokens": prior.get("cumulative_response_tokens", 0) + generated_tokens,
        "response_length": sum(policy_lengths) / len(policy_lengths) if policy_lengths else None,
        "completion_definition": "policy + rubric + reflection outputs; roles logged separately",
        "adjacent_policy_kl": None,
        "adjacent_policy_kl_status": "pending_same_response_checkpoint_logprob_audit",
        "metrics": _plain(metrics),
        "input_policy_step": step - 1,
        "input_evaluator_step": step - 1,
    }
    _write(trainer, f"metrics/step_{step:06d}.json", record)


def _adapter_files(step_dir: Path, adapter: str, *, require_optimizer: bool) -> dict[str, Any]:
    adapter_dir = step_dir / f"lora_adapter_{adapter}"
    weights = adapter_dir / "adapter_model.safetensors"
    config = adapter_dir / "adapter_config.json"
    if not weights.is_file() or weights.stat().st_size < 16 or not config.is_file():
        raise ValueError(f"Incomplete adapter: {adapter_dir}")
    from safetensors import safe_open

    with safe_open(str(weights), framework="pt", device="cpu") as handle:
        if not list(handle.keys()):
            raise ValueError(f"Empty adapter: {weights}")
    optimizer = step_dir / f"optimizer_{adapter}/optimizer_state.pt"
    if require_optimizer and (not optimizer.is_file() or not optimizer.stat().st_size):
        raise ValueError(f"Missing optimizer: {optimizer}")
    return {
        "step_dir": str(step_dir),
        "adapter_path": str(adapter_dir),
        "weights_sha256": sha256_file(weights),
        "config_sha256": sha256_file(config),
        "optimizer_path": str(optimizer),
        "optimizer_present": optimizer.is_file(),
    }


def on_checkpoint(trainer: Any, policy_path: str, rubrics_path: str) -> None:
    """Publish a pair only after both parameters and optimizer files validate."""
    root, domain, seed = _settings(trainer)
    step = int(trainer.global_steps)
    pair = {
        "schema_version": 1,
        "status": "committed",
        "global_step": step,
        "domain": domain,
        "method": "evorubrics",
        "seed": seed,
        "policy": _adapter_files(Path(policy_path), "policy_llm", require_optimizer=True),
        "generator": _adapter_files(
            Path(rubrics_path), "rubrics_generator", require_optimizer=True
        ),
        "sampling_restart": "RNG files recorded separately; vLLM exact replay not certified",
    }
    _write(trainer, f"checkpoints/committed/step_{step:06d}.json", pair)
    write_json_atomic(root / "checkpoints/latest.json", pair, immutable=False)
    # Retain every parameter snapshot, but only the latest optimizer state.
    if trainer.config.rq2.get("prune_old_optimizers", True):
        for committed in (root / "checkpoints/committed").glob("step_*.json"):
            old = json.loads(committed.read_text())
            if int(old["global_step"]) >= step:
                continue
            for role in ("policy", "generator"):
                optimizer = Path(old[role]["optimizer_path"])
                if optimizer.is_relative_to(root):
                    optimizer.unlink(missing_ok=True)


def discover_checkpoint_pairs(run_root: str | Path) -> dict[str, Any]:
    """Discover all committed pairs; incomplete saves are reported, never bridged."""
    root = Path(run_root)
    valid, excluded = [], []
    for path in sorted((root / "checkpoints/committed").glob("step_*.json")):
        row = json.loads(path.read_text())
        try:
            for role, adapter in (("policy", "policy_llm"), ("generator", "rubrics_generator")):
                actual = _adapter_files(
                    Path(row[role]["step_dir"]), adapter, require_optimizer=False
                )
                if any(actual[k] != row[role][k] for k in ("weights_sha256", "config_sha256")):
                    raise ValueError(f"Changed {role} snapshot")
            valid.append(row)
        except (ValueError, OSError) as error:
            excluded.append({"path": str(path), "reason": str(error)})
    steps = sorted(int(r["global_step"]) for r in valid)
    return {
        "checkpoints": valid,
        "steps": steps,
        "excluded": excluded,
        "expected_cells": [[tau, t] for t in steps for tau in steps if tau <= t],
        "analysis_status": "pending_response_generation_and_grading",
    }
