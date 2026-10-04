"""Opt-in, single-step overlap of remote online rewards with frozen log-probs.

The actual reward batch is never reordered or mutated by the background worker.
Only a CPU payload copy is passed through the original actor -> reference calls.
No worker can remain in flight when this function returns or raises.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from time import perf_counter
from typing import Any


def prepare_inference_inputs(trainer: Any, batch: Any, metrics: dict) -> None:
    # This is the original veRL preprocessing, including its exact row order.
    import torch
    from verl.trainer.ppo.ray_trainer import compute_response_mask

    if "response_mask" not in batch.batch.keys():
        batch.batch["response_mask"] = compute_response_mask(batch)
    if trainer.config.trainer.balance_batch:
        trainer._balance_batch(batch, metrics=metrics)
    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
    images_seqlens_all = []
    for multi_modal_input in batch.non_tensor_batch["multi_modal_inputs"]:
        if "image_grid_thw" not in multi_modal_input.keys():
            continue
        images_seqlens_all.extend(multi_modal_input["images_seqlens"].tolist())
    batch.meta_info["images_seqlens"] = images_seqlens_all


def _equal(left: Any, right: Any) -> bool:
    import numpy as np
    import torch

    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and left.dtype == right.dtype and torch.equal(left, right)
    if isinstance(left, np.ndarray):
        if not isinstance(right, np.ndarray) or left.shape != right.shape or left.dtype != right.dtype:
            return False
        return all(_equal(a, b) for a, b in zip(left.flat, right.flat))
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(
            _equal(value, right[key]) for key, value in left.items()
        )
    if isinstance(left, (tuple, list)):
        return type(left) is type(right) and len(left) == len(right) and all(
            _equal(a, b) for a, b in zip(left, right)
        )
    return bool(left == right)


def clone_cpu_payload(batch: Any) -> Any:
    """Copy DataProto payload only; never copy a trainer, model, or Ray handle."""
    for key, tensor in batch.batch.items():
        if tensor.device.type != "cpu":
            raise RuntimeError(f"Log-prob prefetch requires CPU rollout tensors: {key}")
    cloned = type(batch)(batch=batch.batch.clone(), non_tensor_batch=deepcopy(batch.non_tensor_batch),
                         meta_info=deepcopy(batch.meta_info))
    for key, tensor in batch.batch.items():
        if tensor.numel() and tensor.untyped_storage().data_ptr() == cloned.batch[key].untyped_storage().data_ptr():
            raise RuntimeError(f"Prefetch copy aliases live tensor: {key}")
    return cloned


@dataclass
class PrefetchedLogProbs:
    inputs: Any
    input_keys: tuple[str, ...]
    old: Any
    old_mfu: Any
    ref: Any
    timings: dict[str, float]

    def validate_inputs(self, actual: Any) -> None:
        for key in self.input_keys:
            if key not in actual.batch or not _equal(self.inputs.batch[key], actual.batch[key]):
                raise RuntimeError(f"Prefetched log-prob input/order mismatch: {key}")
        for key, value in self.inputs.non_tensor_batch.items():
            if key not in actual.non_tensor_batch or not _equal(value, actual.non_tensor_batch[key]):
                raise RuntimeError(f"Prefetched log-prob row identity mismatch: {key}")
        for key, value in self.inputs.meta_info.items():
            if key not in actual.meta_info or not _equal(value, actual.meta_info[key]):
                raise RuntimeError(f"Prefetched log-prob metadata mismatch: {key}")
        # The reward hook may add ONLY its reward/trace/seal fields. Unexpected
        # new inputs are rejected instead of silently treating them as inert.
        if set(actual.batch.keys()) - set(self.input_keys) - {"rm_scores"}:
            raise RuntimeError("Unexpected tensor added by online reward hook")
        if set(actual.non_tensor_batch) - set(self.inputs.non_tensor_batch) - {"online_trace_ref"}:
            raise RuntimeError("Unexpected row metadata added by online reward hook")
        if set(actual.meta_info) - set(self.inputs.meta_info) - {
            "online_step_sealed", "online_optimizer_update_index", "online_manifest_hash"
        }:
            raise RuntimeError("Unexpected metadata added by online reward hook")

    def matches(self, serial: Any, kind: str) -> bool:
        """Exact gate: first resumed update uses serial outputs regardless."""
        speculative = self.old if kind == "old" else self.ref
        return speculative is not None and set(speculative.batch.keys()) == set(serial.batch.keys()) and all(
            _equal(value, serial.batch[key]) for key, value in speculative.batch.items()
        )


def prepare_with_logprob_prefetch(trainer: Any, batch: Any, metrics: dict) -> tuple[Any, PrefetchedLogProbs]:
    """Overlap one immutable inference payload with the unchanged reward hook.

    Restricted to the validated dense, text-only, non-bypass online path. The
    caller must check all aligned inputs before using these speculative outputs.
    """
    actor = trainer.config.actor_rollout_ref.actor
    correction = trainer.config.algorithm.get("rollout_correction") or {}
    if correction.get("bypass_mode", False) or actor.calculate_entropy or actor.entropy_coeff != 0.0:
        raise RuntimeError("Log-prob prefetch is validated only for non-bypass, entropy-disabled runs")
    if any(value for value in batch.non_tensor_batch["multi_modal_inputs"]):
        raise RuntimeError("Log-prob prefetch is validated only for text-only batches")
    if "routed_experts" in batch.batch or any(key.startswith("teacher_") for key in batch.batch.keys()):
        raise RuntimeError("Log-prob prefetch is not validated for router replay or distillation")
    snapshot = clone_cpu_payload(batch)
    prepare_inference_inputs(trainer, snapshot, {})
    input_keys = tuple(snapshot.batch.keys())
    timings = {}

    def infer() -> PrefetchedLogProbs:
        start = perf_counter()
        old, mfu = trainer._compute_old_log_prob(snapshot, calculate_entropy=False)
        timings["old_log_prob"] = perf_counter() - start
        # Reference sees the same old-logprob union as the serial trainer.
        reference_input = snapshot.union(old)
        start = perf_counter()
        ref = trainer._compute_ref_log_prob(reference_input) if trainer.use_reference_policy else None
        timings["ref"] = perf_counter() - start
        print(
            f"[online-prefetch] step={getattr(trainer, 'global_steps', '?')} log-probs ready "
            f"old_s={timings['old_log_prob']:.3f} ref_s={timings['ref']:.3f}",
            flush=True,
        )
        return PrefetchedLogProbs(snapshot, input_keys, old, mfu, ref, timings)

    start = perf_counter()
    # Context shutdown waits even when reward or log-prob calculation raises.
    # The actual hook stays on the main thread, preserving all runtime state.
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="online-logprob") as pool:
        future = pool.submit(infer)
        reward_start = perf_counter()
        batch = trainer._prepare_online_rewards(batch)
        metrics["overlap/remote_reward_s"] = perf_counter() - reward_start
        result = future.result()
    metrics["overlap/wall_s"] = perf_counter() - start
    metrics["overlap/logprob_s"] = timings["old_log_prob"] + timings["ref"]
    return batch, result
