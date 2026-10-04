"""Discover committed policy checkpoints for post-hoc Phase-1 analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Any


_CHECKPOINT_NAME = re.compile(r"global_step_(\d+)$")
_MODEL_PATTERN = "actor/model_world_size_*_rank_*.pt"


class CheckpointInventoryError(RuntimeError):
    """Raised when a training run has no trustworthy checkpoint inventory."""


@dataclass(frozen=True)
class CheckpointRecord:
    step: int
    path: str
    committed: bool
    model_parameter_files: tuple[str, ...]
    exclusion_reason: str | None = None


def discover_committed_policy_checkpoints(run_root: str | Path) -> dict[str, Any]:
    """Return every numerically ordered checkpoint committed by the trainer.

    A checkpoint is usable only when its step is no newer than the atomic VERL
    tracker and it contains at least one non-empty actor model-parameter shard.
    This excludes a directory that is still being written without guessing from
    its name or modification time.
    """

    root = Path(run_root).resolve()
    checkpoint_root = root / "verl-run" / "checkpoints"
    tracker = checkpoint_root / "latest_checkpointed_iteration.txt"
    if not tracker.is_file():
        raise CheckpointInventoryError(f"missing checkpoint commit tracker: {tracker}")
    try:
        latest_committed = int(tracker.read_text(encoding="utf-8").strip())
    except ValueError as exc:
        raise CheckpointInventoryError(f"invalid checkpoint commit tracker: {tracker}") from exc

    included: list[CheckpointRecord] = []
    excluded: list[CheckpointRecord] = []
    for path in checkpoint_root.iterdir():
        if not path.is_dir():
            continue
        match = _CHECKPOINT_NAME.fullmatch(path.name)
        if match is None:
            continue
        step = int(match.group(1))
        model_files = tuple(
            str(item.resolve())
            for item in sorted(path.glob(_MODEL_PATTERN))
            if item.is_file() and item.stat().st_size > 0
        )
        reason = None
        if step > latest_committed:
            reason = "newer_than_commit_tracker"
        else:
            fsdp_config = path / "actor" / "fsdp_config.json"
            try:
                world_size = int(json.loads(fsdp_config.read_text(encoding="utf-8"))["world_size"])
            except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                reason = "missing_or_invalid_fsdp_metadata"
            else:
                expected = {
                    str((path / "actor" / f"model_world_size_{world_size}_rank_{rank}.pt").resolve())
                    for rank in range(world_size)
                }
                if world_size <= 0 or set(model_files) != expected:
                    reason = "missing_nonempty_model_parameter_shards"
        record = CheckpointRecord(
            step=step,
            path=str(path.resolve()),
            committed=reason is None,
            model_parameter_files=model_files,
            exclusion_reason=reason,
        )
        (included if reason is None else excluded).append(record)

    included.sort(key=lambda item: item.step)
    excluded.sort(key=lambda item: item.step)
    if not included:
        raise CheckpointInventoryError(f"no committed policy checkpoints found under {checkpoint_root}")
    return {
        "run_root": str(root),
        "checkpoint_root": str(checkpoint_root),
        "commit_tracker": str(tracker.resolve()),
        "latest_committed_step": latest_committed,
        "selection": "all_committed_saved_policy_checkpoints",
        "verification_scope": (
            "commit tracker, FSDP world-size metadata, and all non-empty expected model shards; "
            "parameter tensors are not load-tested by this CPU-only inventory"
        ),
        "included": [asdict(item) for item in included],
        "excluded": [asdict(item) for item in excluded],
        "steps": [item.step for item in included],
    }


def comparison_plan(
    steps: list[int] | tuple[int, ...],
    *,
    reuse_horizon_anchors: list[int] | tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """Plan all adjacent comparisons and an optionally anchor-only horizon."""

    ordered = tuple(int(step) for step in steps)
    if not ordered or ordered != tuple(sorted(set(ordered))):
        raise CheckpointInventoryError("checkpoint steps must be non-empty, unique, and increasing")
    requested = ordered if reuse_horizon_anchors is None else tuple(reuse_horizon_anchors)
    if (
        not requested
        or any(type(step) is not int or step < 0 for step in requested)
        or requested != tuple(sorted(set(requested)))
    ):
        raise CheckpointInventoryError(
            "reuse horizon anchors must be non-empty, nonnegative, unique, and increasing"
        )
    missing = [step for step in requested if step <= ordered[-1] and step not in ordered]
    if missing:
        raise CheckpointInventoryError(f"required reuse horizon anchors are not saved: {missing}")
    anchors = tuple(step for step in requested if step <= ordered[-1])
    if not anchors:
        raise CheckpointInventoryError("no reuse horizon anchors are available yet")
    adjacent = [
        {"stale_evaluator_step": stale, "fresh_policy_evaluator_step": current}
        for stale, current in zip(ordered, ordered[1:])
    ]
    triangle = [
        {"evaluator_step": evaluator, "policy_step": policy}
        for policy in anchors
        for evaluator in anchors
        if evaluator <= policy
    ]
    required_cells = {(step, step) for step in ordered}
    required_cells.update(zip(ordered, ordered[1:]))
    required_cells.update((row["evaluator_step"], row["policy_step"]) for row in triangle)
    return {
        "policy_checkpoints": list(ordered),
        "evaluator_checkpoints": list(ordered),
        "reuse_horizon_selection": (
            "all_saved" if reuse_horizon_anchors is None else "selected_anchor_checkpoints_only"
        ),
        "reuse_horizon_anchors": list(anchors),
        "pending_reuse_horizon_anchors": [step for step in requested if step > ordered[-1]],
        "adjacent": adjacent,
        "reuse_horizon_triangle": triangle,
        "required_scoring_cells": [
            {"evaluator_step": evaluator, "policy_step": policy}
            for evaluator, policy in sorted(required_cells, key=lambda cell: (cell[1], cell[0]))
        ],
        "required_scoring_cell_count": len(required_cells),
    }
