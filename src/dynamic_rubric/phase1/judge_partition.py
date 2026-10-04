"""Deterministic whole-policy-column partitioning for heterogeneous judges."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from decimal import Decimal
from typing import Any


class JudgePartitionError(ValueError):
    """Raised when a cell plan cannot be partitioned without changing coverage."""


Cell = tuple[int, int]
BLOCKED_POLICY_STEPS = frozenset({48})


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise JudgePartitionError(f"{label} must be an integer")
    return value


def _cells(values: Iterable[Any], *, label: str) -> tuple[Cell, ...]:
    cells: list[Cell] = []
    for index, value in enumerate(values):
        if isinstance(value, Mapping):
            evaluator = _integer(value.get("evaluator_step"), f"{label}[{index}].evaluator_step")
            policy = _integer(value.get("policy_step"), f"{label}[{index}].policy_step")
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            evaluator = _integer(value[0], f"{label}[{index}][0]")
            policy = _integer(value[1], f"{label}[{index}][1]")
        else:
            raise JudgePartitionError(f"{label}[{index}] must identify one evaluator-policy cell")
        cells.append((evaluator, policy))
    if len(cells) != len(set(cells)):
        raise JudgePartitionError(f"{label} contains duplicate cells")
    return tuple(cells)


def _plan(plan: Mapping[str, Any]) -> tuple[tuple[int, ...], tuple[Cell, ...]]:
    if not isinstance(plan, Mapping) or plan.get("schema_version") != 1:
        raise JudgePartitionError("cell plan must use schema_version 1")
    raw_steps = plan.get("steps")
    raw_cells = plan.get("cells")
    if not isinstance(raw_steps, list) or not isinstance(raw_cells, list):
        raise JudgePartitionError("cell plan must contain list-valued steps and cells")
    steps = tuple(_integer(step, "plan step") for step in raw_steps)
    if not steps or steps != tuple(sorted(set(steps))) or any(step < 0 for step in steps):
        raise JudgePartitionError("plan steps must be non-negative, unique, and increasing")
    cells = _cells(raw_cells, label="plan cells")
    if not cells:
        raise JudgePartitionError("cell plan must contain at least one cell")
    step_set = set(steps)
    if any(evaluator not in step_set or policy not in step_set for evaluator, policy in cells):
        raise JudgePartitionError("cell plan references a checkpoint outside steps")
    if any(evaluator > policy for evaluator, policy in cells):
        raise JudgePartitionError("cell plan contains a future evaluator")
    return steps, cells


def _backend_plan(
    *,
    source: Mapping[str, Any],
    steps: tuple[int, ...],
    cells: Iterable[Cell],
    backend: str,
    partition_policy: str = "whole_policy_column_greedy_v1",
) -> dict[str, Any]:
    ordered = sorted(cells, key=lambda cell: (cell[1], cell[0]))
    return {
        "schema_version": 1,
        "analysis": source.get("analysis", "explicit_cell_plan"),
        "steps": list(steps),
        "cells": [
            {"evaluator_step": evaluator, "policy_step": policy}
            for evaluator, policy in ordered
        ],
        "judge_partition": {
            "backend": backend,
            "policy": partition_policy,
        },
    }


def _backend_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"inference_b", "trainer"}:
        raise JudgePartitionError(f"{label} must contain exactly inference_b and trainer")
    return value


def _rate(value: Any, *, label: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise JudgePartitionError(f"{label} must be a positive finite number")
    if not math.isfinite(float(value)) or value <= 0:
        raise JudgePartitionError(f"{label} must be a positive finite number")
    return Decimal(str(value))


def _completion_counts(
    values: Any,
    *,
    label: str,
    owned_cells: set[Cell],
    all_cells: set[Cell],
    responses_per_cell: int,
) -> dict[Cell, int]:
    if not isinstance(values, Mapping):
        raise JudgePartitionError(f"{label} must map cells to completed response counts")
    result: dict[Cell, int] = {}
    for raw_cell, raw_count in values.items():
        cell = _cells((raw_cell,), label=label)[0]
        if cell not in all_cells:
            raise JudgePartitionError(f"{label} references an out-of-plan cell {cell}")
        if cell not in owned_cells:
            raise JudgePartitionError(
                f"{label} references a cell owned by the other backend {cell}"
            )
        count = _integer(raw_count, f"{label}[{cell}]")
        if count < 0 or count > responses_per_cell:
            raise JudgePartitionError(
                f"{label}[{cell}] must be between 0 and {responses_per_cell}"
            )
        result[cell] = count
    return result


def partition_cells(
    plan: Mapping[str, Any],
    started_policy_steps: Iterable[int],
    completed_cells: Iterable[Any] = (),
) -> dict[str, Any]:
    """Split an explicit plan into disjoint InferenceB/Trainer whole-policy columns.

    A policy column is indivisible, so every fresh/stale comparison over the
    same Pool-B responses uses one judge backend. Columns with any existing
    InferenceB work are pinned to InferenceB. Policy 48 is also pinned to InferenceB so its blocked
    cells remain visible in the original scorer rather than disappearing.
    Remaining columns are assigned greedily by unfinished cell count.
    """

    steps, cells = _plan(plan)
    cell_set = set(cells)
    completed = set(_cells(completed_cells, label="completed cells"))
    if not completed <= cell_set:
        raise JudgePartitionError("completed cells must be a subset of the cell plan")

    columns: dict[int, list[Cell]] = defaultdict(list)
    for cell in cells:
        columns[cell[1]].append(cell)
    policies = set(columns)
    started = {
        _integer(step, "started policy step") for step in started_policy_steps
    }
    if not started <= policies:
        raise JudgePartitionError("started policy steps must exist in the cell plan")
    # A completed InferenceB receipt is itself conclusive evidence that its column
    # started there, even if a caller omitted that policy from its partial-cache scan.
    started.update(policy for _, policy in completed)
    inference_b_policies = started | (policies & BLOCKED_POLICY_STEPS)
    trainer_policies: set[int] = set()

    def unfinished(policy: int) -> int:
        if policy in BLOCKED_POLICY_STEPS:
            return 0
        return sum(cell not in completed for cell in columns[policy])

    blocked_pending = {
        cell
        for policy in policies & BLOCKED_POLICY_STEPS
        for cell in columns[policy]
        if cell not in completed
    }

    inference_b_remaining = sum(unfinished(policy) for policy in inference_b_policies)
    trainer_remaining = 0
    unstarted = policies - inference_b_policies
    for policy in sorted(unstarted, key=lambda step: (-unfinished(step), step)):
        load = unfinished(policy)
        if trainer_remaining <= inference_b_remaining:
            trainer_policies.add(policy)
            trainer_remaining += load
        else:
            inference_b_policies.add(policy)
            inference_b_remaining += load

    inference_b_cells = {cell for cell in cells if cell[1] in inference_b_policies}
    trainer_cells = {cell for cell in cells if cell[1] in trainer_policies}
    if inference_b_cells & trainer_cells or inference_b_cells | trainer_cells != cell_set:
        raise JudgePartitionError("judge partition changed the original cell coverage")
    if {policy for _, policy in inference_b_cells} & {policy for _, policy in trainer_cells}:
        raise JudgePartitionError("a policy column was split across judge backends")
    if not completed <= inference_b_cells:
        raise JudgePartitionError("completed InferenceB cells were not preserved in the InferenceB plan")

    return {
        "schema_version": 1,
        "partition_policy": "whole_policy_column_greedy_v1",
        "inference_b": _backend_plan(
            source=plan,
            steps=steps,
            cells=inference_b_cells,
            backend="inference_b",
        ),
        "trainer": _backend_plan(
            source=plan,
            steps=steps,
            cells=trainer_cells,
            backend="trainer",
        ),
        "summary": {
            "original_cell_count": len(cells),
            "completed_inference_b_cell_count": len(completed),
            "inference_b_policy_steps": sorted(inference_b_policies),
            "trainer_policy_steps": sorted(trainer_policies),
            "inference_b_cell_count": len(inference_b_cells),
            "trainer_cell_count": len(trainer_cells),
            "inference_b_remaining_cell_count": inference_b_remaining,
            "trainer_remaining_cell_count": trainer_remaining,
            "blocked_pending_cell_count": len(blocked_pending),
            "total_remaining_cell_count": (
                inference_b_remaining + trainer_remaining + len(blocked_pending)
            ),
            "blocked_policy_steps_pinned_to_inference_b": sorted(
                policies & BLOCKED_POLICY_STEPS
            ),
        },
    }


def rebalance_cells(
    inference_b_plan: Mapping[str, Any],
    trainer_plan: Mapping[str, Any],
    *,
    started_policy_steps_by_backend: Mapping[str, Iterable[int]],
    completed_counts_by_backend: Mapping[str, Mapping[Cell, int]],
    rates_by_backend: Mapping[str, float],
    blocked_policy_steps: Iterable[int] = (48,),
    responses_per_cell: int = 1600,
) -> dict[str, Any]:
    """Rate-balance unfinished whole-policy columns without moving started work.

    An explicitly supplied completion-count entry is treated as a saved receipt,
    including a zero count. Such columns, currently active columns, and blocked
    columns remain on their original backend. Blocked work remains in its plan
    but is excluded from the remaining-load and ETA calculation.
    """

    responses_per_cell = _integer(responses_per_cell, "responses_per_cell")
    if responses_per_cell <= 0:
        raise JudgePartitionError("responses_per_cell must be positive")

    inference_b_steps, inference_b_cells_tuple = _plan(inference_b_plan)
    trainer_steps, trainer_cells_tuple = _plan(trainer_plan)
    if inference_b_steps != trainer_steps:
        raise JudgePartitionError("InferenceB and Trainer plans must use identical steps")
    inference_b_cells = set(inference_b_cells_tuple)
    trainer_cells = set(trainer_cells_tuple)
    if inference_b_cells & trainer_cells:
        raise JudgePartitionError("InferenceB and Trainer plans contain conflicting cells")
    all_cells = inference_b_cells | trainer_cells

    columns: dict[int, list[Cell]] = defaultdict(list)
    original_owner: dict[int, str] = {}
    for backend, cells in (("inference_b", inference_b_cells), ("trainer", trainer_cells)):
        for cell in cells:
            policy = cell[1]
            previous = original_owner.setdefault(policy, backend)
            if previous != backend:
                raise JudgePartitionError(
                    f"policy column {policy} is split across InferenceB and Trainer"
                )
            columns[policy].append(cell)

    started_values = _backend_mapping(
        started_policy_steps_by_backend,
        label="started_policy_steps_by_backend",
    )
    completed_values = _backend_mapping(
        completed_counts_by_backend,
        label="completed_counts_by_backend",
    )
    rate_values = _backend_mapping(rates_by_backend, label="rates_by_backend")
    rates = {
        backend: _rate(rate_values[backend], label=f"rates_by_backend[{backend}]")
        for backend in ("inference_b", "trainer")
    }

    started: dict[str, set[int]] = {}
    for backend in ("inference_b", "trainer"):
        raw_steps = tuple(started_values[backend])
        parsed = {_integer(step, f"started {backend} policy step") for step in raw_steps}
        if len(parsed) != len(raw_steps):
            raise JudgePartitionError(f"started {backend} policy steps contain duplicates")
        wrong = {policy for policy in parsed if original_owner.get(policy) != backend}
        if wrong:
            raise JudgePartitionError(
                f"started {backend} policy steps are out of plan or owned by the other backend: "
                f"{sorted(wrong)}"
            )
        started[backend] = parsed

    completed = {
        "inference_b": _completion_counts(
            completed_values["inference_b"],
            label="completed_counts_by_backend[inference_b]",
            owned_cells=inference_b_cells,
            all_cells=all_cells,
            responses_per_cell=responses_per_cell,
        ),
        "trainer": _completion_counts(
            completed_values["trainer"],
            label="completed_counts_by_backend[trainer]",
            owned_cells=trainer_cells,
            all_cells=all_cells,
            responses_per_cell=responses_per_cell,
        ),
    }

    raw_blocked = tuple(blocked_policy_steps)
    blocked = {_integer(step, "blocked policy step") for step in raw_blocked}
    if len(blocked) != len(raw_blocked):
        raise JudgePartitionError("blocked policy steps contain duplicates")
    unknown_blocked = blocked - set(columns)
    if unknown_blocked:
        raise JudgePartitionError(
            f"blocked policy steps are outside the plans: {sorted(unknown_blocked)}"
        )

    pinned = blocked | started["inference_b"] | started["trainer"]
    for backend in ("inference_b", "trainer"):
        pinned.update(cell[1] for cell in completed[backend])

    def remaining(policy: int) -> int:
        if policy in blocked:
            return 0
        backend = original_owner[policy]
        counts = completed[backend]
        return sum(responses_per_cell - counts.get(cell, 0) for cell in columns[policy])

    fixed_load = {"inference_b": 0, "trainer": 0}
    for policy in pinned:
        fixed_load[original_owner[policy]] += remaining(policy)

    movable = sorted(set(columns) - pinned)
    # Dynamic programming is bounded by the total integer response load (at
    # most 100 * 1600 for the historical matrix), rather than 2**columns.
    # Each Trainer-added load retains the fewest moves and then the
    # lexicographically smallest Trainer policy tuple for deterministic ties.
    states: dict[int, tuple[int, tuple[int, ...]]] = {0: (0, ())}
    for policy in movable:
        load = remaining(policy)
        owner = original_owner[policy]
        next_states: dict[int, tuple[int, tuple[int, ...]]] = {}
        for trainer_added, (moves, trainer_selected) in states.items():
            choices = (
                (trainer_added, (moves + (owner == "trainer"), trainer_selected)),
                (
                    trainer_added + load,
                    (moves + (owner == "inference_b"), trainer_selected + (policy,)),
                ),
            )
            for new_load, candidate in choices:
                existing = next_states.get(new_load)
                if existing is None or candidate < existing:
                    next_states[new_load] = candidate
        states = next_states

    movable_load = sum(remaining(policy) for policy in movable)
    best: tuple[Any, ...] | None = None
    best_trainer: tuple[int, ...] = ()
    for trainer_added, (moves, trainer_selected) in states.items():
        inference_b_load = fixed_load["inference_b"] + movable_load - trainer_added
        trainer_load = fixed_load["trainer"] + trainer_added
        inference_b_eta = Decimal(inference_b_load) / rates["inference_b"]
        trainer_eta = Decimal(trainer_load) / rates["trainer"]
        moved = tuple(
            policy
            for policy in movable
            if (policy in trainer_selected) != (original_owner[policy] == "trainer")
        )
        objective = (
            max(inference_b_eta, trainer_eta),
            moves,
            abs(inference_b_eta - trainer_eta),
            moved,
            trainer_selected,
        )
        if best is None or objective < best:
            best = objective
            best_trainer = trainer_selected

    trainer_policies = set(best_trainer) | {
        policy for policy in pinned if original_owner[policy] == "trainer"
    }
    inference_b_policies = set(columns) - trainer_policies
    new_inference_b_cells = {cell for cell in all_cells if cell[1] in inference_b_policies}
    new_trainer_cells = all_cells - new_inference_b_cells
    if new_inference_b_cells & new_trainer_cells or new_inference_b_cells | new_trainer_cells != all_cells:
        raise JudgePartitionError("rate-aware partition changed the original coverage")

    inference_b_to_trainer = sorted(
        policy for policy in trainer_policies if original_owner[policy] == "inference_b"
    )
    trainer_to_inference_b = sorted(
        policy for policy in inference_b_policies if original_owner[policy] == "trainer"
    )
    remaining_counts = {
        "inference_b": sum(remaining(policy) for policy in inference_b_policies),
        "trainer": sum(remaining(policy) for policy in trainer_policies),
    }
    eta_hours = {
        backend: float(Decimal(remaining_counts[backend]) / rates[backend])
        for backend in ("inference_b", "trainer")
    }
    blocked_pending = sum(
        responses_per_cell - completed[original_owner[policy]].get(cell, 0)
        for policy in blocked
        for cell in columns[policy]
    )
    partition_policy = "whole_policy_column_rate_aware_v1"
    return {
        "schema_version": 1,
        "partition_policy": partition_policy,
        "inference_b": _backend_plan(
            source=inference_b_plan,
            steps=inference_b_steps,
            cells=new_inference_b_cells,
            backend="inference_b",
            partition_policy=partition_policy,
        ),
        "trainer": _backend_plan(
            source=trainer_plan,
            steps=inference_b_steps,
            cells=new_trainer_cells,
            backend="trainer",
            partition_policy=partition_policy,
        ),
        "moved_policy_steps": {
            "inference_b_to_trainer": inference_b_to_trainer,
            "trainer_to_inference_b": trainer_to_inference_b,
        },
        "summary": {
            "original_cell_count": len(all_cells),
            "original_policy_steps_by_backend": {
                backend: sorted(
                    policy for policy, owner in original_owner.items() if owner == backend
                )
                for backend in ("inference_b", "trainer")
            },
            "new_policy_steps_by_backend": {
                "inference_b": sorted(inference_b_policies),
                "trainer": sorted(trainer_policies),
            },
            "pinned_policy_steps_by_backend": {
                backend: sorted(
                    policy
                    for policy in pinned
                    if original_owner[policy] == backend
                )
                for backend in ("inference_b", "trainer")
            },
            "moved_cell_count": sum(len(columns[policy]) for policy in inference_b_to_trainer + trainer_to_inference_b),
            "remaining_response_counts_by_backend": remaining_counts,
            "rates_responses_per_hour_by_backend": {
                backend: float(rates[backend]) for backend in ("inference_b", "trainer")
            },
            "estimated_remaining_hours_by_backend": eta_hours,
            "estimated_makespan_hours": max(eta_hours.values()),
            "blocked_policy_steps": sorted(blocked),
            "blocked_pending_response_count": blocked_pending,
        },
    }
