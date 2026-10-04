from __future__ import annotations

import pytest

from dynamic_rubric.phase1.judge_partition import (
    JudgePartitionError,
    partition_cells,
    rebalance_cells,
)


STEPS = [0, 3, 6, 9, 12, 13, 15, 16, 18, 21, 24, 27, 30, 32, 33, 34, 36, 39, 40, 42, 45, 48]


def historical_plan() -> dict:
    cells = {(step, step) for step in STEPS}
    cells.update((STEPS[index - 1], step) for index, step in enumerate(STEPS) if index)
    for anchor in (0, 9, 16, 32, 48):
        cells.update((anchor, policy) for policy in STEPS if policy >= anchor)
    assert len(cells) == 100
    return {
        "schema_version": 1,
        "analysis": "historical_100_cell_reuse_matrix",
        "steps": STEPS,
        "cells": [
            {"evaluator_step": evaluator, "policy_step": policy}
            for evaluator, policy in sorted(cells, key=lambda cell: (cell[1], cell[0]))
        ],
    }


def cell_set(plan: dict) -> set[tuple[int, int]]:
    return {
        (cell["evaluator_step"], cell["policy_step"])
        for cell in plan["cells"]
    }


def plan_for_policies(source: dict, policies: set[int], backend: str) -> dict:
    return {
        "schema_version": 1,
        "analysis": source["analysis"],
        "steps": source["steps"],
        "cells": [
            cell for cell in source["cells"] if cell["policy_step"] in policies
        ],
        "judge_partition": {"backend": backend},
    }


def test_historical_100_cells_are_partitioned_as_disjoint_whole_columns():
    source = historical_plan()
    partition = partition_cells(source, started_policy_steps={0, 3, 6, 9})
    inference_b = cell_set(partition["inference_b"])
    trainer = cell_set(partition["trainer"])

    assert inference_b.isdisjoint(trainer)
    assert inference_b | trainer == cell_set(source)
    assert len(inference_b) + len(trainer) == 100
    for policy in STEPS:
        backends = {"inference_b" if cell in inference_b else "trainer" for cell in cell_set(source) if cell[1] == policy}
        assert len(backends) == 1
    assert {0, 3, 6, 9, 48} <= set(partition["summary"]["inference_b_policy_steps"])
    assert 48 not in partition["summary"]["trainer_policy_steps"]
    assert partition["summary"]["blocked_policy_steps_pinned_to_inference_b"] == [48]
    assert partition["summary"]["blocked_pending_cell_count"] == 6
    assert partition["summary"]["total_remaining_cell_count"] == 100


def test_partial_and_completed_inference_b_columns_never_move_to_trainer():
    source = historical_plan()
    completed = {(0, 0), (0, 3), (3, 3), (9, 12)}
    partition = partition_cells(
        source,
        started_policy_steps={6},
        completed_cells=completed,
    )
    inference_b = cell_set(partition["inference_b"])

    assert completed <= inference_b
    assert {0, 3, 6, 12} <= set(partition["summary"]["inference_b_policy_steps"])
    assert all(cell in inference_b for cell in cell_set(source) if cell[1] in {0, 3, 6, 12})
    assert partition["summary"]["completed_inference_b_cell_count"] == 4


def test_partition_is_deterministic_and_balances_unstarted_remaining_work():
    source = historical_plan()
    first = partition_cells(source, started_policy_steps=())
    second = partition_cells(source, started_policy_steps=())

    assert first == second
    difference = abs(
        first["summary"]["inference_b_remaining_cell_count"]
        - first["summary"]["trainer_remaining_cell_count"]
    )
    largest_column = max(
        sum(cell["policy_step"] == policy for cell in source["cells"])
        for policy in STEPS
    )
    assert difference <= largest_column
    assert first["summary"]["inference_b_remaining_cell_count"] + first["summary"]["trainer_remaining_cell_count"] == 94
    assert first["summary"]["blocked_pending_cell_count"] == 6


@pytest.mark.parametrize(
    "change, match",
    [
        (lambda plan: plan["cells"].append(plan["cells"][0]), "duplicate"),
        (
            lambda plan: plan["cells"].append(
                {"evaluator_step": 48, "policy_step": 0}
            ),
            "future evaluator",
        ),
        (
            lambda plan: plan["cells"].append(
                {"evaluator_step": 0, "policy_step": 47}
            ),
            "outside steps",
        ),
    ],
)
def test_invalid_source_plan_fails_closed(change, match):
    source = historical_plan()
    change(source)

    with pytest.raises(JudgePartitionError, match=match):
        partition_cells(source, started_policy_steps=())


def test_unknown_started_or_completed_work_fails_closed():
    source = historical_plan()
    with pytest.raises(JudgePartitionError, match="started policy"):
        partition_cells(source, started_policy_steps={47})
    with pytest.raises(JudgePartitionError, match="completed cells"):
        partition_cells(source, started_policy_steps=(), completed_cells={(3, 0)})


def test_rate_rebalance_moves_17_unstarted_cells_to_faster_trainer():
    source = historical_plan()
    inference_b_policies = {0, 3, 6, 9, 12, 13, 15, 16, 18, 21, 27, 32, 36, 40, 45, 48}
    trainer_policies = {24, 30, 33, 34, 39, 42}
    inference_b = plan_for_policies(source, inference_b_policies, "inference_b")
    trainer = plan_for_policies(source, trainer_policies, "trainer")

    by_policy = {
        policy: sorted(cell for cell in cell_set(source) if cell[1] == policy)
        for policy in STEPS
    }
    inference_b_counts = {
        cell: 1600
        for policy in (0, 3, 6, 9, 12, 13, 15, 16, 18, 21)
        for cell in by_policy[policy]
    }
    inference_b_counts[by_policy[27][0]] = 1600
    inference_b_counts[by_policy[27][1]] = 800
    trainer_counts = {
        cell: 1600
        for policy in (24, 30, 33)
        for cell in by_policy[policy]
    }
    for cell in by_policy[34][:5]:
        trainer_counts[cell] = 1600
    trainer_counts[by_policy[34][5]] = 800

    result = rebalance_cells(
        inference_b,
        trainer,
        started_policy_steps_by_backend={"inference_b": {27}, "trainer": {34}},
        completed_counts_by_backend={"inference_b": inference_b_counts, "trainer": trainer_counts},
        rates_by_backend={"inference_b": 3361, "trainer": 9646},
    )

    new_inference_b = cell_set(result["inference_b"])
    new_trainer = cell_set(result["trainer"])
    assert new_inference_b.isdisjoint(new_trainer)
    assert new_inference_b | new_trainer == cell_set(source)
    assert result["moved_policy_steps"] == {
        "inference_b_to_trainer": [32, 36, 40],
        "trainer_to_inference_b": [],
    }
    assert result["summary"]["moved_cell_count"] == 17
    assert {27, 48} <= set(result["summary"]["new_policy_steps_by_backend"]["inference_b"])
    assert 34 in result["summary"]["new_policy_steps_by_backend"]["trainer"]
    assert result["summary"]["remaining_response_counts_by_backend"] == {
        "inference_b": 15200,
        "trainer": 47200,
    }
    assert result["summary"]["estimated_remaining_hours_by_backend"]["inference_b"] == pytest.approx(4.52246)
    assert result["summary"]["estimated_remaining_hours_by_backend"]["trainer"] == pytest.approx(4.89322)
    assert result["summary"]["blocked_pending_response_count"] == 9600


def test_saved_zero_count_receipt_and_active_columns_are_pinned():
    source = historical_plan()
    inference_b = plan_for_policies(source, set(STEPS) - {42}, "inference_b")
    trainer = plan_for_policies(source, {42}, "trainer")
    saved_cell = next(cell for cell in cell_set(inference_b) if cell[1] == 40)

    result = rebalance_cells(
        inference_b,
        trainer,
        started_policy_steps_by_backend={"inference_b": {45}, "trainer": {42}},
        completed_counts_by_backend={"inference_b": {saved_cell: 0}, "trainer": {}},
        rates_by_backend={"inference_b": 1, "trainer": 100},
    )

    assert {40, 45, 48} <= set(result["summary"]["new_policy_steps_by_backend"]["inference_b"])
    assert 42 in result["summary"]["new_policy_steps_by_backend"]["trainer"]


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda context: context["rates"].update(inference_b=0), "positive finite"),
        (lambda context: context["rates"].update(trainer=float("inf")), "positive finite"),
        (
            lambda context: context["completed"]["inference_b"].update({(0, 0): 1601}),
            "between 0 and 1600",
        ),
        (
            lambda context: context["completed"]["inference_b"].update({(0, 47): 1}),
            "out-of-plan",
        ),
        (
            lambda context: context["completed"]["trainer"].update({(0, 0): 1}),
            "other backend",
        ),
    ],
)
def test_rate_rebalance_rejects_invalid_rates_and_receipt_counts(mutate, match):
    source = historical_plan()
    context = {
        "inference_b": plan_for_policies(source, set(STEPS) - {42}, "inference_b"),
        "trainer": plan_for_policies(source, {42}, "trainer"),
        "completed": {"inference_b": {}, "trainer": {}},
        "rates": {"inference_b": 1, "trainer": 1},
    }
    mutate(context)

    with pytest.raises(JudgePartitionError, match=match):
        rebalance_cells(
            context["inference_b"],
            context["trainer"],
            started_policy_steps_by_backend={"inference_b": set(), "trainer": set()},
            completed_counts_by_backend=context["completed"],
            rates_by_backend=context["rates"],
        )


def test_rate_rebalance_rejects_split_policy_columns():
    source = historical_plan()
    inference_b = plan_for_policies(source, set(STEPS) - {42}, "inference_b")
    trainer = plan_for_policies(source, {42}, "trainer")
    moved_cell = next(cell for cell in inference_b["cells"] if cell["policy_step"] == 40)
    inference_b["cells"].remove(moved_cell)
    trainer["cells"].append(moved_cell)

    with pytest.raises(JudgePartitionError, match="split across"):
        rebalance_cells(
            inference_b,
            trainer,
            started_policy_steps_by_backend={"inference_b": set(), "trainer": set()},
            completed_counts_by_backend={"inference_b": {}, "trainer": {}},
            rates_by_backend={"inference_b": 1, "trainer": 1},
        )
