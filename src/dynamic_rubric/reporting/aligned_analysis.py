"""Shared-pool, candidate-aligned analysis primitives."""

from __future__ import annotations

import dataclasses
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from ..evaluation.bootstrap import BootstrapResult, paired_prompt_bootstrap
from ..evaluation.metrics import kendall_tau_b, reward_resolution, top1_agreement


class AlignmentError(ValueError):
    """Raised when a causal comparison cannot be formed from identical pools."""


def _auc(scores_by_n: Mapping[int, float], n_grid: Sequence[int]) -> float:
    import math

    expected = tuple(int(n) for n in n_grid)
    if not expected or len(expected) != len(set(expected)) or any(n <= 0 for n in expected):
        raise AlignmentError("GT-AUC N grid must contain unique positive values")
    if set(scores_by_n) != set(expected):
        raise AlignmentError(
            f"GT-AUC requires exact configured N grid {expected}, got {tuple(sorted(scores_by_n))}"
        )
    values = sorted((math.log2(int(n)), float(score)) for n, score in scores_by_n.items())
    if len(values) == 1:
        return values[0][1]
    width = values[-1][0] - values[0][0]
    if width <= 0.0:
        raise AlignmentError("GT-AUC grid must contain distinct positive N values")
    return (
        sum(
            (right_x - left_x) * (left_y + right_y) / 2
            for (left_x, left_y), (right_x, right_y) in zip(values, values[1:])
        )
        / width
    )


def analyze_aligned(
    selections: Sequence[Mapping[str, Any]],
    gold: Mapping[tuple[str, str], float],
    rubric_scores: Sequence[Mapping[str, Any]],
    *,
    iterations: int,
    seed: int,
    n_grid: Sequence[int],
    permutations: int,
) -> dict[str, Any]:
    """Analyze only current-vs-static observations with exact shared-pool joins."""

    rows: dict[tuple[str, str, str, int, int, int], Mapping[str, Any]] = {}
    pool_by_cell: dict[tuple[str, str, int, int], str] = {}
    cross_points: dict[tuple[str, int, str, str, int], list[float]] = defaultdict(list)
    for row in selections:
        policy_id = str(row["policy_id"])
        prompt_id = str(row["prompt_id"])
        mode = str(row["mode"])
        rubric_step = int(row["rubric_step"])
        n = int(row["n"])
        permutation = int(row["permutation"])
        pool_hash = str(row["pool_hash"])
        cell = policy_id, prompt_id, n, permutation
        previous_pool = pool_by_cell.setdefault(cell, pool_hash)
        if previous_pool != pool_hash:
            raise AlignmentError(f"rubrics used different candidate pools for {cell}")
        key = policy_id, prompt_id, mode, rubric_step, n, permutation
        if key in rows:
            raise AlignmentError(f"duplicate selection cell: {key}")
        response_key = prompt_id, str(row["response_id"])
        if response_key not in gold:
            raise AlignmentError(f"missing hidden-gold score for {response_key}")
        rows[key] = row
        cross_points[(mode, rubric_step, policy_id, prompt_id, n)].append(gold[response_key])

    if permutations < 1:
        raise AlignmentError("permutation count must be positive")
    expected_cells = {(int(n), permutation) for n in n_grid for permutation in range(permutations)}
    grouped_cells: dict[tuple[str, int, str, str], set[tuple[int, int]]] = defaultdict(set)
    for policy_id, prompt_id, mode, rubric_step, n, permutation in rows:
        grouped_cells[(mode, rubric_step, policy_id, prompt_id)].add((n, permutation))
    for group, cells in grouped_cells.items():
        if cells != expected_cells:
            raise AlignmentError(
                f"selection group {group} does not cover configured N/permutation grid"
            )

    cross_auc: dict[str, float] = {}
    cross_curves: dict[tuple[str, int, str, str], dict[int, float]] = defaultdict(dict)
    for (mode, rubric_step, policy_id, prompt_id, n), values in cross_points.items():
        cross_curves[(mode, rubric_step, policy_id, prompt_id)][n] = sum(values) / len(values)
    cross_grouped: dict[str, list[float]] = defaultdict(list)
    for (mode, rubric_step, policy_id, _prompt_id), curve in cross_curves.items():
        cross_grouped[f"{mode}:R_{rubric_step}:{policy_id}"].append(_auc(curve, n_grid))
    for key, values in cross_grouped.items():
        cross_auc[key] = sum(values) / len(values)

    static: dict[tuple[str, str, int, int], Mapping[str, Any]] = {}
    current: dict[str, dict[tuple[str, str, int, int], Mapping[str, Any]]] = defaultdict(dict)
    for (policy_id, prompt_id, mode, rubric_step, n, permutation), row in rows.items():
        cell = policy_id, prompt_id, n, permutation
        if mode == "static":
            static[cell] = row
        elif rubric_step == int(row["policy_step"]):
            current[mode][cell] = row
    if not static:
        raise AlignmentError("static selections are required")

    auc_by_mode_policy_prompt: dict[str, dict[tuple[str, str], float]] = defaultdict(dict)
    static_curves: dict[tuple[str, str], dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for (policy_id, prompt_id, n, _permutation), row in static.items():
        static_curves[(policy_id, prompt_id)][n].append(gold[(prompt_id, str(row["response_id"]))])
    for key, curve in static_curves.items():
        auc_by_mode_policy_prompt["static"][key] = _auc(
            {n: sum(values) / len(values) for n, values in curve.items()}, n_grid
        )

    agreement: dict[str, float] = {}
    bootstrap: dict[str, BootstrapResult] = {}
    regret: dict[str, float] = {}
    for mode, mode_rows in current.items():
        if set(mode_rows) != set(static):
            missing_static = set(mode_rows) - set(static)
            missing_mode = set(static) - set(mode_rows)
            raise AlignmentError(
                f"{mode}/static selection cells differ: "
                f"missing_static={sorted(missing_static)[:1]}, missing_{mode}={sorted(missing_mode)[:1]}"
            )
        mode_curves: dict[tuple[str, str], dict[int, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        aligned_left: list[str] = []
        aligned_right: list[str] = []
        for cell, row in sorted(mode_rows.items()):
            policy_id, prompt_id, n, _permutation = cell
            baseline = static[cell]
            mode_curves[(policy_id, prompt_id)][n].append(
                gold[(prompt_id, str(row["response_id"]))]
            )
            aligned_left.append(str(row["response_id"]))
            aligned_right.append(str(baseline["response_id"]))
        for key, curve in mode_curves.items():
            auc_by_mode_policy_prompt[mode][key] = _auc(
                {n: sum(values) / len(values) for n, values in curve.items()}, n_grid
            )
        pairs = []
        for (policy_id, prompt_id), value in sorted(auc_by_mode_policy_prompt[mode].items()):
            baseline_value = auc_by_mode_policy_prompt["static"].get((policy_id, prompt_id))
            if baseline_value is None:
                raise AlignmentError(f"missing static GT-AUC for {(policy_id, prompt_id)}")
            pairs.append((prompt_id, value, baseline_value))
        bootstrap[mode] = paired_prompt_bootstrap(pairs, iterations=iterations, seed=seed)
        regret[mode] = bootstrap[mode].point_estimate
        agreement[mode] = top1_agreement(aligned_left, aligned_right)

    gt_auc = {
        mode: sum(values.values()) / len(values)
        for mode, values in auc_by_mode_policy_prompt.items()
        if values
    }

    score_groups: dict[tuple[str, str, str, int], dict[int, float]] = defaultdict(dict)
    repeat_score_groups: dict[tuple[str, str, str, int], dict[int, float]] = defaultdict(dict)
    for row in rubric_scores:
        mode = str(row["mode"])
        rubric_step = int(row["rubric_step"])
        if mode != "static" and rubric_step != int(row["policy_step"]):
            continue
        group = str(row["policy_id"]), str(row["prompt_id"]), mode, rubric_step
        candidate_id = int(row["global_candidate_id"])
        if candidate_id in score_groups[group]:
            raise AlignmentError(f"duplicate candidate score for {group + (candidate_id,)}")
        if "judge_repeat_score" not in row:
            raise AlignmentError(f"missing judge repeat score for {group + (candidate_id,)}")
        score_groups[group][candidate_id] = float(row["score"])
        repeat_score_groups[group][candidate_id] = float(row["judge_repeat_score"])
    static_scores = {
        (policy_id, prompt_id): values
        for (policy_id, prompt_id, mode, _rubric_step), values in score_groups.items()
        if mode == "static"
    }
    tau_values: dict[str, list[float]] = defaultdict(list)
    resolution_values: dict[str, list[float]] = defaultdict(list)
    resolution_repeats: dict[str, list[float]] = defaultdict(list)
    for group, values in score_groups.items():
        mode = group[2]
        candidate_ids = sorted(values)
        resolution_values[mode].extend(values[candidate_id] for candidate_id in candidate_ids)
        repeats = repeat_score_groups[group]
        if set(repeats) != set(values):
            raise AlignmentError(f"judge repeat candidate IDs do not align for {group}")
        resolution_repeats[mode].extend(repeats[candidate_id] for candidate_id in candidate_ids)
    for (policy_id, prompt_id, mode, _rubric_step), values in score_groups.items():
        if mode == "static":
            continue
        baseline = static_scores.get((policy_id, prompt_id))
        if baseline is None or set(baseline) != set(values):
            raise AlignmentError(f"candidate IDs do not align for {(policy_id, prompt_id, mode)}")
        candidate_ids = sorted(values)
        if len(candidate_ids) >= 2:
            tau_values[mode].append(
                kendall_tau_b(
                    [baseline[candidate_id] for candidate_id in candidate_ids],
                    [values[candidate_id] for candidate_id in candidate_ids],
                )
            )
    tau = {mode: sum(values) / len(values) for mode, values in tau_values.items() if values}
    resolution = {
        mode: reward_resolution(values, repeat_scores=[resolution_repeats[mode]])
        for mode, values in resolution_values.items()
        if values
    }
    return {
        "gt_auc": gt_auc,
        "gt_auc_cross_matrix": cross_auc,
        "stale_rubric_regret": regret,
        "top1_agreement": agreement,
        "kendall_tau_b": tau,
        "reward_resolution": resolution,
        "paired_bootstrap": {
            mode: dataclasses.asdict(result) for mode, result in bootstrap.items()
        },
        "bootstrap_results": bootstrap,
    }
