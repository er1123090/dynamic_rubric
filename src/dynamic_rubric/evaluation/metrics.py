"""Metrics for rubric staleness and reward behavior."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import math
import statistics
from typing import Any

from .bon import BON_SIZES


def gt_auc(scores_by_n: Mapping[int, float] | Sequence[float]) -> float:
    """Normalized trapezoidal AUC over the fixed ``u=log2(N)`` grid."""

    if isinstance(scores_by_n, Mapping):
        if set(scores_by_n) != set(BON_SIZES):
            raise ValueError(f"GT-AUC requires exactly N={BON_SIZES}")
        values = [float(scores_by_n[n]) for n in BON_SIZES]
    else:
        values = [float(value) for value in scores_by_n]
        if len(values) != len(BON_SIZES):
            raise ValueError(f"GT-AUC requires {len(BON_SIZES)} scores")
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("gold scores must be finite values in [0, 1]")
    width = math.log2(BON_SIZES[-1]) - math.log2(BON_SIZES[0])
    if width <= 0.0:
        raise ValueError("BoN grid must span at least two distinct sizes")
    return math.fsum((left + right) / 2.0 for left, right in zip(values, values[1:])) / width


def stale_rubric_regret(current_gt_auc: float, stale_gt_auc: float) -> float:
    """Positive values mean the current rubric outperforms the stale rubric."""

    return float(current_gt_auc) - float(stale_gt_auc)


def top1_agreement(left: Sequence[Any], right: Sequence[Any]) -> float:
    if len(left) != len(right):
        raise ValueError("selection sequences must have equal length")
    if not left:
        raise ValueError("selection sequences must not be empty")
    return sum(a == b for a, b in zip(left, right)) / len(left)


def kendall_tau_b(left: Sequence[float], right: Sequence[float]) -> float:
    """Kendall tau-b, including ties in either ranking."""

    if len(left) != len(right):
        raise ValueError("rank sequences must have equal length")
    if len(left) < 2:
        raise ValueError("at least two ranked items are required")
    concordant = discordant = ties_left = ties_right = 0
    for i in range(len(left) - 1):
        for j in range(i + 1, len(left)):
            dx = left[i] - left[j]
            dy = right[i] - right[j]
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                ties_left += 1
            elif dy == 0:
                ties_right += 1
            elif dx * dy > 0:
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt(
        (concordant + discordant + ties_left) * (concordant + discordant + ties_right)
    )
    return 0.0 if denominator == 0.0 else (concordant - discordant) / denominator


def reward_resolution(
    scores: Sequence[float], *, repeat_scores: Sequence[Sequence[float]] | None = None
) -> dict[str, float | None]:
    """Summarize ties, spread, top-vs-median separation, and repeat stability."""

    values = [float(value) for value in scores]
    if not values:
        raise ValueError("scores must not be empty")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("scores must be finite")
    pair_count = len(values) * (len(values) - 1) // 2
    counts = Counter(values)
    tied_pairs = sum(count * (count - 1) // 2 for count in counts.values())
    stability: float | None = None
    if repeat_scores is not None:
        repeats = [tuple(map(float, repeat)) for repeat in repeat_scores]
        if not repeats or any(len(repeat) != len(values) for repeat in repeats):
            raise ValueError("each repeat must align with the original score sequence")
        agreements = []
        base_order = _pairwise_signs(values)
        for repeat in repeats:
            repeat_order = _pairwise_signs(repeat)
            agreements.append(
                sum(a == b for a, b in zip(base_order, repeat_order)) / len(base_order)
                if base_order
                else 1.0
            )
        stability = math.fsum(agreements) / len(agreements)
    return {
        "tie_rate": tied_pairs / pair_count if pair_count else 0.0,
        "variance": statistics.pvariance(values),
        "top_median_margin": max(values) - statistics.median(values),
        "judge_repeat_stability": stability,
    }


def _pairwise_signs(values: Sequence[float]) -> tuple[int, ...]:
    signs = []
    for i in range(len(values) - 1):
        for j in range(i + 1, len(values)):
            difference = values[i] - values[j]
            signs.append((difference > 0) - (difference < 0))
    return tuple(signs)
