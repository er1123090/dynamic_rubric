"""Deterministic count/weight matching for stale and sham rubric controls."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Mapping

from .contracts import WeightedCriterion


@dataclass(frozen=True, slots=True)
class ControlMatch:
    selected: tuple[WeightedCriterion, ...]
    eligible: bool
    target_count: int
    available_count: int
    exact_histogram_match: bool
    target_weight_histogram: tuple[tuple[int, int], ...]
    selected_weight_histogram: tuple[tuple[int, int], ...]
    total_weight_difference: int | None
    reason: str | None = None


def _histogram(criteria: Iterable[WeightedCriterion]) -> tuple[tuple[int, int], ...]:
    counts = Counter(item.weight_units for item in criteria)
    return tuple(sorted(counts.items()))


def match_control_extension(
    current: Iterable[WeightedCriterion],
    control_candidates: Iterable[WeightedCriterion],
    *,
    source_pair_order: Mapping[str, int] | None = None,
) -> ControlMatch:
    """Match count first and weight histogram second, without duplicating criteria.

    Exact weight bins are filled first. Missing bins use the smallest absolute
    weight distance; ties use the oldest source pair and then content hash.
    """

    target = tuple(current)
    available = tuple(control_candidates)
    target_histogram = _histogram(target)
    if len(available) < len(target):
        return ControlMatch(
            selected=(),
            eligible=False,
            target_count=len(target),
            available_count=len(available),
            exact_histogram_match=False,
            target_weight_histogram=target_histogram,
            selected_weight_histogram=(),
            total_weight_difference=None,
            reason="insufficient_control_criteria",
        )
    pair_order = dict(source_pair_order or {})

    def age(item: WeightedCriterion) -> int:
        values = [pair_order.get(candidate, 2**31 - 1) for candidate in item.source_candidate_ids]
        return min(values, default=2**31 - 1)

    remaining = list(available)
    selected: list[WeightedCriterion] = []
    for wanted in sorted((item.weight_units for item in target), reverse=True):
        chosen = min(
            remaining,
            key=lambda item: (
                item.weight_units != wanted,
                abs(item.weight_units - wanted),
                age(item),
                item.canonical_criterion_hash,
                item.criterion_instance_id,
            ),
        )
        selected.append(chosen)
        remaining.remove(chosen)
    selected_tuple = tuple(selected)
    selected_histogram = _histogram(selected_tuple)
    return ControlMatch(
        selected=selected_tuple,
        eligible=True,
        target_count=len(target),
        available_count=len(available),
        exact_histogram_match=selected_histogram == target_histogram,
        target_weight_histogram=target_histogram,
        selected_weight_histogram=selected_histogram,
        total_weight_difference=abs(
            sum(item.weight_units for item in selected_tuple)
            - sum(item.weight_units for item in target)
        ),
    )


match_stale_extension = match_control_extension
match_sham_extension = match_control_extension
