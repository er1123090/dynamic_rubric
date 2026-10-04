"""Prompt-level discriminability metrics for the rubric horizon audit."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from fractions import Fraction
import math
import statistics
from typing import Any


RationalLike = Fraction | int | tuple[int, int]


def as_fraction(value: RationalLike) -> Fraction:
    """Convert an exact stored score to a reduced fraction without using floats."""

    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value, 1)
    if isinstance(value, tuple) and len(value) == 2:
        numerator, denominator = value
        if not isinstance(numerator, int) or not isinstance(denominator, int):
            raise TypeError("rational tuple members must be integers")
        if denominator == 0:
            raise ValueError("rational denominator must be non-zero")
        return Fraction(numerator, denominator)
    raise TypeError("score must be a Fraction, integer, or (numerator, denominator) tuple")


def _fractions(scores: Sequence[RationalLike]) -> tuple[Fraction, ...]:
    if not scores:
        raise ValueError("scores must not be empty")
    return tuple(as_fraction(score) for score in scores)


def exact_zero_advantage(scores: Sequence[RationalLike]) -> bool:
    """Whether every response in one GRPO group has exactly the same reward."""

    values = _fractions(scores)
    return len(set(values)) == 1


def exact_zar_rate(groups: Sequence[Sequence[RationalLike]]) -> float:
    if not groups:
        raise ValueError("at least one score group is required")
    return math.fsum(exact_zero_advantage(group) for group in groups) / len(groups)


def population_sd(scores: Sequence[RationalLike]) -> float:
    values = _fractions(scores)
    floats = [float(value) for value in values]
    return statistics.pstdev(floats)


def low_reward_spread(scores: Sequence[RationalLike], *, epsilon: float) -> bool:
    if not math.isfinite(epsilon) or epsilon < 0.0:
        raise ValueError("epsilon must be finite and non-negative")
    return population_sd(scores) < epsilon


def low_reward_spread_rate(
    groups: Sequence[Sequence[RationalLike]], *, epsilon: float
) -> float:
    if not groups:
        raise ValueError("at least one score group is required")
    return math.fsum(low_reward_spread(group, epsilon=epsilon) for group in groups) / len(groups)


def advantage_degenerate(advantages: Sequence[float], *, delta: float) -> bool:
    """Classify already-computed trainer advantages without reimplementing GRPO."""

    if not advantages:
        raise ValueError("advantages must not be empty")
    if not math.isfinite(delta) or delta < 0.0:
        raise ValueError("delta must be finite and non-negative")
    values = [float(value) for value in advantages]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("advantages must be finite")
    return max(abs(value) for value in values) <= delta


def advantage_degeneracy_rate(groups: Sequence[Sequence[float]], *, delta: float) -> float:
    if not groups:
        raise ValueError("at least one advantage group is required")
    return math.fsum(advantage_degenerate(group, delta=delta) for group in groups) / len(groups)


def _sign(left: Fraction, right: Fraction) -> int:
    return (left > right) - (left < right)


def pairwise_metrics(
    current_scores: Sequence[RationalLike],
    *,
    baseline_scores: Sequence[RationalLike] | None = None,
) -> dict[str, Any]:
    """Compute tie/separation and optional paired ordering-confusion metrics."""

    current = _fractions(current_scores)
    if len(current) < 2:
        raise ValueError("at least two scores are required")
    baseline = None if baseline_scores is None else _fractions(baseline_scores)
    if baseline is not None and len(baseline) != len(current):
        raise ValueError("baseline and current scores must align")

    total = len(current) * (len(current) - 1) // 2
    current_ties = 0
    baseline_ties = 0
    resolved = new_ties = reversals = agreements = 0
    confusion = {f"baseline_{a}_current_{b}": 0 for a in (-1, 0, 1) for b in (-1, 0, 1)}
    for i in range(len(current) - 1):
        for j in range(i + 1, len(current)):
            current_sign = _sign(current[i], current[j])
            current_ties += current_sign == 0
            if baseline is None:
                continue
            baseline_sign = _sign(baseline[i], baseline[j])
            baseline_ties += baseline_sign == 0
            confusion[f"baseline_{baseline_sign}_current_{current_sign}"] += 1
            resolved += baseline_sign == 0 and current_sign != 0
            new_ties += baseline_sign != 0 and current_sign == 0
            reversals += baseline_sign != 0 and current_sign == -baseline_sign
            agreements += baseline_sign == current_sign

    result: dict[str, Any] = {
        "pair_count": total,
        "tie_rate": current_ties / total,
        "separation_rate": 1.0 - current_ties / total,
    }
    if baseline is not None:
        result.update(
            {
                "baseline_tie_rate": baseline_ties / total,
                "incremental_tie_resolution": resolved / total,
                "conditional_tie_resolution": resolved / baseline_ties if baseline_ties else None,
                "new_tie_rate": new_ties / total,
                "ordering_reversal_rate": reversals / total,
                "ordering_agreement": agreements / total,
                "ordering_confusion": confusion,
            }
        )
    return result


def criterion_effectiveness(grades: Sequence[int | bool]) -> str:
    if not grades:
        raise ValueError("criterion grades must not be empty")
    normalized = tuple(int(value) for value in grades)
    if any(value not in (0, 1) for value in normalized):
        raise ValueError("criterion grades must be binary")
    if all(normalized):
        return "saturated"
    if not any(normalized):
        return "dead"
    return "effective"


def criterion_effectiveness_summary(
    criteria: Mapping[str, Sequence[int | bool]],
) -> dict[str, Any]:
    counts = Counter(criterion_effectiveness(grades) for grades in criteria.values())
    total = len(criteria)
    return {
        "criterion_count": total,
        "counts": {name: counts[name] for name in ("saturated", "dead", "effective")},
        "ratios": {
            name: counts[name] / total if total else None
            for name in ("saturated", "dead", "effective")
        },
    }


def criterion_vector_resolution(
    scores: Sequence[RationalLike], response_grade_vectors: Sequence[Sequence[int | bool]]
) -> dict[str, float | int | None]:
    """Measure score-tied response pairs whose binary criterion vectors differ."""

    values = _fractions(scores)
    if len(values) != len(response_grade_vectors):
        raise ValueError("scores and grade vectors must align")
    vectors = []
    for vector in response_grade_vectors:
        normalized = tuple(int(value) for value in vector)
        if any(value not in (0, 1) for value in normalized):
            raise ValueError("criterion grades must be binary")
        vectors.append(normalized)
    if len({len(vector) for vector in vectors}) > 1:
        raise ValueError("all response grade vectors must have equal length")
    pair_count = len(values) * (len(values) - 1) // 2
    tied_pairs = resolved_pairs = 0
    for i in range(len(values) - 1):
        for j in range(i + 1, len(values)):
            if values[i] == values[j]:
                tied_pairs += 1
                resolved_pairs += vectors[i] != vectors[j]
    return {
        "pair_count": pair_count,
        "score_tied_pair_count": tied_pairs,
        "vector_resolved_pair_count": resolved_pairs,
        "unconditional_vector_resolution": resolved_pairs / pair_count if pair_count else None,
        "conditional_vector_resolution": resolved_pairs / tied_pairs if tied_pairs else None,
    }


def _linear_quantile(sorted_values: Sequence[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def score_spread(scores: Sequence[RationalLike]) -> dict[str, float | int]:
    values = _fractions(scores)
    ordered = sorted(float(value) for value in values)
    q1 = _linear_quantile(ordered, 0.25)
    q3 = _linear_quantile(ordered, 0.75)
    return {
        "population_sd": statistics.pstdev(ordered),
        "iqr": q3 - q1,
        "unique_score_count": len(set(values)),
        "unique_score_ratio": len(set(values)) / len(values),
    }


def _kendall_tau_b(left: Sequence[Fraction], right: Sequence[Fraction]) -> float | None:
    if len(set(left)) == 1 or len(set(right)) == 1:
        return None
    concordant = discordant = ties_left = ties_right = 0
    for i in range(len(left) - 1):
        for j in range(i + 1, len(left)):
            dx = _sign(left[i], left[j])
            dy = _sign(right[i], right[j])
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                ties_left += 1
            elif dy == 0:
                ties_right += 1
            elif dx == dy:
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt(
        (concordant + discordant + ties_left) * (concordant + discordant + ties_right)
    )
    return None if denominator == 0.0 else (concordant - discordant) / denominator


def ranking_agreement(
    baseline_scores: Sequence[RationalLike], current_scores: Sequence[RationalLike]
) -> dict[str, Any]:
    baseline = _fractions(baseline_scores)
    current = _fractions(current_scores)
    if len(baseline) != len(current) or len(current) < 2:
        raise ValueError("rankings must align and contain at least two responses")
    pairwise = pairwise_metrics(current, baseline_scores=baseline)
    baseline_top = {index for index, value in enumerate(baseline) if value == max(baseline)}
    current_top = {index for index, value in enumerate(current) if value == max(current)}
    union = baseline_top | current_top
    return {
        "kendall_tau_b": _kendall_tau_b(baseline, current),
        "kendall_defined": len(set(baseline)) > 1 and len(set(current)) > 1,
        "pairwise_ordering_agreement": pairwise["ordering_agreement"],
        "ordering_confusion": pairwise["ordering_confusion"],
        "top_set_jaccard": len(baseline_top & current_top) / len(union),
        "top_set_exact_match": baseline_top == current_top,
        "baseline_top_count": len(baseline_top),
        "current_top_count": len(current_top),
    }


def count_normalized_gain(gain: float, mean_added_criteria: float) -> float | None:
    gain = float(gain)
    mean_added_criteria = float(mean_added_criteria)
    if not math.isfinite(gain) or not math.isfinite(mean_added_criteria):
        raise ValueError("gain and criterion count must be finite")
    if mean_added_criteria < 0.0:
        raise ValueError("criterion count must be non-negative")
    return None if mean_added_criteria == 0.0 else gain / mean_added_criteria


def count_normalized_summary(
    *, delta_zar: float, delta_separation: float, mean_added_criteria: float
) -> dict[str, float | None]:
    return {
        "mean_added_criteria": float(mean_added_criteria),
        "zar_reduction_per_added_criterion": count_normalized_gain(
            delta_zar, mean_added_criteria
        ),
        "separation_gain_per_added_criterion": count_normalized_gain(
            delta_separation, mean_added_criteria
        ),
    }
