"""Discriminability-only metrics for Phase-1 fresh-vs-stale evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import math
import statistics
from typing import Any, Mapping, Sequence


class MetricContractError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ScoreGroup:
    prompt_id: str
    evaluator_checkpoint: str
    policy_checkpoint: str
    response_ids: tuple[str, ...]
    rewards: tuple[float, ...]
    criterion_grades: Mapping[str, tuple[int | bool, ...]] | None = None

    def __post_init__(self) -> None:
        if len(self.response_ids) < 2 or len(self.response_ids) != len(self.rewards):
            raise MetricContractError(
                "a score group needs aligned response IDs and at least two rewards"
            )
        if len(set(self.response_ids)) != len(self.response_ids):
            raise MetricContractError("response IDs must be unique within a group")
        if any(not math.isfinite(float(value)) for value in self.rewards):
            raise MetricContractError("rewards must be finite")
        if self.criterion_grades is not None:
            for criterion_id, grades in self.criterion_grades.items():
                if len(grades) != len(self.rewards):
                    raise MetricContractError(
                        f"criterion {criterion_id} does not align with responses"
                    )
                if any(int(value) not in (0, 1) for value in grades):
                    raise MetricContractError("criterion grades must be binary")


def _validate_epsilon(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise MetricContractError(f"{name} must be finite and non-negative")
    return value


def _pairs(values: Sequence[float]):
    for left in range(len(values) - 1):
        for right in range(left + 1, len(values)):
            yield left, right


def _criterion_ratios(
    criteria: Mapping[str, tuple[int | bool, ...]] | None,
) -> dict[str, float | int | None]:
    if criteria is None:
        return {
            "criterion_count": 0,
            "effective_criterion_ratio": None,
            "saturated_criterion_ratio": None,
            "dead_criterion_ratio": None,
        }
    counts = {"effective": 0, "saturated": 0, "dead": 0}
    for grades in criteria.values():
        normalized = tuple(int(value) for value in grades)
        if all(normalized):
            counts["saturated"] += 1
        elif not any(normalized):
            counts["dead"] += 1
        else:
            counts["effective"] += 1
    total = len(criteria)
    return {
        "criterion_count": total,
        "effective_criterion_ratio": counts["effective"] / total if total else None,
        "saturated_criterion_ratio": counts["saturated"] / total if total else None,
        "dead_criterion_ratio": counts["dead"] / total if total else None,
    }


def group_metrics(
    group: ScoreGroup,
    *,
    epsilon_z: float,
    epsilon_t: float,
) -> dict[str, Any]:
    epsilon_z = _validate_epsilon(epsilon_z, "epsilon_z")
    epsilon_t = _validate_epsilon(epsilon_t, "epsilon_t")
    rewards = tuple(float(value) for value in group.rewards)
    reward_std = statistics.pstdev(rewards)
    pair_indices = tuple(_pairs(rewards))
    tied = sum(abs(rewards[left] - rewards[right]) <= epsilon_t for left, right in pair_indices)
    pair_count = len(pair_indices)
    criterion = _criterion_ratios(group.criterion_grades)
    return {
        "prompt_id": group.prompt_id,
        "evaluator_checkpoint": group.evaluator_checkpoint,
        "policy_checkpoint": group.policy_checkpoint,
        "response_count": len(rewards),
        "exact_zero_advantage": int(len(set(rewards)) == 1),
        "near_zero_advantage": int(reward_std <= epsilon_z),
        "reward_std": reward_std,
        "pair_count": pair_count,
        "pairwise_tie_rate": tied / pair_count,
        "pairwise_separation_rate": 1.0 - tied / pair_count,
        "top_median_margin": max(rewards) - statistics.median(rewards),
        **criterion,
    }


def kendall_tau_b(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        raise MetricContractError("Kendall rankings must align and contain two values")
    concordant = discordant = ties_left = ties_right = 0
    for i, j in _pairs(left):
        dx = (left[i] > left[j]) - (left[i] < left[j])
        dy = (right[i] > right[j]) - (right[i] < right[j])
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
        (concordant + discordant + ties_left)
        * (concordant + discordant + ties_right)
    )
    return None if denominator == 0.0 else (concordant - discordant) / denominator


def compare_fresh_stale(
    stale: ScoreGroup,
    fresh: ScoreGroup,
    *,
    epsilon_z: float,
    epsilon_t: float,
) -> dict[str, Any]:
    if stale.prompt_id != fresh.prompt_id:
        raise MetricContractError("fresh/stale prompt IDs must match")
    if stale.policy_checkpoint != fresh.policy_checkpoint:
        raise MetricContractError("fresh/stale policy checkpoints must match")
    if stale.response_ids != fresh.response_ids:
        raise MetricContractError(
            "fresh/stale metrics require identical ordered response IDs"
        )
    stale_metrics = group_metrics(stale, epsilon_z=epsilon_z, epsilon_t=epsilon_t)
    fresh_metrics = group_metrics(fresh, epsilon_z=epsilon_z, epsilon_t=epsilon_t)
    stale_rewards = tuple(float(value) for value in stale.rewards)
    fresh_rewards = tuple(float(value) for value in fresh.rewards)
    stale_ties = 0
    resolved = 0
    for left, right in _pairs(stale_rewards):
        is_stale_tie = abs(stale_rewards[left] - stale_rewards[right]) <= epsilon_t
        is_fresh_tie = abs(fresh_rewards[left] - fresh_rewards[right]) <= epsilon_t
        stale_ties += is_stale_tie
        resolved += is_stale_tie and not is_fresh_tie
    pair_count = int(stale_metrics["pair_count"])

    def optional_delta(fresh_key: str, stale_key: str) -> float | None:
        fresh_value = fresh_metrics[fresh_key]
        stale_value = stale_metrics[stale_key]
        if fresh_value is None or stale_value is None:
            return None
        return float(fresh_value) - float(stale_value)

    return {
        "prompt_id": fresh.prompt_id,
        "policy_checkpoint": fresh.policy_checkpoint,
        "stale_evaluator_checkpoint": stale.evaluator_checkpoint,
        "fresh_evaluator_checkpoint": fresh.evaluator_checkpoint,
        "response_ids": list(fresh.response_ids),
        "same_pool_b": True,
        "stale": stale_metrics,
        "fresh": fresh_metrics,
        "v_adj_zar": (
            stale_metrics["exact_zero_advantage"]
            - fresh_metrics["exact_zero_advantage"]
        ),
        "delta_near_zero_advantage": (
            stale_metrics["near_zero_advantage"]
            - fresh_metrics["near_zero_advantage"]
        ),
        "delta_tie_rate": (
            stale_metrics["pairwise_tie_rate"]
            - fresh_metrics["pairwise_tie_rate"]
        ),
        "delta_separation_rate": (
            fresh_metrics["pairwise_separation_rate"]
            - stale_metrics["pairwise_separation_rate"]
        ),
        "delta_effective_criterion_ratio": optional_delta(
            "effective_criterion_ratio", "effective_criterion_ratio"
        ),
        "delta_top_median_margin": (
            fresh_metrics["top_median_margin"]
            - stale_metrics["top_median_margin"]
        ),
        "incremental_tie_resolution": (
            resolved / stale_ties if stale_ties else None
        ),
        "incremental_tie_resolution_unconditional": resolved / pair_count,
        "conditional_tie_resolution": resolved / stale_ties if stale_ties else None,
        "kendall_tau_b": kendall_tau_b(stale_rewards, fresh_rewards),
        "ranking_similarity_is_correctness": False,
    }


def _mean(records: Sequence[Mapping[str, Any]], path: Sequence[str]) -> float | None:
    values: list[float] = []
    for record in records:
        value: Any = record
        for key in path:
            value = value[key]
        if value is not None:
            values.append(float(value))
    return math.fsum(values) / len(values) if values else None


def aggregate_comparisons(
    comparisons: Sequence[Mapping[str, Any]],
    *,
    comparison_kind: str,
) -> dict[str, Any]:
    if not comparisons:
        raise MetricContractError("at least one comparison is required")
    if len({str(item["prompt_id"]) for item in comparisons}) != len(comparisons):
        raise MetricContractError("prompt comparisons must be unique")
    if comparison_kind not in {"adjacent_update_value", "reuse_horizon"}:
        raise MetricContractError(f"unknown comparison kind: {comparison_kind}")
    result = {
        "schema_version": 1,
        "comparison_kind": comparison_kind,
        "prompt_count": len(comparisons),
        "delta_near_zero_advantage": _mean(
            comparisons, ("delta_near_zero_advantage",)
        ),
        "delta_tie_rate": _mean(comparisons, ("delta_tie_rate",)),
        "delta_separation_rate": _mean(
            comparisons, ("delta_separation_rate",)
        ),
        "delta_effective_criterion_ratio": _mean(
            comparisons, ("delta_effective_criterion_ratio",)
        ),
        "delta_top_median_margin": _mean(
            comparisons, ("delta_top_median_margin",)
        ),
        "incremental_tie_resolution": _mean(
            comparisons, ("incremental_tie_resolution",)
        ),
        "incremental_tie_resolution_unconditional": _mean(
            comparisons, ("incremental_tie_resolution_unconditional",)
        ),
        "conditional_tie_resolution": _mean(
            comparisons, ("conditional_tie_resolution",)
        ),
        "kendall_tau_b": _mean(comparisons, ("kendall_tau_b",)),
        "stale_zar": _mean(comparisons, ("stale", "exact_zero_advantage")),
        "fresh_zar": _mean(comparisons, ("fresh", "exact_zero_advantage")),
        "absolute_metric_scope": "within_method_only",
    }
    zar_value = _mean(comparisons, ("v_adj_zar",))
    if comparison_kind == "adjacent_update_value":
        result["v_adj_zar"] = zar_value
    else:
        result["l_zar"] = zar_value
    return result


def empirical_reuse_horizon(
    matrix_rows: Sequence[Mapping[str, Any]],
    *,
    tau: int,
    practical_margin_delta_d: float,
) -> int:
    margin = _validate_epsilon(practical_margin_delta_d, "practical_margin_delta_d")
    ordered = sorted(
        (
            (int(row["policy_step"]), float(row["l_zar"]))
            for row in matrix_rows
            if int(row["evaluator_step"]) == tau
        ),
        key=lambda item: item[0],
    )
    if not ordered or ordered[0][0] != tau:
        raise MetricContractError("reuse-horizon rows must begin at their anchor")
    horizon = tau
    for policy_step, stale_loss in ordered:
        if stale_loss > margin:
            break
        horizon = policy_step
    return horizon
