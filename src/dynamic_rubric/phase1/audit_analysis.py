"""Offline aggregation for the Phase-1 checkpoint audit.

The scorer emits one receipt per saved response.  This module is deliberately a
consumer only: it groups those receipts, enforces same-pool comparisons, joins
policy-distance/state artifacts, and writes restartable CPU-only reports.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics
from typing import Any

from ..artifacts import read_json, read_jsonl, write_json_atomic, write_jsonl_atomic
from ..hashing import sha256_file
from .checkpoint_inventory import discover_committed_policy_checkpoints
from .metrics import ScoreGroup, aggregate_comparisons, compare_fresh_stale


PRIMARY_ANCHORS = (0, 9, 16, 32)
EXPLORATORY_STEP = 34
DEFAULT_BOOTSTRAP_ITERATIONS = 10_000
DEFAULT_BOOTSTRAP_SEED = 20_250_807
PRIMARY_POOL = "probe_B"


class AuditAnalysisError(RuntimeError):
    """Raised when audit inputs cannot support an aligned analysis."""


def _mean(values: Iterable[float]) -> float | None:
    materialized = [float(value) for value in values]
    return math.fsum(materialized) / len(materialized) if materialized else None


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _metric_seed(seed: int, *parts: object) -> int:
    suffix = hashlib.sha256("\0".join(map(str, parts)).encode()).digest()[:8]
    return int(seed) ^ int.from_bytes(suffix, "big")


def prompt_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    statistic: Callable[[Sequence[Mapping[str, Any]]], float | None],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any] | None:
    """Prompt-clustered percentile CI, recomputing the statistic after resampling."""

    if iterations <= 0:
        raise AuditAnalysisError("bootstrap iterations must be positive")
    if not rows:
        return None
    prompt_ids = [str(row["prompt_id"]) for row in rows]
    if not all(prompt_ids) or len(prompt_ids) != len(set(prompt_ids)):
        raise AuditAnalysisError("bootstrap rows require unique, non-empty prompt IDs")
    point = statistic(rows)
    if point is None or not math.isfinite(float(point)):
        return None
    rng = random.Random(seed)
    estimates: list[float] = []
    count = len(rows)
    for _ in range(iterations):
        sample = [rows[rng.randrange(count)] for _ in range(count)]
        estimate = statistic(sample)
        if estimate is not None and math.isfinite(float(estimate)):
            estimates.append(float(estimate))
    if not estimates:
        return None
    return {
        "point_estimate": float(point),
        "ci_low": _quantile(estimates, 0.025),
        "ci_high": _quantile(estimates, 0.975),
        "confidence": 0.95,
        "n_prompt_clusters": count,
        "iterations": iterations,
        "seed": seed,
    }


def _checkpoint_step(row: Mapping[str, Any], kind: str) -> int:
    explicit = f"{kind}_step"
    if explicit in row:
        return int(row[explicit])
    if kind == "policy" and "global_step" in row:
        return int(row["global_step"])
    name = str(row.get(f"{kind}_checkpoint", ""))
    matches = re.findall(r"\d+", name)
    if len(matches) != 1:
        raise AuditAnalysisError(
            f"{kind}_step must be explicit when checkpoint is not a numeric label: {name!r}"
        )
    return int(matches[0])


def score_groups(
    score_rows: Sequence[Mapping[str, Any]], *, pool: str = PRIMARY_POOL
) -> dict[tuple[int, int, str], ScoreGroup]:
    """Convert response-level grader receipts into validated prompt score groups."""

    selected = [row for row in score_rows if str(row.get("pool", "")) == pool]
    if not selected:
        raise AuditAnalysisError(f"no scoring rows found for primary pool {pool!r}")
    grouped: dict[tuple[int, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    identities: set[tuple[int, int, str, str]] = set()
    for row in selected:
        policy_step = _checkpoint_step(row, "policy")
        evaluator_step = _checkpoint_step(row, "evaluator")
        prompt_id, response_id = str(row.get("prompt_id", "")), str(row.get("response_id", ""))
        if not prompt_id or not response_id:
            raise AuditAnalysisError("score receipts require prompt_id and response_id")
        identity = policy_step, evaluator_step, prompt_id, response_id
        if identity in identities:
            raise AuditAnalysisError(f"duplicate score receipt: {identity}")
        identities.add(identity)
        grouped[(policy_step, evaluator_step, prompt_id)].append(row)

    output: dict[tuple[int, int, str], ScoreGroup] = {}
    for key, rows in grouped.items():
        policy_step, evaluator_step, prompt_id = key
        ordered = sorted(rows, key=lambda row: str(row["response_id"]))
        criterion_ids: tuple[str, ...] | None = None
        grades_by_criterion: dict[str, list[int]] = defaultdict(list)
        for row in ordered:
            reward = float(row["reward"])
            if not math.isfinite(reward):
                raise AuditAnalysisError(f"non-finite reward in score group {key}")
            grades = [(str(cid), int(grade)) for cid, grade in row.get("grades", ())]
            if len({cid for cid, _ in grades}) != len(grades):
                raise AuditAnalysisError(f"duplicate criterion grade in score group {key}")
            current_ids = tuple(cid for cid, _ in grades)
            if criterion_ids is None:
                criterion_ids = current_ids
            elif current_ids != criterion_ids:
                raise AuditAnalysisError(f"criterion inventory/order drift in score group {key}")
            for criterion_id, grade in grades:
                if grade not in (0, 1):
                    raise AuditAnalysisError("criterion grades must be binary")
                grades_by_criterion[criterion_id].append(grade)
        output[key] = ScoreGroup(
            prompt_id=prompt_id,
            evaluator_checkpoint=str(ordered[0]["evaluator_checkpoint"]),
            policy_checkpoint=str(ordered[0]["policy_checkpoint"]),
            response_ids=tuple(str(row["response_id"]) for row in ordered),
            rewards=tuple(float(row["reward"]) for row in ordered),
            criterion_grades={
                criterion_id: tuple(values) for criterion_id, values in grades_by_criterion.items()
            },
        )
    return output


def criterion_counts(group: ScoreGroup) -> Counter[str]:
    counts: Counter[str] = Counter(effective=0, saturated=0, dead=0)
    for values in (group.criterion_grades or {}).values():
        grades = tuple(int(value) for value in values)
        category = "saturated" if all(grades) else "dead" if not any(grades) else "effective"
        counts[category] += 1
    return counts


def pooled_criterion_summary(groups: Sequence[ScoreGroup]) -> dict[str, Any]:
    """Pool criterion occurrences; never average prompt-level criterion ratios."""

    counts: Counter[str] = Counter(effective=0, saturated=0, dead=0)
    for group in groups:
        counts.update(criterion_counts(group))
    total = sum(counts.values())
    return {
        "criterion_occurrences": total,
        "effective_count": counts["effective"],
        "saturated_count": counts["saturated"],
        "dead_count": counts["dead"],
        "effective_criterion_ratio": counts["effective"] / total if total else None,
        "saturation_ratio": counts["saturated"] / total if total else None,
        "dead_ratio": counts["dead"] / total if total else None,
        "aggregation": "criterion_occurrence_pooled",
    }


def _comparison_summary(
    groups: Mapping[tuple[int, int, str], ScoreGroup],
    *,
    evaluator_step: int,
    policy_step: int,
    prompt_ids: Sequence[str],
    epsilon_z: float,
    epsilon_t: float,
    delta_d: float,
    bootstrap_iterations: int,
    bootstrap_seed: int,
    exploratory: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    comparisons: list[dict[str, Any]] = []
    stale_groups: list[ScoreGroup] = []
    fresh_groups: list[ScoreGroup] = []
    for prompt_id in prompt_ids:
        stale_key = policy_step, evaluator_step, prompt_id
        fresh_key = policy_step, policy_step, prompt_id
        try:
            stale, fresh = groups[stale_key], groups[fresh_key]
        except KeyError as error:
            raise AuditAnalysisError(f"missing aligned score group: {error.args[0]}") from error
        comparison = compare_fresh_stale(stale, fresh, epsilon_z=epsilon_z, epsilon_t=epsilon_t)
        comparison.update(
            evaluator_step=evaluator_step,
            policy_step=policy_step,
            exploratory=exploratory,
        )
        comparisons.append(comparison)
        stale_groups.append(stale)
        fresh_groups.append(fresh)

    aggregate = aggregate_comparisons(comparisons, comparison_kind="reuse_horizon")
    stale_criteria = pooled_criterion_summary(stale_groups)
    fresh_criteria = pooled_criterion_summary(fresh_groups)
    stale_ecr = stale_criteria["effective_criterion_ratio"]
    fresh_ecr = fresh_criteria["effective_criterion_ratio"]

    def comparison_mean(name: str) -> Callable[[Sequence[Mapping[str, Any]]], float | None]:
        return lambda rows: _mean(float(row[name]) for row in rows if row[name] is not None)

    def pooled_ecr_delta(rows: Sequence[Mapping[str, Any]]) -> float | None:
        stale_counts: Counter[str] = Counter()
        fresh_counts: Counter[str] = Counter()
        for row in rows:
            prompt_id = str(row["prompt_id"])
            stale_counts.update(criterion_counts(groups[(policy_step, evaluator_step, prompt_id)]))
            fresh_counts.update(criterion_counts(groups[(policy_step, policy_step, prompt_id)]))
        stale_total, fresh_total = sum(stale_counts.values()), sum(fresh_counts.values())
        if not stale_total or not fresh_total:
            return None
        return fresh_counts["effective"] / fresh_total - stale_counts["effective"] / stale_total

    bootstrap_metrics = {
        "l_zar": "v_adj_zar",
        "delta_tie_rate": "delta_tie_rate",
        "delta_separation_rate": "delta_separation_rate",
        "delta_top_median_margin": "delta_top_median_margin",
        "tie_resolution_unconditional": "incremental_tie_resolution_unconditional",
        "kendall_tau_b": "kendall_tau_b",
    }
    bootstraps = {
        name: prompt_bootstrap(
            comparisons,
            comparison_mean(field),
            iterations=bootstrap_iterations,
            seed=_metric_seed(bootstrap_seed, evaluator_step, policy_step, name),
        )
        for name, field in bootstrap_metrics.items()
    }
    bootstraps["delta_effective_criterion_ratio_pooled"] = prompt_bootstrap(
        comparisons,
        pooled_ecr_delta,
        iterations=bootstrap_iterations,
        seed=_metric_seed(bootstrap_seed, evaluator_step, policy_step, "pooled-ecr"),
    )
    l_zar_ci = bootstraps["l_zar"]
    if l_zar_ci is not None:
        l_zar_ci["ci_excludes_zero_positive"] = l_zar_ci["ci_low"] > 0.0
        l_zar_ci["ci_exceeds_practical_delta_d"] = l_zar_ci["ci_low"] > delta_d

    summary = {
        "schema_version": 1,
        "evaluator_step": evaluator_step,
        "policy_step": policy_step,
        "evaluator_age_steps": policy_step - evaluator_step,
        "prompt_count": len(comparisons),
        "response_count_per_prompt": len(stale_groups[0].response_ids),
        "stale_zar": aggregate["stale_zar"],
        "fresh_zar": aggregate["fresh_zar"],
        "l_zar": aggregate["l_zar"],
        "stale_pairwise_tie_rate": _mean(row["stale"]["pairwise_tie_rate"] for row in comparisons),
        "fresh_pairwise_tie_rate": _mean(row["fresh"]["pairwise_tie_rate"] for row in comparisons),
        "delta_tie_rate": aggregate["delta_tie_rate"],
        "delta_separation_rate": aggregate["delta_separation_rate"],
        "conditional_tie_resolution": aggregate["conditional_tie_resolution"],
        "tie_resolution_unconditional": aggregate["incremental_tie_resolution_unconditional"],
        "stale_top_median_margin": _mean(row["stale"]["top_median_margin"] for row in comparisons),
        "fresh_top_median_margin": _mean(row["fresh"]["top_median_margin"] for row in comparisons),
        "delta_top_median_margin": aggregate["delta_top_median_margin"],
        "kendall_tau_b": aggregate["kendall_tau_b"],
        "criterion_pooled": {
            "stale": stale_criteria,
            "fresh": fresh_criteria,
            "delta_effective_criterion_ratio": (
                float(fresh_ecr) - float(stale_ecr)
                if stale_ecr is not None and fresh_ecr is not None
                else None
            ),
        },
        "bootstrap_95ci": bootstraps,
        "practical_margin_delta_d": delta_d,
        "point_exceeds_practical_delta_d": float(aggregate["l_zar"]) > delta_d,
        "same_pool_b": True,
        "pool": PRIMARY_POOL,
        "exploratory": exploratory,
        "ranking_similarity_is_correctness": False,
    }
    return summary, comparisons


def _validate_prompt_inventory(
    groups: Mapping[tuple[int, int, str], ScoreGroup], policy_steps: Sequence[int]
) -> tuple[str, ...]:
    inventories = []
    for step in policy_steps:
        prompts = {prompt for policy, evaluator, prompt in groups if policy == evaluator == step}
        if not prompts:
            raise AuditAnalysisError(f"no fresh score groups for policy step {step}")
        inventories.append(prompts)
    if any(prompts != inventories[0] for prompts in inventories[1:]):
        raise AuditAnalysisError("fixed probe prompt inventory differs across policy checkpoints")
    return tuple(sorted(inventories[0]))


def _horizons(
    matrix: Sequence[Mapping[str, Any]], anchors: Sequence[int], delta_d: float
) -> list[dict[str, Any]]:
    output = []
    for anchor in anchors:
        rows = sorted(
            (row for row in matrix if int(row["evaluator_step"]) == anchor),
            key=lambda row: int(row["policy_step"]),
        )
        first_failure = next((row for row in rows if float(row["l_zar"]) > delta_d), None)
        acceptable = [
            int(row["policy_step"])
            for row in rows
            if first_failure is None or int(row["policy_step"]) < int(first_failure["policy_step"])
        ]
        output.append(
            {
                "evaluator_step": anchor,
                "last_acceptable_policy_step": max(acceptable) if acceptable else None,
                "first_exceedance_policy_step": (
                    int(first_failure["policy_step"]) if first_failure else None
                ),
                "right_censored": first_failure is None,
                "observed_through_policy_step": int(rows[-1]["policy_step"]),
                "practical_margin_delta_d": delta_d,
                "threshold_basis": "point_estimate_l_zar",
            }
        )
    return output


def _state_rows(rows: Sequence[Mapping[str, Any]]) -> dict[int, dict[str, float]]:
    aliases = {
        "cumulative_prompt_exposures": ("cumulative_prompt_exposures", "cumulative_prompts"),
        "cumulative_completions": ("cumulative_completions",),
        "cumulative_response_tokens": ("cumulative_response_tokens",),
        "response_length": ("response_length", "mean_response_length", "mean_response_tokens"),
    }
    output: dict[int, dict[str, float]] = {}
    for row in rows:
        step = int(row.get("policy_step", row.get("global_step", -1)))
        if step < 0 or step in output:
            raise AuditAnalysisError(f"invalid or duplicate policy state step: {step}")
        values: dict[str, float] = {}
        for canonical, names in aliases.items():
            found = next((row[name] for name in names if name in row), None)
            if found is None or not math.isfinite(float(found)):
                raise AuditAnalysisError(f"policy state {step} lacks finite {canonical}")
            values[canonical] = float(found)
        output[step] = values
    for field in tuple(aliases)[:3]:
        ordered = [output[step][field] for step in sorted(output)]
        if any(right < left for left, right in zip(ordered, ordered[1:])):
            raise AuditAnalysisError(f"policy state field is not cumulative: {field}")
    return output


def _distance_rows(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[int, int], float]:
    output: dict[tuple[int, int], float] = {}
    for row in rows:
        current = _checkpoint_step(row, "current_policy")
        stale = _checkpoint_step(row, "stale_policy")
        response = _checkpoint_step(row, "response_policy")
        if response != current:
            raise AuditAnalysisError("policy distance must use the current policy's pool")
        value = float(row["prompt_balanced_sampled_kl_mean"])
        if not math.isfinite(value):
            raise AuditAnalysisError("policy distance must be finite")
        key = stale, current
        if key in output and output[key] != value:
            raise AuditAnalysisError(f"conflicting policy distance: {key}")
        output[key] = value
    return output


def _preupdate_join(
    comparisons: Sequence[Mapping[str, Any]],
    *,
    states: Mapping[int, Mapping[str, float]],
    distances: Mapping[tuple[int, int], float],
    checkpoint_order: Sequence[int],
) -> list[dict[str, Any]]:
    output = []
    order = {step: index for index, step in enumerate(checkpoint_order)}
    for comparison in comparisons:
        evaluator = int(comparison["evaluator_step"])
        policy = int(comparison["policy_step"])
        if evaluator == policy:
            continue
        if evaluator not in states or policy not in states:
            raise AuditAnalysisError(f"missing policy state for comparison {(evaluator, policy)}")
        current, stale = states[policy], states[evaluator]
        try:
            policy_kl = distances[(evaluator, policy)]
            cumulative_kl = 0.0 if policy == 0 else distances[(0, policy)]
        except KeyError as error:
            raise AuditAnalysisError(f"missing policy distance {error.args[0]}") from error
        criterion = comparison["criterion_pooled"]["stale"]
        row = {
            "global_step": policy,
            "evaluator_step": evaluator,
            "evaluator_age_steps": policy - evaluator,
            "evaluator_age_checkpoints": order[policy] - order[evaluator],
            "cumulative_prompts_since_evaluator": (
                current["cumulative_prompt_exposures"] - stale["cumulative_prompt_exposures"]
            ),
            "cumulative_completions_since_evaluator": (
                current["cumulative_completions"] - stale["cumulative_completions"]
            ),
            "cumulative_response_tokens_since_evaluator": (
                current["cumulative_response_tokens"] - stale["cumulative_response_tokens"]
            ),
            "policy_kl_current_vs_evaluator": policy_kl,
            "cumulative_policy_kl_from_pi0": cumulative_kl,
            "response_length_shift": current["response_length"] - stale["response_length"],
            "stale_zar": comparison["stale_zar"],
            "stale_pairwise_tie_rate": comparison["stale_pairwise_tie_rate"],
            "stale_pairwise_separation_rate": 1.0 - float(comparison["stale_pairwise_tie_rate"]),
            "stale_effective_criterion_ratio": criterion["effective_criterion_ratio"],
            "stale_saturation_ratio": criterion["saturation_ratio"],
            "stale_dead_ratio": criterion["dead_ratio"],
            "stale_reward_std": _mean(
                item["stale"]["reward_std"] for item in comparison["prompt_comparisons"]
            ),
            "stale_top_median_margin": comparison["stale_top_median_margin"],
            "l_zar": comparison["l_zar"],
            "delta_tie_rate": comparison["delta_tie_rate"],
            "delta_effective_criterion_ratio": comparison["criterion_pooled"][
                "delta_effective_criterion_ratio"
            ],
            "delta_top_median_margin": comparison["delta_top_median_margin"],
            "kendall_tau_b": comparison["kendall_tau_b"],
            "exploratory": comparison["exploratory"],
        }
        output.append(row)
    return output


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 3:
        return None
    left_mean, right_mean = statistics.mean(left), statistics.mean(right)
    numerator = math.fsum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_ss = math.fsum((x - left_mean) ** 2 for x in left)
    right_ss = math.fsum((y - right_mean) ** 2 for y in right)
    denominator = math.sqrt(left_ss * right_ss)
    return numerator / denominator if denominator else None


def _associations(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    predictors = (
        "evaluator_age_steps",
        "cumulative_prompts_since_evaluator",
        "cumulative_completions_since_evaluator",
        "cumulative_response_tokens_since_evaluator",
        "policy_kl_current_vs_evaluator",
        "cumulative_policy_kl_from_pi0",
        "response_length_shift",
    )
    outcomes = (
        "l_zar",
        "delta_tie_rate",
        "delta_effective_criterion_ratio",
        "delta_top_median_margin",
        "kendall_tau_b",
    )
    output = []
    for predictor in predictors:
        for outcome in outcomes:
            pairs = [
                (float(row[predictor]), float(row[outcome]))
                for row in rows
                if not row["exploratory"]
                and row.get(predictor) is not None
                and row.get(outcome) is not None
            ]
            x, y = [item[0] for item in pairs], [item[1] for item in pairs]
            correlation = _pearson(x, y)
            slope = intercept = r_squared = None
            if correlation is not None:
                x_mean, y_mean = statistics.mean(x), statistics.mean(y)
                denominator = math.fsum((value - x_mean) ** 2 for value in x)
                if denominator:
                    slope = (
                        math.fsum((left - x_mean) * (right - y_mean) for left, right in pairs)
                        / denominator
                    )
                    intercept = y_mean - slope * x_mean
                    r_squared = correlation**2
            output.append(
                {
                    "predictor": predictor,
                    "outcome": outcome,
                    "n_cells": len(pairs),
                    "pearson_r": correlation,
                    "ols_slope": slope,
                    "ols_intercept": intercept,
                    "ols_r_squared": r_squared,
                    "analysis_role": "exploratory_descriptive_noncausal",
                }
            )
    return output


def analyze_records(
    score_rows: Sequence[Mapping[str, Any]],
    policy_state_rows: Sequence[Mapping[str, Any]],
    policy_distance_rows: Sequence[Mapping[str, Any]],
    *,
    anchors: Sequence[int] = PRIMARY_ANCHORS,
    exploratory_step: int | None = EXPLORATORY_STEP,
    epsilon_z: float,
    epsilon_t: float,
    practical_margin_delta_d: float,
    bootstrap_iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    anchors = tuple(int(step) for step in anchors)
    if anchors != tuple(sorted(set(anchors))) or not anchors:
        raise AuditAnalysisError("anchors must be non-empty, unique, and increasing")
    if any(
        value < 0.0 or not math.isfinite(float(value))
        for value in (epsilon_z, epsilon_t, practical_margin_delta_d)
    ):
        raise AuditAnalysisError("analysis thresholds must be finite and non-negative")
    policy_steps = (*anchors, *((exploratory_step,) if exploratory_step is not None else ()))
    groups = score_groups(score_rows)
    prompt_ids = _validate_prompt_inventory(groups, policy_steps)
    matrix: list[dict[str, Any]] = []
    prompt_comparisons: list[dict[str, Any]] = []

    for policy_step in anchors:
        for evaluator_step in anchors:
            if evaluator_step > policy_step:
                continue
            summary, prompts = _comparison_summary(
                groups,
                evaluator_step=evaluator_step,
                policy_step=policy_step,
                prompt_ids=prompt_ids,
                epsilon_z=epsilon_z,
                epsilon_t=epsilon_t,
                delta_d=practical_margin_delta_d,
                bootstrap_iterations=bootstrap_iterations,
                bootstrap_seed=bootstrap_seed,
                exploratory=False,
            )
            summary["prompt_comparisons"] = prompts
            matrix.append(summary)
            prompt_comparisons.extend(prompts)

    exploratory: list[dict[str, Any]] = []
    if exploratory_step is not None:
        for evaluator_step in (*anchors, exploratory_step):
            summary, prompts = _comparison_summary(
                groups,
                evaluator_step=evaluator_step,
                policy_step=exploratory_step,
                prompt_ids=prompt_ids,
                epsilon_z=epsilon_z,
                epsilon_t=epsilon_t,
                delta_d=practical_margin_delta_d,
                bootstrap_iterations=bootstrap_iterations,
                bootstrap_seed=bootstrap_seed,
                exploratory=True,
            )
            summary["prompt_comparisons"] = prompts
            exploratory.append(summary)
            prompt_comparisons.extend(prompts)

    ordered_steps = tuple(policy_steps)
    previous = {step: ordered_steps[index - 1] for index, step in enumerate(ordered_steps) if index}
    all_comparisons = matrix + exploratory
    adjacent = [
        row
        for row in all_comparisons
        if row["policy_step"] in previous and row["evaluator_step"] == previous[row["policy_step"]]
    ]
    joined = _preupdate_join(
        all_comparisons,
        states=_state_rows(policy_state_rows),
        distances=_distance_rows(policy_distance_rows),
        checkpoint_order=ordered_steps,
    )
    report = {
        "schema_version": 1,
        "primary_dataset": "fixed_train_probe",
        "pool": PRIMARY_POOL,
        "anchors": list(anchors),
        "exploratory_step": exploratory_step,
        "thresholds": {
            "epsilon_z": float(epsilon_z),
            "epsilon_t": float(epsilon_t),
            "practical_margin_delta_d": float(practical_margin_delta_d),
        },
        "bootstrap": {
            "method": "paired_prompt_cluster_percentile",
            "iterations": bootstrap_iterations,
            "seed": bootstrap_seed,
        },
        "prompt_count": len(prompt_ids),
        "triangle": [
            {key: value for key, value in row.items() if key != "prompt_comparisons"}
            for row in matrix
        ],
        "adjacent": [
            {key: value for key, value in row.items() if key != "prompt_comparisons"}
            for row in adjacent
        ],
        "reuse_horizon": _horizons(matrix, anchors, practical_margin_delta_d),
        "exploratory_checkpoint": [
            {key: value for key, value in row.items() if key != "prompt_comparisons"}
            for row in exploratory
        ],
        "preupdate_stale_state": joined,
        "exploratory_associations": _associations(joined),
        "guards": {
            "same_response_pool_b_required": True,
            "probe_a_used_for_scoring": False,
            "training_batches_used_for_scoring": False,
            "heldout_used_for_training": False,
            "step_34_excluded_from_confirmatory_triangle_and_horizon": (
                exploratory_step is not None
            ),
            "non_anchor_exploratory_step_excluded_from_triangle_and_horizon": (
                exploratory_step is not None
            ),
            "stale_zar_excluded_as_predictor_of_l_zar": True,
            "ranking_similarity_is_correctness": False,
        },
        "interpretation": (
            "Associations are exploratory and noncausal. Stale ZAR is a scored outcome, "
            "not an independent predictor of L_ZAR."
        ),
        "_prompt_comparisons": prompt_comparisons,
    }
    return report


def _training_score_group(group: Mapping[str, Any], side: str) -> ScoreGroup:
    rows = sorted(group.get(side, ()), key=lambda row: int(row["rollout_index"]))
    if len(rows) < 2:
        raise AuditAnalysisError(f"training group {group.get('prompt_id')} has no {side} scores")
    criterion_ids: tuple[str, ...] | None = None
    grades_by_criterion: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        grades = tuple((str(item[0]), int(item[1])) for item in row.get("grades", ()))
        current_ids = tuple(item[0] for item in grades)
        if criterion_ids is None:
            criterion_ids = current_ids
        elif current_ids != criterion_ids:
            raise AuditAnalysisError(f"{side} criterion inventory drifted within a group")
        for criterion_id, grade in grades:
            if grade not in (0, 1):
                raise AuditAnalysisError("training criterion grades must be binary")
            grades_by_criterion[criterion_id].append(grade)
    evaluator = (
        str(group["fresh_creation_update"])
        if side == "fresh"
        else str(group["stale_creation_update"])
    )
    return ScoreGroup(
        prompt_id=str(group["prompt_id"]),
        evaluator_checkpoint=evaluator,
        policy_checkpoint=str(group["policy_step"]),
        response_ids=tuple(str(row["response_id"]) for row in rows),
        rewards=tuple(float(row["reward"]) for row in rows),
        criterion_grades={key: tuple(values) for key, values in grades_by_criterion.items()},
    )


def _pair_counts(rewards: Sequence[float], epsilon_t: float) -> tuple[int, int]:
    pairs = ties = 0
    for left in range(len(rewards) - 1):
        for right in range(left + 1, len(rewards)):
            pairs += 1
            ties += abs(float(rewards[left]) - float(rewards[right])) <= epsilon_t
    return pairs, ties


def training_comparison_row(
    group: Mapping[str, Any], *, epsilon_z: float, epsilon_t: float
) -> dict[str, Any]:
    if group.get("pool") != "train_batch":
        raise AuditAnalysisError("training analysis accepts only train_batch groups")
    stale = _training_score_group(group, "stale")
    fresh = _training_score_group(group, "fresh")
    comparison = compare_fresh_stale(stale, fresh, epsilon_z=epsilon_z, epsilon_t=epsilon_t)
    pair_count, stale_ties = _pair_counts(stale.rewards, epsilon_t)
    fresh_pair_count, fresh_ties = _pair_counts(fresh.rewards, epsilon_t)
    if pair_count != fresh_pair_count:
        raise AuditAnalysisError("fresh/stale pair inventories differ")
    stale_counts, fresh_counts = criterion_counts(stale), criterion_counts(fresh)
    row = {
        "schema_version": 1,
        "domain": group.get("domain"),
        "method": group.get("method"),
        "seed": group.get("seed"),
        "global_step": int(group["global_step"]),
        "policy_step": int(group["policy_step"]),
        "evaluator_step": int(group["evaluator_step"]),
        "prompt_id": str(group["prompt_id"]),
        "pool": "train_batch",
        "fresh_creation_update": int(group["fresh_creation_update"]),
        "stale_creation_update": int(group["stale_creation_update"]),
        "evaluator_age_steps": int(group["evaluator_age_steps"]),
        "clock": dict(group.get("clock", {})),
        "fresh": comparison["fresh"],
        "stale": comparison["stale"],
        "v_adj_zar": comparison["v_adj_zar"],
        "delta_near_zero_advantage": comparison["delta_near_zero_advantage"],
        "delta_tie_rate": comparison["delta_tie_rate"],
        "delta_separation_rate": comparison["delta_separation_rate"],
        "delta_effective_criterion_ratio": comparison["delta_effective_criterion_ratio"],
        "delta_top_median_margin": comparison["delta_top_median_margin"],
        "incremental_tie_resolution": comparison["incremental_tie_resolution"],
        "kendall_tau_b": comparison["kendall_tau_b"],
        "pair_count": pair_count,
        "stale_tied_pairs": stale_ties,
        "fresh_tied_pairs": fresh_ties,
        "resolved_stale_tied_pairs": stale_ties
        - sum(
            1
            for left in range(len(stale.rewards) - 1)
            for right in range(left + 1, len(stale.rewards))
            if abs(stale.rewards[left] - stale.rewards[right]) <= epsilon_t
            and abs(fresh.rewards[left] - fresh.rewards[right]) <= epsilon_t
        ),
        "stale_criterion_counts": dict(stale_counts),
        "fresh_criterion_counts": dict(fresh_counts),
        "same_response_pool": True,
        "same_pool_b": False,
        "analysis_role": "operational_in_sample",
    }
    return row


def _cluster_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    statistic: Callable[[Sequence[Mapping[str, Any]]], float | None],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any] | None:
    clusters: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        clusters[str(row["prompt_id"])].append(row)
    point = statistic(rows)
    if point is None or not clusters:
        return None
    keys = sorted(clusters)
    rng = random.Random(seed)
    estimates = []
    for _ in range(iterations):
        sample = [
            row for _index in range(len(keys)) for row in clusters[keys[rng.randrange(len(keys))]]
        ]
        value = statistic(sample)
        if value is not None and math.isfinite(float(value)):
            estimates.append(float(value))
    return {
        "point_estimate": float(point),
        "ci_low": _quantile(estimates, 0.025),
        "ci_high": _quantile(estimates, 0.975),
        "confidence": 0.95,
        "n_prompt_clusters": len(keys),
        "iterations": iterations,
        "seed": seed,
    }


def _training_summary(
    rows: Sequence[Mapping[str, Any]], *, iterations: int, seed: int
) -> dict[str, Any]:
    def mean_field(name: str) -> Callable[[Sequence[Mapping[str, Any]]], float | None]:
        return lambda sample: _mean(row[name] for row in sample if row.get(name) is not None)

    def pooled_delta(sample: Sequence[Mapping[str, Any]], numerator: str) -> float | None:
        total_pairs = sum(int(row["pair_count"]) for row in sample)
        return sum(int(row[numerator]) for row in sample) / total_pairs if total_pairs else None

    def pooled_ecr(sample: Sequence[Mapping[str, Any]]) -> float | None:
        stale_total = sum(sum(row["stale_criterion_counts"].values()) for row in sample)
        fresh_total = sum(sum(row["fresh_criterion_counts"].values()) for row in sample)
        if not stale_total or not fresh_total:
            return None
        stale = sum(row["stale_criterion_counts"]["effective"] for row in sample) / stale_total
        fresh = sum(row["fresh_criterion_counts"]["effective"] for row in sample) / fresh_total
        return fresh - stale

    def tie_delta(sample: Sequence[Mapping[str, Any]]) -> float | None:
        total = sum(int(row["pair_count"]) for row in sample)
        if not total:
            return None
        return (
            sum(int(row["stale_tied_pairs"]) for row in sample)
            - sum(int(row["fresh_tied_pairs"]) for row in sample)
        ) / total

    def tie_resolution(sample: Sequence[Mapping[str, Any]]) -> float | None:
        tied = sum(int(row["stale_tied_pairs"]) for row in sample)
        return sum(int(row["resolved_stale_tied_pairs"]) for row in sample) / tied if tied else None

    metrics: dict[str, Callable[[Sequence[Mapping[str, Any]]], float | None]] = {
        "v_adj_zar": mean_field("v_adj_zar"),
        "delta_near_zero_advantage": mean_field("delta_near_zero_advantage"),
        "delta_tie_rate_pooled": tie_delta,
        "delta_separation_rate_pooled": tie_delta,
        "delta_effective_criterion_ratio_pooled": pooled_ecr,
        "delta_top_median_margin": mean_field("delta_top_median_margin"),
        "incremental_tie_resolution_weighted": tie_resolution,
        "kendall_tau_b": mean_field("kendall_tau_b"),
    }
    stale_criteria = Counter()
    fresh_criteria = Counter()
    for row in rows:
        stale_criteria.update(row["stale_criterion_counts"])
        fresh_criteria.update(row["fresh_criterion_counts"])
    pair_count = sum(int(row["pair_count"]) for row in rows)
    stale_ties = sum(int(row["stale_tied_pairs"]) for row in rows)
    fresh_ties = sum(int(row["fresh_tied_pairs"]) for row in rows)

    def criterion_ratios(counts: Counter[str]) -> dict[str, float | None]:
        total = sum(counts.values())
        return {
            "effective": counts["effective"] / total if total else None,
            "saturated": counts["saturated"] / total if total else None,
            "dead": counts["dead"] / total if total else None,
        }

    return {
        "group_count": len(rows),
        "prompt_count": len({row["prompt_id"] for row in rows}),
        "fresh_zar": _mean(row["fresh"]["exact_zero_advantage"] for row in rows),
        "stale_zar": _mean(row["stale"]["exact_zero_advantage"] for row in rows),
        "fresh_near_zar": _mean(row["fresh"]["near_zero_advantage"] for row in rows),
        "stale_near_zar": _mean(row["stale"]["near_zero_advantage"] for row in rows),
        "fresh_pairwise_tie_rate_pooled": fresh_ties / pair_count,
        "stale_pairwise_tie_rate_pooled": stale_ties / pair_count,
        "fresh_pairwise_separation_rate_pooled": 1 - fresh_ties / pair_count,
        "stale_pairwise_separation_rate_pooled": 1 - stale_ties / pair_count,
        "fresh_criterion_counts": dict(fresh_criteria),
        "stale_criterion_counts": dict(stale_criteria),
        "fresh_criterion_ratios": criterion_ratios(fresh_criteria),
        "stale_criterion_ratios": criterion_ratios(stale_criteria),
        "fresh_top_median_margin": _mean(row["fresh"]["top_median_margin"] for row in rows),
        "stale_top_median_margin": _mean(row["stale"]["top_median_margin"] for row in rows),
        "metrics": {name: statistic(rows) for name, statistic in metrics.items()},
        "paired_prompt_cluster_bootstrap_95ci": {
            name: _cluster_bootstrap(
                rows,
                statistic,
                iterations=iterations,
                seed=_metric_seed(seed, name, *(sorted({row["global_step"] for row in rows}))),
            )
            for name, statistic in metrics.items()
        },
    }


def _training_correlations(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    predictors = {
        "evaluator_age_steps": lambda row: row.get("evaluator_age_steps"),
        "cumulative_prompts_since_evaluator": lambda row: row["clock"].get(
            "cumulative_prompts_since_evaluator"
        ),
        "cumulative_completions_since_evaluator": lambda row: row["clock"].get(
            "cumulative_completions_since_evaluator"
        ),
        "cumulative_response_tokens_since_evaluator": lambda row: row["clock"].get(
            "cumulative_response_tokens_since_evaluator"
        ),
        "stale_zar": lambda row: row["stale"]["exact_zero_advantage"],
        "stale_tie_rate": lambda row: row["stale"]["pairwise_tie_rate"],
        "stale_effective_criterion_ratio": lambda row: row["stale"]["effective_criterion_ratio"],
        "stale_saturation_ratio": lambda row: row["stale"]["saturated_criterion_ratio"],
        "stale_reward_std": lambda row: row["stale"]["reward_std"],
        "stale_top_median_margin": lambda row: row["stale"]["top_median_margin"],
    }
    outcomes = (
        "v_adj_zar",
        "delta_tie_rate",
        "delta_separation_rate",
        "delta_effective_criterion_ratio",
        "delta_top_median_margin",
    )
    output = []
    for predictor, getter in predictors.items():
        for outcome in outcomes:
            pairs = [
                (float(getter(row)), float(row[outcome]))
                for row in rows
                if getter(row) is not None and row.get(outcome) is not None
            ]
            output.append(
                {
                    "predictor": predictor,
                    "outcome": outcome,
                    "n_groups": len(pairs),
                    "pearson_r": _pearson([x for x, _ in pairs], [y for _, y in pairs]),
                    "analysis_role": "exploratory_descriptive_noncausal_training_batch",
                }
            )
    return output


def run_training_group_analysis(
    *,
    training_groups: str | Path,
    output_dir: str | Path,
    expected_groups: int,
    epsilon_z: float,
    epsilon_t: float,
    bootstrap_iterations: int = 2_000,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    source = Path(training_groups).resolve()
    files = sorted(source.rglob("*.json"))
    if not files:
        raise AuditAnalysisError(f"no training group JSON files found: {source}")
    if expected_groups <= 0 or len(files) > expected_groups:
        raise AuditAnalysisError("training group count exceeds or invalidates expected coverage")
    groups = [read_json(path) for path in files]
    rows = [
        training_comparison_row(group, epsilon_z=epsilon_z, epsilon_t=epsilon_t) for group in groups
    ]
    identities = [(row["global_step"], row["prompt_id"]) for row in rows]
    if len(identities) != len(set(identities)):
        raise AuditAnalysisError("duplicate training comparison identity")
    by_step: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_step[int(row["global_step"])].append(row)
    coverage = {
        "status": "final" if len(rows) == expected_groups else "interim",
        "observed_groups": len(rows),
        "expected_groups": expected_groups,
        "coverage_fraction": len(rows) / expected_groups,
        "observed_global_steps": sorted(by_step),
        "missing_groups": expected_groups - len(rows),
    }
    report = {
        "schema_version": 1,
        "analysis": "phase1_actual_training_batch_fresh_vs_prompt_matched_stale",
        "primary_conclusion_dataset": False,
        "dataset_role": "operational_in_sample",
        "coverage": coverage,
        "thresholds": {"epsilon_z": epsilon_z, "epsilon_t": epsilon_t},
        "overall": _training_summary(rows, iterations=bootstrap_iterations, seed=bootstrap_seed),
        "per_step": [
            {
                "global_step": step,
                **_training_summary(
                    step_rows, iterations=bootstrap_iterations, seed=bootstrap_seed
                ),
            }
            for step, step_rows in sorted(by_step.items())
        ],
        "exploratory_correlations": _training_correlations(rows),
        "guards": {
            "fixed_train_probe_used": False,
            "reuse_horizon_triangle_available": False,
            "heldout_used": False,
            "response_level_ground_truth_used": False,
            "initial_static_rubric_used_as_ground_truth": False,
            "same_current_policy_responses_fresh_stale": True,
            "same_prompt_old_rubric_required": True,
            "ranking_similarity_is_correctness": False,
        },
        "limitations": [
            "Actual training batches are operational/in-sample evidence and cannot replace Fixed Train Probe Pool B.",
            "Changing prompt composition prevents construction of the reuse-horizon triangle from these groups.",
            "Metrics measure discriminability and reward usability, not response correctness.",
        ],
        "input": {
            "directory": str(source),
            "files_sha256": {str(path): sha256_file(path) for path in files},
        },
    }
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    report_path = destination / "report.json"
    rows_path = destination / "comparison_rows.jsonl"
    summary_path = destination / "summary.md"
    write_json_atomic(report_path, report, immutable=False)
    write_jsonl_atomic(rows_path, rows, immutable=False)
    from dynamic_rubric.artifacts import write_text_atomic

    summary = (
        f"# Phase-1 actual-training-batch evaluator update analysis\n\n"
        f"- Coverage: **{coverage['status']}**, {len(rows)}/{expected_groups} groups "
        f"({coverage['coverage_fraction']:.1%})\n"
        f"- Stale ZAR − fresh ZAR: **{report['overall']['metrics']['v_adj_zar']:.4f}**\n"
        f"- Dataset role: operational/in-sample; Fixed Train Probe and reuse-horizon triangle are not included.\n"
        f"- No response-level GT or correctness claim is used.\n"
    )
    write_text_atomic(summary_path, summary, immutable=False)
    return {
        "status": coverage["status"],
        "report": str(report_path),
        "comparison_rows": str(rows_path),
        "summary": str(summary_path),
        **coverage,
    }


def _expand_files(paths: Sequence[str | Path], pattern: str) -> list[Path]:
    output: list[Path] = []
    for value in paths:
        path = Path(value)
        if path.is_dir():
            output.extend(sorted(path.rglob(pattern)))
        elif path.is_file():
            output.append(path)
        else:
            raise AuditAnalysisError(f"missing analysis input: {path}")
    unique = sorted(set(path.resolve() for path in output))
    if not unique:
        raise AuditAnalysisError(f"no input files matched {pattern!r}")
    return unique


def _read_distance_files(paths: Sequence[Path]) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for path in paths:
        value = read_json(path)
        if isinstance(value, list):
            rows.extend(value)
        elif isinstance(value, Mapping):
            rows.append(value)
        else:
            raise AuditAnalysisError(f"invalid policy distance JSON: {path}")
    return rows


def run_audit_analysis(
    *,
    score_paths: Sequence[str | Path],
    policy_state_path: str | Path,
    policy_distance_paths: Sequence[str | Path],
    config_path: str | Path,
    output_dir: str | Path,
    through_step: int | None = None,
    checkpoint_steps: Sequence[int] | None = None,
    bootstrap_iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    resume: bool = False,
) -> dict[str, Any]:
    config_file = Path(config_path).resolve()
    state_file = Path(policy_state_path).resolve()
    if not config_file.is_file() or not state_file.is_file():
        raise AuditAnalysisError("config and policy-state inputs must exist")
    scores = _expand_files(score_paths, "*.jsonl")
    distance_files = _expand_files(policy_distance_paths, "policy_distance_summary.json")
    config = read_json(config_file)
    analysis = config["analysis"]
    effective_through_step = (
        int(through_step)
        if through_step is not None
        else (max(checkpoint_steps) if checkpoint_steps is not None else EXPLORATORY_STEP)
    )
    if checkpoint_steps is None:
        anchors = tuple(step for step in PRIMARY_ANCHORS if step <= effective_through_step)
        exploratory_step = (
            EXPLORATORY_STEP if effective_through_step >= EXPLORATORY_STEP else None
        )
        checkpoint_selection = "legacy_preselected_anchors"
    else:
        anchors = tuple(
            int(step) for step in checkpoint_steps if int(step) <= effective_through_step
        )
        if not anchors or anchors != tuple(sorted(set(anchors))):
            raise AuditAnalysisError(
                "checkpoint steps through the requested step must be non-empty, unique, and increasing"
            )
        exploratory_step = None
        checkpoint_selection = "all_committed_saved_policy_checkpoints"
    inputs = [config_file, state_file, *scores, *distance_files]
    manifest = {
        "schema_version": 1,
        "analysis": "phase1_offline_checkpoint_audit",
        "through_step": effective_through_step,
        "anchors": list(anchors),
        "checkpoint_selection": checkpoint_selection,
        "exploratory_step": exploratory_step,
        "input_sha256": {str(path): sha256_file(path) for path in inputs},
        "bootstrap_iterations": bootstrap_iterations,
        "bootstrap_seed": bootstrap_seed,
        "cpu_only": True,
    }
    destination = Path(output_dir)
    manifest_path = destination / "manifest.json"
    result_path = destination / "result.json"
    if manifest_path.exists():
        if not resume:
            raise AuditAnalysisError("analysis output already exists; pass --resume")
        if read_json(manifest_path) != manifest:
            raise AuditAnalysisError("analysis resume manifest differs from current inputs")
        required = (
            destination / "report.json",
            destination / "prompt_comparisons.jsonl",
            destination / "preupdate_stale_state.jsonl",
            result_path,
        )
        if not all(path.is_file() for path in required):
            raise AuditAnalysisError("analysis manifest exists but output inventory is incomplete")
        result = dict(read_json(result_path))
        result["resumed"] = True
        return result

    score_rows = [row for path in scores for row in read_jsonl(path)]
    report = analyze_records(
        score_rows,
        read_jsonl(state_file),
        _read_distance_files(distance_files),
        anchors=anchors,
        exploratory_step=exploratory_step,
        epsilon_z=float(analysis["epsilon_z"]),
        epsilon_t=float(analysis["epsilon_t"]),
        practical_margin_delta_d=float(analysis["practical_margin_delta_d"]),
        bootstrap_iterations=bootstrap_iterations,
        bootstrap_seed=bootstrap_seed,
    )
    prompt_rows = report.pop("_prompt_comparisons")
    report_path = destination / "report.json"
    prompt_path = destination / "prompt_comparisons.jsonl"
    state_path = destination / "preupdate_stale_state.jsonl"
    write_json_atomic(report_path, report)
    write_jsonl_atomic(prompt_path, prompt_rows)
    write_jsonl_atomic(state_path, report["preupdate_stale_state"])
    result = {
        "schema_version": 1,
        "status": "completed",
        "resumed": False,
        "report": {"path": str(report_path.resolve()), "sha256": sha256_file(report_path)},
        "prompt_comparisons": {
            "path": str(prompt_path.resolve()),
            "sha256": sha256_file(prompt_path),
        },
        "preupdate_stale_state": {
            "path": str(state_path.resolve()),
            "sha256": sha256_file(state_path),
        },
    }
    write_json_atomic(result_path, result)
    write_json_atomic(manifest_path, manifest)
    return result


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-groups", type=Path)
    parser.add_argument("--expected-groups", type=int, default=1692)
    parser.add_argument("--epsilon-z", type=float, default=0.01)
    parser.add_argument("--epsilon-t", type=float, default=0.01)
    parser.add_argument("--scores", action="append", type=Path)
    parser.add_argument("--policy-state", type=Path)
    parser.add_argument("--policy-distance", action="append", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--through-step",
        type=int,
        help=(
            "optional explicit snapshot cutoff; defaults to 34 for the legacy anchor path and "
            "to the latest discovered checkpoint for --all-saved-checkpoints-from"
        ),
    )
    parser.add_argument(
        "--all-saved-checkpoints-from",
        type=Path,
        help="discover every committed model checkpoint from this training run",
    )
    parser.add_argument("--bootstrap-iterations", type=int)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.training_groups is not None:
        if any(
            (
                args.scores,
                args.policy_state,
                args.policy_distance,
                args.config,
                args.all_saved_checkpoints_from,
            )
        ):
            parser.error("--training-groups cannot be mixed with fixed-probe inputs")
        result = run_training_group_analysis(
            training_groups=args.training_groups,
            output_dir=args.output,
            expected_groups=args.expected_groups,
            epsilon_z=args.epsilon_z,
            epsilon_t=args.epsilon_t,
            bootstrap_iterations=args.bootstrap_iterations or 2_000,
            bootstrap_seed=args.bootstrap_seed,
        )
    else:
        missing = [
            name
            for name, value in (
                ("--scores", args.scores),
                ("--policy-state", args.policy_state),
                ("--policy-distance", args.policy_distance),
                ("--config", args.config),
            )
            if not value
        ]
        if missing:
            parser.error("fixed-probe analysis requires " + ", ".join(missing))
        checkpoint_steps = None
        if args.all_saved_checkpoints_from is not None:
            inventory = discover_committed_policy_checkpoints(args.all_saved_checkpoints_from)
            checkpoint_steps = inventory["steps"]
        through_step = (
            args.through_step
            if args.through_step is not None
            else (max(checkpoint_steps) if checkpoint_steps is not None else EXPLORATORY_STEP)
        )
        result = run_audit_analysis(
            score_paths=args.scores,
            policy_state_path=args.policy_state,
            policy_distance_paths=args.policy_distance,
            config_path=args.config,
            output_dir=args.output,
            through_step=through_step,
            checkpoint_steps=checkpoint_steps,
            bootstrap_iterations=args.bootstrap_iterations or DEFAULT_BOOTSTRAP_ITERATIONS,
            bootstrap_seed=args.bootstrap_seed,
            resume=args.resume,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
