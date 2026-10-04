#!/usr/bin/env python3
"""Reconstruct GRPO advantages and relate reward diagnostics to policy performance.

This analysis is intentionally offline. It reads immutable committed training
artifacts, the completed fixed-probe matrix, and checkpoint policy evaluations.
It does not call a model, grade responses, or update policy parameters.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from fractions import Fraction
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/medicine/online_rubrics/seed-11/phase1-online-rubrics-medicine-full-20260905-seed11-final"
ONLINE_STEPS = RUN / "verl-run/online_steps"
CORE = ROOT / "results/rq2_online_medicine_full253_20260910/training_core_reanalysis_20260910"
PROBE_PROMPTS = ROOT / "results/rq2_online_medicine_full253_20260910/core_metric_reanalysis_20260910/prompt_metrics.csv"
PERFORMANCE = ROOT / "outputs/policy_eval/medicine_checkpoint_trajectory_hb500_20260912/full-f0cdda051c001185/summary/checkpoint_performance_trajectory.csv"
OUTPUT = ROOT / "results/mad_ptr_zar_grpo_advantage_20260917"

EPSILON = 1e-6
PTR_THRESHOLD = Fraction(1, 100)
EXPECTED_STEPS = list(range(1, 49))
PERFORMANCE_DATASETS = ("rar_medicine_test", "healthbench")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def safe_corr(x: pd.Series, y: pd.Series, method: str) -> tuple[float, float]:
    frame = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(frame) < 3 or frame["x"].nunique() < 2 or frame["y"].nunique() < 2:
        return math.nan, math.nan
    result = spearmanr(frame["x"], frame["y"]) if method == "spearman" else pearsonr(frame["x"], frame["y"])
    return float(result.statistic), float(result.pvalue)


def loo_sign_stability(x: pd.Series, y: pd.Series, full_r: float) -> tuple[int, int, float, float]:
    frame = pd.DataFrame({"x": x, "y": y}).dropna().reset_index(drop=True)
    estimates: list[float] = []
    for omitted in range(len(frame)):
        sample = frame.drop(index=omitted)
        r, _ = safe_corr(sample["x"], sample["y"], "spearman")
        if not math.isnan(r):
            estimates.append(r)
    if not estimates or math.isnan(full_r) or full_r == 0:
        return len(estimates), 0, math.nan, math.nan
    same = sum(np.sign(value) == np.sign(full_r) for value in estimates)
    return len(estimates), same, float(min(estimates)), float(max(estimates))


def benjamini_hochberg(values: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=values.index, dtype=float)
    valid = values.dropna().sort_values()
    if valid.empty:
        return result
    m = len(valid)
    adjusted = np.empty(m, dtype=float)
    running = 1.0
    for reverse_index in range(m - 1, -1, -1):
        rank = reverse_index + 1
        running = min(running, float(valid.iloc[reverse_index]) * m / rank)
        adjusted[reverse_index] = running
    result.loc[valid.index] = adjusted
    return result


def reconstruct_training_advantages() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    response_rows: list[dict] = []
    group_rows: list[dict] = []
    source_hashes: list[dict] = []

    for step in EXPECTED_STEPS:
        step_dir = ONLINE_STEPS / f"step-{step:06d}"
        reward_path = step_dir / "rewards.jsonl"
        batch_path = step_dir / "batch.json"
        union_path = step_dir / "rubric_unions.jsonl"
        commit_path = step_dir / "commit.json"
        for path in (reward_path, batch_path, union_path, commit_path):
            if not path.is_file():
                raise RuntimeError(f"missing committed artifact: {path}")
            source_hashes.append({"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "bytes": path.stat().st_size})

        batch = json.loads(batch_path.read_text(encoding="utf-8"))
        commit = json.loads(commit_path.read_text(encoding="utf-8"))
        if commit["state"] != "committed" or int(commit["optimizer_update_index"]) != step:
            raise RuntimeError(f"invalid commit at update {step}")
        policy_version = int(batch["current_policy"]["policy_version"])
        if policy_version != step - 1:
            raise RuntimeError(f"policy/update mismatch at update {step}")

        grouped: dict[str, list[dict]] = defaultdict(list)
        for row in read_jsonl(reward_path):
            grouped[row["prompt_occurrence_id"]].append(row)

        for prompt_id, rows in grouped.items():
            rows.sort(key=lambda row: int(row["rollout_index"]))
            if [int(row["rollout_index"]) for row in rows] != list(range(16)):
                raise RuntimeError(f"invalid rollout indices at update {step}, prompt {prompt_id}")

            exact_rewards = [Fraction(int(row["numerator"]), int(row["denominator"])) for row in rows]
            rewards = np.asarray([float(value) for value in exact_rewards], dtype=np.float64)
            exact_mean = sum(exact_rewards, Fraction()) / len(exact_rewards)
            reward_mean = float(exact_mean)
            reward_std = float(np.std(rewards, ddof=1))
            advantages = (rewards - reward_mean) / (reward_std + EPSILON)

            exact_pairs = list(combinations(exact_rewards, 2))
            exact_tie_rate = sum(left == right for left, right in exact_pairs) / len(exact_pairs)
            ptr_001 = sum(abs(left - right) <= PTR_THRESHOLD for left, right in exact_pairs) / len(exact_pairs)
            reward_mad = float(sum(abs(value - exact_mean) for value in exact_rewards) / len(exact_rewards))
            exact_zar = int(all(value == exact_rewards[0] for value in exact_rewards))

            criterion_ids = [criterion_id for criterion_id, _ in rows[0]["grades"]]
            grade_maps = [{criterion_id: int(grade) for criterion_id, grade in row["grades"]} for row in rows]
            if any(list(grade_map) != criterion_ids for grade_map in grade_maps):
                raise RuntimeError(f"criterion order mismatch at update {step}, prompt {prompt_id}")
            effective_count = sum(0 < sum(grade_map[key] for grade_map in grade_maps) < 16 for key in criterion_ids)
            ecr = effective_count / len(criterion_ids)

            exact_centered = [value - exact_mean for value in exact_rewards]
            positive = sum(value > 0 for value in exact_centered)
            negative = sum(value < 0 for value in exact_centered)
            zero = sum(value == 0 for value in exact_centered)
            effective_response_rate = 1.0 - zero / 16

            if exact_zar:
                if not np.allclose(advantages, 0.0, atol=1e-12):
                    raise RuntimeError("ZAR group produced non-zero advantage")
            elif positive == 0 or negative == 0:
                raise RuntimeError("non-ZAR group lacks both positive and negative advantages")
            if abs(float(np.mean(advantages))) > 1e-10:
                raise RuntimeError("group advantage mean is not zero")

            group_rows.append(
                {
                    "optimizer_update": step,
                    "response_policy_version": policy_version,
                    "prompt_occurrence_id": prompt_id,
                    "response_count": 16,
                    "reward_mean": reward_mean,
                    "reward_mad": reward_mad,
                    "reward_std_sample": reward_std,
                    "exact_pairwise_tie_rate": exact_tie_rate,
                    "epsilon_01_pairwise_tie_rate": ptr_001,
                    "exact_zar": exact_zar,
                    "criterion_count": len(criterion_ids),
                    "effective_criterion_count": effective_count,
                    "effective_criterion_ratio": ecr,
                    "grpo_advantage_mad": float(np.mean(np.abs(advantages))),
                    "grpo_advantage_abs_max": float(np.max(np.abs(advantages))),
                    "grpo_advantage_range": float(np.max(advantages) - np.min(advantages)),
                    "grpo_advantage_std_sample": float(np.std(advantages, ddof=1)),
                    "positive_advantage_count": positive,
                    "negative_advantage_count": negative,
                    "zero_advantage_response_count": zero,
                    "effective_response_rate": effective_response_rate,
                    "unique_reward_count": len(set(exact_rewards)),
                }
            )

            for row, reward, advantage, centered in zip(rows, rewards, advantages, exact_centered):
                response_rows.append(
                    {
                        "optimizer_update": step,
                        "response_policy_version": policy_version,
                        "prompt_occurrence_id": prompt_id,
                        "response_id": row["response_id"],
                        "rollout_index": int(row["rollout_index"]),
                        "reward": reward,
                        "grpo_advantage": float(advantage),
                        "advantage_sign": "positive" if centered > 0 else "negative" if centered < 0 else "zero",
                    }
                )

    responses = pd.DataFrame(response_rows)
    groups = pd.DataFrame(group_rows)
    if len(responses) != 72_000 or len(groups) != 4_500:
        raise RuntimeError(f"coverage mismatch: responses={len(responses)}, groups={len(groups)}")

    step_metrics = [
        "reward_mean",
        "reward_mad",
        "reward_std_sample",
        "exact_pairwise_tie_rate",
        "epsilon_01_pairwise_tie_rate",
        "exact_zar",
        "effective_criterion_ratio",
        "grpo_advantage_mad",
        "grpo_advantage_abs_max",
        "grpo_advantage_range",
        "grpo_advantage_std_sample",
        "effective_response_rate",
        "unique_reward_count",
    ]
    steps = groups.groupby(["optimizer_update", "response_policy_version"], as_index=False).agg(
        prompt_visits=("prompt_occurrence_id", "size"),
        **{metric: (metric, "mean") for metric in step_metrics},
    )

    canonical = pd.read_csv(CORE / "training_by_prompt.csv")
    comparison = groups.merge(
        canonical,
        on=["optimizer_update", "response_policy_version", "prompt_occurrence_id"],
        validate="one_to_one",
        suffixes=("_reconstructed", "_canonical"),
    )
    validation = {
        "response_rows": len(responses),
        "prompt_groups": len(groups),
        "optimizer_updates": len(steps),
        "max_abs_mad_error": float(np.max(np.abs(comparison["reward_mad"] - comparison["normalized_group_mad"]))),
        "max_abs_exact_tie_error": float(
            np.max(
                np.abs(
                    comparison["exact_pairwise_tie_rate_reconstructed"]
                    - comparison["exact_pairwise_tie_rate_canonical"]
                )
            )
        ),
        "exact_zar_mismatches": int(
            np.sum(comparison["exact_zar_reconstructed"] != comparison["exact_zar_canonical"])
        ),
        "max_abs_group_advantage_mean": float(
            responses.groupby(["optimizer_update", "prompt_occurrence_id"])["grpo_advantage"].mean().abs().max()
        ),
        "source_hashes": source_hashes,
    }
    if validation["max_abs_mad_error"] > 1e-12 or validation["max_abs_exact_tie_error"] > 1e-12:
        raise RuntimeError(f"canonical metric validation failed: {validation}")
    if validation["exact_zar_mismatches"]:
        raise RuntimeError(f"canonical ZAR validation failed: {validation}")
    return responses, groups, steps, validation


def reconstruct_static_r0_on_same_rollouts(
    dynamic_responses: pd.DataFrame,
    dynamic_groups: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Replay each OnlineRubrics rollout with only its stored offline-R0 criteria.

    The online grader receipt contains a binary grade for every offline and online
    criterion in the rubric union.  Restricting the exact weighted-rational score
    to ``offline_criteria`` therefore gives a same-response static-R0
    counterfactual without another judge call.  It does not emulate a separate
    R0-only grader prompt; that distinction is recorded in the summary guards.
    """

    response_rows: list[dict] = []
    group_rows: list[dict] = []
    complete_offline_inventories = 0
    complete_union_inventories = 0

    def pair_sign(left: Fraction, right: Fraction) -> int:
        return int(left > right) - int(left < right)

    for step in EXPECTED_STEPS:
        step_dir = ONLINE_STEPS / f"step-{step:06d}"
        reward_path = step_dir / "rewards.jsonl"
        union_path = step_dir / "rubric_unions.jsonl"
        unions = {row["prompt_occurrence_id"]: row for row in read_jsonl(union_path)}
        grouped: dict[str, list[dict]] = defaultdict(list)
        for row in read_jsonl(reward_path):
            grouped[row["prompt_occurrence_id"]].append(row)
        if set(unions) != set(grouped):
            raise RuntimeError(f"union/reward prompt mismatch at update {step}")

        for prompt_id, rows in grouped.items():
            rows.sort(key=lambda row: int(row["rollout_index"]))
            union = unions[prompt_id]
            if union["content_hash"] != rows[0]["rubric_hash"]:
                raise RuntimeError(f"rubric hash mismatch at update {step}, prompt {prompt_id}")

            offline = union["offline_criteria"]
            online = union["online_criteria"]
            offline_ids = [item["criterion_id"] for item in offline]
            union_ids = offline_ids + [item["criterion_id"] for item in online]
            weights = {
                item["criterion_id"]: Fraction(str(item["weight"]))
                for item in offline
            }
            denominator = sum((weight for weight in weights.values() if weight > 0), Fraction())
            if not offline_ids or denominator <= 0:
                raise RuntimeError(f"invalid static R0 rubric at update {step}, prompt {prompt_id}")
            if any(item.get("source") != "offline_r0" for item in offline):
                raise RuntimeError(f"non-R0 criterion in offline inventory at update {step}")

            exact_dynamic_rewards: list[Fraction] = []
            exact_static_rewards: list[Fraction] = []
            grade_maps: list[dict[str, int]] = []
            for row in rows:
                grade_map = {criterion_id: int(grade) for criterion_id, grade in row["grades"]}
                if all(criterion_id in grade_map for criterion_id in offline_ids):
                    complete_offline_inventories += 1
                else:
                    raise RuntimeError(f"missing static R0 grade at update {step}, prompt {prompt_id}")
                if set(grade_map) == set(union_ids):
                    complete_union_inventories += 1
                else:
                    raise RuntimeError(f"grade/union inventory mismatch at update {step}, prompt {prompt_id}")
                numerator = sum(
                    (weights[criterion_id] * grade_map[criterion_id] for criterion_id in offline_ids),
                    Fraction(),
                )
                static_reward = numerator / denominator
                dynamic_reward = Fraction(int(row["numerator"]), int(row["denominator"]))
                exact_static_rewards.append(static_reward)
                exact_dynamic_rewards.append(dynamic_reward)
                grade_maps.append(grade_map)

            static_rewards = np.asarray([float(value) for value in exact_static_rewards], dtype=np.float64)
            static_mean_exact = sum(exact_static_rewards, Fraction()) / len(exact_static_rewards)
            static_mean = float(static_mean_exact)
            static_std = float(np.std(static_rewards, ddof=1))
            static_advantages = (static_rewards - static_mean) / (static_std + EPSILON)
            static_pairs = list(combinations(exact_static_rewards, 2))
            static_exact_tie = sum(left == right for left, right in static_pairs) / len(static_pairs)
            static_ptr = sum(abs(left - right) <= PTR_THRESHOLD for left, right in static_pairs) / len(static_pairs)
            static_mad = float(
                sum(abs(value - static_mean_exact) for value in exact_static_rewards)
                / len(exact_static_rewards)
            )
            static_zar = int(all(value == exact_static_rewards[0] for value in exact_static_rewards))
            static_effective_count = sum(
                0 < sum(grade_map[key] for grade_map in grade_maps) < 16
                for key in offline_ids
            )
            static_centered = [value - static_mean_exact for value in exact_static_rewards]
            static_zero = sum(value == 0 for value in static_centered)

            dynamic_mean_exact = sum(exact_dynamic_rewards, Fraction()) / len(exact_dynamic_rewards)
            pair_states = [
                (
                    pair_sign(exact_dynamic_rewards[left], exact_dynamic_rewards[right]),
                    pair_sign(exact_static_rewards[left], exact_static_rewards[right]),
                )
                for left, right in combinations(range(16), 2)
            ]
            pair_count = len(pair_states)
            group_rows.append(
                {
                    "optimizer_update": step,
                    "response_policy_version": step - 1,
                    "prompt_occurrence_id": prompt_id,
                    "response_count": 16,
                    "static_reward_mean": static_mean,
                    "static_reward_mad": static_mad,
                    "static_reward_std_sample": static_std,
                    "static_exact_pairwise_tie_rate": static_exact_tie,
                    "static_epsilon_01_pairwise_tie_rate": static_ptr,
                    "static_exact_zar": static_zar,
                    "static_criterion_count": len(offline_ids),
                    "static_effective_criterion_count": static_effective_count,
                    "static_effective_criterion_ratio": static_effective_count / len(offline_ids),
                    "static_grpo_advantage_mad": float(np.mean(np.abs(static_advantages))),
                    "static_grpo_advantage_abs_max": float(np.max(np.abs(static_advantages))),
                    "static_grpo_advantage_std_sample": float(np.std(static_advantages, ddof=1)),
                    "static_effective_response_rate": 1.0 - static_zero / 16,
                    "static_unique_reward_count": len(set(exact_static_rewards)),
                    "pair_order_agreement_rate": sum(left == right for left, right in pair_states) / pair_count,
                    "strict_pair_reversal_rate": sum(left * right < 0 for left, right in pair_states) / pair_count,
                    "dynamic_non_tie_static_tie_rate": sum(left != 0 and right == 0 for left, right in pair_states)
                    / pair_count,
                    "dynamic_tie_static_non_tie_rate": sum(left == 0 and right != 0 for left, right in pair_states)
                    / pair_count,
                }
            )

            for row, dynamic_reward, static_reward, static_advantage, centered in zip(
                rows,
                exact_dynamic_rewards,
                exact_static_rewards,
                static_advantages,
                static_centered,
            ):
                response_rows.append(
                    {
                        "optimizer_update": step,
                        "response_policy_version": step - 1,
                        "prompt_occurrence_id": prompt_id,
                        "response_id": row["response_id"],
                        "rollout_index": int(row["rollout_index"]),
                        "static_reward_numerator": static_reward.numerator,
                        "static_reward_denominator": static_reward.denominator,
                        "static_reward": float(static_reward),
                        "static_grpo_advantage": float(static_advantage),
                        "static_advantage_sign": (
                            "positive" if centered > 0 else "negative" if centered < 0 else "zero"
                        ),
                        "reward_exact_equal": int(dynamic_reward == static_reward),
                    }
                )

    static_responses = pd.DataFrame(response_rows)
    merge_keys = [
        "optimizer_update",
        "response_policy_version",
        "prompt_occurrence_id",
        "response_id",
        "rollout_index",
    ]
    responses = dynamic_responses.rename(
        columns={
            "reward": "dynamic_reward",
            "grpo_advantage": "dynamic_grpo_advantage",
            "advantage_sign": "dynamic_advantage_sign",
        }
    ).merge(static_responses, on=merge_keys, validate="one_to_one")
    responses["advantage_sign_agreement"] = (
        responses["dynamic_advantage_sign"] == responses["static_advantage_sign"]
    ).astype(int)
    responses["static_minus_dynamic_reward"] = responses["static_reward"] - responses["dynamic_reward"]
    responses["static_minus_dynamic_advantage"] = (
        responses["static_grpo_advantage"] - responses["dynamic_grpo_advantage"]
    )

    dynamic_columns = {
        "reward_mean": "dynamic_reward_mean",
        "reward_mad": "dynamic_reward_mad",
        "reward_std_sample": "dynamic_reward_std_sample",
        "exact_pairwise_tie_rate": "dynamic_exact_pairwise_tie_rate",
        "epsilon_01_pairwise_tie_rate": "dynamic_epsilon_01_pairwise_tie_rate",
        "exact_zar": "dynamic_exact_zar",
        "criterion_count": "dynamic_criterion_count",
        "effective_criterion_count": "dynamic_effective_criterion_count",
        "effective_criterion_ratio": "dynamic_effective_criterion_ratio",
        "grpo_advantage_mad": "dynamic_grpo_advantage_mad",
        "grpo_advantage_abs_max": "dynamic_grpo_advantage_abs_max",
        "grpo_advantage_std_sample": "dynamic_grpo_advantage_std_sample",
        "effective_response_rate": "dynamic_effective_response_rate",
        "unique_reward_count": "dynamic_unique_reward_count",
    }
    group_keys = ["optimizer_update", "response_policy_version", "prompt_occurrence_id"]
    groups = pd.DataFrame(group_rows).merge(
        dynamic_groups[group_keys + list(dynamic_columns)].rename(columns=dynamic_columns),
        on=group_keys,
        validate="one_to_one",
    )
    metric_columns = [
        column
        for column in groups.columns
        if column.startswith("static_")
        or column.startswith("dynamic_")
        or column.endswith("_rate")
    ]
    steps = groups.groupby(["optimizer_update", "response_policy_version"], as_index=False).agg(
        prompt_visits=("prompt_occurrence_id", "size"),
        **{column: (column, "mean") for column in metric_columns},
    )

    trajectory_rows: list[dict] = []
    for metric, dynamic_column, static_column in [
        ("MAD", "dynamic_reward_mad", "static_reward_mad"),
        ("PTR@0.01", "dynamic_epsilon_01_pairwise_tie_rate", "static_epsilon_01_pairwise_tie_rate"),
        ("ZAR", "dynamic_exact_zar", "static_exact_zar"),
        ("ECR", "dynamic_effective_criterion_ratio", "static_effective_criterion_ratio"),
        ("Advantage MAD", "dynamic_grpo_advantage_mad", "static_grpo_advantage_mad"),
        ("Max |advantage|", "dynamic_grpo_advantage_abs_max", "static_grpo_advantage_abs_max"),
        ("Effective response rate", "dynamic_effective_response_rate", "static_effective_response_rate"),
    ]:
        spearman_r, spearman_p = safe_corr(steps[dynamic_column], steps[static_column], "spearman")
        pearson_r, pearson_p = safe_corr(steps[dynamic_column], steps[static_column], "pearson")
        trajectory_rows.append(
            {
                "metric": metric,
                "dynamic_column": dynamic_column,
                "static_column": static_column,
                "n_updates": len(steps),
                "spearman_r": spearman_r,
                "spearman_p": spearman_p,
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
            }
        )
    trajectories = pd.DataFrame(trajectory_rows)

    validation = {
        "response_rows": len(responses),
        "prompt_groups": len(groups),
        "optimizer_updates": len(steps),
        "complete_offline_grade_inventories": complete_offline_inventories,
        "complete_union_grade_inventories": complete_union_inventories,
        "max_abs_dynamic_reward_alignment_error": float(
            np.max(np.abs(responses["dynamic_reward"] - dynamic_responses["reward"]))
        ),
        "max_abs_static_group_advantage_mean": float(
            responses.groupby(["optimizer_update", "prompt_occurrence_id"])["static_grpo_advantage"]
            .mean()
            .abs()
            .max()
        ),
    }
    if len(responses) != 72_000 or len(groups) != 4_500 or len(steps) != 48:
        raise RuntimeError(f"static replay coverage mismatch: {validation}")
    if validation["max_abs_dynamic_reward_alignment_error"] > 1e-12:
        raise RuntimeError(f"dynamic response alignment failed: {validation}")
    return responses, groups, steps, trajectories, validation


def metric_advantage_relationships(groups: pd.DataFrame, steps: pd.DataFrame) -> pd.DataFrame:
    predictors = {
        "MAD": "reward_mad",
        "PTR@0.01": "epsilon_01_pairwise_tie_rate",
        "Exact tie rate": "exact_pairwise_tie_rate",
        "ZAR": "exact_zar",
        "ECR": "effective_criterion_ratio",
        "Reward std": "reward_std_sample",
    }
    outcomes = {
        "Advantage MAD": "grpo_advantage_mad",
        "Max |advantage|": "grpo_advantage_abs_max",
        "Effective response rate": "effective_response_rate",
    }
    rows: list[dict] = []
    for level, frame in (("prompt_group", groups), ("optimizer_update", steps)):
        for predictor_label, predictor in predictors.items():
            for outcome_label, outcome in outcomes.items():
                spearman_r, spearman_p = safe_corr(frame[predictor], frame[outcome], "spearman")
                pearson_r, pearson_p = safe_corr(frame[predictor], frame[outcome], "pearson")
                rows.append(
                    {
                        "level": level,
                        "predictor": predictor_label,
                        "outcome": outcome_label,
                        "n": len(frame),
                        "spearman_r": spearman_r,
                        "spearman_p": spearman_p,
                        "pearson_r": pearson_r,
                        "pearson_p": pearson_p,
                    }
                )
    return pd.DataFrame(rows)


def fixed_probe_states() -> pd.DataFrame:
    prompt_metrics = pd.read_csv(PROBE_PROMPTS)
    diagonal = prompt_metrics[prompt_metrics["policy"] == prompt_metrics["evaluator"]].copy()
    states = (
        diagonal.groupby("policy", as_index=False)
        .agg(
            prompt_count=("prompt_id", "size"),
            probe_mad=("group_mad", "mean"),
            probe_ptr_001=("epsilon_01_tie", "mean"),
            probe_zar=("exact_zar", "mean"),
            effective_count=("effective_count", "sum"),
            criterion_count=("criterion_count", "sum"),
        )
        .rename(columns={"policy": "global_step"})
    )
    states["probe_ecr_pooled"] = states["effective_count"] / states["criterion_count"]
    if len(states) != 22 or set(states["prompt_count"]) != {100}:
        raise RuntimeError("fixed-probe diagonal coverage mismatch")
    return states


def build_performance_intervals(groups: pd.DataFrame, probe: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    performance = pd.read_csv(PERFORMANCE)
    performance = performance[
        (performance["method"] == "online") & performance["dataset"].isin(PERFORMANCE_DATASETS)
    ].copy()
    checkpoints = sorted(performance[performance["dataset"] == PERFORMANCE_DATASETS[0]]["global_step"].tolist())
    if checkpoints != sorted(performance[performance["dataset"] == PERFORMANCE_DATASETS[1]]["global_step"].tolist()):
        raise RuntimeError("performance checkpoint sets differ by dataset")
    if checkpoints != probe["global_step"].tolist():
        raise RuntimeError("fixed-probe and policy-eval checkpoint sets differ")

    group_metrics = [
        "reward_mad",
        "epsilon_01_pairwise_tie_rate",
        "exact_zar",
        "effective_criterion_ratio",
        "grpo_advantage_mad",
        "grpo_advantage_abs_max",
        "effective_response_rate",
        "reward_std_sample",
    ]
    interval_rows: list[dict] = []
    state_rows: list[dict] = []
    probe_by_step = probe.set_index("global_step")

    for dataset in PERFORMANCE_DATASETS:
        trajectory = performance[performance["dataset"] == dataset].sort_values("global_step")
        for row in trajectory.itertuples(index=False):
            probe_row = probe_by_step.loc[int(row.global_step)]
            state_rows.append(
                {
                    "dataset": dataset,
                    "global_step": int(row.global_step),
                    "performance": float(row.final_mean),
                    "base_performance": float(row.base_mean),
                    "delta_from_base": float(row.delta_final_minus_base),
                    "probe_mad": float(probe_row.probe_mad),
                    "probe_ptr_001": float(probe_row.probe_ptr_001),
                    "probe_zar": float(probe_row.probe_zar),
                    "probe_ecr_pooled": float(probe_row.probe_ecr_pooled),
                }
            )

        records = list(trajectory.itertuples(index=False))
        for previous, current in zip(records, records[1:]):
            start = int(previous.global_step)
            end = int(current.global_step)
            interval_groups = groups[
                (groups["optimizer_update"] >= start + 1) & (groups["optimizer_update"] <= end)
            ]
            if interval_groups.empty:
                raise RuntimeError(f"empty training interval {start}->{end}")
            start_probe = probe_by_step.loc[start]
            end_probe = probe_by_step.loc[end]
            result = {
                "dataset": dataset,
                "start_step": start,
                "end_step": end,
                "updates_in_interval": end - start,
                "train_prompt_groups": len(interval_groups),
                "performance_start": float(previous.final_mean),
                "performance_end": float(current.final_mean),
                "performance_delta": float(current.final_mean - previous.final_mean),
                "performance_delta_per_update": float(
                    (current.final_mean - previous.final_mean) / (end - start)
                ),
            }
            result.update({f"train_{metric}": float(interval_groups[metric].mean()) for metric in group_metrics})
            for metric in ("probe_mad", "probe_ptr_001", "probe_zar", "probe_ecr_pooled"):
                result[f"start_{metric}"] = float(start_probe[metric])
                result[f"end_{metric}"] = float(end_probe[metric])
                result[f"delta_{metric}"] = float(end_probe[metric] - start_probe[metric])
            interval_rows.append(result)

    intervals = pd.DataFrame(interval_rows)
    states = pd.DataFrame(state_rows)
    if len(intervals) != 42 or len(states) != 44:
        raise RuntimeError("performance join coverage mismatch")
    return intervals, states


def performance_correlations(intervals: pd.DataFrame, states: pd.DataFrame) -> pd.DataFrame:
    families = {
        "actual_training_interval": {
            "MAD": "train_reward_mad",
            "PTR@0.01": "train_epsilon_01_pairwise_tie_rate",
            "ZAR": "train_exact_zar",
            "ECR": "train_effective_criterion_ratio",
            "Advantage MAD": "train_grpo_advantage_mad",
            "Max |advantage|": "train_grpo_advantage_abs_max",
            "Effective response rate": "train_effective_response_rate",
        },
        "fixed_probe_start_predictive": {
            "MAD": "start_probe_mad",
            "PTR@0.01": "start_probe_ptr_001",
            "ZAR": "start_probe_zar",
            "ECR": "start_probe_ecr_pooled",
        },
        "fixed_probe_change_concurrent": {
            "ΔMAD": "delta_probe_mad",
            "ΔPTR@0.01": "delta_probe_ptr_001",
            "ΔZAR": "delta_probe_zar",
            "ΔECR": "delta_probe_ecr_pooled",
        },
    }
    rows: list[dict] = []
    for dataset in PERFORMANCE_DATASETS:
        frame = intervals[intervals["dataset"] == dataset]
        for family, predictors in families.items():
            for label, column in predictors.items():
                spearman_r, spearman_p = safe_corr(
                    frame[column], frame["performance_delta_per_update"], "spearman"
                )
                pearson_r, pearson_p = safe_corr(
                    frame[column], frame["performance_delta_per_update"], "pearson"
                )
                loo_n, loo_same, loo_min, loo_max = loo_sign_stability(
                    frame[column], frame["performance_delta_per_update"], spearman_r
                )
                rows.append(
                    {
                        "analysis_family": family,
                        "dataset": dataset,
                        "predictor": label,
                        "outcome": "next_checkpoint_performance_delta_per_update",
                        "n": len(frame),
                        "spearman_r": spearman_r,
                        "spearman_p": spearman_p,
                        "pearson_r": pearson_r,
                        "pearson_p": pearson_p,
                        "loo_defined": loo_n,
                        "loo_same_sign": loo_same,
                        "loo_spearman_min": loo_min,
                        "loo_spearman_max": loo_max,
                    }
                )

        state_frame = states[states["dataset"] == dataset]
        for label, column in {
            "MAD": "probe_mad",
            "PTR@0.01": "probe_ptr_001",
            "ZAR": "probe_zar",
            "ECR": "probe_ecr_pooled",
        }.items():
            spearman_r, spearman_p = safe_corr(state_frame[column], state_frame["performance"], "spearman")
            pearson_r, pearson_p = safe_corr(state_frame[column], state_frame["performance"], "pearson")
            loo_n, loo_same, loo_min, loo_max = loo_sign_stability(
                state_frame[column], state_frame["performance"], spearman_r
            )
            rows.append(
                {
                    "analysis_family": "fixed_probe_same_checkpoint",
                    "dataset": dataset,
                    "predictor": label,
                    "outcome": "same_checkpoint_performance",
                    "n": len(state_frame),
                    "spearman_r": spearman_r,
                    "spearman_p": spearman_p,
                    "pearson_r": pearson_r,
                    "pearson_p": pearson_p,
                    "loo_defined": loo_n,
                    "loo_same_sign": loo_same,
                    "loo_spearman_min": loo_min,
                    "loo_spearman_max": loo_max,
                }
            )

    correlations = pd.DataFrame(rows)
    correlations["spearman_q_bh_within_family_dataset"] = correlations.groupby(
        ["analysis_family", "dataset"]
    )["spearman_p"].transform(benjamini_hochberg)
    return correlations


def plot_advantage_bridge(steps: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    panels = [
        ("dynamic_reward_mad", "static_reward_mad", "Reward MAD"),
        (
            "dynamic_epsilon_01_pairwise_tie_rate",
            "static_epsilon_01_pairwise_tie_rate",
            "PTR @ 0.01",
        ),
        ("dynamic_exact_zar", "static_exact_zar", "Exact ZAR"),
        (
            "dynamic_grpo_advantage_mad",
            "static_grpo_advantage_mad",
            "Mean |GRPO advantage|",
        ),
    ]
    for axis, (dynamic_column, static_column, title) in zip(axes.flat, panels):
        axis.plot(
            steps["optimizer_update"],
            steps[dynamic_column],
            color="#3366cc",
            linewidth=1.8,
            marker="o",
            markersize=3,
            label="Online union",
        )
        axis.plot(
            steps["optimizer_update"],
            steps[static_column],
            color="#dd7711",
            linewidth=1.8,
            marker="s",
            markersize=3,
            label="Static R0",
        )
        axis.set_title(title)
        axis.grid(alpha=0.25)
    for axis in axes[1]:
        axis.set_xlabel("Optimizer update")
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Same OnlineRubrics rollouts: online union vs static R0")
    fig.tight_layout()
    fig.savefig(output, format="svg", bbox_inches="tight")
    plt.close(fig)


def plot_same_rollout_static_comparison(steps: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    panels = [
        ("dynamic_reward_mad", "static_reward_mad", "Reward MAD"),
        (
            "dynamic_epsilon_01_pairwise_tie_rate",
            "static_epsilon_01_pairwise_tie_rate",
            "PTR @ 0.01",
        ),
        ("dynamic_exact_zar", "static_exact_zar", "Exact ZAR"),
        (
            "dynamic_effective_criterion_ratio",
            "static_effective_criterion_ratio",
            "ECR",
        ),
        (
            "dynamic_grpo_advantage_mad",
            "static_grpo_advantage_mad",
            "Mean |GRPO advantage|",
        ),
        (
            "dynamic_effective_response_rate",
            "static_effective_response_rate",
            "Effective response rate",
        ),
    ]
    for axis, (dynamic_column, static_column, title) in zip(axes.flat, panels):
        axis.plot(
            steps["optimizer_update"],
            steps[dynamic_column],
            color="#3366cc",
            linewidth=1.8,
            label="Online union",
        )
        axis.plot(
            steps["optimizer_update"],
            steps[static_column],
            color="#dd7711",
            linewidth=1.8,
            label="Static R0 replay",
        )
        axis.set_title(title)
        axis.grid(alpha=0.25)
    for axis in axes[-1]:
        axis.set_xlabel("Optimizer update")
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Same OnlineRubrics rollouts scored with online union vs static R0")
    fig.tight_layout()
    fig.savefig(output, format="svg", bbox_inches="tight")
    plt.close(fig)


def plot_performance(states: pd.DataFrame, output: Path) -> None:
    labels = {"rar_medicine_test": "RaR-Medicine heldout 300", "healthbench": "HealthBench 500"}
    colors = {"rar_medicine_test": "#3366cc", "healthbench": "#dd7711"}
    fig, axis = plt.subplots(figsize=(10.5, 4.8))
    for dataset in PERFORMANCE_DATASETS:
        frame = states[states["dataset"] == dataset].sort_values("global_step")
        axis.plot(
            frame["global_step"],
            frame["performance"],
            marker="o",
            linewidth=2,
            markersize=4,
            label=labels[dataset],
            color=colors[dataset],
        )
    axis.set_xlabel("Policy checkpoint")
    axis.set_ylabel("Evaluation score")
    axis.set_title("OnlineRubrics checkpoint performance trajectory")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, format="svg", bbox_inches="tight")
    plt.close(fig)


def plot_correlation_heatmap(correlations: pd.DataFrame, output: Path) -> None:
    families = ["actual_training_interval", "fixed_probe_start_predictive"]
    family_titles = ["Actual training signal during interval", "Fixed-probe state at interval start"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), constrained_layout=True)
    for axis, family, title in zip(axes, families, family_titles):
        frame = correlations[correlations["analysis_family"] == family]
        matrix = frame.pivot(index="predictor", columns="dataset", values="spearman_r")
        matrix = matrix.reindex(columns=list(PERFORMANCE_DATASETS))
        image = axis.imshow(matrix.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
        axis.set_xticks(range(len(matrix.columns)), ["Medicine", "HealthBench"])
        axis.set_yticks(range(len(matrix.index)), matrix.index)
        axis.set_title(title)
        for row in range(len(matrix.index)):
            for col in range(len(matrix.columns)):
                value = matrix.iloc[row, col]
                axis.text(col, row, "NA" if pd.isna(value) else f"{value:+.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(image, ax=axes, shrink=0.8, label="Spearman correlation with next performance change per update")
    fig.savefig(output, format="svg", bbox_inches="tight")
    plt.close(fig)


def summarize(
    groups: pd.DataFrame,
    steps: pd.DataFrame,
    advantage_correlations: pd.DataFrame,
    same_rollout_responses: pd.DataFrame,
    same_rollout_groups: pd.DataFrame,
    same_rollout_steps: pd.DataFrame,
    same_rollout_trajectories: pd.DataFrame,
    same_rollout_validation: dict,
    intervals: pd.DataFrame,
    performance_correlations_frame: pd.DataFrame,
    validation: dict,
) -> dict:
    active = groups[groups["exact_zar"] == 0]
    endpoints = steps[steps["optimizer_update"].isin([1, 48])].to_dict(orient="records")
    top_performance = {}
    for dataset in PERFORMANCE_DATASETS:
        dataset_intervals = intervals[intervals["dataset"] == dataset]
        best = dataset_intervals.loc[dataset_intervals["performance_end"].idxmax()]
        top_performance[dataset] = {
            "best_checkpoint": int(best["end_step"]),
            "best_score": float(best["performance_end"]),
            "final_score": float(dataset_intervals.iloc[-1]["performance_end"]),
            "base_score": float(dataset_intervals.iloc[0]["performance_start"]),
        }
    interval_wide = intervals.pivot(
        index=["start_step", "end_step"],
        columns="dataset",
        values="performance_delta_per_update",
    )
    cross_dataset_spearman, cross_dataset_p = safe_corr(
        interval_wide[PERFORMANCE_DATASETS[0]],
        interval_wide[PERFORMANCE_DATASETS[1]],
        "spearman",
    )
    strongest = {}
    for (family, dataset), frame in performance_correlations_frame.groupby(["analysis_family", "dataset"]):
        ranked = frame.dropna(subset=["spearman_r"]).assign(abs_r=lambda value: value["spearman_r"].abs())
        ranked = ranked.sort_values("abs_r", ascending=False)
        strongest[f"{family}:{dataset}"] = ranked.head(3)[
            ["predictor", "spearman_r", "spearman_p", "spearman_q_bh_within_family_dataset", "loo_same_sign", "loo_defined"]
        ].to_dict(orient="records")

    same_rollout_metric_comparison = {}
    for metric, dynamic_column, static_column in [
        ("MAD", "dynamic_reward_mad", "static_reward_mad"),
        ("PTR@0.01", "dynamic_epsilon_01_pairwise_tie_rate", "static_epsilon_01_pairwise_tie_rate"),
        ("ZAR", "dynamic_exact_zar", "static_exact_zar"),
        ("ECR", "dynamic_effective_criterion_ratio", "static_effective_criterion_ratio"),
        ("Advantage MAD", "dynamic_grpo_advantage_mad", "static_grpo_advantage_mad"),
        ("Max |advantage|", "dynamic_grpo_advantage_abs_max", "static_grpo_advantage_abs_max"),
        ("Effective response rate", "dynamic_effective_response_rate", "static_effective_response_rate"),
    ]:
        delta = same_rollout_groups[static_column] - same_rollout_groups[dynamic_column]
        same_rollout_metric_comparison[metric] = {
            "dynamic_mean": float(same_rollout_groups[dynamic_column].mean()),
            "static_mean": float(same_rollout_groups[static_column].mean()),
            "paired_mean_delta_static_minus_dynamic": float(delta.mean()),
            "static_higher_group_rate": float((delta > 1e-12).mean()),
            "equal_group_rate": float((delta.abs() <= 1e-12).mean()),
            "static_lower_group_rate": float((delta < -1e-12).mean()),
        }

    sign_crosstab = pd.crosstab(
        same_rollout_responses["dynamic_advantage_sign"],
        same_rollout_responses["static_advantage_sign"],
    )
    same_rollout_active = same_rollout_groups[same_rollout_groups["static_exact_zar"] == 0]
    same_rollout_endpoints = same_rollout_steps[
        same_rollout_steps["optimizer_update"].isin([1, 48])
    ].to_dict(orient="records")
    return {
        "schema_version": 1,
        "analysis_scope": "OnlineRubrics RaR-Medicine seed11 actual training plus fixed probe and policy evaluation",
        "grpo_formula": "(reward - group_mean) / (sample_std + 1e-6)",
        "coverage": {
            "optimizer_updates": int(steps["optimizer_update"].nunique()),
            "prompt_groups": len(groups),
            "responses": int(groups["response_count"].sum()),
            "zar_groups": int(groups["exact_zar"].sum()),
            "active_groups": len(active),
            "performance_intervals_per_dataset": int(len(intervals) / len(PERFORMANCE_DATASETS)),
        },
        "advantage_properties": {
            "active_group_advantage_std_mean": float(active["grpo_advantage_std_sample"].mean()),
            "active_group_advantage_std_min": float(active["grpo_advantage_std_sample"].min()),
            "active_group_advantage_std_max": float(active["grpo_advantage_std_sample"].max()),
            "all_group_advantage_mad_mean": float(groups["grpo_advantage_mad"].mean()),
            "all_group_effective_response_rate_mean": float(groups["effective_response_rate"].mean()),
        },
        "endpoint_step_metrics": endpoints,
        "metric_advantage_correlations": advantage_correlations.to_dict(orient="records"),
        "same_rollout_static_r0": {
            "definition": (
                "Counterfactual static-R0 reward computed on the exact OnlineRubrics responses by "
                "restricting each stored union-grade receipt to offline_criteria and applying the "
                "same exact weighted-rational reward followed by GRPO group normalization."
            ),
            "coverage": {
                "optimizer_updates": int(same_rollout_steps["optimizer_update"].nunique()),
                "prompt_groups": len(same_rollout_groups),
                "responses": len(same_rollout_responses),
                "static_zar_groups": int(same_rollout_groups["static_exact_zar"].sum()),
                "dynamic_zar_groups": int(same_rollout_groups["dynamic_exact_zar"].sum()),
            },
            "metric_comparison": same_rollout_metric_comparison,
            "response_level": {
                "exact_reward_equality_rate": float(same_rollout_responses["reward_exact_equal"].mean()),
                "advantage_sign_agreement_rate": float(
                    same_rollout_responses["advantage_sign_agreement"].mean()
                ),
                "reward_spearman": safe_corr(
                    same_rollout_responses["dynamic_reward"],
                    same_rollout_responses["static_reward"],
                    "spearman",
                )[0],
                "advantage_spearman": safe_corr(
                    same_rollout_responses["dynamic_grpo_advantage"],
                    same_rollout_responses["static_grpo_advantage"],
                    "spearman",
                )[0],
                "advantage_sign_crosstab": {
                    str(row): {str(column): int(value) for column, value in values.items()}
                    for row, values in sign_crosstab.to_dict(orient="index").items()
                },
            },
            "pairwise_ordering": {
                "mean_exact_order_agreement_rate": float(
                    same_rollout_groups["pair_order_agreement_rate"].mean()
                ),
                "mean_strict_reversal_rate": float(
                    same_rollout_groups["strict_pair_reversal_rate"].mean()
                ),
                "mean_dynamic_non_tie_static_tie_rate": float(
                    same_rollout_groups["dynamic_non_tie_static_tie_rate"].mean()
                ),
                "mean_dynamic_tie_static_non_tie_rate": float(
                    same_rollout_groups["dynamic_tie_static_non_tie_rate"].mean()
                ),
            },
            "advantage_properties": {
                "active_static_group_advantage_std_mean": float(
                    same_rollout_active["static_grpo_advantage_std_sample"].mean()
                ),
                "all_static_group_advantage_mad_mean": float(
                    same_rollout_groups["static_grpo_advantage_mad"].mean()
                ),
                "all_static_group_effective_response_rate_mean": float(
                    same_rollout_groups["static_effective_response_rate"].mean()
                ),
            },
            "endpoint_step_metrics": same_rollout_endpoints,
            "update_trajectory_correlations": same_rollout_trajectories.to_dict(orient="records"),
            "validation": same_rollout_validation,
            "interpretation_guard": (
                "This removes rollout identity as a confound but reuses offline-R0 binary grades "
                "from the union-grading call; it is not an independent R0-only judge invocation "
                "and does not estimate the policy that static-R0 training would have produced."
            ),
        },
        "policy_performance": top_performance,
        "cross_dataset_performance_change": {
            "outcome": "checkpoint_interval_delta_per_update",
            "spearman_r": cross_dataset_spearman,
            "spearman_p": cross_dataset_p,
            "n_intervals": len(interval_wide),
        },
        "strongest_performance_correlations": strongest,
        "validation": validation,
        "interpretation_guards": [
            "Correlations use one training run and are exploratory, not causal.",
            "Twenty-one checkpoint intervals are serially dependent; prompt count does not increase the time-series sample size.",
            "Fixed-probe scores were not the rewards used by the optimizer.",
            "Advantage reconstruction does not reconstruct PPO clipping, KL terms, token-length weighting, gradients, or parameter updates.",
        ],
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    responses, groups, steps, validation = reconstruct_training_advantages()
    advantage_correlations = metric_advantage_relationships(groups, steps)
    (
        same_rollout_responses,
        same_rollout_groups,
        same_rollout_steps,
        same_rollout_trajectories,
        same_rollout_validation,
    ) = reconstruct_static_r0_on_same_rollouts(responses, groups)
    probe = fixed_probe_states()
    intervals, states = build_performance_intervals(groups, probe)
    performance_correlations_frame = performance_correlations(intervals, states)

    responses.to_csv(OUTPUT / "advantage_by_response.csv", index=False)
    groups.to_csv(OUTPUT / "advantage_by_prompt_group.csv", index=False)
    steps.to_csv(OUTPUT / "advantage_by_update.csv", index=False)
    advantage_correlations.to_csv(OUTPUT / "metric_advantage_correlations.csv", index=False)
    same_rollout_responses.to_csv(OUTPUT / "same_rollout_static_by_response.csv", index=False)
    same_rollout_groups.to_csv(OUTPUT / "same_rollout_static_by_prompt_group.csv", index=False)
    same_rollout_steps.to_csv(OUTPUT / "same_rollout_static_by_update.csv", index=False)
    same_rollout_trajectories.to_csv(
        OUTPUT / "same_rollout_static_trajectory_correlations.csv", index=False
    )
    probe.to_csv(OUTPUT / "fixed_probe_checkpoint_states.csv", index=False)
    intervals.to_csv(OUTPUT / "performance_intervals.csv", index=False)
    states.to_csv(OUTPUT / "checkpoint_states_and_performance.csv", index=False)
    performance_correlations_frame.to_csv(OUTPUT / "metric_performance_correlations.csv", index=False)

    plot_advantage_bridge(
        same_rollout_steps,
        OUTPUT / "figure1_actual_training_advantage_bridge.svg",
    )
    plot_same_rollout_static_comparison(
        same_rollout_steps,
        OUTPUT / "figure1b_same_rollout_static_comparison.svg",
    )
    plot_performance(states, OUTPUT / "figure2_checkpoint_performance.svg")
    plot_correlation_heatmap(performance_correlations_frame, OUTPUT / "figure3_metric_performance_correlations.svg")

    summary = summarize(
        groups,
        steps,
        advantage_correlations,
        same_rollout_responses,
        same_rollout_groups,
        same_rollout_steps,
        same_rollout_trajectories,
        same_rollout_validation,
        intervals,
        performance_correlations_frame,
        validation,
    )
    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    verification = {
        "status": "passed",
        "checks": {
            "source_coverage": validation,
            "all_expected_outputs_exist": True,
            "performance_datasets": list(PERFORMANCE_DATASETS),
            "performance_intervals_each": 21,
            "fixed_probe_checkpoints": 22,
            "same_rollout_static_replay": same_rollout_validation,
        },
    }
    (OUTPUT / "verification.json").write_text(
        json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"output": str(OUTPUT), "coverage": summary["coverage"]}, indent=2))


if __name__ == "__main__":
    main()
