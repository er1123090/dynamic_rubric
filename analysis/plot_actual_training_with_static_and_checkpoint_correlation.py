#!/usr/bin/env python3
"""Compare actual online-training signals with a same-rollout Static R0 replay.

No prompt, response, rubric, or grade is generated here. Static R0 rewards are
reconstructed from the offline-R0 criteria and criterion grades already stored
in each committed online-training artifact.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/mad_ptr_zar_grpo_advantage_20260917"
ONLINE_INPUT = RESULTS / "advantage_by_update.csv"
STATIC_INPUT = RESULTS / "same_rollout_static_by_update.csv"
PERFORMANCE_INPUT = RESULTS / "checkpoint_states_and_performance.csv"

TRAJECTORY_SVG = RESULTS / "figure5_actual_training_reward_metrics_by_update.svg"
TRAJECTORY_PNG = RESULTS / "figure5_actual_training_reward_metrics_by_update.png"
CORRELATION_SVG = RESULTS / "figure6_actual_training_checkpoint_healthbench_correlations.svg"
CORRELATION_PNG = RESULTS / "figure6_actual_training_checkpoint_healthbench_correlations.png"
SUMMARY_CSV = RESULTS / "actual_training_online_static_metrics_summary.csv"
SUMMARY_JSON = RESULTS / "actual_training_online_static_checkpoint_correlation_summary.json"
PAIRS_CSV = RESULTS / "actual_training_checkpoint_healthbench_pairs.csv"
CORRELATIONS_CSV = RESULTS / "actual_training_checkpoint_healthbench_correlations.csv"

METRICS = (
    ("MAD", "dynamic_reward_mad", "static_reward_mad"),
    (
        "PTR",
        "dynamic_epsilon_01_pairwise_tie_rate",
        "static_epsilon_01_pairwise_tie_rate",
    ),
    (
        "ECR",
        "dynamic_effective_criterion_ratio",
        "static_effective_criterion_ratio",
    ),
)


def benjamini_hochberg(values: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=values.index, dtype=float)
    valid = values.dropna().sort_values()
    adjusted = np.empty(len(valid), dtype=float)
    running = 1.0
    for reverse_index in range(len(valid) - 1, -1, -1):
        rank = reverse_index + 1
        running = min(running, float(valid.iloc[reverse_index]) * len(valid) / rank)
        adjusted[reverse_index] = running
    result.loc[valid.index] = adjusted
    return result


def loo_sign_stability(x: pd.Series, y: pd.Series, full_rho: float) -> tuple[int, int]:
    estimates: list[float] = []
    for omitted in range(len(x)):
        keep = np.arange(len(x)) != omitted
        estimates.append(float(spearmanr(x[keep], y[keep]).statistic))
    same = sum(np.sign(value) == np.sign(full_rho) for value in estimates)
    return len(estimates), same


def load_and_validate() -> tuple[pd.DataFrame, pd.DataFrame]:
    canonical = pd.read_csv(ONLINE_INPUT).sort_values("optimizer_update").reset_index(drop=True)
    frame = pd.read_csv(STATIC_INPUT).sort_values("optimizer_update").reset_index(drop=True)
    expected = list(range(1, 49))
    if frame["optimizer_update"].tolist() != expected:
        raise RuntimeError("expected optimizer updates 1 through 48")
    if not (frame["response_policy_version"] == frame["optimizer_update"] - 1).all():
        raise RuntimeError("response policy version must equal optimizer update minus one")
    if int(frame["prompt_visits"].sum()) != 4500:
        raise RuntimeError("expected 4,500 actual training prompt groups")

    joined = frame.merge(
        canonical[
            [
                "optimizer_update",
                "reward_mad",
                "epsilon_01_pairwise_tie_rate",
                "effective_criterion_ratio",
            ]
        ],
        on="optimizer_update",
        validate="one_to_one",
    )
    for reconstructed, expected_column in (
        ("dynamic_reward_mad", "reward_mad"),
        ("dynamic_epsilon_01_pairwise_tie_rate", "epsilon_01_pairwise_tie_rate"),
        ("dynamic_effective_criterion_ratio", "effective_criterion_ratio"),
    ):
        if not np.allclose(joined[reconstructed], joined[expected_column], atol=1e-12, rtol=0):
            raise RuntimeError(f"dynamic reconstruction mismatch for {reconstructed}")

    performance = pd.read_csv(PERFORMANCE_INPUT)
    healthbench = performance.loc[
        (performance["dataset"] == "healthbench") & (performance["global_step"] > 0),
        ["global_step", "performance"],
    ].rename(columns={"global_step": "optimizer_update", "performance": "healthbench"})
    pairs = healthbench.merge(frame, on="optimizer_update", validate="one_to_one")
    if len(pairs) != 21:
        raise RuntimeError(f"expected 21 nonzero HealthBench checkpoints, found {len(pairs)}")
    return frame, pairs


def summarize_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    update_33 = frame.loc[frame["optimizer_update"] == 33].iloc[0]
    rows: list[dict] = []
    for label, online_column, static_column in METRICS:
        for evaluator, column in (
            ("Online union", online_column),
            ("Static R0 replay", static_column),
        ):
            minimum = frame[column].idxmin()
            maximum = frame[column].idxmax()
            rows.append(
                {
                    "evaluator": evaluator,
                    "metric": label,
                    "mean": float(frame[column].mean()),
                    "std_across_updates": float(frame[column].std(ddof=1)),
                    "min": float(frame.loc[minimum, column]),
                    "min_update": int(frame.loc[minimum, "optimizer_update"]),
                    "max": float(frame.loc[maximum, column]),
                    "max_update": int(frame.loc[maximum, "optimizer_update"]),
                    "update_33": float(update_33[column]),
                }
            )
    return pd.DataFrame(rows)


def compute_correlations(pairs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for evaluator, selector in (
        ("Online union", 1),
        ("Static R0 replay", 2),
    ):
        for metric in METRICS:
            label = metric[0]
            column = metric[selector]
            pearson = pearsonr(pairs[column], pairs["healthbench"])
            spearman = spearmanr(pairs[column], pairs["healthbench"])
            loo_n, loo_same = loo_sign_stability(
                pairs[column], pairs["healthbench"], float(spearman.statistic)
            )
            rows.append(
                {
                    "evaluator": evaluator,
                    "predictor": label,
                    "outcome": "same_checkpoint_healthbench",
                    "n": len(pairs),
                    "pearson_r": float(pearson.statistic),
                    "pearson_p": float(pearson.pvalue),
                    "spearman_r": float(spearman.statistic),
                    "spearman_p": float(spearman.pvalue),
                    "loo_defined": loo_n,
                    "loo_same_sign": loo_same,
                }
            )
    result = pd.DataFrame(rows)
    result["spearman_q_bh_three_metrics"] = result.groupby("evaluator")[
        "spearman_p"
    ].transform(benjamini_hochberg)
    return result


def plot_trajectories(frame: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.6), constrained_layout=True)
    for axis, (label, online_column, static_column) in zip(axes, METRICS):
        axis.plot(
            frame["optimizer_update"],
            frame[online_column],
            color="#3366cc",
            linewidth=1.7,
            marker="o",
            markersize=3.6,
            markeredgewidth=0,
            label="Online union (used for update)",
        )
        axis.plot(
            frame["optimizer_update"],
            frame[static_column],
            color="#dd7711",
            linewidth=1.6,
            linestyle="--",
            marker="s",
            markersize=3.1,
            markeredgewidth=0,
            label="Static R0 replay (same rollout)",
        )
        axis.set_title(label)
        axis.set_xlabel("Optimizer update")
        axis.set_ylabel(label)
        axis.set_xlim(0.5, 48.5)
        axis.set_xticks([1, 6, 12, 18, 24, 30, 33, 36, 42, 48])
        axis.grid(alpha=0.22)
        axis.legend(frameon=False, fontsize=8.0)
    fig.suptitle("Same actual training rollouts: Online union vs Static R0 replay")
    fig.savefig(TRAJECTORY_SVG, format="svg", bbox_inches="tight")
    fig.savefig(TRAJECTORY_PNG, format="png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_checkpoint_correlations(pairs: pd.DataFrame, correlations: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.8), constrained_layout=True, sharey=True)
    for axis, (label, online_column, static_column) in zip(axes, METRICS):
        for evaluator, column, color, marker in (
            ("Online union", online_column, "#3366cc", "o"),
            ("Static R0 replay", static_column, "#dd7711", "s"),
        ):
            x = pairs[column].to_numpy()
            y = pairs["healthbench"].to_numpy()
            axis.scatter(x, y, color=color, marker=marker, s=44, alpha=0.82, label=evaluator)
            slope, intercept = np.polyfit(x, y, deg=1)
            grid = np.linspace(float(x.min()), float(x.max()), 160)
            axis.plot(grid, slope * grid + intercept, color=color, linewidth=1.5)

        metric_rows = correlations.loc[correlations["predictor"] == label]
        online_rho = metric_rows.loc[
            metric_rows["evaluator"] == "Online union", "spearman_r"
        ].iloc[0]
        static_rho = metric_rows.loc[
            metric_rows["evaluator"] == "Static R0 replay", "spearman_r"
        ].iloc[0]
        axis.text(
            0.03,
            0.97,
            f"Online rho = {online_rho:+.3f}\nStatic R0 rho = {static_rho:+.3f}",
            transform=axis.transAxes,
            va="top",
            fontsize=9,
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "white",
                "alpha": 0.85,
                "edgecolor": "#cccccc",
            },
        )
        axis.set_title(label)
        axis.set_xlabel(label)
        axis.grid(alpha=0.22)
        axis.legend(frameon=False, fontsize=8.2)
    axes[0].set_ylabel("HealthBench score at resulting checkpoint")
    fig.suptitle("Actual training signal at available updates vs HealthBench performance (n=21)")
    fig.savefig(CORRELATION_SVG, format="svg", bbox_inches="tight")
    fig.savefig(CORRELATION_PNG, format="png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    frame, pairs = load_and_validate()
    metric_summary = summarize_metrics(frame)
    correlations = compute_correlations(pairs)
    metric_summary.to_csv(SUMMARY_CSV, index=False)
    correlations.to_csv(CORRELATIONS_CSV, index=False)

    pair_columns = [
        "optimizer_update",
        "response_policy_version",
        "healthbench",
        *[
            column
            for _, online_column, static_column in METRICS
            for column in (online_column, static_column)
        ],
    ]
    pairs[pair_columns].to_csv(PAIRS_CSV, index=False)
    plot_trajectories(frame)
    plot_checkpoint_correlations(pairs, correlations)

    update_33 = frame.loc[frame["optimizer_update"] == 33].iloc[0]
    summary = {
        "data_provenance": {
            "new_prompt_set": False,
            "new_response_generation": False,
            "new_rubric_generation": False,
            "new_grading": False,
            "online_union": "actual normalized reward used for each OnlineRubrics optimizer update",
            "static_r0_replay": (
                "same training responses, stored offline-R0 criteria, and already-recorded grades; "
                "no new judge call; counterfactual and not used for the online optimizer update"
            ),
        },
        "coverage": {
            "optimizer_updates": 48,
            "training_prompt_groups": int(frame["prompt_visits"].sum()),
            "responses": int(frame["prompt_visits"].sum() * 16),
            "healthbench_checkpoints": len(pairs),
            "checkpoint_steps": pairs["optimizer_update"].astype(int).tolist(),
        },
        "alignment": (
            "update t metrics use policy t-1 rollouts and are paired with HealthBench at resulting "
            "checkpoint t; checkpoint 0 is excluded because no update-0 training reward exists"
        ),
        "update_33": {
            "response_policy_version": int(update_33["response_policy_version"]),
            "online_MAD": float(update_33["dynamic_reward_mad"]),
            "online_PTR": float(update_33["dynamic_epsilon_01_pairwise_tie_rate"]),
            "online_ECR": float(update_33["dynamic_effective_criterion_ratio"]),
            "static_r0_MAD": float(update_33["static_reward_mad"]),
            "static_r0_PTR": float(update_33["static_epsilon_01_pairwise_tie_rate"]),
            "static_r0_ECR": float(update_33["static_effective_criterion_ratio"]),
        },
        "metric_summary": metric_summary.to_dict(orient="records"),
        "checkpoint_healthbench_correlations": correlations.to_dict(orient="records"),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(correlations.to_string(index=False))
    print(json.dumps(summary["update_33"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
