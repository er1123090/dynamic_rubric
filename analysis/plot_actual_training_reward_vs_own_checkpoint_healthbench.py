#!/usr/bin/env python3
"""Correlate each run's actual training rewards with its own checkpoints.

For optimizer update t, the reward metrics were computed on rollouts from
policy t-1 and used to create checkpoint t.  The corresponding HealthBench
outcome is therefore the score of checkpoint t from the same training run.
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
REWARD_INPUT = (
    ROOT
    / "results/latest_dense_online_vs_static_actual_training_metrics_20260926"
    / "actual_training_mad_ptr_ecr_by_step.csv"
)
ONLINE_HEALTHBENCH_INPUT = (
    ROOT
    / "outputs/policy_eval/medicine_dense_all48_healthbench500_20260926"
    / "full-6b52dfbc30980b62/summary/checkpoint_performance_trajectory.csv"
)
STATIC_HEALTHBENCH_INPUT = (
    ROOT
    / "outputs/policy_eval/medicine_static_matched_checkpoint_trajectory_hb500_20260926"
    / "full-3f9ab542f8a7f87d/summary/checkpoint_performance_trajectory.csv"
)

FIGURE_SVG = RESULTS / "figure6_actual_training_checkpoint_healthbench_correlations.svg"
FIGURE_PNG = RESULTS / "figure6_actual_training_checkpoint_healthbench_correlations.png"
PAIRS_CSV = RESULTS / "actual_training_checkpoint_healthbench_pairs.csv"
CORRELATIONS_CSV = RESULTS / "actual_training_checkpoint_healthbench_correlations.csv"
SUMMARY_JSON = RESULTS / "actual_training_checkpoint_healthbench_correlation_summary.json"

METRICS = ("MAD", "PTR", "ECR")
METHODS = {
    "Online Rubrics": {
        "label": "Online Rubrics actual",
        "healthbench": ONLINE_HEALTHBENCH_INPUT,
        "expected_steps": list(range(1, 49)),
        "color": "#3366cc",
        "marker": "o",
    },
    "Static R0 matched": {
        "label": "Static R0 actual",
        "healthbench": STATIC_HEALTHBENCH_INPUT,
        "expected_steps": list(range(3, 43, 3)),
        "color": "#dd7711",
        "marker": "s",
    },
}


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
        estimate = float(spearmanr(x[keep], y[keep]).statistic)
        if np.isfinite(estimate):
            estimates.append(estimate)
    same = sum(np.sign(value) == np.sign(full_rho) for value in estimates)
    return len(estimates), same


def load_pairs() -> pd.DataFrame:
    rewards = pd.read_csv(REWARD_INPUT)
    required = {"method", "training_step", *METRICS}
    missing = required.difference(rewards.columns)
    if missing:
        raise RuntimeError(f"reward input is missing columns: {sorted(missing)}")

    frames: list[pd.DataFrame] = []
    for reward_method, settings in METHODS.items():
        reward_frame = (
            rewards.loc[
                rewards["method"] == reward_method,
                ["training_step", *METRICS],
            ]
            .sort_values("training_step")
            .copy()
        )
        if reward_frame.empty:
            raise RuntimeError(f"no reward rows found for {reward_method}")

        healthbench_path = Path(settings["healthbench"])
        if not healthbench_path.is_file():
            raise RuntimeError(f"HealthBench summary is not ready: {healthbench_path}")
        performance = pd.read_csv(healthbench_path)
        healthbench = (
            performance.loc[
                performance["dataset"] == "healthbench",
                ["global_step", "model_name", "final_mean", "n_prompts"],
            ]
            .rename(
                columns={
                    "global_step": "training_step",
                    "final_mean": "healthbench",
                }
            )
            .sort_values("training_step")
        )

        expected_steps = settings["expected_steps"]
        actual_steps = healthbench["training_step"].astype(int).tolist()
        if actual_steps != expected_steps:
            raise RuntimeError(
                f"{reward_method} HealthBench steps mismatch: "
                f"expected {expected_steps}, found {actual_steps}"
            )
        if healthbench["n_prompts"].astype(int).tolist() != [500] * len(expected_steps):
            raise RuntimeError(f"{reward_method} must have 500 HealthBench prompts per checkpoint")

        pairs = healthbench.merge(
            reward_frame,
            on="training_step",
            how="left",
            validate="one_to_one",
        )
        if pairs[list(METRICS)].isna().any().any():
            raise RuntimeError(f"missing reward metric after pairing {reward_method}")
        pairs.insert(0, "method", str(settings["label"]))
        pairs.insert(2, "response_policy_version", pairs["training_step"] - 1)
        frames.append(pairs)

    return pd.concat(frames, ignore_index=True)


def compute_correlations(pairs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method, frame in pairs.groupby("method", sort=False):
        for metric in METRICS:
            pearson = pearsonr(frame[metric], frame["healthbench"])
            spearman = spearmanr(frame[metric], frame["healthbench"])
            loo_defined, loo_same_sign = loo_sign_stability(
                frame[metric], frame["healthbench"], float(spearman.statistic)
            )
            rows.append(
                {
                    "method": method,
                    "predictor": metric,
                    "outcome": "own_checkpoint_healthbench",
                    "n": len(frame),
                    "pearson_r": float(pearson.statistic),
                    "pearson_p": float(pearson.pvalue),
                    "spearman_rho": float(spearman.statistic),
                    "spearman_p": float(spearman.pvalue),
                    "loo_defined": loo_defined,
                    "loo_same_sign": loo_same_sign,
                }
            )
    result = pd.DataFrame(rows)
    result["spearman_q_bh_three_metrics"] = result.groupby("method")[
        "spearman_p"
    ].transform(benjamini_hochberg)
    return result


def plot(pairs: pd.DataFrame, correlations: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.9), constrained_layout=True, sharey=True)
    for axis, metric in zip(axes, METRICS, strict=True):
        annotation_lines: list[str] = []
        for reward_method, settings in METHODS.items():
            label = str(settings["label"])
            frame = pairs.loc[pairs["method"] == label]
            x = frame[metric].to_numpy(dtype=float)
            y = frame["healthbench"].to_numpy(dtype=float)
            axis.scatter(
                x,
                y,
                color=str(settings["color"]),
                marker=str(settings["marker"]),
                s=42,
                alpha=0.82,
                label=f"{label} (n={len(frame)})",
            )
            if np.unique(x).size > 1:
                slope, intercept = np.polyfit(x, y, deg=1)
                grid = np.linspace(float(x.min()), float(x.max()), 160)
                axis.plot(
                    grid,
                    slope * grid + intercept,
                    color=str(settings["color"]),
                    linewidth=1.5,
                )
            row = correlations.loc[
                (correlations["method"] == label)
                & (correlations["predictor"] == metric)
            ].iloc[0]
            short_label = "Online" if reward_method == "Online Rubrics" else "Static"
            annotation_lines.append(
                f"{short_label} rho = {row['spearman_rho']:+.3f} (n={int(row['n'])})"
            )

        axis.text(
            0.03,
            0.97,
            "\n".join(annotation_lines),
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
        axis.set_title(metric)
        axis.set_xlabel(f"Actual training-reward {metric} at update t")
        axis.grid(alpha=0.22)
        axis.legend(frameon=False, fontsize=8.0, loc="best")

    axes[0].set_ylabel("HealthBench score of checkpoint t")
    fig.suptitle("Actual reward at update t vs HealthBench of its own checkpoint t")
    fig.savefig(FIGURE_SVG, format="svg", bbox_inches="tight")
    fig.savefig(FIGURE_PNG, format="png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    pairs = load_pairs()
    correlations = compute_correlations(pairs)
    pairs.to_csv(PAIRS_CSV, index=False)
    correlations.to_csv(CORRELATIONS_CSV, index=False)
    plot(pairs, correlations)

    summary = {
        "alignment": (
            "reward metrics at optimizer update t are computed from policy t-1 rollouts, "
            "used to create checkpoint t, and paired with HealthBench of checkpoint t"
        ),
        "reward_source": "actual normalized rewards used during each training run",
        "healthbench": {
            "prompts_per_checkpoint": 500,
            "sample_seed": 11,
            "judge": "Qwen/Qwen3-32B@9216db5781bf21249d130ec9da846c4624c16137",
        },
        "coverage": {
            method: {
                "n": int(len(frame)),
                "checkpoint_steps": frame["training_step"].astype(int).tolist(),
            }
            for method, frame in pairs.groupby("method", sort=False)
        },
        "correlations": correlations.to_dict(orient="records"),
    }
    SUMMARY_JSON.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(correlations.to_string(index=False))


if __name__ == "__main__":
    main()
