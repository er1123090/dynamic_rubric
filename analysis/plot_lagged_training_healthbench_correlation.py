#!/usr/bin/env python3
"""Plot actual OnlineRubrics training signals against the next HealthBench change.

The source table already aligns each saved-checkpoint interval ``s -> t`` with
the online training updates ``s + 1, ..., t``.  In particular, the ``32 -> 33``
row contains update 33 metrics generated from policy version 32 and compares
them with the HealthBench change from checkpoint 32 to checkpoint 33.
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
INTERVALS = RESULTS / "performance_intervals.csv"
OUTPUT_TABLE = RESULTS / "lagged_training_healthbench_correlations.csv"
OUTPUT_INTERVALS = RESULTS / "lagged_training_healthbench_intervals.csv"
OUTPUT_FIGURE_SVG = RESULTS / "figure4_lagged_training_healthbench_correlations.svg"
OUTPUT_FIGURE_PNG = RESULTS / "figure4_lagged_training_healthbench_correlations.png"
OUTPUT_SUMMARY = RESULTS / "lagged_training_healthbench_summary.json"

METRICS = (
    ("MAD", "train_reward_mad", "Training reward MAD"),
    ("PTR@0.01", "train_epsilon_01_pairwise_tie_rate", "Training PTR@0.01"),
    ("ECR", "train_effective_criterion_ratio", "Training ECR"),
)


def benjamini_hochberg(values: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=values.index, dtype=float)
    valid = values.dropna().sort_values()
    if valid.empty:
        return result
    adjusted = np.empty(len(valid), dtype=float)
    running = 1.0
    for reverse_index in range(len(valid) - 1, -1, -1):
        rank = reverse_index + 1
        running = min(running, float(valid.iloc[reverse_index]) * len(valid) / rank)
        adjusted[reverse_index] = running
    result.loc[valid.index] = adjusted
    return result


def leave_one_out_sign_stability(x: pd.Series, y: pd.Series, full_rho: float) -> tuple[int, int, float, float]:
    estimates: list[float] = []
    for omitted in range(len(x)):
        keep = np.arange(len(x)) != omitted
        estimates.append(float(spearmanr(x[keep], y[keep]).statistic))
    same_sign = sum(np.sign(value) == np.sign(full_rho) for value in estimates)
    return len(estimates), same_sign, min(estimates), max(estimates)


def main() -> None:
    intervals = pd.read_csv(INTERVALS)
    frame = intervals.loc[intervals["dataset"] == "healthbench"].copy()
    frame = frame.sort_values(["start_step", "end_step"]).reset_index(drop=True)
    if len(frame) != 21:
        raise RuntimeError(f"expected 21 HealthBench intervals, found {len(frame)}")

    frame["interval"] = frame["start_step"].astype(str) + "→" + frame["end_step"].astype(str)
    export_columns = [
        "interval",
        "start_step",
        "end_step",
        "updates_in_interval",
        "train_prompt_groups",
        "performance_start",
        "performance_end",
        "performance_delta",
        "performance_delta_per_update",
        *[column for _, column, _ in METRICS],
    ]
    frame[export_columns].to_csv(OUTPUT_INTERVALS, index=False)

    rows: list[dict] = []
    outcome = frame["performance_delta_per_update"]
    for label, column, _ in METRICS:
        pearson = pearsonr(frame[column], outcome)
        spearman = spearmanr(frame[column], outcome)
        loo_n, loo_same, loo_min, loo_max = leave_one_out_sign_stability(
            frame[column], outcome, float(spearman.statistic)
        )
        rows.append(
            {
                "predictor": label,
                "outcome": "next_checkpoint_healthbench_delta_per_update",
                "n": len(frame),
                "pearson_r": float(pearson.statistic),
                "pearson_p": float(pearson.pvalue),
                "spearman_r": float(spearman.statistic),
                "spearman_p": float(spearman.pvalue),
                "loo_defined": loo_n,
                "loo_same_sign": loo_same,
                "loo_spearman_min": loo_min,
                "loo_spearman_max": loo_max,
            }
        )
    correlations = pd.DataFrame(rows)
    correlations["spearman_q_bh_three_metrics"] = benjamini_hochberg(correlations["spearman_p"])
    correlations.to_csv(OUTPUT_TABLE, index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.8), constrained_layout=True, sharey=True)
    end_steps = frame["end_step"].to_numpy()
    norm = plt.Normalize(end_steps.min(), end_steps.max())
    cmap = plt.get_cmap("viridis")
    highlighted = (frame["start_step"] == 32) & (frame["end_step"] == 33)

    for axis, (label, column, x_label), result in zip(axes, METRICS, rows):
        x = frame[column].to_numpy()
        y = outcome.to_numpy()
        axis.scatter(
            x,
            y,
            c=end_steps,
            cmap=cmap,
            norm=norm,
            s=52,
            edgecolors="white",
            linewidths=0.7,
            alpha=0.9,
            zorder=3,
        )
        axis.scatter(
            frame.loc[highlighted, column],
            frame.loc[highlighted, "performance_delta_per_update"],
            marker="*",
            s=220,
            color="#e66101",
            edgecolors="black",
            linewidths=0.8,
            label="32→33",
            zorder=5,
        )
        slope, intercept = np.polyfit(x, y, deg=1)
        grid = np.linspace(float(x.min()), float(x.max()), 200)
        axis.plot(grid, slope * grid + intercept, color="#444444", linewidth=1.6, zorder=2)
        axis.axhline(0.0, color="#999999", linewidth=1.0, linestyle="--", zorder=1)
        axis.set_xlabel(x_label)
        axis.set_title(label)
        axis.grid(alpha=0.2)
        axis.text(
            0.03,
            0.97,
            (
                f"Pearson r = {result['pearson_r']:+.3f}  (p = {result['pearson_p']:.3f})\n"
                f"Spearman ρ = {result['spearman_r']:+.3f}  (p = {result['spearman_p']:.3f})"
            ),
            transform=axis.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
        )
        axis.legend(loc="lower left", frameon=False)

    axes[0].set_ylabel("Next HealthBench change per optimizer update")
    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=axes,
        shrink=0.82,
        pad=0.02,
    )
    colorbar.set_label("End checkpoint")
    fig.suptitle("Actual training rubric signal vs next HealthBench change (21 checkpoint intervals)")
    fig.savefig(OUTPUT_FIGURE_SVG, format="svg", bbox_inches="tight")
    fig.savefig(OUTPUT_FIGURE_PNG, format="png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    step_32_to_33 = frame.loc[highlighted].iloc[0]
    summary = {
        "analysis": "actual training interval signal vs next checkpoint HealthBench delta per update",
        "alignment": "interval s->t uses optimizer updates s+1..t; update u uses policy version u-1",
        "n_intervals": len(frame),
        "n_optimizer_updates": int(frame["updates_in_interval"].sum()),
        "n_training_prompt_groups": int(frame["train_prompt_groups"].sum()),
        "multiple_testing": "Benjamini-Hochberg over the three prespecified Spearman tests",
        "step_32_to_33": {
            "training_update": 33,
            "response_policy_version": 32,
            "MAD": float(step_32_to_33["train_reward_mad"]),
            "PTR@0.01": float(step_32_to_33["train_epsilon_01_pairwise_tie_rate"]),
            "ECR": float(step_32_to_33["train_effective_criterion_ratio"]),
            "healthbench_start": float(step_32_to_33["performance_start"]),
            "healthbench_end": float(step_32_to_33["performance_end"]),
            "healthbench_delta_per_update": float(step_32_to_33["performance_delta_per_update"]),
        },
        "correlations": correlations.to_dict(orient="records"),
    }
    OUTPUT_SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(correlations.to_string(index=False))
    print(json.dumps(summary["step_32_to_33"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
