#!/usr/bin/env python3
"""Plot actual normalized training-reward diagnostics for Online and Static runs.

The Online series comes from the historical run used by the 21-checkpoint
HealthBench correlation analysis.  The Static series comes from the matched
Static R0 run's own training rollouts, not from a replay on Online responses.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/mad_ptr_zar_grpo_advantage_20260917"
ONLINE_INPUT = RESULTS / "same_rollout_static_by_update.csv"
STATIC_INPUT = (
    ROOT
    / "results/latest_dense_online_vs_static_actual_training_metrics_20260926/"
    "actual_training_mad_ptr_ecr_by_step.csv"
)
OUTPUT_CSV = RESULTS / "actual_online_and_static_training_reward_metrics_by_update.csv"
OUTPUT_SVG = RESULTS / "figure5_actual_online_and_static_training_reward_metrics_by_update.svg"
OUTPUT_PNG = RESULTS / "figure5_actual_online_and_static_training_reward_metrics_by_update.png"


def load_metrics() -> pd.DataFrame:
    online_raw = pd.read_csv(ONLINE_INPUT).sort_values("optimizer_update")
    if online_raw["optimizer_update"].tolist() != list(range(1, 49)):
        raise RuntimeError("historical Online run must contain updates 1 through 48")
    if not (
        online_raw["response_policy_version"] == online_raw["optimizer_update"] - 1
    ).all():
        raise RuntimeError("Online response policy must equal optimizer update minus one")

    online = pd.DataFrame(
        {
            "method": "Online Rubrics actual",
            "training_step": online_raw["optimizer_update"],
            "MAD": online_raw["dynamic_reward_mad"],
            "PTR": online_raw["dynamic_epsilon_01_pairwise_tie_rate"],
            "ECR": online_raw["dynamic_effective_criterion_ratio"],
        }
    )

    static_all = pd.read_csv(STATIC_INPUT)
    static = (
        static_all.loc[
            static_all["method"] == "Static R0 matched",
            ["training_step", "MAD", "PTR", "ECR"],
        ]
        .sort_values("training_step")
        .copy()
    )
    if static["training_step"].tolist() != list(range(1, 43)):
        raise RuntimeError("matched Static R0 run must contain steps 1 through 42")
    static.insert(0, "method", "Static R0 actual")

    metrics = pd.concat([online, static], ignore_index=True)
    if metrics[["MAD", "PTR", "ECR"]].isna().any().any():
        raise RuntimeError("reward metrics contain missing values")
    return metrics


def plot(metrics: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.8), constrained_layout=True)
    styles = {
        "Online Rubrics actual": {
            "color": "#3366cc",
            "linestyle": "-",
            "marker": "o",
            "label": "Online Rubrics actual training (updates 1–48)",
        },
        "Static R0 actual": {
            "color": "#dd7711",
            "linestyle": "--",
            "marker": "s",
            "label": "Static R0 actual training (updates 1–42)",
        },
    }
    for axis, metric in zip(axes, ("MAD", "PTR", "ECR"), strict=True):
        for method, style in styles.items():
            frame = metrics.loc[metrics["method"] == method]
            axis.plot(
                frame["training_step"],
                frame[metric],
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                markersize=3.2,
                markeredgewidth=0,
                linewidth=1.7,
                label=style["label"],
            )
        axis.set_title(metric)
        axis.set_xlabel("Optimizer update")
        axis.set_ylabel(metric)
        axis.set_xlim(0.5, 48.5)
        axis.set_xticks([1, 6, 12, 18, 24, 30, 36, 42, 48])
        axis.grid(alpha=0.22)
        axis.legend(frameon=False, fontsize=8.0, loc="best")

    fig.suptitle(
        "Actual normalized training rewards: Online Rubrics vs matched Static R0"
    )
    fig.savefig(OUTPUT_SVG, format="svg", bbox_inches="tight")
    fig.savefig(OUTPUT_PNG, format="png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    metrics = load_metrics()
    metrics.to_csv(OUTPUT_CSV, index=False)
    plot(metrics)
    print(
        metrics.groupby("method", sort=False)
        .agg(
            first_step=("training_step", "min"),
            last_step=("training_step", "max"),
            rows=("training_step", "size"),
        )
        .to_string()
    )


if __name__ == "__main__":
    main()
