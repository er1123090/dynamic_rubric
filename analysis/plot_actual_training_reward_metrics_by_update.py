#!/usr/bin/env python3
"""Plot MAD, PTR, and ECR reconstructed from actual online-training artifacts.

This script performs no generation, rubric construction, or grading.  It reads
the previously reconstructed per-update table whose source is the committed
``rewards.jsonl`` and ``rubric_unions.jsonl`` files from optimizer updates
1 through 48.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/mad_ptr_zar_grpo_advantage_20260917"
INPUT = RESULTS / "advantage_by_update.csv"
OUTPUT_FIGURE_SVG = RESULTS / "figure5_actual_training_reward_metrics_by_update.svg"
OUTPUT_FIGURE_PNG = RESULTS / "figure5_actual_training_reward_metrics_by_update.png"
OUTPUT_TABLE = RESULTS / "actual_training_reward_metrics_summary.csv"
OUTPUT_SUMMARY = RESULTS / "actual_training_reward_metrics_summary.json"

METRICS = (
    ("MAD", "reward_mad", "Actual training reward MAD", "#3366cc"),
    ("PTR@0.01", "epsilon_01_pairwise_tie_rate", "Actual training PTR@0.01", "#cc6677"),
    ("ECR", "effective_criterion_ratio", "Actual training ECR", "#228833"),
)


def main() -> None:
    frame = pd.read_csv(INPUT).sort_values("optimizer_update").reset_index(drop=True)
    expected_updates = list(range(1, 49))
    if frame["optimizer_update"].tolist() != expected_updates:
        raise RuntimeError("actual training update coverage must be exactly 1..48")
    if not (frame["response_policy_version"] == frame["optimizer_update"] - 1).all():
        raise RuntimeError("every update must use responses from policy version update-1")
    if int(frame["prompt_visits"].sum()) != 4500:
        raise RuntimeError("expected 4,500 actual training prompt groups")

    update_33 = frame.loc[frame["optimizer_update"] == 33].iloc[0]
    summary_rows: list[dict] = []
    for label, column, _, _ in METRICS:
        minimum_index = frame[column].idxmin()
        maximum_index = frame[column].idxmax()
        summary_rows.append(
            {
                "metric": label,
                "mean": float(frame[column].mean()),
                "std_across_updates": float(frame[column].std(ddof=1)),
                "min": float(frame.loc[minimum_index, column]),
                "min_update": int(frame.loc[minimum_index, "optimizer_update"]),
                "max": float(frame.loc[maximum_index, column]),
                "max_update": int(frame.loc[maximum_index, "optimizer_update"]),
                "update_33": float(update_33[column]),
            }
        )
    summary_table = pd.DataFrame(summary_rows)
    summary_table.to_csv(OUTPUT_TABLE, index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.6), constrained_layout=True)
    for axis, (label, column, title, color) in zip(axes, METRICS):
        axis.plot(
            frame["optimizer_update"],
            frame[column],
            color=color,
            linewidth=1.7,
            marker="o",
            markersize=3.7,
            markeredgewidth=0,
            alpha=0.92,
        )
        axis.axvline(33, color="#e66101", linestyle="--", linewidth=1.2, alpha=0.8)
        axis.scatter(
            [33],
            [update_33[column]],
            marker="*",
            s=210,
            color="#e66101",
            edgecolors="black",
            linewidths=0.7,
            label="Update 33 (policy 32)",
            zorder=5,
        )
        axis.set_title(title)
        axis.set_xlabel("Optimizer update")
        axis.set_ylabel(label)
        axis.set_xlim(0.5, 48.5)
        axis.set_xticks([1, 6, 12, 18, 24, 30, 33, 36, 42, 48])
        axis.grid(alpha=0.22)
        axis.legend(loc="best", frameon=False, fontsize=8.5)

    fig.suptitle("MAD, PTR, and ECR from actual OnlineRubrics training rewards (updates 1–48)")
    fig.savefig(OUTPUT_FIGURE_SVG, format="svg", bbox_inches="tight")
    fig.savefig(OUTPUT_FIGURE_PNG, format="png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "data_provenance": {
            "new_prompt_set": False,
            "new_response_generation": False,
            "new_rubric_generation": False,
            "new_grading": False,
            "source": "committed online-training rewards.jsonl and rubric_unions.jsonl",
        },
        "coverage": {
            "optimizer_updates": 48,
            "training_prompt_groups": int(frame["prompt_visits"].sum()),
            "responses": int(frame["prompt_visits"].sum() * 16),
        },
        "alignment": "optimizer update u uses response_policy_version u-1",
        "update_33": {
            "response_policy_version": int(update_33["response_policy_version"]),
            "MAD": float(update_33["reward_mad"]),
            "PTR@0.01": float(update_33["epsilon_01_pairwise_tie_rate"]),
            "ECR": float(update_33["effective_criterion_ratio"]),
        },
        "metrics": summary_rows,
    }
    OUTPUT_SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(summary_table.to_string(index=False))


if __name__ == "__main__":
    main()
