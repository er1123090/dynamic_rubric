#!/usr/bin/env python3
"""Create compact own-base downstream performance trajectories from a completed run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    summary_path = args.run_root / "summary" / "summary.json"
    summary = json.loads(summary_path.read_text())
    rows = [
        {"dataset": dataset, **row}
        for dataset, methods in summary["trajectories"].items()
        for values in methods.values()
        for row in values
    ]
    output = args.run_root / "summary"
    with (output / "checkpoint_performance_trajectory.csv").open("w", newline="") as stream:
        fields = ["dataset", "method", "global_step", "model_name", "role", "final_mean",
                  "base_mean", "delta_final_minus_base", "ci95_low", "ci95_high", "n_prompts"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "dataset": row["dataset"], "method": row["method"],
                "global_step": row["global_step"], "model_name": row["model_name"],
                "role": row["role"], "final_mean": row["final_mean"],
                "base_mean": row["base_mean"],
                "delta_final_minus_base": row["delta_final_minus_base"],
                "ci95_low": row["bootstrap_95_ci"][0],
                "ci95_high": row["bootstrap_95_ci"][1], "n_prompts": row["n_prompts"],
            })
    datasets = list(summary["trajectories"])
    methods = ["static", "online"]
    labels = {"static": "Static rubric GRPO (Qwen3-1.7B)",
              "online": "OnlineRubrics GRPO (Qwen3-4B)"}
    dataset_labels = {"rar_medicine_test": "RaR-Medicine held-out (300)",
                      "healthbench": "HealthBench (5,000)"}
    colors = {"static": "#2f6f9f", "online": "#d46a4c"}
    fig, axes = plt.subplots(len(methods), len(datasets), figsize=(13, 8), sharex=False)
    for r_index, method in enumerate(methods):
        for c_index, dataset in enumerate(datasets):
            axis = axes[r_index][c_index]
            values = summary["trajectories"][dataset][method]
            x = [row["global_step"] for row in values]
            y = [row["delta_final_minus_base"] for row in values]
            low = [row["bootstrap_95_ci"][0] for row in values]
            high = [row["bootstrap_95_ci"][1] for row in values]
            axis.fill_between(x, low, high, color=colors[method], alpha=0.18,
                              label="95% paired bootstrap CI")
            axis.plot(x, y, marker="o", linewidth=2, color=colors[method],
                      label="checkpoint − own base")
            axis.axhline(0, color="#555555", linewidth=1, linestyle="--")
            axis.set_title(f"{labels[method]}\n{dataset_labels.get(dataset, dataset)}")
            axis.set_xlabel("GRPO global step")
            axis.set_ylabel("Score gain vs own step-0 base")
            axis.grid(alpha=0.2)
    handles, labels_legend = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels_legend, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Downstream policy performance across saved checkpoints", fontsize=16)
    fig.text(0.5, 0.025,
             "One deterministic response per prompt; no BoN. Static and OnlineRubrics use different backbones, so compare each curve with its own base.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(output / "checkpoint_performance_trajectory.png", dpi=220)
    fig.savefig(output / "checkpoint_performance_trajectory.svg")


if __name__ == "__main__":
    main()
