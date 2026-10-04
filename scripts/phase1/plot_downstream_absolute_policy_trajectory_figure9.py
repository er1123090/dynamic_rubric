#!/usr/bin/env python3
"""Plot absolute held-out and HealthBench policy-score trajectories for Notion."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


DATASET_ORDER = ["rar_medicine_test", "healthbench"]
METHOD_ORDER = ["static", "online"]
DATASET_LABELS = {
    "rar_medicine_test": "RaR-Medicine held-out (n=300)",
    "healthbench": "HealthBench subset (n=500)",
}
METHOD_LABELS = {
    "static": "Static rubric GRPO · Qwen3-1.7B",
    "online": "OnlineRubrics GRPO · Qwen3-4B",
}
COLORS = {"static": "#2F6F9F", "online": "#D46A4C"}


def read_rows(path: Path) -> dict[tuple[str, str], list[dict[str, float | str]]]:
    grouped: dict[tuple[str, str], list[dict[str, float | str]]] = defaultdict(list)
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            parsed: dict[str, float | str] = dict(row)
            for field in (
                "global_step",
                "final_mean",
                "base_mean",
                "delta_final_minus_base",
                "ci95_low",
                "ci95_high",
                "n_prompts",
            ):
                parsed[field] = float(row[field])
            grouped[(row["dataset"], row["method"])].append(parsed)
    for rows in grouped.values():
        rows.sort(key=lambda item: float(item["global_step"]))
    return grouped


def plot_panel(axis: plt.Axes, rows: list[dict[str, float | str]], method: str) -> None:
    color = COLORS[method]
    steps = [float(row["global_step"]) for row in rows]
    scores = [float(row["final_mean"]) for row in rows]
    base = float(rows[0]["base_mean"])
    ci_low = [base + float(row["ci95_low"]) for row in rows]
    ci_high = [base + float(row["ci95_high"]) for row in rows]

    axis.fill_between(steps, ci_low, ci_high, color=color, alpha=0.18, linewidth=0)
    axis.plot(steps, scores, color=color, marker="o", markersize=4.5, linewidth=2.2)
    axis.axhline(base, color="#666666", linestyle="--", linewidth=1.1)

    peak = max(rows, key=lambda row: float(row["final_mean"]))
    final = next(row for row in rows if row["role"] == "final")
    axis.scatter(
        [float(peak["global_step"])],
        [float(peak["final_mean"])],
        marker="*",
        s=155,
        color="#16856A",
        edgecolor="white",
        linewidth=0.8,
        zorder=5,
    )
    axis.scatter(
        [float(final["global_step"])],
        [float(final["final_mean"])],
        marker="D",
        s=55,
        color=color,
        edgecolor="#222222",
        linewidth=0.8,
        zorder=5,
    )
    axis.annotate(
        f"peak s{int(float(peak['global_step']))}: {float(peak['final_mean']):.3f}",
        (float(peak["global_step"]), float(peak["final_mean"])),
        xytext=(6, 9),
        textcoords="offset points",
        fontsize=8.5,
        color="#126B57",
    )
    axis.annotate(
        f"final: {float(final['final_mean']):.3f}",
        (float(final["global_step"]), float(final["final_mean"])),
        xytext=(-4, -17),
        textcoords="offset points",
        ha="right",
        fontsize=8.5,
        color="#333333",
    )
    axis.set_xlim(-1.5, 50.5)
    axis.set_ylim(0.46, 0.72)
    axis.set_xticks([0, 8, 16, 24, 32, 40, 48])
    axis.grid(axis="both", alpha=0.2)
    axis.spines[["top", "right"]].set_visible(False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_png", type=Path)
    args = parser.parse_args()

    grouped = read_rows(args.input_csv)
    missing = [
        key
        for key in ((dataset, method) for method in METHOD_ORDER for dataset in DATASET_ORDER)
        if key not in grouped
    ]
    if missing:
        raise ValueError(f"Missing dataset/method trajectories: {missing}")

    fig, axes = plt.subplots(2, 2, figsize=(13.6, 8.8), sharex=True, sharey=True)
    for row_index, method in enumerate(METHOD_ORDER):
        for col_index, dataset in enumerate(DATASET_ORDER):
            axis = axes[row_index, col_index]
            plot_panel(axis, grouped[(dataset, method)], method)
            if row_index == 0:
                axis.set_title(DATASET_LABELS[dataset], fontsize=12.5, fontweight="bold")
            if col_index == 0:
                axis.set_ylabel(f"{METHOD_LABELS[method]}\nAbsolute policy score", fontsize=10.5)
            if row_index == len(METHOD_ORDER) - 1:
                axis.set_xlabel("GRPO global step", fontsize=10.5)

    legend = [
        Line2D([0], [0], color="#555555", linestyle="--", label="own step-0 base"),
        Line2D([0], [0], color="#777777", linewidth=7, alpha=0.22,
               label="95% paired-bootstrap CI for change vs own base"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor="#16856A",
               markeredgecolor="white", markersize=12, label="best observed checkpoint"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor="#888888",
               markeredgecolor="#222222", markersize=7, label="step-48 final"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle("Figure 9. Absolute downstream policy performance across checkpoints",
                 fontsize=16, fontweight="bold", y=0.985)
    fig.text(
        0.5,
        0.942,
        "One deterministic response per prompt · Qwen3-32B judge · HealthBench is a seeded 500/5,000 subset",
        ha="center",
        fontsize=10,
        color="#444444",
    )
    fig.text(
        0.5,
        0.026,
        "Static and OnlineRubrics use different backbones. Interpret each trajectory against its own base; row-height differences are not a rubric-method effect.",
        ha="center",
        fontsize=9.3,
        color="#8A3A2E",
    )
    fig.tight_layout(rect=(0.025, 0.085, 0.985, 0.91), h_pad=2.1, w_pad=1.8)
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
