#!/usr/bin/env python3
"""Correlate actual Online training rewards with HealthBench of checkpoint t."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/latest_dense_online_vs_static_actual_training_metrics_20260926"
REWARD_INPUT = RESULTS / "actual_training_mad_ptr_ecr_by_step.csv"
HEALTHBENCH_INPUT = (
    ROOT
    / "outputs/policy_eval/medicine_dense_all48_healthbench500_20260926"
    / "full-6b52dfbc30980b62/summary/checkpoint_trajectory.jsonl"
)

PAIRS_CSV = RESULTS / "online_dense_actual_reward_healthbench_pairs.csv"
CORRELATIONS_CSV = RESULTS / "online_dense_actual_reward_healthbench_correlations.csv"
SUMMARY_JSON = RESULTS / "online_dense_actual_reward_healthbench_correlation_summary.json"
FIGURE_PNG = RESULTS / "online_dense_actual_reward_healthbench_correlation.png"
FIGURE_SVG = RESULTS / "online_dense_actual_reward_healthbench_correlation.svg"

METRICS = ("MAD", "PTR", "ECR")


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


def load_pairs() -> pd.DataFrame:
    rewards = pd.read_csv(REWARD_INPUT)
    rewards = (
        rewards.loc[
            rewards["method"] == "Online Rubrics",
            ["training_step", "prompt_groups", "responses", *METRICS],
        ]
        .sort_values("training_step")
        .reset_index(drop=True)
    )
    expected_steps = list(range(1, 49))
    if rewards["training_step"].astype(int).tolist() != expected_steps:
        raise RuntimeError("Online reward input must contain steps 1 through 48")

    healthbench = pd.read_json(HEALTHBENCH_INPUT, lines=True)
    healthbench = (
        healthbench.loc[
            (healthbench["dataset"] == "healthbench")
            & (healthbench["method"] == "online"),
            ["global_step", "model_name", "final_mean", "n_prompts"],
        ]
        .rename(
            columns={
                "global_step": "training_step",
                "final_mean": "healthbench",
            }
        )
        .sort_values("training_step")
        .reset_index(drop=True)
    )
    if healthbench["training_step"].astype(int).tolist() != expected_steps:
        raise RuntimeError("HealthBench input must contain checkpoints 1 through 48")
    if healthbench["n_prompts"].astype(int).tolist() != [500] * 48:
        raise RuntimeError("every checkpoint must have 500 HealthBench prompts")

    pairs = rewards.merge(healthbench, on="training_step", validate="one_to_one")
    pairs.insert(1, "response_policy_version", pairs["training_step"] - 1)
    if pairs[[*METRICS, "healthbench"]].isna().any().any():
        raise RuntimeError("paired data contains missing values")
    return pairs


def compute_correlations(pairs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metric in METRICS:
        pearson = pearsonr(pairs[metric], pairs["healthbench"])
        spearman = spearmanr(pairs[metric], pairs["healthbench"])
        loo_rhos = []
        for omitted in range(len(pairs)):
            keep = np.arange(len(pairs)) != omitted
            loo_rhos.append(
                float(spearmanr(pairs.loc[keep, metric], pairs.loc[keep, "healthbench"]).statistic)
            )
        same_sign = sum(
            np.sign(value) == np.sign(float(spearman.statistic)) for value in loo_rhos
        )
        rows.append(
            {
                "predictor": metric,
                "outcome": "checkpoint_t_healthbench",
                "n": len(pairs),
                "pearson_r": float(pearson.statistic),
                "pearson_p": float(pearson.pvalue),
                "spearman_rho": float(spearman.statistic),
                "spearman_p": float(spearman.pvalue),
                "loo_same_sign": same_sign,
                "loo_total": len(loo_rhos),
                "loo_spearman_min": float(min(loo_rhos)),
                "loo_spearman_max": float(max(loo_rhos)),
            }
        )
    result = pd.DataFrame(rows)
    result["pearson_q_bh"] = benjamini_hochberg(result["pearson_p"])
    result["spearman_q_bh"] = benjamini_hochberg(result["spearman_p"])
    return result


def plot(pairs: pd.DataFrame, correlations: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 4.9), sharey=True, constrained_layout=True)
    scatter = None
    for axis, metric in zip(axes, METRICS, strict=True):
        x = pairs[metric].to_numpy(dtype=float)
        y = pairs["healthbench"].to_numpy(dtype=float)
        steps = pairs["training_step"].to_numpy(dtype=int)
        scatter = axis.scatter(
            x,
            y,
            c=steps,
            cmap="viridis",
            vmin=1,
            vmax=48,
            s=47,
            alpha=0.88,
            edgecolor="white",
            linewidth=0.35,
        )
        slope, intercept = np.polyfit(x, y, deg=1)
        grid = np.linspace(float(x.min()), float(x.max()), 200)
        axis.plot(grid, slope * grid + intercept, color="#313695", linewidth=1.7)

        row = correlations.loc[correlations["predictor"] == metric].iloc[0]
        axis.text(
            0.03,
            0.97,
            (
                f"Pearson r = {row['pearson_r']:+.3f} (p={row['pearson_p']:.3g})\n"
                f"Spearman rho = {row['spearman_rho']:+.3f} (p={row['spearman_p']:.3g})"
            ),
            transform=axis.transAxes,
            va="top",
            fontsize=9,
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "white",
                "alpha": 0.88,
                "edgecolor": "#cccccc",
            },
        )
        axis.set_title(metric)
        axis.set_xlabel(f"Actual training-reward {metric} at update t")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("HealthBench score of Online checkpoint t")
    if scatter is not None:
        colorbar = fig.colorbar(scatter, ax=axes, shrink=0.86, pad=0.015)
        colorbar.set_label("Checkpoint t")
    fig.suptitle(
        "Online Rubrics: actual reward at update t vs HealthBench of checkpoint t (n=48)"
    )
    fig.savefig(FIGURE_PNG, dpi=220, bbox_inches="tight")
    fig.savefig(FIGURE_SVG, format="svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    pairs = load_pairs()
    correlations = compute_correlations(pairs)
    pairs.to_csv(PAIRS_CSV, index=False)
    correlations.to_csv(CORRELATIONS_CSV, index=False)
    plot(pairs, correlations)

    summary = {
        "alignment": (
            "policy t-1 rollout -> actual normalized reward at optimizer update t -> "
            "HealthBench of resulting Online checkpoint t"
        ),
        "coverage": {
            "checkpoints": 48,
            "checkpoint_steps": pairs["training_step"].astype(int).tolist(),
            "healthbench_prompts_per_checkpoint": 500,
            "training_prompt_groups": int(pairs["prompt_groups"].sum()),
            "training_responses": int(pairs["responses"].sum()),
        },
        "healthbench_judge": (
            "Qwen/Qwen3-32B@9216db5781bf21249d130ec9da846c4624c16137"
        ),
        "correlations": correlations.to_dict(orient="records"),
        "interpretation_limit": (
            "checkpoints are serially related observations from one trajectory; ordinary "
            "correlation p-values do not establish independence or causality"
        ),
    }
    SUMMARY_JSON.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(correlations.to_string(index=False))


if __name__ == "__main__":
    main()
