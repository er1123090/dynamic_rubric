#!/usr/bin/env python3
"""Correlate actual Online rewards with per-prompt min-max HealthBench scores."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/latest_dense_online_vs_static_actual_training_metrics_20260926"
EVAL_ROOT = (
    ROOT
    / "outputs/policy_eval/medicine_dense_all48_healthbench500_20260926"
    / "full-6b52dfbc30980b62"
)
REWARD_INPUT = RESULTS / "actual_training_mad_ptr_ecr_by_step.csv"
PROMPT_INPUT = EVAL_ROOT / "prepared/prompts.jsonl"
PROMPT_SCORE_INPUT = EVAL_ROOT / "summary/prompt_scores.jsonl"

PROMPT_SCORES_CSV = RESULTS / "online_dense_healthbench_prompt_minmax_scores.csv"
PAIRS_CSV = RESULTS / "online_dense_prompt_minmax_healthbench_pairs.csv"
CORRELATIONS_CSV = RESULTS / "online_dense_prompt_minmax_healthbench_correlations.csv"
SUMMARY_JSON = RESULTS / "online_dense_prompt_minmax_healthbench_correlation_summary.json"
FIGURE_PNG = RESULTS / "online_dense_prompt_minmax_healthbench_correlation.png"
FIGURE_SVG = RESULTS / "online_dense_prompt_minmax_healthbench_correlation.svg"

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


def load_prompt_bounds() -> pd.DataFrame:
    prompts = pd.read_json(PROMPT_INPUT, lines=True)
    rows: list[dict[str, object]] = []
    for prompt in prompts.to_dict(orient="records"):
        points = [float(item["points"]) for item in prompt["criteria"]]
        positive_total = sum(point for point in points if point > 0)
        negative_total = sum(point for point in points if point < 0)
        if positive_total <= 0:
            raise RuntimeError(f"prompt has no positive denominator: {prompt['prompt_id']}")
        lower_bound = negative_total / positive_total
        rows.append(
            {
                "prompt_id": prompt["prompt_id"],
                "positive_point_total": positive_total,
                "negative_point_total": negative_total,
                "theoretical_min": lower_bound,
                "theoretical_max": 1.0,
            }
        )
    result = pd.DataFrame(rows)
    if len(result) != 500 or result["prompt_id"].duplicated().any():
        raise RuntimeError("expected 500 unique HealthBench prompts")
    return result


def load_checkpoint_scores(bounds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    prompt_scores = pd.read_json(PROMPT_SCORE_INPUT, lines=True)
    prompt_scores = prompt_scores.loc[
        (prompt_scores["dataset"] == "healthbench")
        & (prompt_scores["model_method"] == "online"),
        ["global_step", "model_name", "prompt_id", "raw_score_unclipped"],
    ].copy()
    if len(prompt_scores) != 48 * 500:
        raise RuntimeError(f"expected 24,000 prompt scores, found {len(prompt_scores)}")
    prompt_scores = prompt_scores.merge(bounds, on="prompt_id", validate="many_to_one")
    prompt_scores["prompt_minmax_score"] = (
        prompt_scores["raw_score_unclipped"] - prompt_scores["theoretical_min"]
    ) / (prompt_scores["theoretical_max"] - prompt_scores["theoretical_min"])
    tolerance = 1e-12
    if (
        (prompt_scores["prompt_minmax_score"] < -tolerance)
        | (prompt_scores["prompt_minmax_score"] > 1 + tolerance)
    ).any():
        raise RuntimeError("per-prompt min-max score fell outside [0, 1]")
    prompt_scores["prompt_minmax_score"] = prompt_scores["prompt_minmax_score"].clip(0, 1)

    checkpoint_scores = (
        prompt_scores.groupby(["global_step", "model_name"], as_index=False)
        .agg(
            healthbench_prompt_minmax=("prompt_minmax_score", "mean"),
            healthbench_official_raw_mean=("raw_score_unclipped", "mean"),
            n_prompts=("prompt_id", "size"),
        )
        .rename(columns={"global_step": "training_step"})
        .sort_values("training_step")
        .reset_index(drop=True)
    )
    if checkpoint_scores["training_step"].astype(int).tolist() != list(range(1, 49)):
        raise RuntimeError("expected checkpoints 1 through 48")
    if checkpoint_scores["n_prompts"].astype(int).tolist() != [500] * 48:
        raise RuntimeError("each checkpoint must contain 500 prompts")
    return prompt_scores, checkpoint_scores


def load_pairs(checkpoint_scores: pd.DataFrame) -> pd.DataFrame:
    rewards = pd.read_csv(REWARD_INPUT)
    rewards = (
        rewards.loc[
            rewards["method"] == "Online Rubrics",
            ["training_step", "prompt_groups", "responses", *METRICS],
        ]
        .sort_values("training_step")
        .reset_index(drop=True)
    )
    if rewards["training_step"].astype(int).tolist() != list(range(1, 49)):
        raise RuntimeError("Online rewards must contain steps 1 through 48")
    pairs = rewards.merge(checkpoint_scores, on="training_step", validate="one_to_one")
    pairs.insert(1, "response_policy_version", pairs["training_step"] - 1)
    return pairs


def compute_correlations(pairs: pd.DataFrame) -> pd.DataFrame:
    outcome = "healthbench_prompt_minmax"
    rows: list[dict[str, object]] = []
    for metric in METRICS:
        pearson = pearsonr(pairs[metric], pairs[outcome])
        spearman = spearmanr(pairs[metric], pairs[outcome])
        loo_rhos = []
        for omitted in range(len(pairs)):
            keep = np.arange(len(pairs)) != omitted
            loo_rhos.append(
                float(spearmanr(pairs.loc[keep, metric], pairs.loc[keep, outcome]).statistic)
            )
        rows.append(
            {
                "predictor": metric,
                "outcome": "checkpoint_t_healthbench_prompt_minmax",
                "n": len(pairs),
                "pearson_r": float(pearson.statistic),
                "pearson_p": float(pearson.pvalue),
                "spearman_rho": float(spearman.statistic),
                "spearman_p": float(spearman.pvalue),
                "loo_same_sign": sum(
                    np.sign(value) == np.sign(float(spearman.statistic))
                    for value in loo_rhos
                ),
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
    outcome = "healthbench_prompt_minmax"
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 4.9), sharey=True, constrained_layout=True)
    scatter = None
    for axis, metric in zip(axes, METRICS, strict=True):
        x = pairs[metric].to_numpy(dtype=float)
        y = pairs[outcome].to_numpy(dtype=float)
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
    axes[0].set_ylabel("Per-prompt min-max HealthBench mean of checkpoint t")
    if scatter is not None:
        colorbar = fig.colorbar(scatter, ax=axes, shrink=0.86, pad=0.015)
        colorbar.set_label("Checkpoint t")
    fig.suptitle(
        "Online Rubrics: actual reward vs per-prompt min-max HealthBench (n=48)"
    )
    fig.savefig(FIGURE_PNG, dpi=220, bbox_inches="tight")
    fig.savefig(FIGURE_SVG, format="svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    bounds = load_prompt_bounds()
    prompt_scores, checkpoint_scores = load_checkpoint_scores(bounds)
    pairs = load_pairs(checkpoint_scores)
    correlations = compute_correlations(pairs)

    prompt_scores.to_csv(PROMPT_SCORES_CSV, index=False)
    pairs.to_csv(PAIRS_CSV, index=False)
    correlations.to_csv(CORRELATIONS_CSV, index=False)
    plot(pairs, correlations)

    summary = {
        "normalization": {
            "prompt_theoretical_min": "sum(negative points) / sum(positive points)",
            "prompt_theoretical_max": 1.0,
            "transformation": "(raw_score - theoretical_min) / (1 - theoretical_min)",
            "aggregation": "arithmetic mean of 500 transformed prompt scores per checkpoint",
            "official_healthbench_metric": False,
        },
        "prompt_inventory": {
            "prompts": len(bounds),
            "prompts_with_negative_points": int((bounds["negative_point_total"] < 0).sum()),
            "prompts_without_negative_points": int((bounds["negative_point_total"] == 0).sum()),
            "theoretical_min_range": [
                float(bounds["theoretical_min"].min()),
                float(bounds["theoretical_min"].max()),
            ],
        },
        "alignment": (
            "policy t-1 rollout -> actual normalized reward at optimizer update t -> "
            "per-prompt min-max HealthBench mean of checkpoint t"
        ),
        "coverage": {"checkpoints": 48, "healthbench_prompts_per_checkpoint": 500},
        "checkpoint_score_range": [
            float(pairs["healthbench_prompt_minmax"].min()),
            float(pairs["healthbench_prompt_minmax"].max()),
        ],
        "correlations": correlations.to_dict(orient="records"),
        "interpretation_limit": (
            "This custom normalization changes prompt weighting and is not the official "
            "HealthBench aggregate; checkpoints are serially related observations."
        ),
    }
    SUMMARY_JSON.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(correlations.to_string(index=False))
    print(
        pairs[["healthbench_official_raw_mean", "healthbench_prompt_minmax"]]
        .agg(["min", "max", "mean"])
        .to_string()
    )


if __name__ == "__main__":
    main()
