#!/usr/bin/env python3
"""Appendix analysis: correlate R_t reward metrics with checkpoint t HealthBench.

The committed training-reward row for optimizer update ``u`` was produced by
applying ``R_{u-1}`` to the rollout from policy/checkpoint ``u-1``.  Therefore,
the same-index appendix pairing ``R_t`` with checkpoint ``t`` uses reward row
``u=t+1`` and checkpoint row ``t``.  The last usable pair is t=47 because the
run has no optimizer update 49 in which R_48 was used.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_online_dense_prompt_minmax_healthbench_correlation import (
    METRICS,
    RESULTS,
    compute_correlations,
    load_checkpoint_scores,
    load_prompt_bounds,
)


REWARD_INPUT = RESULTS / "actual_training_mad_ptr_ecr_by_step.csv"
PAIRS_CSV = RESULTS / "online_dense_rt_checkpoint_t_prompt_minmax_healthbench_pairs.csv"
CORRELATIONS_CSV = (
    RESULTS / "online_dense_rt_checkpoint_t_prompt_minmax_healthbench_correlations.csv"
)
SUMMARY_JSON = (
    RESULTS / "online_dense_rt_checkpoint_t_prompt_minmax_healthbench_summary.json"
)
FIGURE_PNG = RESULTS / "online_dense_rt_checkpoint_t_prompt_minmax_healthbench_correlation.png"
FIGURE_SVG = RESULTS / "online_dense_rt_checkpoint_t_prompt_minmax_healthbench_correlation.svg"


def load_same_index_pairs(checkpoint_scores: pd.DataFrame) -> pd.DataFrame:
    rewards = pd.read_csv(REWARD_INPUT)
    rewards = (
        rewards.loc[
            rewards["method"] == "Online Rubrics",
            ["training_step", "prompt_groups", "responses", *METRICS],
        ]
        .rename(columns={"training_step": "reward_update_step"})
        .sort_values("reward_update_step")
        .reset_index(drop=True)
    )
    if rewards["reward_update_step"].astype(int).tolist() != list(range(1, 49)):
        raise RuntimeError("Online rewards must contain optimizer updates 1 through 48")

    # Reward row t+1 was generated with R_t on the rollout from checkpoint t.
    rewards["rubric_step"] = rewards["reward_update_step"] - 1
    rewards = rewards.loc[rewards["rubric_step"].between(1, 47)].copy()

    checkpoints = checkpoint_scores.rename(columns={"training_step": "checkpoint_step"})
    checkpoints = checkpoints.loc[checkpoints["checkpoint_step"].between(1, 47)].copy()
    pairs = rewards.merge(
        checkpoints,
        left_on="rubric_step",
        right_on="checkpoint_step",
        validate="one_to_one",
    ).sort_values("checkpoint_step")
    if pairs["checkpoint_step"].astype(int).tolist() != list(range(1, 48)):
        raise RuntimeError("expected same-index R_t/checkpoint t pairs for t=1 through 47")
    if not (pairs["reward_update_step"] == pairs["checkpoint_step"] + 1).all():
        raise RuntimeError("reward update must equal checkpoint step + 1")
    return pairs.reset_index(drop=True)


def plot(pairs: pd.DataFrame, correlations: pd.DataFrame) -> None:
    outcome = "healthbench_prompt_minmax"
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 4.9), sharey=True, constrained_layout=True)
    scatter = None
    for axis, metric in zip(axes, METRICS, strict=True):
        x = pairs[metric].to_numpy(dtype=float)
        y = pairs[outcome].to_numpy(dtype=float)
        steps = pairs["checkpoint_step"].to_numpy(dtype=int)
        scatter = axis.scatter(
            x,
            y,
            c=steps,
            cmap="viridis",
            vmin=1,
            vmax=47,
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
        axis.set_xlabel(f"Actual reward {metric} from R_t (update t+1)")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Per-prompt min-max HealthBench mean of checkpoint t")
    if scatter is not None:
        colorbar = fig.colorbar(scatter, ax=axes, shrink=0.86, pad=0.015)
        colorbar.set_label("Checkpoint / rubric index t")
    fig.suptitle("Appendix: R_t reward vs checkpoint t HealthBench (t=1..47)")
    fig.savefig(FIGURE_PNG, dpi=220, bbox_inches="tight")
    fig.savefig(FIGURE_SVG, format="svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    bounds = load_prompt_bounds()
    _, checkpoint_scores = load_checkpoint_scores(bounds)
    pairs = load_same_index_pairs(checkpoint_scores)
    correlations = compute_correlations(pairs)
    correlations["outcome"] = "checkpoint_t_healthbench_prompt_minmax"

    pairs.to_csv(PAIRS_CSV, index=False)
    correlations.to_csv(CORRELATIONS_CSV, index=False)
    plot(pairs, correlations)

    summary = {
        "analysis": "appendix same-index R_t reward metrics vs checkpoint t HealthBench",
        "alignment": (
            "reward row at optimizer update t+1 uses policy/checkpoint t rollout and R_t; "
            "pair that reward's MAD/PTR/ECR with prompt-min-max HealthBench of checkpoint t"
        ),
        "coverage": {
            "pairs": len(pairs),
            "checkpoint_and_rubric_steps": [1, 47],
            "healthbench_prompts_per_checkpoint": 500,
            "excluded_checkpoint": 48,
            "exclusion_reason": "no optimizer update 49 exists, so no actual training reward from R_48",
        },
        "normalization": (
            "same per-prompt theoretical min-max HealthBench transformation as the main analysis"
        ),
        "correlations": correlations.to_dict(orient="records"),
        "interpretation_limit": (
            "This is a same-index descriptive pairing, not the causal training alignment: "
            "R_t is generated after checkpoint t already exists."
        ),
    }
    SUMMARY_JSON.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(correlations.to_string(index=False))


if __name__ == "__main__":
    main()
