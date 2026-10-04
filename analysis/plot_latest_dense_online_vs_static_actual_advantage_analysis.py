#!/usr/bin/env python3
"""Analyze trainer-equivalent GRPO advantages from actual training rewards.

This script reconstructs the scalar GRPO advantage used by the pinned veRL
configuration from immutable Online/Static training reward artifacts.  It does
not regenerate responses, rubrics, grades, rewards, or HealthBench judgments.

The first two diagnostics are defined in advantage space:

* Advantage MAD: mean absolute deviation of the 16 scalar advantages.
* Advantage PTR: fraction of the 120 response pairs with |A_i - A_j| <= 0.01.

ECR cannot be defined from one scalar advantage per response, so the third
panel deliberately retains the criterion-level ECR from the same reward
artifact and labels it ``Reward ECR``.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from fractions import Fraction
from itertools import combinations
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from dynamic_rubric.horizon.advantage import DEFAULT_EPSILON, grpo_scalar_advantages


ROOT = Path(__file__).resolve().parents[1]
ONLINE_RUN = (
    ROOT
    / "outputs/medicine/online_rubrics/seed-11/"
    "phase1-online-rubrics-medicine-full-dense-20260919-seed11"
)
STATIC_RUN = (
    ROOT
    / "outputs/medicine/static_r0_matched/seed-11/"
    "phase1-static-r0-medicine-qwen3-4b-matched-20260914"
)
REWARD_RESULTS = ROOT / "results/latest_dense_online_vs_static_actual_training_metrics_20260926"
OUTPUT = ROOT / "results/latest_dense_online_vs_static_actual_advantage_metrics_20260927"

ONLINE_HEALTHBENCH_PAIRS = REWARD_RESULTS / "online_dense_prompt_minmax_healthbench_pairs.csv"
STATIC_HEALTHBENCH_PAIRS = REWARD_RESULTS / "static_matched_prompt_minmax_healthbench_pairs.csv"

ONLINE_STEPS = tuple(range(1, 49))
STATIC_STEPS = tuple(range(1, 43))
STATIC_CHECKPOINT_STEPS = tuple(range(3, 43, 3))
RESPONSES_PER_PROMPT = 16
FULL_PROMPTS_PER_STEP = 96
REMAINDER_PROMPTS_PER_STEP = 60
STEPS_PER_EPOCH = 16
ADVANTAGE_PTR_THRESHOLD = 0.01

METRICS = ("advantage_MAD", "advantage_PTR", "reward_ECR")
METRIC_LABELS = {
    "advantage_MAD": "Advantage MAD",
    "advantage_PTR": "Advantage PTR",
    "reward_ECR": "Reward ECR",
}


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_group(rewards: list[Fraction], grade_matrix: list[list[int]]) -> dict[str, float]:
    if len(rewards) != RESPONSES_PER_PROMPT:
        raise RuntimeError(f"expected {RESPONSES_PER_PROMPT} responses, found {len(rewards)}")
    if len(grade_matrix) != RESPONSES_PER_PROMPT:
        raise RuntimeError("criterion grade matrix does not match response count")
    criterion_count = len(grade_matrix[0])
    if criterion_count == 0 or any(len(row) != criterion_count for row in grade_matrix):
        raise RuntimeError("criterion inventory is empty or changes within a prompt group")

    advantages = np.asarray(
        grpo_scalar_advantages(
            [float(value) for value in rewards],
            epsilon=DEFAULT_EPSILON,
            normalize_by_std=True,
        ),
        dtype=np.float64,
    )
    advantage_mean = float(advantages.mean())
    pairs = list(combinations(advantages.tolist(), 2))
    mixed_criteria = sum(
        0 < sum(response[index] for response in grade_matrix) < RESPONSES_PER_PROMPT
        for index in range(criterion_count)
    )
    return {
        "advantage_MAD": float(np.mean(np.abs(advantages - advantage_mean))),
        "advantage_PTR": sum(
            abs(left - right) <= ADVANTAGE_PTR_THRESHOLD for left, right in pairs
        )
        / len(pairs),
        "reward_ECR": mixed_criteria / criterion_count,
        "criterion_count": criterion_count,
        "reward_sample_std": float(np.std([float(value) for value in rewards], ddof=1)),
        "advantage_sample_std": float(np.std(advantages, ddof=1)),
        "max_abs_advantage_mean": abs(advantage_mean),
        "degenerate_advantage_group": float(np.max(np.abs(advantages)) <= 1e-12),
    }


def aggregate_step(method: str, step: int, groups: Iterable[dict[str, float]]) -> dict:
    rows = list(groups)
    expected_prompts = (
        REMAINDER_PROMPTS_PER_STEP
        if step % STEPS_PER_EPOCH == 0
        else FULL_PROMPTS_PER_STEP
    )
    if len(rows) != expected_prompts:
        raise RuntimeError(
            f"{method} step {step}: expected {expected_prompts} prompts, found {len(rows)}"
        )
    return {
        "method": method,
        "training_step": step,
        "response_policy_version": step - 1,
        "prompt_groups": len(rows),
        "responses": len(rows) * RESPONSES_PER_PROMPT,
        **{metric: float(np.mean([row[metric] for row in rows])) for metric in METRICS},
        "mean_criterion_count": float(np.mean([row["criterion_count"] for row in rows])),
        "mean_reward_sample_std": float(np.mean([row["reward_sample_std"] for row in rows])),
        "mean_advantage_sample_std": float(
            np.mean([row["advantage_sample_std"] for row in rows])
        ),
        "degenerate_advantage_group_rate": float(
            np.mean([row["degenerate_advantage_group"] for row in rows])
        ),
        "max_abs_group_advantage_mean": float(
            np.max([row["max_abs_advantage_mean"] for row in rows])
        ),
    }


def online_metrics() -> tuple[list[dict], list[dict], list[dict]]:
    steps: list[dict] = []
    groups_output: list[dict] = []
    sources: list[dict] = []
    step_root = ONLINE_RUN / "verl-run/online_steps"
    for step in ONLINE_STEPS:
        directory = step_root / f"step-{step:06d}"
        reward_path = directory / "rewards.jsonl"
        batch_path = directory / "batch.json"
        commit_path = directory / "commit.json"
        for path in (reward_path, batch_path, commit_path):
            if not path.is_file():
                raise RuntimeError(f"missing Online training artifact: {path}")

        batch = json.loads(batch_path.read_text(encoding="utf-8"))
        commit = json.loads(commit_path.read_text(encoding="utf-8"))
        if int(batch["current_policy"]["policy_version"]) != step - 1:
            raise RuntimeError(f"Online step {step}: response policy must be checkpoint t-1")
        if commit.get("state") != "committed" or int(commit["optimizer_update_index"]) != step:
            raise RuntimeError(f"Online step {step}: invalid commit record")

        grouped: dict[str, list[dict]] = defaultdict(list)
        for reward in read_jsonl(reward_path):
            grouped[str(reward["prompt_occurrence_id"])].append(reward)

        group_metrics: list[dict[str, float]] = []
        for prompt_id, prompt_rows in grouped.items():
            prompt_rows.sort(key=lambda row: int(row["rollout_index"]))
            if [int(row["rollout_index"]) for row in prompt_rows] != list(
                range(RESPONSES_PER_PROMPT)
            ):
                raise RuntimeError(f"Online step {step}, {prompt_id}: invalid rollout indices")
            criterion_ids = [str(item[0]) for item in prompt_rows[0]["grades"]]
            rewards: list[Fraction] = []
            grade_matrix: list[list[int]] = []
            for row in prompt_rows:
                if [str(item[0]) for item in row["grades"]] != criterion_ids:
                    raise RuntimeError(f"Online step {step}, {prompt_id}: criterion order changed")
                rewards.append(Fraction(int(row["numerator"]), int(row["denominator"])))
                grade_matrix.append([int(item[1]) for item in row["grades"]])
            summary = summarize_group(rewards, grade_matrix)
            group_metrics.append(summary)
            groups_output.append(
                {
                    "method": "Online Rubrics",
                    "training_step": step,
                    "response_policy_version": step - 1,
                    "prompt_group_id": prompt_id,
                    **summary,
                }
            )

        steps.append(aggregate_step("Online Rubrics", step, group_metrics))
        sources.append(
            {
                "method": "Online Rubrics",
                "training_step": step,
                "path": str(reward_path.relative_to(ROOT)),
                "sha256": sha256(reward_path),
                "bytes": reward_path.stat().st_size,
            }
        )
    return steps, groups_output, sources


def static_metrics() -> tuple[list[dict], list[dict], list[dict]]:
    steps: list[dict] = []
    groups_output: list[dict] = []
    sources: list[dict] = []
    rollout_root = STATIC_RUN / "verl-run/rollouts"
    for step in STATIC_STEPS:
        path = rollout_root / f"{step}.jsonl"
        if not path.is_file():
            raise RuntimeError(f"missing Static training artifact: {path}")
        grouped: dict[str, list[dict]] = defaultdict(list)
        for rollout in read_jsonl(path):
            if int(rollout["policy_step"]) != step:
                raise RuntimeError(f"Static step {step}: policy_step mismatch")
            grouped[str(rollout["prompt_id"])].append(rollout)

        group_metrics: list[dict[str, float]] = []
        for prompt_id, prompt_rows in grouped.items():
            prompt_rows.sort(key=lambda row: int(row["sample_index"]))
            if [int(row["sample_index"]) for row in prompt_rows] != list(
                range(RESPONSES_PER_PROMPT)
            ):
                raise RuntimeError(f"Static step {step}, {prompt_id}: invalid sample indices")
            rewards: list[Fraction] = []
            grade_matrix: list[list[int]] = []
            criterion_count = int(prompt_rows[0]["criterion_count"])
            for row in prompt_rows:
                probabilities = [
                    float(value) for value in json.loads(str(row["criterion_probabilities_json"]))
                ]
                if len(probabilities) != criterion_count or int(row["criterion_count"]) != criterion_count:
                    raise RuntimeError(f"Static step {step}, {prompt_id}: criterion layout changed")
                reward = Fraction(int(row["score_num"]), int(row["score_den"]))
                if abs(float(reward) - float(row["static_reward"])) > 1e-12:
                    raise RuntimeError(f"Static step {step}, {prompt_id}: exact reward mismatch")
                rewards.append(reward)
                grade_matrix.append([int(probability > 0.5) for probability in probabilities])
            summary = summarize_group(rewards, grade_matrix)
            group_metrics.append(summary)
            groups_output.append(
                {
                    "method": "Static R0 matched",
                    "training_step": step,
                    "response_policy_version": step - 1,
                    "prompt_group_id": prompt_id,
                    **summary,
                }
            )

        steps.append(aggregate_step("Static R0 matched", step, group_metrics))
        sources.append(
            {
                "method": "Static R0 matched",
                "training_step": step,
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    return steps, groups_output, sources


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


def compute_correlations(pairs: pd.DataFrame, outcome: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metric in METRICS:
        pearson = pearsonr(pairs[metric], pairs[outcome])
        spearman = spearmanr(pairs[metric], pairs[outcome])
        loo_rhos: list[float] = []
        for omitted in range(len(pairs)):
            keep = np.arange(len(pairs)) != omitted
            loo_rhos.append(
                float(spearmanr(pairs.loc[keep, metric], pairs.loc[keep, outcome]).statistic)
            )
        rows.append(
            {
                "predictor": metric,
                "predictor_label": METRIC_LABELS[metric],
                "outcome": outcome,
                "n": len(pairs),
                "pearson_r": float(pearson.statistic),
                "pearson_p": float(pearson.pvalue),
                "spearman_rho": float(spearman.statistic),
                "spearman_p": float(spearman.pvalue),
                "loo_same_sign": sum(
                    np.sign(value) == np.sign(float(spearman.statistic)) for value in loo_rhos
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


def trajectory_plot(metrics: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 4.9), constrained_layout=True)
    styles = {
        "Online Rubrics": {"color": "#3366cc", "marker": "o", "linestyle": "-"},
        "Static R0 matched": {"color": "#dd7711", "marker": "s", "linestyle": "--"},
    }
    labels = {
        "Online Rubrics": "Online Rubrics (steps 1–48)",
        "Static R0 matched": "Static R0 (steps 1–42)",
    }
    for axis, metric in zip(axes, METRICS, strict=True):
        for method in styles:
            frame = metrics.loc[metrics["method"] == method]
            style = styles[method]
            axis.plot(
                frame["training_step"],
                frame[metric],
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                markersize=3.2,
                markeredgewidth=0,
                linewidth=1.7,
                label=labels[method],
            )
        axis.set_title(METRIC_LABELS[metric])
        axis.set_xlabel("Optimizer update t")
        axis.set_ylabel(METRIC_LABELS[metric])
        axis.set_xlim(0.5, 48.5)
        axis.set_xticks([1, 6, 12, 18, 24, 30, 36, 42, 48])
        axis.grid(alpha=0.22)
    axes[0].legend(frameon=False, fontsize=8.4, loc="best")
    fig.suptitle("Actual-training GRPO advantage diagnostics: Online vs Static")
    fig.savefig(OUTPUT / "online_vs_static_actual_advantage_mad_ptr_reward_ecr.png", dpi=220)
    fig.savefig(OUTPUT / "online_vs_static_actual_advantage_mad_ptr_reward_ecr.svg")
    plt.close(fig)


def correlation_plot(
    pairs: pd.DataFrame,
    correlations: pd.DataFrame,
    *,
    title: str,
    step_column: str,
    xlabel_suffix: str,
    stem: str,
    cmap: str,
) -> None:
    outcome = "healthbench_prompt_minmax"
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 4.9), sharey=True, constrained_layout=True)
    scatter = None
    steps = pairs[step_column].to_numpy(dtype=int)
    for axis, metric in zip(axes, METRICS, strict=True):
        x = pairs[metric].to_numpy(dtype=float)
        y = pairs[outcome].to_numpy(dtype=float)
        scatter = axis.scatter(
            x,
            y,
            c=steps,
            cmap=cmap,
            vmin=int(steps.min()),
            vmax=int(steps.max()),
            s=50,
            alpha=0.88,
            edgecolor="white",
            linewidth=0.4,
        )
        if np.unique(x).size >= 2:
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
        axis.set_title(METRIC_LABELS[metric])
        axis.set_xlabel(f"{METRIC_LABELS[metric]} {xlabel_suffix}")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Checkpoint t HealthBench (per-prompt min-max mean)")
    if scatter is not None:
        colorbar = fig.colorbar(scatter, ax=axes, shrink=0.86, pad=0.015)
        colorbar.set_label("Checkpoint t")
    fig.suptitle(title)
    fig.savefig(OUTPUT / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUTPUT / f"{stem}.svg", format="svg", bbox_inches="tight")
    plt.close(fig)


def load_main_pairs(metrics: pd.DataFrame, method: str, healthbench_path: Path) -> pd.DataFrame:
    hb = pd.read_csv(healthbench_path)[
        [
            "training_step",
            "model_name",
            "healthbench_prompt_minmax",
            "healthbench_official_raw_mean",
            "n_prompts",
        ]
    ]
    selected = metrics.loc[
        metrics["method"] == method,
        [
            "training_step",
            "response_policy_version",
            "prompt_groups",
            "responses",
            *METRICS,
        ],
    ]
    pairs = selected.merge(hb, on="training_step", validate="one_to_one")
    if not (pairs["n_prompts"] == 500).all():
        raise RuntimeError("every HealthBench checkpoint score must contain 500 prompts")
    return pairs.sort_values("training_step").reset_index(drop=True)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    online_steps, online_groups, online_sources = online_metrics()
    static_steps, static_groups, static_sources = static_metrics()
    metrics = pd.DataFrame(online_steps + static_steps)
    groups = pd.DataFrame(online_groups + static_groups)
    sources = pd.DataFrame(online_sources + static_sources)

    if metrics.loc[metrics["method"] == "Online Rubrics", "training_step"].tolist() != list(
        ONLINE_STEPS
    ):
        raise RuntimeError("Online coverage must be steps 1 through 48")
    if metrics.loc[
        metrics["method"] == "Static R0 matched", "training_step"
    ].tolist() != list(STATIC_STEPS):
        raise RuntimeError("Static coverage must be steps 1 through 42")
    # Extremely low-variance rational rewards can leave a sub-nanoscopic
    # centering residual after conversion to float and division by epsilon.
    if float(groups["max_abs_advantage_mean"].max()) > 1e-8:
        raise RuntimeError("reconstructed group advantages are not centered at zero")

    metrics.to_csv(OUTPUT / "actual_training_advantage_metrics_by_step.csv", index=False)
    groups.to_csv(OUTPUT / "actual_training_advantage_metrics_by_prompt_group.csv", index=False)
    sources.to_csv(OUTPUT / "source_artifact_hashes.csv", index=False)
    trajectory_plot(metrics)

    online_pairs = load_main_pairs(metrics, "Online Rubrics", ONLINE_HEALTHBENCH_PAIRS)
    online_correlations = compute_correlations(online_pairs, "healthbench_prompt_minmax")
    online_pairs.to_csv(OUTPUT / "online_advantage_checkpoint_t_healthbench_pairs.csv", index=False)
    online_correlations.to_csv(
        OUTPUT / "online_advantage_checkpoint_t_healthbench_correlations.csv", index=False
    )
    correlation_plot(
        online_pairs,
        online_correlations,
        title="Online: actual GRPO advantage at update t vs checkpoint t HealthBench (n=48)",
        step_column="training_step",
        xlabel_suffix="at update t",
        stem="online_advantage_checkpoint_t_healthbench_correlation",
        cmap="viridis",
    )

    static_pairs = load_main_pairs(metrics, "Static R0 matched", STATIC_HEALTHBENCH_PAIRS)
    if static_pairs["training_step"].astype(int).tolist() != list(STATIC_CHECKPOINT_STEPS):
        raise RuntimeError("Static evaluated checkpoints must be steps 3, 6, ..., 42")
    static_correlations = compute_correlations(static_pairs, "healthbench_prompt_minmax")
    static_pairs.to_csv(OUTPUT / "static_advantage_checkpoint_t_healthbench_pairs.csv", index=False)
    static_correlations.to_csv(
        OUTPUT / "static_advantage_checkpoint_t_healthbench_correlations.csv", index=False
    )
    correlation_plot(
        static_pairs,
        static_correlations,
        title="Static R0: actual GRPO advantage at update t vs checkpoint t HealthBench (n=14)",
        step_column="training_step",
        xlabel_suffix="at update t",
        stem="static_advantage_checkpoint_t_healthbench_correlation",
        cmap="plasma",
    )

    # Same-index appendix: the reward/advantage row at update t+1 was generated
    # with checkpoint t's rollouts and Online rubric state R_t.
    appendix_metrics = metrics.loc[
        (metrics["method"] == "Online Rubrics") & metrics["training_step"].between(2, 48),
        ["training_step", "prompt_groups", "responses", *METRICS],
    ].rename(columns={"training_step": "advantage_update_step"})
    appendix_metrics["checkpoint_step"] = appendix_metrics["advantage_update_step"] - 1
    appendix_hb = online_pairs[
        [
            "training_step",
            "model_name",
            "healthbench_prompt_minmax",
            "healthbench_official_raw_mean",
            "n_prompts",
        ]
    ].rename(columns={"training_step": "checkpoint_step"})
    appendix_pairs = appendix_metrics.merge(
        appendix_hb, on="checkpoint_step", validate="one_to_one"
    ).sort_values("checkpoint_step")
    if appendix_pairs["checkpoint_step"].astype(int).tolist() != list(range(1, 48)):
        raise RuntimeError("appendix must contain R_t/checkpoint t pairs for t=1 through 47")
    appendix_correlations = compute_correlations(
        appendix_pairs, "healthbench_prompt_minmax"
    )
    appendix_pairs.to_csv(OUTPUT / "online_rt_advantage_checkpoint_t_healthbench_pairs.csv", index=False)
    appendix_correlations.to_csv(
        OUTPUT / "online_rt_advantage_checkpoint_t_healthbench_correlations.csv", index=False
    )
    correlation_plot(
        appendix_pairs,
        appendix_correlations,
        title="Appendix: Online R_t advantage (update t+1) vs checkpoint t HealthBench (n=47)",
        step_column="checkpoint_step",
        xlabel_suffix="from R_t / update t+1",
        stem="online_rt_advantage_checkpoint_t_healthbench_correlation",
        cmap="cividis",
    )

    method_summaries: list[dict] = []
    for method, frame in metrics.groupby("method", sort=False):
        method_summary: dict[str, object] = {
            "method": method,
            "steps": [int(value) for value in frame["training_step"]],
            "prompt_groups": int(frame["prompt_groups"].sum()),
            "responses": int(frame["responses"].sum()),
        }
        for metric in METRICS:
            method_summary[metric] = {
                "mean_across_steps": float(frame[metric].mean()),
                "first_step": float(frame.iloc[0][metric]),
                "last_available_step": float(frame.iloc[-1][metric]),
                "min": float(frame[metric].min()),
                "max": float(frame[metric].max()),
            }
        method_summary["degenerate_advantage_group_rate_overall"] = float(
            groups.loc[groups["method"] == method, "degenerate_advantage_group"].mean()
        )
        method_summaries.append(method_summary)

    summary = {
        "schema_version": 1,
        "analysis": "actual-training GRPO advantage MAD/PTR plus criterion-side reward ECR",
        "advantage_reconstruction": {
            "formula": "A_i = (r_i - group_mean) / (sample_std + 1e-6)",
            "group_size": RESPONSES_PER_PROMPT,
            "normalize_by_std": True,
            "epsilon": DEFAULT_EPSILON,
            "source": "actual normalized rewards in immutable committed training artifacts",
            "caveat": (
                "The per-response trainer tensor was not dumped. Values are reconstructed with the "
                "repository's pinned pure-Python reproduction of the veRL scalar formula; this is "
                "formula/configuration equivalent, not a claim of bitwise tensor identity."
            ),
        },
        "definitions": {
            "advantage_MAD": "group mean absolute deviation of the 16 reconstructed advantages",
            "advantage_PTR": (
                "fraction of 120 response pairs with absolute advantage difference <= 0.01"
            ),
            "reward_ECR": (
                "fraction of rubric criteria whose hard grade varies across the 16 responses; "
                "retained from the same reward artifact because scalar advantage has no criterion axis"
            ),
            "step_aggregation": "equal-weight mean over prompt groups within each optimizer update",
            "healthbench": (
                "per-prompt theoretical min-max score, then arithmetic mean over 500 prompts; "
                "this is not the official HealthBench aggregate"
            ),
        },
        "main_alignment": (
            "Online R_(t-1) / policy checkpoint t-1 -> actual reward and advantage at optimizer "
            "update t -> checkpoint t HealthBench"
        ),
        "coverage": {
            "online_updates_and_checkpoints": 48,
            "static_updates": 42,
            "static_evaluated_checkpoints": list(STATIC_CHECKPOINT_STEPS),
            "healthbench_prompts_per_checkpoint": 500,
        },
        "method_summaries": method_summaries,
        "online_correlations": online_correlations.to_dict(orient="records"),
        "static_correlations": static_correlations.to_dict(orient="records"),
        "appendix_same_index_rt_correlations": appendix_correlations.to_dict(orient="records"),
        "limits": [
            "Checkpoint observations are serially related, so p/q values are descriptive.",
            "Static has only 14 evaluated checkpoints and therefore low statistical power.",
            "Advantage standardization removes most absolute reward-scale information in non-degenerate groups.",
            "Reward ECR is a criterion-level companion metric, not a scalar-advantage metric.",
        ],
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print("ONLINE")
    print(online_correlations.to_string(index=False))
    print("\nSTATIC")
    print(static_correlations.to_string(index=False))
    print("\nAPPENDIX R_t")
    print(appendix_correlations.to_string(index=False))
    print("\nTRAJECTORY SUMMARY")
    print(metrics.groupby("method")[list(METRICS)].agg(["mean", "min", "max"]).to_string())
    print(f"\noutputs: {OUTPUT}")


if __name__ == "__main__":
    main()
