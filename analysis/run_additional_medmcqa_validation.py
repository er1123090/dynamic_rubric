#!/usr/bin/env python3
"""Repeat the reward/advantage trajectory analyses with MedMCQA accuracy.

The analysis is conditional on one observed OnlineRubrics training trajectory.
It preserves the causal ordering

    policy checkpoint t-1 -> update t metric -> checkpoint t.

Four related questions are reported for normalized reward and GRPO advantage:

1. Does the update-t metric move with accuracy at checkpoint t?
2. Does that association remain after a prespecified linear step adjustment?
3. Does the update-t metric move with the one-update change t-1 -> t?
4. Does the update-(s+1) metric predict checkpoint s+3 after controlling for
   checkpoint s accuracy and the starting step?

Prompt-paired bootstrap intervals resample the same 4,183 MedMCQA IDs across
all checkpoints.  They quantify benchmark-item composition uncertainty only;
they do not capture training-seed or checkpoint-generation uncertainty.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from run_additional_healthbench_validation import (
    nested_partial_r2,
    ols_hac,
    partial_correlation,
    rowwise_correlation,
    rowwise_step_partial,
    zscore,
)


ROOT = Path(__file__).resolve().parents[1]
EVAL_ROOT = ROOT / "outputs/policy_eval/medicine_dense_all48_medmcqa_20260928"
REWARD_RESULTS = ROOT / "results/latest_dense_online_vs_static_actual_training_metrics_20260926"
ADVANTAGE_RESULTS = ROOT / "results/latest_dense_online_vs_static_actual_advantage_metrics_20260927"
OUTPUT_ROOT = ROOT / "results/medmcqa_additional_validation_20260928"
REWARD_METRICS = REWARD_RESULTS / "actual_training_mad_ptr_ecr_by_step.csv"
ADVANTAGE_METRICS = ADVANTAGE_RESULTS / "actual_training_advantage_metrics_by_step.csv"

BOOTSTRAP_REPLICATES = 5_000
BOOTSTRAP_SEED = 20260928
HAC_LAGS = 3
COLOR = "#2563eb"


@dataclass(frozen=True)
class Representation:
    key: str
    title: str
    metrics_path: Path
    metrics: tuple[str, str, str]
    labels: tuple[str, str, str]


REPRESENTATIONS = (
    Representation(
        key="normalized_reward",
        title="Normalized reward",
        metrics_path=REWARD_METRICS,
        metrics=("MAD", "PTR", "ECR"),
        labels=("MAD", "PTR", "ECR"),
    ),
    Representation(
        key="advantage",
        title="GRPO advantage",
        metrics_path=ADVANTAGE_METRICS,
        metrics=("advantage_MAD", "advantage_PTR", "reward_ECR"),
        labels=("Advantage MAD", "Advantage PTR", "Reward ECR"),
    ),
)


def load_medmcqa() -> dict[str, object]:
    manifest = json.loads((EVAL_ROOT / "manifest.json").read_text(encoding="utf-8"))
    if not manifest.get("complete"):
        raise RuntimeError("MedMCQA checkpoint evaluation is not complete")
    if manifest.get("completed_checkpoints") != list(range(1, 49)):
        raise RuntimeError("MedMCQA evaluation must cover checkpoints 1 through 48")

    prompt_ids: list[str] | None = None
    rows: list[np.ndarray] = []
    for step in range(1, 49):
        path = EVAL_ROOT / "prompt_scores" / f"checkpoint_{step:03d}.csv"
        frame = pd.read_csv(path)
        current_ids = frame["prompt_id"].astype(str).tolist()
        if len(frame) != 4_183 or not frame["correct"].isin([0, 1]).all():
            raise RuntimeError(f"invalid prompt-level MedMCQA result: {path}")
        if prompt_ids is None:
            prompt_ids = current_ids
        elif current_ids != prompt_ids:
            raise RuntimeError(f"prompt IDs are not paired at checkpoint {step}")
        rows.append(frame["correct"].to_numpy(dtype=float))
    values = np.stack(rows)
    observed = values.mean(axis=1)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    counts = rng.multinomial(
        values.shape[1],
        np.full(values.shape[1], 1.0 / values.shape[1]),
        size=BOOTSTRAP_REPLICATES,
    ).astype(np.float32)
    bootstrap = (values.astype(np.float32) @ counts.T / values.shape[1]).T
    return {
        "steps": np.arange(1, 49, dtype=int),
        "prompt_ids": prompt_ids,
        "values": values,
        "accuracy": observed,
        "bootstrap": bootstrap,
        "manifest": manifest,
    }


def load_metrics(representation: Representation) -> pd.DataFrame:
    frame = pd.read_csv(representation.metrics_path)
    frame = frame.loc[frame["method"] == "Online Rubrics"].sort_values("training_step")
    if frame["training_step"].astype(int).tolist() != list(range(1, 49)):
        raise RuntimeError(f"{representation.key} metrics must cover updates 1 through 48")
    return frame.reset_index(drop=True)


def interval(values: np.ndarray) -> tuple[float, float]:
    low, high = np.quantile(np.asarray(values, dtype=float), [0.025, 0.975])
    return float(low), float(high)


def rowwise_fixed_control_partial(
    y_rows: np.ndarray, metric: np.ndarray, controls: np.ndarray
) -> np.ndarray:
    design = np.column_stack([np.ones(len(metric)), controls])
    residual_maker = np.eye(len(metric)) - design @ np.linalg.pinv(design)
    metric_residual = residual_maker @ metric
    y_residual = y_rows @ residual_maker.T
    return rowwise_correlation(y_residual, metric_residual)


def bootstrap_adjusted_beta(
    outcomes: np.ndarray,
    metric: np.ndarray,
    baseline: np.ndarray,
    steps: np.ndarray,
) -> np.ndarray:
    """Standardized metric beta for each prompt bootstrap replicate."""

    metric_z = zscore(metric)
    step_z = zscore(steps)
    betas = np.empty(outcomes.shape[0], dtype=float)
    for index, (outcome, start) in enumerate(zip(outcomes, baseline)):
        design = np.column_stack(
            [np.ones(len(metric_z)), metric_z, zscore(start), step_z]
        )
        coefficients = np.linalg.lstsq(design, zscore(outcome), rcond=None)[0]
        betas[index] = coefficients[1]
    return betas


def trajectory_tables(data: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame]:
    steps = np.asarray(data["steps"])
    observed = np.asarray(data["accuracy"])
    bootstrap = np.asarray(data["bootstrap"])
    low, high = np.quantile(bootstrap, [0.025, 0.975], axis=0)
    checkpoint = pd.DataFrame(
        {
            "checkpoint": steps,
            "accuracy": observed,
            "correct": np.rint(observed * 4_183).astype(int),
            "prompt_bootstrap_ci_low": low,
            "prompt_bootstrap_ci_high": high,
        }
    )

    difference_rows: list[dict[str, object]] = []
    for horizon in (1, 3):
        for start in range(1, 49 - horizon):
            start_index = start - 1
            end_index = start_index + horizon
            differences = bootstrap[:, end_index] - bootstrap[:, start_index]
            ci_low, ci_high = interval(differences)
            difference_rows.append(
                {
                    "horizon": horizon,
                    "start_checkpoint": start,
                    "end_checkpoint": start + horizon,
                    "accuracy_difference": observed[end_index] - observed[start_index],
                    "prompt_bootstrap_ci_low": ci_low,
                    "prompt_bootstrap_ci_high": ci_high,
                    "probability_positive": float(np.mean(differences > 0)),
                    "ci_excludes_zero": bool(ci_low > 0 or ci_high < 0),
                }
            )
    return checkpoint, pd.DataFrame(difference_rows)


def analyze_representation(
    representation: Representation, data: dict[str, object]
) -> dict[str, pd.DataFrame]:
    metric_frame = load_metrics(representation)
    steps = np.asarray(data["steps"], dtype=float)
    accuracy = np.asarray(data["accuracy"], dtype=float)
    bootstrap = np.asarray(data["bootstrap"], dtype=float)
    current_rows: list[dict[str, object]] = []
    step_rows: list[dict[str, object]] = []
    one_step_rows: list[dict[str, object]] = []
    future_rows: list[dict[str, object]] = []

    for metric_name, label in zip(representation.metrics, representation.labels):
        metric = metric_frame[metric_name].to_numpy(dtype=float)

        pearson = float(pearsonr(metric, accuracy).statistic)
        spearman = float(spearmanr(metric, accuracy).statistic)
        boot_pearson = rowwise_correlation(bootstrap, metric)
        boot_spearman = rowwise_correlation(bootstrap, metric, rank=True)
        pearson_low, pearson_high = interval(boot_pearson)
        spearman_low, spearman_high = interval(boot_spearman)
        current_rows.append(
            {
                "representation": representation.key,
                "metric": metric_name,
                "metric_label": label,
                "n": 48,
                "pearson_r": pearson,
                "pearson_prompt_bootstrap_ci_low": pearson_low,
                "pearson_prompt_bootstrap_ci_high": pearson_high,
                "spearman_rho": spearman,
                "spearman_prompt_bootstrap_ci_low": spearman_low,
                "spearman_prompt_bootstrap_ci_high": spearman_high,
            }
        )

        partial_r = partial_correlation(metric, accuracy, steps[:, None])
        partial_rho = partial_correlation(metric, accuracy, steps[:, None], rank=True)
        boot_partial = rowwise_step_partial(bootstrap, metric, steps)
        partial_low, partial_high = interval(boot_partial)
        regression = ols_hac(
            zscore(accuracy),
            np.column_stack([zscore(metric), zscore(steps)]),
            metric_index=0,
            max_lags=HAC_LAGS,
        )
        quadratic_controls = np.column_stack([steps, steps**2])
        quadratic_partial_r = partial_correlation(metric, accuracy, quadratic_controls)
        quadratic_boot_partial = rowwise_fixed_control_partial(
            bootstrap, metric, quadratic_controls
        )
        quadratic_partial_low, quadratic_partial_high = interval(quadratic_boot_partial)
        quadratic_regression = ols_hac(
            zscore(accuracy),
            np.column_stack([zscore(metric), zscore(steps), zscore(steps**2)]),
            metric_index=0,
            max_lags=HAC_LAGS,
        )
        step_rows.append(
            {
                "representation": representation.key,
                "metric": metric_name,
                "metric_label": label,
                "n": 48,
                "partial_pearson_r": partial_r,
                "partial_pearson_prompt_bootstrap_ci_low": partial_low,
                "partial_pearson_prompt_bootstrap_ci_high": partial_high,
                "partial_spearman_rho": partial_rho,
                "metric_standardized_beta": regression["beta"],
                "hac_ci_low": regression["ci_low"],
                "hac_ci_high": regression["ci_high"],
                "hac_p_value": regression["p_value"],
                "partial_r2": nested_partial_r2(
                    zscore(accuracy), zscore(steps)[:, None], zscore(metric)
                ),
                "residual_acf1": regression["residual_acf1"],
                "quadratic_step_partial_pearson_r": quadratic_partial_r,
                "quadratic_step_partial_prompt_bootstrap_ci_low": quadratic_partial_low,
                "quadratic_step_partial_prompt_bootstrap_ci_high": quadratic_partial_high,
                "quadratic_step_metric_standardized_beta": quadratic_regression["beta"],
                "quadratic_step_hac_ci_low": quadratic_regression["ci_low"],
                "quadratic_step_hac_ci_high": quadratic_regression["ci_high"],
                "quadratic_step_hac_p_value": quadratic_regression["p_value"],
                "quadratic_step_partial_r2": nested_partial_r2(
                    zscore(accuracy),
                    np.column_stack([zscore(steps), zscore(steps**2)]),
                    zscore(metric),
                ),
            }
        )

        current = accuracy[1:]
        baseline = accuracy[:-1]
        current_bootstrap = bootstrap[:, 1:]
        baseline_bootstrap = bootstrap[:, :-1]
        one_metric = metric[1:]
        one_steps = steps[1:]
        delta = current - baseline
        boot_delta = current_bootstrap - baseline_bootstrap
        boot_delta_r = rowwise_correlation(boot_delta, one_metric)
        delta_low, delta_high = interval(boot_delta_r)
        one_regression = ols_hac(
            zscore(current),
            np.column_stack([zscore(one_metric), zscore(baseline), zscore(one_steps)]),
            metric_index=0,
            max_lags=HAC_LAGS,
        )
        one_boot_beta = bootstrap_adjusted_beta(
            current_bootstrap, one_metric, baseline_bootstrap, one_steps
        )
        one_beta_low, one_beta_high = interval(one_boot_beta)
        one_step_rows.append(
            {
                "representation": representation.key,
                "metric": metric_name,
                "metric_label": label,
                "n": 47,
                "delta_pearson_r": float(pearsonr(one_metric, delta).statistic),
                "delta_spearman_rho": float(spearmanr(one_metric, delta).statistic),
                "delta_pearson_prompt_bootstrap_ci_low": delta_low,
                "delta_pearson_prompt_bootstrap_ci_high": delta_high,
                "baseline_step_adjusted_partial_r": partial_correlation(
                    one_metric,
                    current,
                    np.column_stack([baseline, one_steps]),
                ),
                "metric_standardized_beta": one_regression["beta"],
                "hac_ci_low": one_regression["ci_low"],
                "hac_ci_high": one_regression["ci_high"],
                "hac_p_value": one_regression["p_value"],
                "metric_beta_prompt_bootstrap_ci_low": one_beta_low,
                "metric_beta_prompt_bootstrap_ci_high": one_beta_high,
                "partial_r2": nested_partial_r2(
                    zscore(current),
                    np.column_stack([zscore(baseline), zscore(one_steps)]),
                    zscore(one_metric),
                ),
            }
        )

        starts = np.arange(1, 46, dtype=int)
        future_metric = metric[starts]
        future_baseline = accuracy[starts - 1]
        future_outcome = accuracy[starts + 2]
        future_baseline_bootstrap = bootstrap[:, starts - 1]
        future_outcome_bootstrap = bootstrap[:, starts + 2]
        future_delta = future_outcome - future_baseline
        future_boot_delta = future_outcome_bootstrap - future_baseline_bootstrap
        future_boot_r = rowwise_correlation(future_boot_delta, future_metric)
        future_delta_low, future_delta_high = interval(future_boot_r)
        future_regression = ols_hac(
            zscore(future_outcome),
            np.column_stack(
                [zscore(future_metric), zscore(future_baseline), zscore(starts)]
            ),
            metric_index=0,
            max_lags=HAC_LAGS,
        )
        future_boot_beta = bootstrap_adjusted_beta(
            future_outcome_bootstrap,
            future_metric,
            future_baseline_bootstrap,
            starts.astype(float),
        )
        future_beta_low, future_beta_high = interval(future_boot_beta)
        future_rows.append(
            {
                "representation": representation.key,
                "metric": metric_name,
                "metric_label": label,
                "n": 45,
                "delta_pearson_r": float(pearsonr(future_metric, future_delta).statistic),
                "delta_spearman_rho": float(spearmanr(future_metric, future_delta).statistic),
                "delta_pearson_prompt_bootstrap_ci_low": future_delta_low,
                "delta_pearson_prompt_bootstrap_ci_high": future_delta_high,
                "baseline_step_adjusted_partial_r": partial_correlation(
                    future_metric,
                    future_outcome,
                    np.column_stack([future_baseline, starts]),
                ),
                "metric_standardized_beta": future_regression["beta"],
                "hac_ci_low": future_regression["ci_low"],
                "hac_ci_high": future_regression["ci_high"],
                "hac_p_value": future_regression["p_value"],
                "metric_beta_prompt_bootstrap_ci_low": future_beta_low,
                "metric_beta_prompt_bootstrap_ci_high": future_beta_high,
                "partial_r2": nested_partial_r2(
                    zscore(future_outcome),
                    np.column_stack([zscore(future_baseline), zscore(starts)]),
                    zscore(future_metric),
                ),
            }
        )

    return {
        "current": pd.DataFrame(current_rows),
        "step_adjusted": pd.DataFrame(step_rows),
        "one_step": pd.DataFrame(one_step_rows),
        "future_3step": pd.DataFrame(future_rows),
    }


def add_fit(ax: plt.Axes, x: np.ndarray, y: np.ndarray) -> None:
    order = np.argsort(x)
    coefficients = np.polyfit(x, y, deg=1)
    ax.plot(x[order], np.polyval(coefficients, x[order]), color="#dc2626", lw=1.5)


def plot_trajectory(checkpoints: pd.DataFrame) -> None:
    figure, axis = plt.subplots(figsize=(10.5, 4.8))
    axis.plot(checkpoints["checkpoint"], checkpoints["accuracy"], color=COLOR, marker="o", ms=3)
    axis.fill_between(
        checkpoints["checkpoint"],
        checkpoints["prompt_bootstrap_ci_low"],
        checkpoints["prompt_bootstrap_ci_high"],
        color=COLOR,
        alpha=0.16,
        linewidth=0,
    )
    axis.set(xlabel="Checkpoint", ylabel="MedMCQA accuracy", title="OnlineRubrics MedMCQA trajectory")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(OUTPUT_ROOT / "figure_1_medmcqa_checkpoint_trajectory.png", dpi=220)
    plt.close(figure)


def plot_representation(
    representation: Representation,
    data: dict[str, object],
    tables: dict[str, pd.DataFrame],
) -> None:
    metric_frame = load_metrics(representation)
    steps = np.asarray(data["steps"], dtype=float)
    accuracy = np.asarray(data["accuracy"], dtype=float)

    plot_specs = (
        ("current", "Checkpoint t accuracy", "metric_t vs checkpoint_t", 2),
        ("step_adjusted", "Step-adjusted accuracy residual", "linear step adjusted", 3),
        ("one_step", "Accuracy(t) - accuracy(t-1)", "one-update change", 4),
        ("future_3step", "Accuracy(s+3) - accuracy(s)", "three-step future change", 5),
    )
    for table_key, y_label, subtitle, figure_number in plot_specs:
        figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        for axis, metric_name, label in zip(
            axes, representation.metrics, representation.labels
        ):
            metric = metric_frame[metric_name].to_numpy(dtype=float)
            row = tables[table_key].loc[tables[table_key]["metric"] == metric_name].iloc[0]
            if table_key == "current":
                x, y = metric, accuracy
                annotation = f"Pearson r={row['pearson_r']:.2f}"
            elif table_key == "step_adjusted":
                x = metric - np.polyval(np.polyfit(steps, metric, 1), steps)
                y = accuracy - np.polyval(np.polyfit(steps, accuracy, 1), steps)
                annotation = f"partial r={row['partial_pearson_r']:.2f}"
            elif table_key == "one_step":
                x, y = metric[1:], accuracy[1:] - accuracy[:-1]
                annotation = f"delta r={row['delta_pearson_r']:.2f}"
            else:
                starts = np.arange(1, 46, dtype=int)
                x = metric[starts]
                y = accuracy[starts + 2] - accuracy[starts - 1]
                annotation = f"delta r={row['delta_pearson_r']:.2f}"
            axis.scatter(x, y, color=COLOR, alpha=0.8, s=28)
            add_fit(axis, x, y)
            axis.set(xlabel=label, ylabel=y_label, title=annotation)
            axis.grid(alpha=0.2)
        figure.suptitle(f"{representation.title}: {subtitle}", y=1.02)
        figure.tight_layout()
        figure.savefig(
            OUTPUT_ROOT / f"figure_{figure_number}_{representation.key}_{table_key}.png",
            dpi=220,
            bbox_inches="tight",
        )
        plt.close(figure)


def plot_quadratic_step_adjusted(
    representation: Representation,
    data: dict[str, object],
    table: pd.DataFrame,
) -> None:
    metric_frame = load_metrics(representation)
    steps = np.asarray(data["steps"], dtype=float)
    accuracy = np.asarray(data["accuracy"], dtype=float)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for axis, metric_name, label in zip(
        axes, representation.metrics, representation.labels
    ):
        metric = metric_frame[metric_name].to_numpy(dtype=float)
        row = table.loc[table["metric"] == metric_name].iloc[0]
        metric_residual = metric - np.polyval(np.polyfit(steps, metric, 2), steps)
        accuracy_residual = accuracy - np.polyval(np.polyfit(steps, accuracy, 2), steps)
        axis.scatter(metric_residual, accuracy_residual, color=COLOR, alpha=0.8, s=28)
        add_fit(axis, metric_residual, accuracy_residual)
        axis.set(
            xlabel=label,
            ylabel="Quadratic-step-adjusted accuracy residual",
            title=f"partial r={row['quadratic_step_partial_pearson_r']:.2f}",
        )
        axis.grid(alpha=0.2)
    figure.suptitle(f"{representation.title}: quadratic step sensitivity", y=1.02)
    figure.tight_layout()
    figure.savefig(
        OUTPUT_ROOT / f"figure_3b_{representation.key}_quadratic_step_adjusted.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(figure)


def write_summary(
    checkpoint: pd.DataFrame,
    differences: pd.DataFrame,
    analyses: dict[str, dict[str, pd.DataFrame]],
) -> None:
    one_step = differences.loc[differences["horizon"] == 1]
    three_step = differences.loc[differences["horizon"] == 3]
    full_difference = checkpoint["accuracy"].iloc[-1] - checkpoint["accuracy"].iloc[0]
    trajectory_steps = checkpoint["checkpoint"].to_numpy(dtype=float)
    trajectory_accuracy = checkpoint["accuracy"].to_numpy(dtype=float)
    total_sum_squares = float(np.sum((trajectory_accuracy - trajectory_accuracy.mean()) ** 2))
    linear_fit = np.polyval(np.polyfit(trajectory_steps, trajectory_accuracy, 1), trajectory_steps)
    quadratic_fit = np.polyval(np.polyfit(trajectory_steps, trajectory_accuracy, 2), trajectory_steps)
    linear_r2 = 1.0 - float(np.sum((trajectory_accuracy - linear_fit) ** 2)) / total_sum_squares
    quadratic_r2 = 1.0 - float(np.sum((trajectory_accuracy - quadratic_fit) ** 2)) / total_sum_squares
    minimum = checkpoint.loc[checkpoint["accuracy"].idxmin()]
    lines = [
        "# MedMCQA checkpoint trajectory and rubric-metric validation",
        "",
        "## Evaluation contract",
        "",
        "- OnlineRubrics checkpoints: 1–48.",
        "- Benchmark: MedMCQA validation, 4,183 fixed prompt IDs.",
        "- Scoring: zero-shot A–D next-token conditional log likelihood with the official lm-evaluation-harness prompt and the fixed Qwen chat template.",
        "- Alignment: update t metric (generated by policy t-1) is paired with checkpoint t.",
        "- Bootstrap: 5,000 prompt-paired resamples; it does not represent seed or training-run uncertainty.",
        "- No multiple-testing/BH adjustment is used; all analyses are descriptive and trajectory-conditional.",
        "",
        "## Accuracy trajectory",
        "",
        f"- Checkpoint 1 accuracy: {checkpoint['accuracy'].iloc[0]:.4f}.",
        f"- Checkpoint 48 accuracy: {checkpoint['accuracy'].iloc[-1]:.4f}.",
        f"- Minimum: checkpoint {int(minimum['checkpoint'])}, accuracy {minimum['accuracy']:.4f}.",
        f"- Step-only trend fit: linear R²={linear_r2:.3f}; quadratic R²={quadratic_r2:.3f}.",
        f"- Net change (48 - 1): {full_difference:+.4f}.",
        f"- One-step paired-bootstrap CIs excluding zero: {int(one_step['ci_excludes_zero'].sum())}/{len(one_step)}.",
        f"- Three-step paired-bootstrap CIs excluding zero: {int(three_step['ci_excludes_zero'].sum())}/{len(three_step)}.",
        "",
        "## Correlation and update-level results",
        "",
    ]
    for representation in REPRESENTATIONS:
        tables = analyses[representation.key]
        lines.extend([f"### {representation.title}", ""])
        combined = tables["current"][
            ["metric_label", "pearson_r", "spearman_rho"]
        ].merge(
            tables["step_adjusted"][
                [
                    "metric_label",
                    "partial_pearson_r",
                    "quadratic_step_partial_pearson_r",
                    "metric_standardized_beta",
                ]
            ],
            on="metric_label",
        ).merge(
            tables["one_step"][
                ["metric_label", "delta_pearson_r", "baseline_step_adjusted_partial_r"]
            ],
            on="metric_label",
        ).merge(
            tables["future_3step"][
                ["metric_label", "delta_pearson_r", "baseline_step_adjusted_partial_r"]
            ],
            on="metric_label",
            suffixes=("_one_step", "_future_3step"),
        )
        lines.extend([combined.to_markdown(index=False, floatfmt=".3f"), ""])
    lines.extend(
        [
            "## Interpretation guardrails",
            "",
            "- The raw metric–accuracy correlation mixes metric variation with the checkpoint trend.",
            "- The step-adjusted result asks whether a metric contains information beyond a linear training-step trend; it is not a causal estimate.",
            "- Because the MedMCQA trajectory is U-shaped, quadratic-step adjustment is reported as a prespecified sensitivity; disagreement with the linear result means the conclusion is trend-model dependent.",
            "- The one-step analysis is the closest available alignment to one optimizer update, but adjacent checkpoint scores and metric batches remain serially dependent.",
            "- The three-step analysis asks about near-future improvement and controls for starting accuracy and step; overlapping windows are not independent runs.",
            "- A prompt-bootstrap interval that excludes zero only shows stability to MedMCQA item composition for this trajectory.",
        ]
    )
    (OUTPUT_ROOT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    data = load_medmcqa()
    checkpoint, differences = trajectory_tables(data)
    checkpoint.to_csv(OUTPUT_ROOT / "checkpoint_accuracy_with_prompt_bootstrap_ci.csv", index=False)
    differences.to_csv(OUTPUT_ROOT / "checkpoint_difference_prompt_bootstrap.csv", index=False)
    plot_trajectory(checkpoint)

    analyses: dict[str, dict[str, pd.DataFrame]] = {}
    for representation in REPRESENTATIONS:
        tables = analyze_representation(representation, data)
        analyses[representation.key] = tables
        for name, frame in tables.items():
            frame.to_csv(OUTPUT_ROOT / f"{representation.key}_{name}.csv", index=False)
        plot_representation(representation, data, tables)
        plot_quadratic_step_adjusted(representation, data, tables["step_adjusted"])
    write_summary(checkpoint, differences, analyses)
    print((OUTPUT_ROOT / "README.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
