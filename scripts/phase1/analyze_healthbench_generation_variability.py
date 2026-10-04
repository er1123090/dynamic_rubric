#!/usr/bin/env python3
"""Analyze three-seed HealthBench generation variability across checkpoints."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from dynamic_rubric.artifacts import read_jsonl, write_json_atomic
from scripts.phase1 import evaluate_final_policies as evaluation
from scripts.phase1.run_healthbench_generation_variability_trainer1 import (
    DEFAULT_CONFIG,
    _repeat_configs,
)


def _clip_mean(values: Sequence[float]) -> float:
    return min(1.0, max(0.0, float(np.mean(values))))


def _load_scores(
    configs: Sequence[Mapping[str, Any]],
) -> tuple[dict[int, dict[int, dict[str, float]]], dict[int, dict[int, dict[str, str]]]]:
    scores: dict[int, dict[int, dict[str, float]]] = {}
    responses: dict[int, dict[int, dict[str, str]]] = {}
    expected_ids: set[str] | None = None
    for config in configs:
        seed = int(config["generation"]["seed"])
        run = evaluation.run_directory(config)
        rows = read_jsonl(run / "summary" / "prompt_scores.jsonl")
        seed_scores: dict[int, dict[str, float]] = {}
        seed_responses: dict[int, dict[str, str]] = {}
        for row in rows:
            if row["dataset"] != "healthbench":
                continue
            step = int(row["global_step"])
            prompt_id = str(row["prompt_id"])
            seed_scores.setdefault(step, {})[prompt_id] = float(row["raw_score_unclipped"])
            seed_responses.setdefault(step, {})[prompt_id] = str(row["response_id"])
        if not seed_scores:
            raise RuntimeError(f"no HealthBench prompt scores found: {run}")
        for step, by_prompt in seed_scores.items():
            ids = set(by_prompt)
            if len(ids) != 500:
                raise RuntimeError(f"seed={seed} step={step}: expected 500 prompts, found {len(ids)}")
            if expected_ids is None:
                expected_ids = ids
            elif ids != expected_ids:
                raise RuntimeError("prompt IDs differ across seed/checkpoint cells")
        scores[seed] = seed_scores
        responses[seed] = seed_responses
    return scores, responses


def _hierarchical_delta(
    scores: Mapping[int, Mapping[int, Mapping[str, float]]],
    step_a: int,
    step_b: int,
    *,
    seed: int,
    replicates: int,
) -> tuple[float, float, float]:
    generation_seeds = sorted(scores)
    prompt_ids = sorted(scores[generation_seeds[0]][step_a])
    point = float(
        np.mean(
            [
                _clip_mean(list(scores[value][step_b].values()))
                - _clip_mean(list(scores[value][step_a].values()))
                for value in generation_seeds
            ]
        )
    )
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=float)
    for index in range(replicates):
        sampled_seeds = rng.choice(generation_seeds, size=len(generation_seeds), replace=True)
        seed_deltas = []
        sampled_prompt_indices = rng.integers(0, len(prompt_ids), size=len(prompt_ids))
        sampled_prompts = [prompt_ids[position] for position in sampled_prompt_indices]
        for generation_seed in sampled_seeds:
            before = [scores[int(generation_seed)][step_a][prompt] for prompt in sampled_prompts]
            after = [scores[int(generation_seed)][step_b][prompt] for prompt in sampled_prompts]
            seed_deltas.append(_clip_mean(after) - _clip_mean(before))
        draws[index] = float(np.mean(seed_deltas))
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return point, float(lower), float(upper)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"refusing to write empty table: {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(config_path: Path) -> dict[str, Any]:
    configs = _repeat_configs(config_path.resolve())
    scores, responses = _load_scores(configs)
    seeds = sorted(scores)
    steps = sorted(next(iter(scores.values())))
    expected_steps = [int(value) for value in configs[0]["experiment"]["checkpoints"]]
    if steps != expected_steps:
        raise RuntimeError(f"checkpoint mismatch: expected={expected_steps}, found={steps}")
    output_root = Path(str(evaluation.load_config(config_path.resolve(), None)["output_root"]))
    analysis_root = output_root / "analysis"
    analysis_root.mkdir(parents=True, exist_ok=True)

    checkpoint_rows = []
    means_by_seed: dict[int, list[float]] = {seed: [] for seed in seeds}
    changed_fractions = []
    for step in steps:
        repeat_means = []
        for seed in seeds:
            value = _clip_mean(list(scores[seed][step].values()))
            repeat_means.append(value)
            means_by_seed[seed].append(value)
        prompt_ids = sorted(scores[seeds[0]][step])
        changed_fraction = float(
            np.mean(
                [
                    len({responses[seed][step][prompt] for seed in seeds}) > 1
                    for prompt in prompt_ids
                ]
            )
        )
        changed_fractions.append(changed_fraction)
        checkpoint_rows.append(
            {
                "checkpoint": step,
                **{f"healthbench_seed_{seed}": repeat_means[index] for index, seed in enumerate(seeds)},
                "healthbench_mean": float(np.mean(repeat_means)),
                "healthbench_sd_across_generation_seeds": float(np.std(repeat_means, ddof=1)),
                "healthbench_min": min(repeat_means),
                "healthbench_max": max(repeat_means),
                "fraction_prompts_with_distinct_responses": changed_fraction,
                "n_prompts": len(prompt_ids),
                "n_generation_seeds": len(seeds),
            }
        )

    pair_rows = []
    comparison_pairs = list(zip(steps[:-1], steps[1:])) + [
        (steps[0], step) for step in steps[1:] if step != steps[1]
    ]
    seen = set()
    for step_a, step_b in comparison_pairs:
        if (step_a, step_b) in seen:
            continue
        seen.add((step_a, step_b))
        point, lower, upper = _hierarchical_delta(
            scores,
            step_a,
            step_b,
            seed=20260928 + step_a * 100 + step_b,
            replicates=int(configs[0]["bootstrap_replicates"]),
        )
        pair_rows.append(
            {
                "checkpoint_before": step_a,
                "checkpoint_after": step_b,
                "healthbench_delta": point,
                "hierarchical_bootstrap_95_ci_lower": lower,
                "hierarchical_bootstrap_95_ci_upper": upper,
                "ci_excludes_zero": lower > 0 or upper < 0,
                "bootstrap_replicates": int(configs[0]["bootstrap_replicates"]),
                "n_prompts": 500,
                "n_generation_seeds": len(seeds),
            }
        )

    rank_rows = []
    for first, second in itertools.combinations(seeds, 2):
        result = spearmanr(means_by_seed[first], means_by_seed[second])
        rank_rows.append(
            {
                "generation_seed_a": first,
                "generation_seed_b": second,
                "spearman_checkpoint_trajectory": float(result.statistic),
                "p_value_descriptive_only": float(result.pvalue),
                "n_checkpoints": len(steps),
            }
        )

    _write_csv(analysis_root / "checkpoint_repeat_summary.csv", checkpoint_rows)
    _write_csv(analysis_root / "paired_checkpoint_differences.csv", pair_rows)
    _write_csv(analysis_root / "trajectory_rank_consistency.csv", rank_rows)

    colors = ["#4472C4", "#ED7D31", "#70AD47"]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for color, seed in zip(colors, seeds):
        axes[0].plot(steps, means_by_seed[seed], marker="o", color=color, alpha=0.72, label=f"seed {seed}")
    means = np.array([row["healthbench_mean"] for row in checkpoint_rows])
    sds = np.array([row["healthbench_sd_across_generation_seeds"] for row in checkpoint_rows])
    axes[0].errorbar(steps, means, yerr=sds, color="black", marker="o", linewidth=2, capsize=3, label="mean ± SD")
    axes[0].set_title("A. HealthBench across generation seeds")
    axes[0].set_xlabel("Checkpoint")
    axes[0].set_ylabel("HealthBench score")
    axes[0].legend(frameon=False, fontsize=8)

    versus_first = [row for row in pair_rows if row["checkpoint_before"] == steps[0]]
    x = [row["checkpoint_after"] for row in versus_first]
    y = np.array([row["healthbench_delta"] for row in versus_first])
    low = np.array([row["hierarchical_bootstrap_95_ci_lower"] for row in versus_first])
    high = np.array([row["hierarchical_bootstrap_95_ci_upper"] for row in versus_first])
    axes[1].axhline(0, color="#777777", linewidth=1)
    axes[1].errorbar(x, y, yerr=np.vstack([y - low, high - y]), fmt="o-", color="#4472C4", capsize=4)
    axes[1].set_title("B. Difference from checkpoint 1")
    axes[1].set_xlabel("Checkpoint")
    axes[1].set_ylabel("HealthBench delta (95% CI)")

    axes[2].bar(steps, np.array(changed_fractions) * 100, width=5.5, color="#70AD47")
    axes[2].set_title("C. Response variation across seeds")
    axes[2].set_xlabel("Checkpoint")
    axes[2].set_ylabel("Prompts with >1 response (%)")
    axes[2].set_ylim(0, 105)
    for axis in axes:
        axis.grid(axis="y", alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(analysis_root / "healthbench_generation_variability.png", dpi=220, bbox_inches="tight")
    figure.savefig(analysis_root / "healthbench_generation_variability.pdf", bbox_inches="tight")
    plt.close(figure)

    first_last = next(
        row
        for row in pair_rows
        if row["checkpoint_before"] == steps[0] and row["checkpoint_after"] == steps[-1]
    )
    summary = {
        "schema_version": 1,
        "question": "How stable are HealthBench checkpoint scores under repeated response generation?",
        "generation_seeds": seeds,
        "checkpoints": steps,
        "n_prompts": 500,
        "sampling": {
            "temperature": float(configs[0]["generation"]["temperature"]),
            "top_p": float(configs[0]["generation"]["top_p"]),
            "max_output_tokens": int(configs[0]["generation"]["max_output_tokens"]),
        },
        "judge": {
            "model": str(configs[0]["grading"]["served_model"]),
            "temperature": 0.0,
            "same_judge_contract_for_all_cells": True,
        },
        "max_checkpoint_sd_across_generation_seeds": max(
            row["healthbench_sd_across_generation_seeds"] for row in checkpoint_rows
        ),
        "mean_fraction_prompts_with_distinct_responses": float(np.mean(changed_fractions)),
        "minimum_pairwise_seed_trajectory_spearman": min(
            row["spearman_checkpoint_trajectory"] for row in rank_rows
        ),
        "checkpoint_1_to_48": first_last,
        "interpretation_limits": [
            "Three generation seeds are a pilot estimate, not a precise variance-component estimate.",
            "The hierarchical bootstrap resamples generation seeds and paired prompt IDs; it does not represent independent training-run uncertainty.",
            "Judge decoding is deterministic, so this experiment targets response-generation and prompt-sampling uncertainty, not repeated-judge variability.",
        ],
        "artifacts": {
            "checkpoint_table": str((analysis_root / "checkpoint_repeat_summary.csv").resolve()),
            "difference_table": str((analysis_root / "paired_checkpoint_differences.csv").resolve()),
            "rank_table": str((analysis_root / "trajectory_rank_consistency.csv").resolve()),
            "figure_png": str((analysis_root / "healthbench_generation_variability.png").resolve()),
            "figure_pdf": str((analysis_root / "healthbench_generation_variability.pdf").resolve()),
        },
    }
    write_json_atomic(analysis_root / "analysis_summary.json", summary, immutable=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    print(json.dumps(analyze(args.config), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
