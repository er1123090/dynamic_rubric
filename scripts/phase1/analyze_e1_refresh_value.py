#!/usr/bin/env python3
"""E1: measure refresh value on aligned adjacent OnlineRubrics cells.

The analysis is CPU-only and consumes the sealed validation100 grade matrix.
For every saved policy checkpoint after the first, it compares the immediately
previous evaluator rubric with the current evaluator rubric on the exact same
16 Pool-B responses for each prompt.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = (
    ROOT
    / "outputs/analysis/medicine_static_online_heldout_validation100_full_20260910_v4"
)
DEFAULT_OUT = ROOT / "outputs/analysis/medicine_online_e1_refresh_value_20260921"
PAIR_PATTERN = re.compile(r"policy-(\d+)/evaluator-(\d+)\.jsonl$")
SATURATION_BINS = (-1e-12, 1.0 / 3.0, 2.0 / 3.0, 1.0 + 1e-12)
SATURATION_LABELS = ("low_[0,1/3]", "mid_(1/3,2/3]", "high_(2/3,1]")
EPSILON = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--bootstrap", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20260921)
    return parser.parse_args()


def discover_steps(source: Path) -> list[int]:
    steps = []
    for path in (source / "grades/cells/online").glob("policy-*/evaluator-*.jsonl"):
        match = PAIR_PATTERN.search(path.as_posix())
        if match and match.group(1) == match.group(2):
            steps.append(int(match.group(1)))
    result = sorted(set(steps))
    if len(result) < 2:
        raise ValueError("at least two diagonal OnlineRubrics cells are required")
    return result


def cell_path(source: Path, policy_step: int, evaluator_step: int) -> Path:
    return (
        source
        / "grades/cells/online"
        / f"policy-{policy_step:03d}"
        / f"evaluator-{evaluator_step:03d}.jsonl"
    )


def load_cell(path: Path) -> dict[str, list[dict]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    groups: dict[str, list[dict]] = defaultdict(list)
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row["data_role"] != "heldout_validation" or row["pool"] != "probe_B":
                raise ValueError(f"unexpected analysis role in {path}")
            if row["used_for_gradient"]:
                raise ValueError(f"evaluation response marked as gradient data in {path}")
            groups[str(row["prompt_id"])].append(row)
    if len(groups) != 100:
        raise ValueError(f"expected 100 prompts in {path}, found {len(groups)}")
    for prompt_id, rows in groups.items():
        if len(rows) != 16:
            raise ValueError(f"expected 16 responses for {prompt_id} in {path}")
        if len({str(row["response_id"]) for row in rows}) != 16:
            raise ValueError(f"duplicate response IDs for {prompt_id} in {path}")
    return groups


def criterion_counts(rows: list[dict]) -> dict[str, int]:
    values: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        for criterion_id, grade in row["grades"]:
            values[str(criterion_id)].append(int(grade))
    if not values or any(len(grades) != len(rows) for grades in values.values()):
        raise ValueError("criterion grades do not align with responses")
    counts = {"effective": 0, "saturated": 0, "dead": 0}
    for grades in values.values():
        if all(grades):
            counts["saturated"] += 1
        elif not any(grades):
            counts["dead"] += 1
        else:
            counts["effective"] += 1
    counts["total"] = len(values)
    return counts


def advantages(rewards: np.ndarray) -> np.ndarray:
    std = float(np.std(rewards, ddof=1))
    if std == 0.0:
        return np.zeros_like(rewards)
    return (rewards - float(np.mean(rewards))) / (std + EPSILON)


def exact_ptr(rewards: np.ndarray) -> float:
    upper = rewards[:, None] == rewards[None, :]
    return float(np.mean(upper[np.triu_indices(len(rewards), k=1)]))


def align_rows(old_rows: list[dict], new_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    old_by_id = {str(row["response_id"]): row for row in old_rows}
    new_by_id = {str(row["response_id"]): row for row in new_rows}
    if old_by_id.keys() != new_by_id.keys():
        raise ValueError("old/current response pools differ")
    response_ids = sorted(old_by_id)
    return [old_by_id[key] for key in response_ids], [new_by_id[key] for key in response_ids]


def analyze_pair(
    *,
    policy_step: int,
    old_step: int,
    prompt_id: str,
    old_rows: list[dict],
    new_rows: list[dict],
) -> dict:
    old_rows, new_rows = align_rows(old_rows, new_rows)
    old_rewards = np.asarray([float(row["reward"]) for row in old_rows])
    new_rewards = np.asarray([float(row["reward"]) for row in new_rows])
    old_adv = advantages(old_rewards)
    new_adv = advantages(new_rewards)
    old_counts = criterion_counts(old_rows)
    new_counts = criterion_counts(new_rows)
    old_zero = old_adv == 0.0
    new_zero = new_adv == 0.0
    direct_flip = ((old_adv > 0.0) & (new_adv < 0.0)) | (
        (old_adv < 0.0) & (new_adv > 0.0)
    )
    both_nonzero = (~old_zero) & (~new_zero)
    return {
        "policy_step": policy_step,
        "old_evaluator_step": old_step,
        "current_evaluator_step": policy_step,
        "step_gap": policy_step - old_step,
        "prompt_id": prompt_id,
        "response_count": len(old_rows),
        "old_criterion_count": old_counts["total"],
        "new_criterion_count": new_counts["total"],
        "old_effective_count": old_counts["effective"],
        "new_effective_count": new_counts["effective"],
        "old_saturated_count": old_counts["saturated"],
        "new_saturated_count": new_counts["saturated"],
        "old_dead_count": old_counts["dead"],
        "new_dead_count": new_counts["dead"],
        "old_ecr": old_counts["effective"] / old_counts["total"],
        "new_ecr": new_counts["effective"] / new_counts["total"],
        "delta_ecr_prompt": (
            new_counts["effective"] / new_counts["total"]
            - old_counts["effective"] / old_counts["total"]
        ),
        "old_saturation_ratio": old_counts["saturated"] / old_counts["total"],
        "new_saturation_ratio": new_counts["saturated"] / new_counts["total"],
        "old_dead_ratio": old_counts["dead"] / old_counts["total"],
        "new_dead_ratio": new_counts["dead"] / new_counts["total"],
        "old_exact_zar": int(bool(np.all(old_zero))),
        "new_exact_zar": int(bool(np.all(new_zero))),
        "old_zero_to_new_nonzero_group": int(bool(np.all(old_zero) and not np.all(new_zero))),
        "old_nonzero_to_new_zero_group": int(bool(not np.all(old_zero) and np.all(new_zero))),
        "old_exact_ptr": exact_ptr(old_rewards),
        "new_exact_ptr": exact_ptr(new_rewards),
        "delta_ptr_new_minus_old": exact_ptr(new_rewards) - exact_ptr(old_rewards),
        "tie_reduction_old_minus_new": exact_ptr(old_rewards) - exact_ptr(new_rewards),
        "mean_abs_advantage_change": float(np.mean(np.abs(new_adv - old_adv))),
        "direct_sign_flip_fraction_all": float(np.mean(direct_flip)),
        "direct_sign_flip_fraction_both_nonzero": (
            float(np.mean(direct_flip[both_nonzero])) if np.any(both_nonzero) else math.nan
        ),
        "old_zero_to_new_nonzero_response_fraction": float(np.mean(old_zero & ~new_zero)),
        "old_nonzero_to_new_zero_response_fraction": float(np.mean(~old_zero & new_zero)),
        "any_zero_nonzero_response_transition_fraction": float(np.mean(old_zero != new_zero)),
    }


def pooled_delta_ecr(frame: pd.DataFrame) -> float:
    old_total = frame.old_criterion_count.sum()
    new_total = frame.new_criterion_count.sum()
    return float(
        frame.new_effective_count.sum() / new_total
        - frame.old_effective_count.sum() / old_total
    )


def summarize(frame: pd.DataFrame) -> dict[str, float | int]:
    return {
        "analysis_units": int(len(frame)),
        "prompts": int(frame.prompt_id.nunique()),
        "checkpoint_pairs": int(frame[["old_evaluator_step", "policy_step"]].drop_duplicates().shape[0]),
        "old_saturation_ratio_mean": float(frame.old_saturation_ratio.mean()),
        "old_ecr_pooled": float(frame.old_effective_count.sum() / frame.old_criterion_count.sum()),
        "new_ecr_pooled": float(frame.new_effective_count.sum() / frame.new_criterion_count.sum()),
        "delta_ecr_pooled": pooled_delta_ecr(frame),
        "old_exact_zar": float(frame.old_exact_zar.mean()),
        "new_exact_zar": float(frame.new_exact_zar.mean()),
        "old_zero_to_new_nonzero_group": float(frame.old_zero_to_new_nonzero_group.mean()),
        "old_nonzero_to_new_zero_group": float(frame.old_nonzero_to_new_zero_group.mean()),
        "old_exact_ptr": float(frame.old_exact_ptr.mean()),
        "new_exact_ptr": float(frame.new_exact_ptr.mean()),
        "tie_reduction_old_minus_new": float(frame.tie_reduction_old_minus_new.mean()),
        "mean_abs_advantage_change": float(frame.mean_abs_advantage_change.mean()),
        "direct_sign_flip_fraction_all": float(frame.direct_sign_flip_fraction_all.mean()),
        "direct_sign_flip_fraction_both_nonzero": float(
            frame.direct_sign_flip_fraction_both_nonzero.mean()
        ),
        "old_zero_to_new_nonzero_response_fraction": float(
            frame.old_zero_to_new_nonzero_response_fraction.mean()
        ),
        "old_nonzero_to_new_zero_response_fraction": float(
            frame.old_nonzero_to_new_zero_response_fraction.mean()
        ),
    }


def cluster_bootstrap(
    frame: pd.DataFrame, *, draws: int, seed: int
) -> dict[str, dict[str, float]]:
    prompt_ids = sorted(frame.prompt_id.unique())
    by_prompt = {prompt_id: frame[frame.prompt_id == prompt_id] for prompt_id in prompt_ids}
    rng = np.random.default_rng(seed)
    metrics = {
        "delta_ecr_pooled": [],
        "old_zero_to_new_nonzero_group": [],
        "tie_reduction_old_minus_new": [],
        "mean_abs_advantage_change": [],
        "direct_sign_flip_fraction_all": [],
        "old_zero_to_new_nonzero_response_fraction": [],
        "old_nonzero_to_new_zero_response_fraction": [],
    }
    for _ in range(draws):
        sampled = rng.choice(prompt_ids, size=len(prompt_ids), replace=True)
        boot = pd.concat([by_prompt[prompt_id] for prompt_id in sampled], ignore_index=True)
        metrics["delta_ecr_pooled"].append(pooled_delta_ecr(boot))
        for name in metrics:
            if name != "delta_ecr_pooled":
                metrics[name].append(float(boot[name].mean()))
    result = {}
    for name, values in metrics.items():
        array = np.asarray(values, dtype=float)
        result[name] = {
            "mean": float(np.nanmean(array)),
            "ci_low": float(np.nanquantile(array, 0.025)),
            "ci_high": float(np.nanquantile(array, 0.975)),
        }
    return result


def saturation_summary(frame: pd.DataFrame, draws: int, seed: int) -> tuple[pd.DataFrame, dict]:
    frame = frame.copy()
    frame["old_saturation_bin"] = pd.cut(
        frame.old_saturation_ratio,
        bins=SATURATION_BINS,
        labels=SATURATION_LABELS,
        include_lowest=True,
    )
    rows = []
    bootstraps = {}
    for index, label in enumerate(SATURATION_LABELS):
        subset = frame[frame.old_saturation_bin == label]
        if subset.empty:
            continue
        point = summarize(subset)
        point["old_saturation_bin"] = label
        point["old_saturation_ratio_mean"] = float(subset.old_saturation_ratio.mean())
        rows.append(point)
        bootstraps[label] = cluster_bootstrap(subset, draws=draws, seed=seed + index + 1)
    return pd.DataFrame(rows), bootstraps


def plot_saturation(summary: pd.DataFrame, bootstraps: dict, out: Path) -> None:
    panels = (
        ("delta_ecr_pooled", "ΔECR (current − old)"),
        ("old_zero_to_new_nonzero_group", "Old zero → current nonzero groups"),
        ("mean_abs_advantage_change", "Mean |A_current − A_old|"),
        ("direct_sign_flip_fraction_all", "Direct +↔− response flips"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    labels = summary.old_saturation_bin.tolist()
    x = np.arange(len(labels))
    for axis, (metric, title) in zip(axes.flat, panels):
        values = summary[metric].to_numpy(float)
        lows = np.asarray([bootstraps[label][metric]["ci_low"] for label in labels])
        highs = np.asarray([bootstraps[label][metric]["ci_high"] for label in labels])
        axis.errorbar(
            x,
            values,
            yerr=np.vstack([values - lows, highs - values]),
            fmt="o-",
            capsize=4,
            color="#2A6F97",
        )
        axis.axhline(0.0, color="#777777", linewidth=0.8)
        axis.set_xticks(x, labels, rotation=12)
        axis.set_title(title)
        axis.set_xlabel("Old-rubric all-pass saturation")
        axis.grid(alpha=0.2)
    fig.suptitle("E1: adjacent rubric refresh value on fixed Pool-B responses")
    fig.savefig(out / "figure_e1_saturation_vs_refresh.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.bootstrap <= 0:
        raise ValueError("--bootstrap must be positive")
    source = args.source.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    steps = discover_steps(source)
    records = []
    inputs = []
    for old_step, policy_step in zip(steps, steps[1:]):
        old_path = cell_path(source, policy_step, old_step)
        new_path = cell_path(source, policy_step, policy_step)
        old_groups = load_cell(old_path)
        new_groups = load_cell(new_path)
        if old_groups.keys() != new_groups.keys():
            raise ValueError(f"prompt inventories differ for {old_step}->{policy_step}")
        inputs.extend((old_path, new_path))
        for prompt_id in sorted(old_groups):
            records.append(
                analyze_pair(
                    policy_step=policy_step,
                    old_step=old_step,
                    prompt_id=prompt_id,
                    old_rows=old_groups[prompt_id],
                    new_rows=new_groups[prompt_id],
                )
            )
    frame = pd.DataFrame(records)
    if len(frame) != (len(steps) - 1) * 100:
        raise AssertionError("unexpected adjacent-pair inventory")
    overall = summarize(frame)
    overall_bootstrap = cluster_bootstrap(frame, draws=args.bootstrap, seed=args.seed)
    by_saturation, saturation_bootstrap = saturation_summary(
        frame, draws=args.bootstrap, seed=args.seed
    )
    checkpoint_rows = []
    for (old_step, policy_step), group in frame.groupby(
        ["old_evaluator_step", "policy_step"], sort=True
    ):
        row = summarize(group)
        row.update(old_evaluator_step=int(old_step), policy_step=int(policy_step))
        checkpoint_rows.append(row)
    by_checkpoint = pd.DataFrame(checkpoint_rows)
    frame["old_saturation_bin"] = pd.cut(
        frame.old_saturation_ratio,
        bins=SATURATION_BINS,
        labels=SATURATION_LABELS,
        include_lowest=True,
    )
    frame.to_csv(out / "prompt_pair_metrics.csv", index=False)
    by_checkpoint.to_csv(out / "checkpoint_pair_summary.csv", index=False)
    by_saturation.to_csv(out / "saturation_bin_summary.csv", index=False)
    plot_saturation(by_saturation, saturation_bootstrap, out)
    report = {
        "schema_version": 1,
        "analysis": "E1_adjacent_rubric_refresh_value",
        "source": str(source),
        "method": "online",
        "data_role": "heldout_validation",
        "pool": "probe_B",
        "saved_checkpoints": steps,
        "adjacent_pairs": [[old, new] for old, new in zip(steps, steps[1:])],
        "advantage": {"std": "sample_ddof1", "epsilon": EPSILON},
        "ptr": "exact reward equality",
        "saturation": "criterion all-pass across the 16 fixed responses",
        "bootstrap": {
            "draws": args.bootstrap,
            "seed": args.seed,
            "unit": "prompt trajectory; all checkpoint-pair rows retained per sampled prompt",
        },
        "overall": overall,
        "overall_bootstrap_95ci": overall_bootstrap,
        "saturation_bins": by_saturation.to_dict(orient="records"),
        "saturation_bin_bootstrap_95ci": saturation_bootstrap,
        "input_files": [
            {"path": str(path), "size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
            for path in dict.fromkeys(inputs)
        ],
        "interpretation_boundary": (
            "Paired fixed-response sensitivity analysis only; it does not establish policy-level "
            "causal benefit or harm from rubric refreshes."
        ),
    }
    (out / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps({"out": str(out), "overall": overall}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
