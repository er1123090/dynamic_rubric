#!/usr/bin/env python3
"""Recompute through-34 RQ2 discriminability statistics from raw audit grades."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from fractions import Fraction
import json
import math
from pathlib import Path
import random
import statistics
from typing import Any, Callable, Iterable, Sequence


from dynamic_rubric.artifacts import read_json, write_json_atomic
from dynamic_rubric.hashing import sha256_file
from dynamic_rubric.phase1.metrics import kendall_tau_b


METRIC_KEYS = (
    "v_adj_zar",
    "delta_near_zar",
    "delta_tie_rate",
    "delta_exact_tie_rate",
    "delta_separation_rate",
    "delta_effective_ratio",
    "delta_margin",
    "delta_reward_std",
    "kendall_tau_b",
    "tie_resolution_rate",
    "new_tie_rate",
    "new_tie_rate_among_stale_separated",
)


def _pairs(size: int) -> Iterable[tuple[int, int]]:
    for left in range(size - 1):
        for right in range(left + 1, size):
            yield left, right


def _reward(row: dict[str, Any]) -> Fraction:
    numerator, denominator = int(row["numerator"]), int(row["denominator"])
    if denominator <= 0:
        raise ValueError("reward denominator must be positive")
    exact = Fraction(numerator, denominator)
    if not math.isclose(float(exact), float(row["reward"]), rel_tol=1e-14, abs_tol=1e-15):
        raise ValueError("stored reward differs from exact numerator/denominator")
    return exact


def evaluator_metrics(rows: Sequence[dict[str, Any]], epsilon_z: float, epsilon_t: float) -> dict:
    ordered = sorted(rows, key=lambda row: row["response_id"])
    if len(ordered) < 2 or len({row["response_id"] for row in ordered}) != len(ordered):
        raise ValueError("response group must contain unique IDs")
    rewards = [_reward(row) for row in ordered]
    floats = [float(value) for value in rewards]
    criterion_maps = [dict(row["grades"]) for row in ordered]
    criterion_ids = set(criterion_maps[0])
    if any(set(item) != criterion_ids for item in criterion_maps):
        raise ValueError("criterion inventory differs within evaluator group")
    counts = Counter()
    for criterion_id in criterion_ids:
        grades = [int(item[criterion_id]) for item in criterion_maps]
        if any(value not in (0, 1) for value in grades):
            raise ValueError("criterion grades must be binary")
        counts["saturated" if all(grades) else "dead" if not any(grades) else "effective"] += 1
    pair_list = list(_pairs(len(rewards)))
    tied = sum(abs(floats[a] - floats[b]) <= epsilon_t for a, b in pair_list)
    criterion_count = len(criterion_ids)
    return {
        "response_ids": [row["response_id"] for row in ordered],
        "rewards": rewards,
        "zar": int(len(set(rewards)) == 1),
        "near_zar": int(statistics.pstdev(floats) <= epsilon_z),
        "reward_std": statistics.pstdev(floats),
        "tie_count": tied,
        "exact_tie_count": sum(rewards[a] == rewards[b] for a, b in pair_list),
        "pair_count": len(pair_list),
        "tie_rate": tied / len(pair_list),
        "separation_rate": 1 - tied / len(pair_list),
        "top_median_margin": max(floats) - statistics.median(floats),
        "criterion_count": criterion_count,
        "effective_count": counts["effective"],
        "saturated_count": counts["saturated"],
        "dead_count": counts["dead"],
        "effective_ratio": counts["effective"] / criterion_count,
        "saturated_ratio": counts["saturated"] / criterion_count,
        "dead_ratio": counts["dead"] / criterion_count,
    }


def group_row(group: dict[str, Any], stratum: str, epsilon_z: float, epsilon_t: float) -> dict:
    fresh = evaluator_metrics(group["fresh"], epsilon_z, epsilon_t)
    stale = evaluator_metrics(group["stale"], epsilon_z, epsilon_t)
    if fresh["response_ids"] != stale["response_ids"]:
        raise ValueError("fresh/stale response IDs differ")
    prompt_id = str(group["prompt_id"])
    if any(str(row["prompt_id"]) != prompt_id for row in [*group["fresh"], *group["stale"]]):
        raise ValueError("row prompt ID differs from group prompt ID")
    age = int(group["global_step"]) - int(group["stale_creation_update"])
    if age != int(group["evaluator_age_steps"]):
        raise ValueError("evaluator age metadata is inconsistent")
    resolved = created = 0
    for left, right in _pairs(len(fresh["rewards"])):
        stale_tie = abs(float(stale["rewards"][left]) - float(stale["rewards"][right])) <= epsilon_t
        fresh_tie = abs(float(fresh["rewards"][left]) - float(fresh["rewards"][right])) <= epsilon_t
        resolved += stale_tie and not fresh_tie
        created += fresh_tie and not stale_tie
    contingency = (
        "rescue"
        if stale["zar"] and not fresh["zar"]
        else "harm"
        if fresh["zar"] and not stale["zar"]
        else "both_zero"
        if fresh["zar"] and stale["zar"]
        else "both_usable"
    )
    tau = kendall_tau_b([float(x) for x in stale["rewards"]], [float(x) for x in fresh["rewards"]])
    clock = group.get("clock", {})
    row = {
        "stratum": stratum,
        "global_step": int(group["global_step"]),
        "response_count": len(fresh["response_ids"]),
        "prompt_id": prompt_id,
        "evaluator_age_steps": age,
        "fresh_creation_update": int(group["fresh_creation_update"]),
        "stale_creation_update": int(group["stale_creation_update"]),
        "contingency": contingency,
        "v_adj_zar": stale["zar"] - fresh["zar"],
        "delta_near_zar": stale["near_zar"] - fresh["near_zar"],
        "delta_tie_rate": stale["tie_rate"] - fresh["tie_rate"],
        "delta_separation_rate": fresh["separation_rate"] - stale["separation_rate"],
        "delta_effective_ratio": fresh["effective_ratio"] - stale["effective_ratio"],
        "delta_margin": fresh["top_median_margin"] - stale["top_median_margin"],
        "delta_reward_std": fresh["reward_std"] - stale["reward_std"],
        "kendall_tau_b": tau,
        "stale_ties": stale["tie_count"],
        "fresh_ties": fresh["tie_count"],
        "stale_exact_ties": stale["exact_tie_count"],
        "fresh_exact_ties": fresh["exact_tie_count"],
        "delta_exact_tie_rate": (stale["exact_tie_count"] - fresh["exact_tie_count"])
        / stale["pair_count"],
        "pair_count": stale["pair_count"],
        "stale_separated": stale["pair_count"] - stale["tie_count"],
        "ties_resolved": resolved,
        "ties_created": created,
        "tie_resolution_rate": resolved / stale["tie_count"] if stale["tie_count"] else None,
        "new_tie_rate": created / stale["pair_count"],
        "new_tie_rate_among_stale_separated": (
            created / (stale["pair_count"] - stale["tie_count"])
            if stale["pair_count"] > stale["tie_count"]
            else None
        ),
    }
    for side, metrics in (("fresh", fresh), ("stale", stale)):
        for key in (
            "zar",
            "near_zar",
            "reward_std",
            "tie_rate",
            "separation_rate",
            "top_median_margin",
            "criterion_count",
            "effective_count",
            "saturated_count",
            "dead_count",
            "effective_ratio",
            "saturated_ratio",
            "dead_ratio",
        ):
            row[f"{side}_{key}"] = metrics[key]
    for key, value in clock.items():
        row[key] = value
    if stale["tie_count"] - fresh["tie_count"] != resolved - created:
        raise ValueError("tie transition counts do not reconcile")
    saved = group.get("comparison")
    if saved:
        checks = {
            "v_adj_zar": row["v_adj_zar"],
            "delta_near_zero_advantage": row["delta_near_zar"],
            "delta_tie_rate": row["delta_tie_rate"],
            "delta_separation_rate": row["delta_separation_rate"],
            "delta_effective_criterion_ratio": row["delta_effective_ratio"],
            "delta_top_median_margin": row["delta_margin"],
            "kendall_tau_b": row["kendall_tau_b"],
        }
        for key, expected in checks.items():
            actual = saved.get(key)
            equal = (
                actual is expected
                if expected is None
                else actual is not None
                and math.isclose(float(actual), float(expected), rel_tol=1e-14, abs_tol=1e-15)
            )
            if not equal:
                raise ValueError(f"saved metric {key} differs from raw-grade recomputation")
    return row


def _cluster_contributions(
    rows: Sequence[dict], specs: dict[str, Callable[[dict], tuple[float, float]]]
):
    prompts = sorted({row["prompt_id"] for row in rows})
    by_prompt = defaultdict(list)
    for row in rows:
        by_prompt[row["prompt_id"]].append(row)
    numerator = [[0.0 for _ in specs] for _ in prompts]
    denominator = [[0.0 for _ in specs] for _ in prompts]
    for p_index, prompt in enumerate(prompts):
        for row in by_prompt[prompt]:
            for m_index, function in enumerate(specs.values()):
                value, weight = function(row)
                if weight:
                    numerator[p_index][m_index] += value
                    denominator[p_index][m_index] += weight
    return prompts, numerator, denominator


def cluster_bootstrap(
    rows: Sequence[dict],
    iterations: int,
    seed: int,
    specs: dict[str, Callable[[dict], tuple[float, float]]],
) -> dict:
    prompts, numerator, denominator = _cluster_contributions(rows, specs)
    try:
        import numpy as np
    except ImportError:
        rng = random.Random(seed)
        draws = [[] for _ in specs]
        for _ in range(iterations):
            indices = [rng.randrange(len(prompts)) for _ in prompts]
            for metric in range(len(specs)):
                num = sum(numerator[index][metric] for index in indices)
                den = sum(denominator[index][metric] for index in indices)
                if den:
                    draws[metric].append(num / den)

        def quantile(values, q):
            return sorted(values)[round(q * (len(values) - 1))]
    else:
        rng = np.random.default_rng(seed)
        arrays_num, arrays_den = np.asarray(numerator), np.asarray(denominator)
        array_draws = np.empty((iterations, len(specs)))
        for start in range(0, iterations, 500):
            stop = min(iterations, start + 500)
            indices = rng.integers(0, len(prompts), size=(stop - start, len(prompts)))
            num = arrays_num[indices].sum(axis=1)
            den = arrays_den[indices].sum(axis=1)
            array_draws[start:stop] = np.divide(
                num, den, out=np.full_like(num, np.nan), where=den != 0
            )
        draws = [
            list(array_draws[:, index][~np.isnan(array_draws[:, index])])
            for index in range(len(specs))
        ]

        def quantile(values, q):
            return float(np.quantile(values, q))

    result = {}
    for index, key in enumerate(specs):
        point_num = sum(row[index] for row in numerator)
        point_den = sum(row[index] for row in denominator)
        valid = draws[index]
        result[key] = {
            "estimate": point_num / point_den if point_den else None,
            "ci95": [quantile(valid, 0.025), quantile(valid, 0.975)] if valid else None,
            "bootstrap_defined": len(valid),
            "cluster_count": len(prompts),
            "visit_count": len(rows),
        }
    return result


def cluster_bootstrap_ratio_differences(
    rows: Sequence[dict],
    iterations: int,
    seed: int,
    specs: dict[str, Callable[[dict], tuple[float, float, float, float]]],
) -> dict:
    """Cluster-bootstrap differences between two separately pooled ratios."""
    prompts = sorted({row["prompt_id"] for row in rows})
    by_prompt = defaultdict(list)
    for row in rows:
        by_prompt[row["prompt_id"]].append(row)
    contributions = []
    for prompt in prompts:
        prompt_values = []
        for function in specs.values():
            values = [function(row) for row in by_prompt[prompt]]
            prompt_values.append(tuple(sum(value[index] for value in values) for index in range(4)))
        contributions.append(prompt_values)

    def evaluate(sample: Sequence[int], metric: int) -> float | None:
        totals = [sum(contributions[index][metric][part] for index in sample) for part in range(4)]
        return totals[0] / totals[1] - totals[2] / totals[3] if totals[1] and totals[3] else None

    rng = random.Random(seed)
    draws = [[] for _ in specs]
    for _ in range(iterations):
        sample = [rng.randrange(len(prompts)) for _ in prompts]
        for metric in range(len(specs)):
            value = evaluate(sample, metric)
            if value is not None:
                draws[metric].append(value)

    result = {}
    all_prompts = list(range(len(prompts)))
    for index, key in enumerate(specs):
        values = sorted(draws[index])
        result[key] = {
            "estimate": evaluate(all_prompts, index),
            "ci95": (
                [values[round(0.025 * (len(values) - 1))], values[round(0.975 * (len(values) - 1))]]
                if values
                else None
            ),
            "bootstrap_defined": len(values),
            "cluster_count": len(prompts),
            "visit_count": len(rows),
            "estimand": "criterion-pooled fresh ratio minus criterion-pooled stale ratio",
        }
    return result


def _specs() -> dict[str, Callable[[dict], tuple[float, float]]]:
    ratio_keys = {
        "kendall_tau_b",
        "tie_resolution_rate",
        "new_tie_rate",
        "new_tie_rate_among_stale_separated",
    }
    specs = {
        key: (lambda row, key=key: (float(row[key]), 1.0))
        for key in METRIC_KEYS
        if key not in ratio_keys
    }
    specs["kendall_tau_b"] = lambda row: (
        (float(row["kendall_tau_b"]), 1.0) if row["kendall_tau_b"] is not None else (0.0, 0.0)
    )
    specs["tie_resolution_rate"] = lambda row: (row["ties_resolved"], row["stale_ties"])
    specs["new_tie_rate"] = lambda row: (row["ties_created"], row["pair_count"])
    specs["new_tie_rate_among_stale_separated"] = lambda row: (
        row["ties_created"],
        row["stale_separated"],
    )
    return specs


def summarize(rows: Sequence[dict], iterations: int, seed: int) -> dict:
    bootstrap = cluster_bootstrap(rows, iterations, seed, _specs())
    pooled_criterion_bootstrap = cluster_bootstrap_ratio_differences(
        rows,
        iterations,
        seed,
        {
            f"delta_{kind}_ratio": (
                lambda row, kind=kind: (
                    row[f"fresh_{kind}_count"],
                    row["fresh_criterion_count"],
                    row[f"stale_{kind}_count"],
                    row["stale_criterion_count"],
                )
            )
            for kind in ("effective", "saturated", "dead")
        },
    )
    contingency = Counter(row["contingency"] for row in rows)
    result = {
        "groups": len(rows),
        "unique_prompts": len({row["prompt_id"] for row in rows}),
        "responses": sum(row["response_count"] for row in rows),
        "pairs": sum(row["pair_count"] for row in rows),
        "contingency": dict(contingency),
        "ties": {
            "stale": sum(row["stale_ties"] for row in rows),
            "fresh": sum(row["fresh_ties"] for row in rows),
            "stale_exact_epsilon0": sum(row["stale_exact_ties"] for row in rows),
            "fresh_exact_epsilon0": sum(row["fresh_exact_ties"] for row in rows),
            "resolved": sum(row["ties_resolved"] for row in rows),
            "newly_created": sum(row["ties_created"] for row in rows),
            "stale_separated": sum(row["stale_separated"] for row in rows),
        },
        "tie_rate_definitions": {
            "new_tie_rate": "newly created ties / all pairs",
            "new_tie_rate_among_stale_separated": "newly created ties / stale-separated pairs",
            "tie_resolution_rate": "resolved ties / stale-tied pairs",
        },
        "kendall_tau_b_undefined": sum(row["kendall_tau_b"] is None for row in rows),
        "bootstrap": bootstrap,
        "criterion_pooled_bootstrap_primary": pooled_criterion_bootstrap,
        "criterion_pooled_primary": {},
        "criterion_prompt_mean_sensitivity": {},
        "age_distribution": {},
    }
    for side in ("fresh", "stale"):
        total = sum(row[f"{side}_criterion_count"] for row in rows)
        result["criterion_pooled_primary"][side] = {
            key: sum(row[f"{side}_{key}_count"] for row in rows) / total
            for key in ("effective", "saturated", "dead")
        } | {
            "criteria": total,
            **{
                f"{key}_count": sum(row[f"{side}_{key}_count"] for row in rows)
                for key in ("effective", "saturated", "dead")
            },
        }
        result["criterion_prompt_mean_sensitivity"][side] = {
            key: statistics.fmean(row[f"{side}_{key}_ratio"] for row in rows)
            for key in ("effective", "saturated", "dead")
        } | {"prompt_visits": len(rows)}
    ages = [row["evaluator_age_steps"] for row in rows]
    result["age_distribution"] = {
        "counts": {str(k): v for k, v in sorted(Counter(ages).items())},
        "min": min(ages),
        "median": statistics.median(ages),
        "max": max(ages),
    }
    return result


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _per_step(rows: Sequence[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["global_step"]].append(row)
    output = []
    for step, values in sorted(grouped.items()):
        item = {
            "global_step": step,
            "groups": len(values),
            "unique_prompts": len({x["prompt_id"] for x in values}),
        }
        for key in METRIC_KEYS:
            present = [float(row[key]) for row in values if row[key] is not None]
            item[key] = statistics.fmean(present) if present else None
        item.update(
            {
                f"contingency_{key}": sum(row["contingency"] == key for row in values)
                for key in ("rescue", "harm", "both_zero", "both_usable")
            }
        )
        output.append(item)
    return output


def _correlations(rows: Sequence[dict]) -> list[dict]:
    from scipy.stats import pearsonr, spearmanr

    predictors = (
        "evaluator_age_steps",
        "cumulative_prompts_since_evaluator",
        "cumulative_completions_since_evaluator",
        "cumulative_prompt_exposures",
    )
    output = []
    for predictor in predictors:
        for outcome in ("v_adj_zar", "delta_tie_rate", "delta_effective_ratio", "delta_margin"):
            pairs = [
                (float(row[predictor]), float(row[outcome])) for row in rows if predictor in row
            ]
            x, y = zip(*pairs)
            output.append(
                {
                    "predictor": predictor,
                    "outcome": outcome,
                    "n_visits": len(pairs),
                    "pearson_r": float(pearsonr(x, y).statistic)
                    if len(set(x)) > 1 and len(set(y)) > 1
                    else None,
                    "spearman_r": float(spearmanr(x, y).statistic)
                    if len(set(x)) > 1 and len(set(y)) > 1
                    else None,
                    "role": "exploratory descriptive association; no threshold or causal claim",
                }
            )
    return output


def analyze(args: argparse.Namespace) -> dict:
    roots = {"inference_b": args.inference_b.resolve(), "trainer": args.trainer.resolve()}
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    strata = {}
    files_by_stratum = {}
    for stratum, root in roots.items():
        files = sorted((root / "groups").glob("step-*/*.json"))
        if len(files) != args.expected_groups:
            raise ValueError(f"{stratum} coverage is {len(files)}/{args.expected_groups}")
        files_by_stratum[stratum] = files
        rows = [
            group_row(read_json(path), stratum, args.epsilon_z, args.epsilon_t) for path in files
        ]
        if len({(row["global_step"], row["prompt_id"]) for row in rows}) != len(rows):
            raise ValueError(f"duplicate {stratum} group identity")
        strata[stratum] = rows

    inference_b_by_key = {(row["global_step"], row["prompt_id"]): row for row in strata["inference_b"]}
    trainer_by_key = {(row["global_step"], row["prompt_id"]): row for row in strata["trainer"]}
    if inference_b_by_key.keys() != trainer_by_key.keys():
        raise ValueError("Inference B and Trainer group identities differ")
    paired = []
    for key in sorted(inference_b_by_key):
        inference_b, trainer = inference_b_by_key[key], trainer_by_key[key]
        if inference_b["evaluator_age_steps"] != trainer["evaluator_age_steps"]:
            raise ValueError("Inference B/Trainer evaluator age differs")
        paired.append(
            {
                "global_step": key[0],
                "prompt_id": key[1],
                "evaluator_age_steps": inference_b["evaluator_age_steps"],
                **{
                    f"trainer_minus_inference_b_{metric}": (
                        None
                        if inference_b[metric] is None or trainer[metric] is None
                        else trainer[metric] - inference_b[metric]
                    )
                    for metric in METRIC_KEYS
                },
            }
        )

    summary = {
        "schema_version": 1,
        "analysis": "through34_actual_training_batch_dynamic_rubric_update",
        "thresholds": {"epsilon_z": args.epsilon_z, "epsilon_t": args.epsilon_t},
        "bootstrap": {
            "iterations": args.bootstrap_iterations,
            "seed": args.seed,
            "unit": "prompt_id cluster",
            "estimand": "visit-weighted",
        },
        "strata": {
            name: summarize(rows, args.bootstrap_iterations, args.seed)
            for name, rows in strata.items()
        },
        "hardware_effect_sensitivity": cluster_bootstrap(
            paired,
            args.bootstrap_iterations,
            args.seed,
            {
                metric: (
                    lambda row, metric=metric: (
                        float(row[f"trainer_minus_inference_b_{metric}"]),
                        1.0,
                    )
                    if row[f"trainer_minus_inference_b_{metric}"] is not None
                    else (0.0, 0.0)
                )
                for metric in METRIC_KEYS
            },
        ),
        "exploratory_correlations": {name: _correlations(rows) for name, rows in strata.items()},
        "count_weight_matched_subset": {
            "computed_in_this_script": False,
            "external_join_path": "analysis/count_weight_sensitivity.json",
            "reason": "per-criterion weights require a separate canonical rubric-union join",
        },
        "raw_recomputation_validation": {
            name: {
                "group_receipts_checked": len(rows),
                "saved_comparisons_checked": sum(
                    bool(read_json(path).get("comparison"))
                    for path in files_by_stratum[name]
                ),
                "rational_reward_rows_checked": sum(
                    row["response_count"] * 2 for row in rows
                ),
                "saved_comparison_mismatches": 0,
                "response_pool_identity_mismatches": 0,
                "evaluator_age_mismatches": 0,
                "policy": "fail closed before output on any mismatch",
            }
            for name, rows in strata.items()
        },
        "guards": {
            "fixed_probe": False,
            "ground_truth": False,
            "correctness_claim": False,
            "strata_merged": False,
            "same_response_pool": True,
        },
    }
    for name, rows in strata.items():
        _write_csv(output / "analysis" / f"per_group_{name}.csv", rows)
        _write_csv(output / "analysis" / f"per_step_{name}.csv", _per_step(rows))
    _write_csv(output / "paired" / "same_group_hardware_sensitivity.csv", paired)
    write_json_atomic(output / "analysis" / "summary.json", summary, immutable=False)
    write_json_atomic(
        output / "verification" / "inputs.json",
        {
            name: {"groups": len(files), "sha256": {str(path): sha256_file(path) for path in files}}
            for name, files in files_by_stratum.items()
        },
        immutable=False,
    )
    try:
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(8, 4.5))
        for name, rows in strata.items():
            per_step = _per_step(rows)
            axis.plot(
                [x["global_step"] for x in per_step],
                [x["v_adj_zar"] for x in per_step],
                marker="o",
                label=name,
            )
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set(
            xlabel="Global step",
            ylabel="Stale ZAR − fresh ZAR",
            title="Adjacent update value (actual training batches)",
        )
        axis.legend()
        figure.tight_layout()
        (output / "figures").mkdir(exist_ok=True)
        figure.savefig(output / "figures" / "update_value_by_step.png", dpi=180)
        plt.close(figure)
    except ImportError:
        summary["figures_unavailable"] = "matplotlib unavailable; numerical outputs complete"
        write_json_atomic(output / "analysis" / "summary.json", summary, immutable=False)
    return summary


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    base = Path("outputs/medicine/online_rubrics/seed-11")
    value.add_argument("--inference_b", type=Path, default=base / "phase1-audit-through34-20260907")
    value.add_argument(
        "--trainer", type=Path, default=base / "phase1-trainer-paired-audit-through34-20260907"
    )
    value.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/phase1/medicine/online_rubrics/seed-11/reanalysis-through34-20260908"
        ),
    )
    value.add_argument("--expected-groups", type=int, default=1692)
    value.add_argument("--epsilon-z", type=float, default=0.01)
    value.add_argument("--epsilon-t", type=float, default=0.01)
    value.add_argument("--bootstrap-iterations", type=int, default=20_000)
    value.add_argument("--seed", type=int, default=11)
    return value


def main(argv: Sequence[str] | None = None) -> None:
    result = analyze(parser().parse_args(argv))
    print(json.dumps({"strata": list(result["strata"]), "output": "complete"}, sort_keys=True))


if __name__ == "__main__":
    main()
