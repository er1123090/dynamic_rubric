#!/usr/bin/env python3
"""Build a sealed-source interim R0/current report while control grading runs."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import dataclasses
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from dynamic_rubric.config import load_config
from dynamic_rubric.evaluation.bootstrap import paired_prompt_bootstrap
from dynamic_rubric.horizon.observations import _verified_summary


AUDIT_CHECKPOINTS = (0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0)
BOOTSTRAP_ITERATIONS = 10_000
STEP_BY_CHECKPOINT = {0.2: 3, 0.4: 6, 0.6: 9, 0.8: 13, 1.0: 16, 1.5: 24, 2.0: 32, 2.5: 40, 3.0: 48}


def _mean(values: Iterable[float | int | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(value)]
    return math.fsum(finite) / len(finite) if finite else None


def _metric(row: Mapping[str, Any], variant: str, *keys: str) -> Any:
    value: Any = row["variants"][variant]
    for key in keys:
        value = value[key]
    return value


def _criterion_summary(rows: Sequence[Mapping[str, Any]], group: str) -> dict[str, Any]:
    counts = {
        state: sum(int(row["criterion_effectiveness"][group]["counts"][state]) for row in rows)
        for state in ("effective", "saturated", "dead")
    }
    criterion_count = sum(
        int(row["criterion_effectiveness"][group]["criterion_count"]) for row in rows
    )
    return {
        "mean_count_per_prompt": criterion_count / len(rows),
        "counts": counts,
        "ratios": {
            state: counts[state] / criterion_count if criterion_count else None for state in counts
        },
    }


def _paired_ci(
    rows: Sequence[Mapping[str, Any]],
    *,
    current_value: Any,
    baseline_value: Any,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    pairs = {
        str(row["prompt_id"]): (float(current_value(row)), float(baseline_value(row)))
        for row in rows
    }
    return dataclasses.asdict(
        paired_prompt_bootstrap(pairs, iterations=iterations, seed=seed)
    )


def aggregate_checkpoint(
    rows: Sequence[Mapping[str, Any]], *, bootstrap_iterations: int, bootstrap_seed: int
) -> dict[str, Any]:
    valid = [row for row in rows if row.get("analysis_status", "valid") == "valid"]
    if not valid:
        raise ValueError("checkpoint has no valid prompt summaries")
    checkpoint = float(valid[0]["checkpoint"])
    pool_family = str(valid[0]["pool_family"])
    if any(float(row["checkpoint"]) != checkpoint or row["pool_family"] != pool_family for row in valid):
        raise ValueError("checkpoint aggregation received mixed checkpoint or pool rows")

    r0_zar = _mean(_metric(row, "r0", "exact_zar") for row in valid)
    current_zar = _mean(_metric(row, "current", "exact_zar") for row in valid)
    r0_separation = _mean(
        _metric(row, "r0", "pairwise", "separation_rate") for row in valid
    )
    current_separation = _mean(
        _metric(row, "current", "pairwise", "separation_rate") for row in valid
    )
    online_count = _mean(int(row["online_criterion_count"]) for row in valid)
    assert r0_zar is not None and current_zar is not None
    assert r0_separation is not None and current_separation is not None
    assert online_count is not None
    refresh_gain = r0_zar - current_zar
    separation_gain = current_separation - r0_separation

    variants: dict[str, Any] = {}
    for variant in ("r0", "current"):
        variants[variant] = {
            "exact_zar": _mean(_metric(row, variant, "exact_zar") for row in valid),
            "near_zero_zar": _mean(
                _metric(row, variant, "near_zero_zar") for row in valid
            ),
            "advantage_degenerate_rate": _mean(
                _metric(row, variant, "advantage_degenerate") for row in valid
            ),
            "tie_rate": _mean(
                _metric(row, variant, "pairwise", "tie_rate") for row in valid
            ),
            "separation_rate": _mean(
                _metric(row, variant, "pairwise", "separation_rate") for row in valid
            ),
            "spread": {
                metric: _mean(_metric(row, variant, "spread", metric) for row in valid)
                for metric in ("population_sd", "iqr", "unique_score_ratio")
            },
        }

    ranking = {
        "kendall_tau_b": _mean(
            _metric(row, "current", "ranking_vs_r0", "kendall_tau_b") for row in valid
        ),
        "kendall_defined_prompt_rate": _mean(
            _metric(row, "current", "ranking_vs_r0", "kendall_defined") for row in valid
        ),
        "pairwise_ordering_agreement": _mean(
            _metric(row, "current", "ranking_vs_r0", "pairwise_ordering_agreement")
            for row in valid
        ),
        "top_set_jaccard": _mean(
            _metric(row, "current", "ranking_vs_r0", "top_set_jaccard") for row in valid
        ),
        "top_set_exact_match_rate": _mean(
            _metric(row, "current", "ranking_vs_r0", "top_set_exact_match") for row in valid
        ),
    }
    tie_resolution = {
        metric: _mean(
            _metric(row, "current", "pairwise_vs_r0", metric) for row in valid
        )
        for metric in (
            "incremental_tie_resolution",
            "conditional_tie_resolution",
            "new_tie_rate",
            "ordering_reversal_rate",
        )
    }
    return {
        "checkpoint": checkpoint,
        "policy_step": int(valid[0]["policy_step"]),
        "pool_family": pool_family,
        "prompt_count": len(valid),
        "response_count_per_prompt": int(valid[0]["response_count"]),
        "variants": variants,
        "gains": {
            "refresh_zar_r0_minus_current": refresh_gain,
            "separation_current_minus_r0": separation_gain,
            "refresh_per_added_criterion": refresh_gain / online_count if online_count else None,
            "separation_per_added_criterion": (
                separation_gain / online_count if online_count else None
            ),
        },
        "bootstrap_95ci": {
            "refresh_zar_r0_minus_current": _paired_ci(
                valid,
                current_value=lambda row: _metric(row, "r0", "exact_zar"),
                baseline_value=lambda row: _metric(row, "current", "exact_zar"),
                iterations=bootstrap_iterations,
                seed=bootstrap_seed,
            ),
            "separation_current_minus_r0": _paired_ci(
                valid,
                current_value=lambda row: _metric(
                    row, "current", "pairwise", "separation_rate"
                ),
                baseline_value=lambda row: _metric(row, "r0", "pairwise", "separation_rate"),
                iterations=bootstrap_iterations,
                seed=bootstrap_seed,
            ),
        },
        "tie_resolution": tie_resolution,
        "criterion_effectiveness": {
            "r0": _criterion_summary(valid, "r0"),
            "extension": _criterion_summary(valid, "extension"),
        },
        "mean_online_criterion_count": online_count,
        "ranking_current_vs_r0": ranking,
    }


def _load_series(
    root: Path,
    checkpoints: Sequence[float],
    *,
    bootstrap_iterations: int,
    bootstrap_seed: int,
    expected_prompt_count: int,
) -> tuple[list[dict[str, Any]], dict[float, list[Mapping[str, Any]]], list[dict[str, Any]]]:
    aggregates = []
    rows_by_checkpoint = {}
    sources = []
    expected_prompt_ids: set[str] | None = None
    for checkpoint in checkpoints:
        path = root / f"epoch-{checkpoint:.1f}" / "prompt_summary.jsonl"
        rows, seal = _verified_summary(path)
        prompt_ids = {str(row["prompt_id"]) for row in rows}
        if len(rows) != expected_prompt_count or len(prompt_ids) != expected_prompt_count:
            raise ValueError(
                f"expected exactly {expected_prompt_count} unique prompts at {path}, got {len(rows)} rows/{len(prompt_ids)} IDs"
            )
        if expected_prompt_ids is None:
            expected_prompt_ids = prompt_ids
        elif prompt_ids != expected_prompt_ids:
            raise ValueError(f"prompt ID set drift across checkpoints at {path}")
        rows_by_checkpoint[checkpoint] = rows
        aggregates.append(
            aggregate_checkpoint(
                rows,
                bootstrap_iterations=bootstrap_iterations,
                bootstrap_seed=bootstrap_seed,
            )
        )
        sources.append(
            {
                "path": str(path),
                "checkpoint": checkpoint,
                "comparison_scope": seal["comparison_scope"],
                "prompt_count": seal["prompt_count"],
                "config_hash": seal.get("config_hash"),
                "grader_model_revision": seal.get("grader_model_revision"),
                "tokenizer_revision": seal.get("tokenizer_revision"),
                "target_encoding_version": seal.get("target_encoding_version"),
                "seed_id": str(seal.get("seed_id")),
            }
        )
    return aggregates, rows_by_checkpoint, sources


def _generalization_gaps(
    pool_a: Mapping[float, Sequence[Mapping[str, Any]]],
    pool_b: Mapping[float, Sequence[Mapping[str, Any]]],
    *,
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    output = []
    for checkpoint in sorted(set(pool_a) & set(pool_b)):
        a = {str(row["prompt_id"]): row for row in pool_a[checkpoint]}
        b = {str(row["prompt_id"]): row for row in pool_b[checkpoint]}
        if set(a) != set(b):
            raise ValueError(f"Pool A/B prompt ID sets differ at checkpoint {checkpoint}")
        prompt_ids = sorted(a)

        def refresh(row: Mapping[str, Any]) -> float:
            return float(_metric(row, "r0", "exact_zar")) - float(
                _metric(row, "current", "exact_zar")
            )

        def separation(row: Mapping[str, Any]) -> float:
            return float(_metric(row, "current", "pairwise", "separation_rate")) - float(
                _metric(row, "r0", "pairwise", "separation_rate")
            )

        refresh_ci = paired_prompt_bootstrap(
            {prompt_id: (refresh(b[prompt_id]), refresh(a[prompt_id])) for prompt_id in prompt_ids},
            iterations=bootstrap_iterations,
            seed=bootstrap_seed,
        )
        separation_ci = paired_prompt_bootstrap(
            {
                prompt_id: (separation(b[prompt_id]), separation(a[prompt_id]))
                for prompt_id in prompt_ids
            },
            iterations=bootstrap_iterations,
            seed=bootstrap_seed,
        )
        mean_count = _mean(int(b[prompt_id]["online_criterion_count"]) for prompt_id in prompt_ids)
        assert mean_count is not None
        output.append(
            {
                "checkpoint": checkpoint,
                "paired_prompt_count": len(prompt_ids),
                "pool_b_minus_pool_a_refresh_gain": refresh_ci.point_estimate,
                "pool_b_minus_pool_a_separation_gain": separation_ci.point_estimate,
                "pool_b_minus_pool_a_refresh_per_added_criterion": (
                    refresh_ci.point_estimate / mean_count if mean_count else None
                ),
                "pool_b_minus_pool_a_separation_per_added_criterion": (
                    separation_ci.point_estimate / mean_count if mean_count else None
                ),
                "bootstrap_95ci": {
                    "refresh_gain_gap": dataclasses.asdict(refresh_ci),
                    "separation_gain_gap": dataclasses.asdict(separation_ci),
                },
            }
        )
    return output


def _kl_series(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {
        "source": str(path),
        "estimator_direction": raw["estimator_direction"],
        "pairs": [
            {
                key: pair[key]
                for key in (
                    "old_policy_step",
                    "new_policy_step",
                    "step_gap",
                    "prompt_count",
                    "response_count",
                    "k1_prompt_balanced_mean",
                    "k3_clipped_prompt_balanced_mean",
                    "k3_clipped_prompt_balanced_per_step_proxy",
                )
            }
            for pair in raw["pairs"]
        ],
    }


def _read_rubric_rows(path: Path) -> list[Mapping[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    required = {"prompt_id", "r0", "extension", "rejected"}
    if not rows or any(not required <= set(row) for row in rows):
        raise ValueError(f"rubric artifact does not match the expected schema: {path}")
    if len({str(row["prompt_id"]) for row in rows}) != len(rows):
        raise ValueError(f"rubric artifact has duplicate prompt IDs: {path}")
    return rows


def rubric_structural_analysis(
    rubric_root: Path, checkpoints: Sequence[float]
) -> dict[str, Any] | None:
    """Analyze only fields present in canonical rubric artifacts, without judge outputs."""
    if not rubric_root.is_dir():
        return None
    checkpoint_rows: dict[float, dict[str, Mapping[str, Any]]] = {}
    summaries = []
    for checkpoint in checkpoints:
        step = STEP_BY_CHECKPOINT[checkpoint]
        path = rubric_root / f"step-{step}.jsonl"
        if not path.is_file():
            continue
        rows = _read_rubric_rows(path)
        by_prompt = {str(row["prompt_id"]): row for row in rows}
        checkpoint_rows[checkpoint] = by_prompt
        extensions = [criterion for row in rows for criterion in row["extension"]]
        weights = Counter(str(c["weight_units"]) for c in extensions if "weight_units" in c)
        importance = Counter(str(c["importance_class"]) for c in extensions if "importance_class" in c)
        criterion_types = Counter(str(c["criterion_type"]) for c in extensions if "criterion_type" in c)
        rejection_reasons = Counter(
            str(rejection[1])
            for row in rows
            for rejection in row["rejected"]
            if isinstance(rejection, list) and len(rejection) >= 2
        )
        independent_path = rubric_root / "independent" / f"step-{step}.jsonl"
        independent_check = None
        if independent_path.is_file():
            independent = {str(row["prompt_id"]): row for row in _read_rubric_rows(independent_path)}
            shared = sorted(set(by_prompt) & set(independent))
            mismatch_count = sum(
                {str(item.get("canonical_criterion_hash")) for item in by_prompt[prompt_id]["extension"]}
                != {str(item.get("canonical_criterion_hash")) for item in independent[prompt_id]["extension"]}
                for prompt_id in shared
            )
            independent_check = {
                "path": str(independent_path),
                "paired_prompt_count": len(shared),
                "extension_hash_set_mismatch_count": mismatch_count,
            }
        extension_counts = [len(row["extension"]) for row in rows]
        summaries.append(
            {
                "checkpoint": checkpoint,
                "policy_step": step,
                "source": str(path),
                "prompt_count": len(rows),
                "extension_count": sum(extension_counts),
                "mean_extension_count": _mean(extension_counts),
                "zero_extension_prompt_rate": sum(count == 0 for count in extension_counts) / len(rows),
                "max_extension_count": max(extension_counts),
                "weight_units_histogram": dict(sorted(weights.items())),
                "importance_class_distribution": dict(sorted(importance.items())),
                "criterion_type_distribution": dict(sorted(criterion_types.items())),
                "rejection_reason_distribution": dict(sorted(rejection_reasons.items())),
                "independent_artifact_check": independent_check,
            }
        )
    churn = []
    ordered = sorted(checkpoint_rows)
    for previous, current in zip(ordered, ordered[1:]):
        old = checkpoint_rows[previous]
        new = checkpoint_rows[current]
        prompt_ids = sorted(set(old) & set(new))
        per_prompt = []
        for prompt_id in prompt_ids:
            old_hashes = {str(item["canonical_criterion_hash"]) for item in old[prompt_id]["extension"] if "canonical_criterion_hash" in item}
            new_hashes = {str(item["canonical_criterion_hash"]) for item in new[prompt_id]["extension"] if "canonical_criterion_hash" in item}
            union = old_hashes | new_hashes
            overlap = old_hashes & new_hashes
            per_prompt.append(
                {
                    "jaccard": len(overlap) / len(union) if union else 1.0,
                    "previous_retention": len(overlap) / len(old_hashes) if old_hashes else None,
                    "added": len(new_hashes - old_hashes),
                    "dropped": len(old_hashes - new_hashes),
                }
            )
        churn.append(
            {
                "previous_checkpoint": previous,
                "current_checkpoint": current,
                "paired_prompt_count": len(prompt_ids),
                "mean_extension_hash_jaccard": _mean(row["jaccard"] for row in per_prompt),
                "mean_previous_extension_retention": _mean(row["previous_retention"] for row in per_prompt),
                "mean_added_criteria": _mean(row["added"] for row in per_prompt),
                "mean_dropped_criteria": _mean(row["dropped"] for row in per_prompt),
            }
        )
    return {
        "status": "preliminary_structure_only",
        "note": "Canonical extension fields are analyzed without criterion grades; this is not a stale-control effectiveness result.",
        "checkpoints": summaries,
        "adjacent_extension_hash_churn": churn,
    }


def build_report(
    *,
    pool_b_root: Path,
    pool_a_root: Path,
    checkpoints: Sequence[float],
    baseline_path: Path | None,
    kl_path: Path,
    rubric_root: Path | None,
    bootstrap_iterations: int,
    bootstrap_seed: int,
    expected_config_hash: str,
    expected_seed_id: str,
    expected_grader_revision: str,
    expected_prompt_count: int,
) -> dict[str, Any]:
    pool_b, pool_b_rows, sources_b = _load_series(
        pool_b_root,
        checkpoints,
        bootstrap_iterations=bootstrap_iterations,
        bootstrap_seed=bootstrap_seed,
        expected_prompt_count=expected_prompt_count,
    )
    pool_a, pool_a_rows, sources_a = _load_series(
        pool_a_root,
        checkpoints,
        bootstrap_iterations=bootstrap_iterations,
        bootstrap_seed=bootstrap_seed,
        expected_prompt_count=expected_prompt_count,
    )
    all_sources = sources_a + sources_b
    for field, expected in (
        ("config_hash", expected_config_hash),
        ("seed_id", expected_seed_id),
        ("grader_model_revision", expected_grader_revision),
    ):
        observed = {source[field] for source in all_sources}
        if observed != {expected}:
            raise ValueError(f"score seal {field} mismatch: expected {expected!r}, got {observed!r}")
    for field in ("tokenizer_revision", "target_encoding_version"):
        observed = {source[field] for source in all_sources}
        if len(observed) != 1 or None in observed:
            raise ValueError(f"score seal {field} is absent or inconsistent: {observed!r}")
    expected_cells = len(checkpoints) * expected_prompt_count
    if sum(len(rows) for rows in pool_a_rows.values()) != expected_cells:
        raise ValueError("Pool A does not have the exact expected checkpoint/prompt grid")
    if sum(len(rows) for rows in pool_b_rows.values()) != expected_cells:
        raise ValueError("Pool B does not have the exact expected checkpoint/prompt grid")
    for checkpoint in checkpoints:
        if {str(row["prompt_id"]) for row in pool_a_rows[checkpoint]} != {
            str(row["prompt_id"]) for row in pool_b_rows[checkpoint]
        }:
            raise ValueError(f"Pool A/B prompt ID sets differ at checkpoint {checkpoint}")
    baseline = None
    baseline_source = None
    if baseline_path is not None and baseline_path.is_file():
        rows, seal = _verified_summary(baseline_path)
        baseline = aggregate_checkpoint(
            rows,
            bootstrap_iterations=bootstrap_iterations,
            bootstrap_seed=bootstrap_seed,
        )
        baseline_source = {
            "path": str(baseline_path),
            "comparison_scope": seal.get("comparison_scope"),
            "prompt_count": seal["prompt_count"],
        }
    return {
        "schema_version": 1,
        "artifact_type": "horizon_interim_r0_current_analysis",
        "status": "interim_without_stale_control",
        "bootstrap": {
            "unit": "paired_prompt_id",
            "iterations": bootstrap_iterations,
            "seed": bootstrap_seed,
            "confidence": 0.95,
        },
        "checkpoints": list(checkpoints),
        "coverage": {
            "expected_prompts_per_checkpoint": expected_prompt_count,
            "pool_a_cells": expected_cells,
            "pool_b_cells": expected_cells,
        },
        "baseline_pool_b": baseline,
        "pool_b": pool_b,
        "pool_a_auxiliary": pool_a,
        "pool_a_vs_b_generalization": _generalization_gaps(
            pool_a_rows,
            pool_b_rows,
            bootstrap_iterations=bootstrap_iterations,
            bootstrap_seed=bootstrap_seed,
        ),
        "adjacent_kl": _kl_series(kl_path),
        "rubric_structure": rubric_structural_analysis(rubric_root, checkpoints) if rubric_root is not None else None,
        "sources": {"pool_b": sources_b, "pool_a_auxiliary": sources_a, "baseline": baseline_source},
    }


def _flat_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    gaps = {row["checkpoint"]: row for row in report["pool_a_vs_b_generalization"]}
    rows = []
    for pool_key in ("pool_b", "pool_a_auxiliary"):
        for item in report[pool_key]:
            gap = gaps[item["checkpoint"]]
            rows.append(
                {
                    "pool": item["pool_family"],
                    "checkpoint": item["checkpoint"],
                    "policy_step": item["policy_step"],
                    "prompt_count": item["prompt_count"],
                    "r0_exact_zar": item["variants"]["r0"]["exact_zar"],
                    "current_exact_zar": item["variants"]["current"]["exact_zar"],
                    "r0_advantage_degenerate_rate": item["variants"]["r0"]["advantage_degenerate_rate"],
                    "current_advantage_degenerate_rate": item["variants"]["current"]["advantage_degenerate_rate"],
                    "refresh_gain": item["gains"]["refresh_zar_r0_minus_current"],
                    "refresh_ci_low": item["bootstrap_95ci"]["refresh_zar_r0_minus_current"]["ci_low"],
                    "refresh_ci_high": item["bootstrap_95ci"]["refresh_zar_r0_minus_current"]["ci_high"],
                    "r0_near_zar": item["variants"]["r0"]["near_zero_zar"],
                    "current_near_zar": item["variants"]["current"]["near_zero_zar"],
                    "r0_tie_rate": item["variants"]["r0"]["tie_rate"],
                    "current_tie_rate": item["variants"]["current"]["tie_rate"],
                    "r0_separation_rate": item["variants"]["r0"]["separation_rate"],
                    "current_separation_rate": item["variants"]["current"]["separation_rate"],
                    "separation_gain": item["gains"]["separation_current_minus_r0"],
                    "separation_ci_low": item["bootstrap_95ci"]["separation_current_minus_r0"]["ci_low"],
                    "separation_ci_high": item["bootstrap_95ci"]["separation_current_minus_r0"]["ci_high"],
                    "incremental_tie_resolution": item["tie_resolution"]["incremental_tie_resolution"],
                    "r0_effective_ratio": item["criterion_effectiveness"]["r0"]["ratios"]["effective"],
                    "extension_effective_ratio": item["criterion_effectiveness"]["extension"]["ratios"]["effective"],
                    "mean_online_criterion_count": item["mean_online_criterion_count"],
                    "refresh_per_added_criterion": item["gains"]["refresh_per_added_criterion"],
                    "separation_per_added_criterion": item["gains"]["separation_per_added_criterion"],
                    "r0_population_sd": item["variants"]["r0"]["spread"]["population_sd"],
                    "current_population_sd": item["variants"]["current"]["spread"]["population_sd"],
                    "r0_iqr": item["variants"]["r0"]["spread"]["iqr"],
                    "current_iqr": item["variants"]["current"]["spread"]["iqr"],
                    "r0_unique_score_ratio": item["variants"]["r0"]["spread"]["unique_score_ratio"],
                    "current_unique_score_ratio": item["variants"]["current"]["spread"]["unique_score_ratio"],
                    "kendall_tau_b": item["ranking_current_vs_r0"]["kendall_tau_b"],
                    "pairwise_ordering_agreement": item["ranking_current_vs_r0"]["pairwise_ordering_agreement"],
                    "top_set_jaccard": item["ranking_current_vs_r0"]["top_set_jaccard"],
                    "pool_b_minus_a_refresh_gap": gap["pool_b_minus_pool_a_refresh_gain"],
                    "pool_b_minus_a_separation_gap": gap["pool_b_minus_pool_a_separation_gain"],
                }
            )
    return rows


def write_outputs(report: Mapping[str, Any], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "interim_r0_current.json"
    csv_path = output_dir / "interim_r0_current.csv"
    markdown_path = output_dir / "interim_r0_current.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rows = _flat_rows(report)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Medicine interim R0/current analysis",
        "",
        "This report uses only digest-verified Pool A/B summaries; stale-control results are intentionally excluded.",
        "",
        "| Pool | Epoch | Step | R0 ZAR | Rt ZAR | Refresh gain [95% CI] | Separation gain [95% CI] | Added criteria |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['pool']} | {row['checkpoint']:.1f} | {row['policy_step']} | "
            f"{row['r0_exact_zar']:.3f} | {row['current_exact_zar']:.3f} | "
            f"{row['refresh_gain']:.3f} [{row['refresh_ci_low']:.3f}, {row['refresh_ci_high']:.3f}] | "
            f"{row['separation_gain']:.3f} [{row['separation_ci_low']:.3f}, {row['separation_ci_high']:.3f}] | "
            f"{row['mean_online_criterion_count']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "Pool A is criterion-elicitation data, so its gains are in-sample. Pool B is the leakage-safe primary evaluation pool. Pool B minus Pool A gaps quantify this generalization difference but do not replace the stale-control comparison.",
            "",
            "Adjacent KL values in the JSON are diagnostic policy-drift estimates, not causal explanations of rubric refresh gain.",
        ]
    )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": json_path, "csv": csv_path, "markdown": markdown_path}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/horizon_medicine.yaml"))
    parser.add_argument(
        "--pool-b-root",
        type=Path,
        default=Path("artifacts/horizon/medicine/scores/seed-11/immediate"),
    )
    parser.add_argument(
        "--pool-a-root",
        type=Path,
        default=Path("artifacts/horizon/medicine/scores_pool_a_auxiliary/seed-11"),
    )
    parser.add_argument(
        "--baseline-path",
        type=Path,
        default=Path("artifacts/horizon/medicine/scores/seed-11/epoch-0.0/prompt_summary.jsonl"),
    )
    parser.add_argument(
        "--kl-path",
        type=Path,
        default=Path("artifacts/horizon/medicine/checkpoint_kl/seed-11/adjacent_kl_summary.json"),
    )
    parser.add_argument(
        "--rubric-root",
        type=Path,
        default=Path("artifacts/horizon/medicine/rubrics/seed-11"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/horizon/medicine/interim_r0_current")
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=BOOTSTRAP_ITERATIONS)
    args = parser.parse_args()
    config = load_config(args.config)
    report = build_report(
        pool_b_root=args.pool_b_root,
        pool_a_root=args.pool_a_root,
        checkpoints=AUDIT_CHECKPOINTS,
        baseline_path=args.baseline_path,
        kl_path=args.kl_path,
        rubric_root=args.rubric_root,
        bootstrap_iterations=args.bootstrap_iterations,
        bootstrap_seed=config.bootstrap_seed,
        expected_config_hash=config.config_hash,
        expected_seed_id=str(config.horizon.training_seeds[0]),
        expected_grader_revision=str(config.models["proxy_grader"]["revision"]),
        expected_prompt_count=config.horizon.final_count,
    )
    outputs = write_outputs(report, args.output_dir)
    print(json.dumps({key: str(path) for key, path in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
