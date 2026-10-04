#!/usr/bin/env python3
"""Checkpoint-wise analysis for sealed horizon scores with incomplete stale controls.

This intentionally does not apply the complete-grid filter used by the confirmatory
observation builder.  Each checkpoint is analyzed on its own valid, count-matched
control subset and the excluded prompts remain explicit in the output.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

from dynamic_rubric.config import load_config
from dynamic_rubric.horizon.metrics import pairwise_metrics, ranking_agreement


CHECKPOINTS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0)
EXPECTED_VARIANTS = ("r0", "current", "control")
GAIN_FIELDS = (
    "exact_zar_reduction",
    "near_zero_zar_reduction",
    "tie_rate_reduction",
    "separation_gain",
    "population_sd_gain",
    "iqr_gain",
    "unique_score_ratio_gain",
)


class AnalysisError(RuntimeError):
    """Raised when a score shard is incomplete or its sealed lineage has drifted."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _epoch_label(checkpoint: float) -> str:
    return f"{checkpoint:.1f}"


def _mean(values: Sequence[float]) -> float | None:
    return math.fsum(values) / len(values) if values else None


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _variant_metrics(row: Mapping[str, Any], name: str) -> dict[str, float]:
    variant = row["variants"][name]
    return {
        "exact_zar": float(bool(variant["exact_zar"])),
        "near_zero_zar": float(bool(variant["near_zero_zar"])),
        "tie_rate": float(variant["pairwise"]["tie_rate"]),
        "separation_rate": float(variant["pairwise"]["separation_rate"]),
        "population_sd": float(variant["spread"]["population_sd"]),
        "iqr": float(variant["spread"]["iqr"]),
        "unique_score_ratio": float(variant["spread"]["unique_score_ratio"]),
    }


def _gain(current: Mapping[str, float], comparator: Mapping[str, float]) -> dict[str, float]:
    return {
        "exact_zar_reduction": comparator["exact_zar"] - current["exact_zar"],
        "near_zero_zar_reduction": comparator["near_zero_zar"] - current["near_zero_zar"],
        "tie_rate_reduction": comparator["tie_rate"] - current["tie_rate"],
        "separation_gain": current["separation_rate"] - comparator["separation_rate"],
        "population_sd_gain": current["population_sd"] - comparator["population_sd"],
        "iqr_gain": current["iqr"] - comparator["iqr"],
        "unique_score_ratio_gain": (
            current["unique_score_ratio"] - comparator["unique_score_ratio"]
        ),
    }


def _aggregate_effectiveness(
    rows: Sequence[Mapping[str, Any]], groups: Sequence[str]
) -> dict[str, Any]:
    counts = {name: 0 for name in ("saturated", "dead", "effective")}
    criterion_count = 0
    per_prompt_counts: list[int] = []
    for row in rows:
        prompt_count = 0
        for group in groups:
            summary = row["criterion_effectiveness"][group]
            prompt_count += int(summary["criterion_count"])
            criterion_count += int(summary["criterion_count"])
            for name in counts:
                counts[name] += int(summary["counts"][name])
        per_prompt_counts.append(prompt_count)
    return {
        "criterion_count": criterion_count,
        "mean_criteria_per_prompt": _mean([float(value) for value in per_prompt_counts]),
        "counts": counts,
        "ratios": {
            name: counts[name] / criterion_count if criterion_count else None for name in counts
        },
    }


def _bootstrap_gains(
    prompt_gains: Sequence[Mapping[str, float]],
    criterion_counts: Sequence[float],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    if not prompt_gains:
        raise AnalysisError("cannot bootstrap a checkpoint with zero valid prompts")
    point = {field: _mean([row[field] for row in prompt_gains]) for field in GAIN_FIELDS}
    count_mean = _mean(criterion_counts)
    normalized_point = {
        field: (point[field] / count_mean if count_mean else None) for field in GAIN_FIELDS
    }
    samples = {field: [] for field in GAIN_FIELDS}
    normalized_samples = {field: [] for field in GAIN_FIELDS}
    rng = random.Random(seed)
    n = len(prompt_gains)
    for _ in range(replicates):
        indices = [rng.randrange(n) for _ in range(n)]
        sampled_count = math.fsum(criterion_counts[index] for index in indices) / n
        for field in GAIN_FIELDS:
            value = math.fsum(prompt_gains[index][field] for index in indices) / n
            samples[field].append(value)
            if sampled_count:
                normalized_samples[field].append(value / sampled_count)
    return {
        "bootstrap_replicates": replicates,
        "point_estimate": point,
        "ci95": {
            field: [_quantile(samples[field], 0.025), _quantile(samples[field], 0.975)]
            for field in GAIN_FIELDS
        },
        "count_normalized": {
            "mean_current_extension_criteria": count_mean,
            "point_estimate": normalized_point,
            "ci95": {
                field: (
                    [
                        _quantile(normalized_samples[field], 0.025),
                        _quantile(normalized_samples[field], 0.975),
                    ]
                    if normalized_samples[field]
                    else None
                )
                for field in GAIN_FIELDS
            },
        },
    }


def _score_index(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], list[tuple[int, int]]]:
    grouped: dict[tuple[str, str], list[tuple[int, int, str]]] = {}
    for row in rows:
        key = (str(row["prompt_id"]), str(row["variant"]))
        grouped.setdefault(key, []).append(
            (
                int(row["score_numerator"]),
                int(row["score_denominator"]),
                str(row["response_id"]),
            )
        )
    result: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for key, values in grouped.items():
        ordered = sorted(values, key=lambda item: item[2])
        if len(ordered) != 16 or len({item[2] for item in ordered}) != 16:
            raise AnalysisError(f"variant score response grid is invalid: {key}")
        result[key] = [(item[0], item[1]) for item in ordered]
    return result


def _verify_shard(directory: Path, checkpoint: float) -> dict[str, Any]:
    paths = {
        "criterion_grades": directory / "criterion_grades.jsonl",
        "variant_scores": directory / "variant_scores.jsonl",
        "prompt_summary": directory / "prompt_summary.jsonl",
    }
    seal_path = directory / "score_seal.json"
    if not seal_path.is_file():
        raise AnalysisError(f"missing score seal: {seal_path}")
    seal = _read_json(seal_path)
    if seal.get("artifact_type") != "horizon_score_seal":
        raise AnalysisError(f"wrong score seal type: {seal_path}")
    if float(seal.get("checkpoint", -1)) != checkpoint:
        raise AnalysisError(f"checkpoint lineage mismatch: {seal_path}")
    if str(seal.get("pool_family")) != "pool_b" or str(seal.get("seed_id")) != "11":
        raise AnalysisError(f"pool or seed lineage mismatch: {seal_path}")
    if checkpoint != 0.0 and seal.get("comparison_scope") != "full":
        raise AnalysisError(f"nonzero checkpoint is not a full stale-control shard: {seal_path}")
    outputs = seal.get("outputs")
    if not isinstance(outputs, Mapping):
        raise AnalysisError(f"score seal outputs are malformed: {seal_path}")
    for name, path in paths.items():
        if not path.is_file() or outputs.get(name) != _sha256(path):
            raise AnalysisError(f"sealed output digest mismatch: {path}")
    sources = seal.get("sources")
    if not isinstance(sources, Mapping) or not sources:
        raise AnalysisError(f"score seal has no source lineage: {seal_path}")
    for source in sources.values():
        path = Path(str(source["path"]))
        if not path.is_file() or source.get("sha256") != _sha256(path):
            raise AnalysisError(f"sealed source digest mismatch: {path}")
    summary = _read_jsonl(paths["prompt_summary"])
    scores = _read_jsonl(paths["variant_scores"])
    grades = _read_jsonl(paths["criterion_grades"])
    if len(summary) != int(seal.get("prompt_count", -1)) or len(summary) != 100:
        raise AnalysisError(f"prompt count mismatch: {directory}")
    expected_score_count = sum(
        int(row["response_count"]) * (3 if str(row.get("analysis_status")) == "valid" else 2)
        for row in summary
    )
    if len(scores) != expected_score_count:
        raise AnalysisError(f"variant score count mismatch: {directory}")
    if len(grades) != int(seal.get("grade_count", -1)):
        raise AnalysisError(f"criterion grade count mismatch: {directory}")
    ids = [str(row["prompt_id"]) for row in summary]
    if len(ids) != len(set(ids)):
        raise AnalysisError(f"duplicate prompt summary row: {directory}")
    for row in summary:
        if float(row["checkpoint"]) != checkpoint or int(row["response_count"]) != 16:
            raise AnalysisError(f"prompt summary lineage mismatch: {directory}")
        status = str(row.get("analysis_status"))
        if status not in {"valid", "na"}:
            raise AnalysisError(f"unknown analysis status: {directory}")
        required = set(EXPECTED_VARIANTS if status == "valid" else ("r0", "current"))
        if not required <= set(row.get("variants", {})):
            raise AnalysisError(f"prompt variants are incomplete: {directory}")
    return {"seal": seal, "seal_path": seal_path, "summary": summary, "scores": scores}


def _aggregate_variant(rows: Sequence[Mapping[str, Any]], name: str) -> dict[str, Any]:
    metrics = [_variant_metrics(row, name) for row in rows]
    return {key: _mean([row[key] for row in metrics]) for key in metrics[0]}


def _aggregate_ranking(
    rows: Sequence[Mapping[str, Any]],
    score_index: Mapping[tuple[str, str], Sequence[tuple[int, int]]],
    comparator: str,
) -> dict[str, Any]:
    rankings = []
    pairings = []
    for row in rows:
        prompt_id = str(row["prompt_id"])
        baseline = score_index[(prompt_id, comparator)]
        current = score_index[(prompt_id, "current")]
        rankings.append(ranking_agreement(baseline, current))
        pairings.append(pairwise_metrics(current, baseline_scores=baseline))
    tau = [float(item["kendall_tau_b"]) for item in rankings if item["kendall_tau_b"] is not None]
    return {
        "kendall_tau_b_mean_defined": _mean(tau),
        "kendall_defined_prompts": len(tau),
        "pairwise_ordering_agreement": _mean(
            [float(item["pairwise_ordering_agreement"]) for item in rankings]
        ),
        "top_set_jaccard": _mean([float(item["top_set_jaccard"]) for item in rankings]),
        "top_set_exact_match_rate": _mean(
            [float(bool(item["top_set_exact_match"])) for item in rankings]
        ),
        "incremental_tie_resolution": _mean(
            [float(item["incremental_tie_resolution"]) for item in pairings]
        ),
        "conditional_tie_resolution_mean_defined": _mean(
            [
                float(item["conditional_tie_resolution"])
                for item in pairings
                if item["conditional_tie_resolution"] is not None
            ]
        ),
        "ordering_reversal_rate": _mean(
            [float(item["ordering_reversal_rate"]) for item in pairings]
        ),
    }


def analyze(
    scores_root: Path,
    *,
    output_dir: Path,
    config_path: Path,
    bootstrap_replicates: int | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    if config.horizon is None or config.horizon.domain != "medicine":
        raise AnalysisError(
            "checkpoint-wise Medicine analysis requires the Medicine horizon config"
        )
    if bootstrap_replicates is None:
        bootstrap_replicates = int(config.raw["horizon"]["bootstrap_replicates"])
    if bootstrap_replicates <= 0:
        raise ValueError("bootstrap_replicates must be positive")
    shards = {
        checkpoint: _verify_shard(scores_root / f"epoch-{_epoch_label(checkpoint)}", checkpoint)
        for checkpoint in CHECKPOINTS
    }
    lineage_fields = (
        "config_hash",
        "grader_model_revision",
        "tokenizer_revision",
        "target_encoding_version",
    )
    lineage = {}
    for field in lineage_fields:
        observed = {str(shard["seal"].get(field)) for shard in shards.values()}
        if len(observed) != 1 or "None" in observed:
            raise AnalysisError(f"inconsistent score-seal {field}: {sorted(observed)}")
        lineage[field] = next(iter(observed))
    if lineage["config_hash"] != config.config_hash:
        raise AnalysisError(
            "score-seal config hash differs from the loaded Medicine horizon config"
        )
    grader_config = config.models["proxy_grader"]
    expected_grader_lineage = {
        "grader_model_revision": str(grader_config["revision"]),
        "tokenizer_revision": str(grader_config["tokenizer_revision"]),
    }
    for field, expected in expected_grader_lineage.items():
        if lineage[field] != expected:
            raise AnalysisError(f"score-seal {field} differs from the loaded grader config")

    checkpoint_results = []
    valid_sets: list[set[str]] = []
    source_lineage = []
    for checkpoint in CHECKPOINTS:
        shard = shards[checkpoint]
        all_rows = shard["summary"]
        valid_rows = [row for row in all_rows if row["analysis_status"] == "valid"]
        na_rows = [row for row in all_rows if row["analysis_status"] == "na"]
        valid_ids = sorted(str(row["prompt_id"]) for row in valid_rows)
        na_ids = sorted(str(row["prompt_id"]) for row in na_rows)
        if checkpoint != 0.0:
            valid_sets.append(set(valid_ids))
        reasons: dict[str, int] = {}
        for row in na_rows:
            reason = str(row.get("na_reason") or "unspecified")
            reasons[reason] = reasons.get(reason, 0) + 1
        score_index = _score_index(shard["scores"])
        entry: dict[str, Any] = {
            "checkpoint": checkpoint,
            "policy_step": int(shard["seal"]["policy_step"]),
            "coverage": {
                "total": len(all_rows),
                "valid": len(valid_rows),
                "na": len(na_rows),
                "valid_prompt_ids": valid_ids,
                "na_prompt_ids": na_ids,
                "na_reasons": dict(sorted(reasons.items())),
            },
            "variants": {name: _aggregate_variant(valid_rows, name) for name in EXPECTED_VARIANTS},
            "criterion_effectiveness": {
                "r0": _aggregate_effectiveness(valid_rows, ("r0",)),
                "extension": _aggregate_effectiveness(valid_rows, ("extension",)),
                "control_extension": _aggregate_effectiveness(valid_rows, ("control_extension",)),
                "current_combined": _aggregate_effectiveness(valid_rows, ("r0", "extension")),
                "control_combined": _aggregate_effectiveness(
                    valid_rows, ("r0", "control_extension")
                ),
            },
            "ranking": {
                "current_vs_r0": _aggregate_ranking(valid_rows, score_index, "r0"),
                "current_vs_control": _aggregate_ranking(valid_rows, score_index, "control"),
            },
        }
        if checkpoint == 0.0:
            entry["g_refresh_r0_minus_current"] = None
            entry["g_count_control_minus_current"] = None
        else:
            current_metrics = [_variant_metrics(row, "current") for row in valid_rows]
            counts = [float(row["online_criterion_count"]) for row in valid_rows]
            refresh = [
                _gain(current, _variant_metrics(row, "r0"))
                for current, row in zip(current_metrics, valid_rows)
            ]
            count = [
                _gain(current, _variant_metrics(row, "control"))
                for current, row in zip(current_metrics, valid_rows)
            ]
            base_seed = config.bootstrap_seed + int(round(checkpoint * 1000)) * 10
            entry["g_refresh_r0_minus_current"] = _bootstrap_gains(
                refresh, counts, replicates=bootstrap_replicates, seed=base_seed + 1
            )
            entry["g_count_control_minus_current"] = _bootstrap_gains(
                count, counts, replicates=bootstrap_replicates, seed=base_seed + 2
            )
        checkpoint_results.append(entry)
        source_lineage.append(
            {
                "checkpoint": checkpoint,
                "score_seal_path": str(shard["seal_path"].resolve()),
                "score_seal_sha256": _sha256(shard["seal_path"]),
            }
        )

    intersection = sorted(set.intersection(*valid_sets)) if valid_sets else []
    result = {
        "schema_version": 1,
        "analysis": "medicine_stale_control_checkpointwise",
        "seed_id": "11",
        "config_path": str(config_path.resolve()),
        "bootstrap_seed": config.bootstrap_seed,
        "bootstrap_replicates": bootstrap_replicates,
        "lineage": lineage,
        "source_score_seals": source_lineage,
        "checkpoint_results": checkpoint_results,
        "complete_case": {
            "nonzero_checkpoint_count": len(valid_sets),
            "prompt_count": len(intersection),
            "prompt_ids": intersection,
            "official_horizon_estimable": bool(intersection),
            "explanation": (
                "No prompt is valid at every nonzero stale-control checkpoint; the "
                "official simultaneous crossed t* decision is therefore not estimable."
                if not intersection
                else "A complete-case prompt intersection exists; checkpoint-wise estimates "
                "remain descriptive and do not replace the preregistered inference build."
            ),
        },
    }
    _write_outputs(result, output_dir)
    return result


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _write_outputs(result: Mapping[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_text = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    (output_dir / "stale_checkpointwise.json").write_text(json_text, encoding="utf-8")

    columns = [
        "checkpoint",
        "policy_step",
        "valid_prompts",
        "na_prompts",
        "mean_online_criteria",
        "r0_exact_zar",
        "current_exact_zar",
        "control_exact_zar",
        "g_refresh_exact_zar",
        "g_refresh_exact_zar_ci_low",
        "g_refresh_exact_zar_ci_high",
        "g_count_exact_zar",
        "g_count_exact_zar_ci_low",
        "g_count_exact_zar_ci_high",
        "r0_separation",
        "current_separation",
        "control_separation",
        "g_refresh_separation",
        "g_count_separation",
        "current_vs_r0_tau_b",
        "current_vs_control_tau_b",
    ]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for entry in result["checkpoint_results"]:
        refresh = entry["g_refresh_r0_minus_current"]
        count = entry["g_count_control_minus_current"]
        writer.writerow(
            {
                "checkpoint": _fmt(entry["checkpoint"]),
                "policy_step": entry["policy_step"],
                "valid_prompts": entry["coverage"]["valid"],
                "na_prompts": entry["coverage"]["na"],
                "mean_online_criteria": _fmt(
                    entry["criterion_effectiveness"]["extension"]["mean_criteria_per_prompt"]
                ),
                "r0_exact_zar": _fmt(entry["variants"]["r0"]["exact_zar"]),
                "current_exact_zar": _fmt(entry["variants"]["current"]["exact_zar"]),
                "control_exact_zar": _fmt(entry["variants"]["control"]["exact_zar"]),
                "g_refresh_exact_zar": _fmt(
                    refresh["point_estimate"]["exact_zar_reduction"] if refresh else None
                ),
                "g_refresh_exact_zar_ci_low": _fmt(
                    refresh["ci95"]["exact_zar_reduction"][0] if refresh else None
                ),
                "g_refresh_exact_zar_ci_high": _fmt(
                    refresh["ci95"]["exact_zar_reduction"][1] if refresh else None
                ),
                "g_count_exact_zar": _fmt(
                    count["point_estimate"]["exact_zar_reduction"] if count else None
                ),
                "g_count_exact_zar_ci_low": _fmt(
                    count["ci95"]["exact_zar_reduction"][0] if count else None
                ),
                "g_count_exact_zar_ci_high": _fmt(
                    count["ci95"]["exact_zar_reduction"][1] if count else None
                ),
                "r0_separation": _fmt(entry["variants"]["r0"]["separation_rate"]),
                "current_separation": _fmt(entry["variants"]["current"]["separation_rate"]),
                "control_separation": _fmt(entry["variants"]["control"]["separation_rate"]),
                "g_refresh_separation": _fmt(
                    refresh["point_estimate"]["separation_gain"] if refresh else None
                ),
                "g_count_separation": _fmt(
                    count["point_estimate"]["separation_gain"] if count else None
                ),
                "current_vs_r0_tau_b": _fmt(
                    entry["ranking"]["current_vs_r0"]["kendall_tau_b_mean_defined"]
                ),
                "current_vs_control_tau_b": _fmt(
                    entry["ranking"]["current_vs_control"]["kendall_tau_b_mean_defined"]
                ),
            }
        )
    (output_dir / "stale_checkpointwise.csv").write_text(buffer.getvalue(), encoding="utf-8")

    complete = result["complete_case"]
    lines = [
        "# Medicine stale-control checkpoint-wise analysis",
        "",
        "All ten Pool B score seals, output/source digests, config lineage, and grader/tokenizer "
        "lineage were verified before analysis. Nonzero checkpoints use only valid, "
        "count-matched stale-control prompts; NA prompts are retained in JSON coverage records.",
        "",
        "| Epoch | Valid | NA | R0 ZAR | Current ZAR | Control ZAR | g_refresh | g_count | Current−R0 Sep | Current−Control Sep |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for entry in result["checkpoint_results"]:
        refresh = entry["g_refresh_r0_minus_current"]
        count = entry["g_count_control_minus_current"]
        lines.append(
            "| "
            + " | ".join(
                [
                    _fmt(entry["checkpoint"]),
                    str(entry["coverage"]["valid"]),
                    str(entry["coverage"]["na"]),
                    _fmt(entry["variants"]["r0"]["exact_zar"]),
                    _fmt(entry["variants"]["current"]["exact_zar"]),
                    _fmt(entry["variants"]["control"]["exact_zar"]),
                    _fmt(refresh["point_estimate"]["exact_zar_reduction"] if refresh else None),
                    _fmt(count["point_estimate"]["exact_zar_reduction"] if count else None),
                    _fmt(refresh["point_estimate"]["separation_gain"] if refresh else None),
                    _fmt(count["point_estimate"]["separation_gain"] if count else None),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            f"Complete-case intersection: **{complete['prompt_count']} prompts**. "
            + complete["explanation"],
            "",
            "`g_refresh = R0 − current` for ZAR/tie and `current − R0` for spread/separation. "
            "`g_count = stale control − current` for ZAR/tie and `current − stale control` "
            "for spread/separation. Thus positive g_count is current-policy-specific advantage "
            "beyond a count/weight-matched stale rubric. JSON contains deterministic "
            "10,000-replicate paired-prompt bootstrap CIs, count-normalized gains, rankings, "
            "criterion effectiveness, and "
            "the exact valid/NA prompt IDs and reasons.",
            "",
        ]
    )
    (output_dir / "stale_checkpointwise.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/horizon_medicine.yaml"),
    )
    parser.add_argument(
        "--scores-root",
        type=Path,
        default=Path("artifacts/horizon/medicine/scores/seed-11"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/horizon/medicine/stale_checkpointwise"),
    )
    parser.add_argument("--bootstrap-replicates", type=int)
    args = parser.parse_args()
    analyze(
        args.scores_root,
        output_dir=args.output_dir,
        config_path=args.config,
        bootstrap_replicates=args.bootstrap_replicates,
    )


if __name__ == "__main__":
    main()
