"""Balanced prompt-subset analysis for fast minimum-experiment decisions."""

from __future__ import annotations

import csv
import io
import math
import os
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
import tempfile
from typing import Any

from .artifacts import artifact_record, read_json, write_json_atomic, write_text_atomic
from .evaluation.bootstrap import paired_prompt_bootstrap
from .hashing import sha256_file
from .minimum_staleness import (
    FOCAL_STEPS,
    MODE,
    N_GRID,
    PERMUTATIONS,
    MinimumExperimentError,
    _bon_groups,
    _jsonl,
    _shard_name,
)
from .reporting.aligned_analysis import analyze_aligned


def ordered_prompt_subset(run_root: Path, count: int) -> tuple[str, ...]:
    """Select the first prompts in immutable pi_3 BoN order, before seeing scores."""

    if count < 1:
        raise MinimumExperimentError("interim prompt count must be positive")
    bon_path = run_root / "generate-bon" / "bon_pool.jsonl"
    prompts: list[str] = []
    for (policy_id, prompt_id), _ in _bon_groups(bon_path):
        if policy_id == "pi_3":
            prompts.append(prompt_id)
            if len(prompts) == count:
                break
    if len(prompts) != count:
        raise MinimumExperimentError(f"requested {count} interim prompts, found {len(prompts)}")
    expected = {(f"pi_{step}", prompt_id) for step in FOCAL_STEPS for prompt_id in prompts}
    actual = {key for key, _ in _bon_groups(bon_path) if key in expected}
    if actual != expected:
        raise MinimumExperimentError("interim prompt subset is absent at some checkpoints")
    return tuple(prompts)


def _normalize_steps(steps: Sequence[int]) -> tuple[int, ...]:
    normalized = tuple(int(step) for step in steps)
    if (
        not normalized
        or len(set(normalized)) != len(normalized)
        or any(step not in FOCAL_STEPS for step in normalized)
    ):
        raise MinimumExperimentError(f"invalid interim policy steps: {normalized}")
    return normalized


def target_groups(
    prompt_ids: Sequence[str], steps: Sequence[int] = FOCAL_STEPS
) -> frozenset[tuple[str, str]]:
    policy_steps = _normalize_steps(steps)
    return frozenset(
        (f"pi_{step}", str(prompt_id)) for step in policy_steps for prompt_id in prompt_ids
    )


def _target_paths(
    run_root: Path, prompt_ids: Sequence[str], steps: Sequence[int] = FOCAL_STEPS
) -> list[tuple[str, str, Path, Path, Path]]:
    paths = []
    for policy_id, prompt_id in sorted(target_groups(prompt_ids, steps)):
        stem = f"{policy_id}-{_shard_name(policy_id, prompt_id)}"
        paths.append(
            (
                policy_id,
                prompt_id,
                run_root / "select-bon-minimum" / "shards" / f"{stem}.jsonl",
                run_root / "score-proxy-minimum" / "shards" / f"{stem}.jsonl",
                run_root / "audit-gold-streaming-private" / "groups" / stem / "gold_scores.jsonl",
            )
        )
    return paths


def target_gold_complete(
    run_root: Path, prompt_ids: Sequence[str], steps: Sequence[int] = FOCAL_STEPS
) -> int:
    return sum(
        gold_path.is_file() for *_, gold_path in _target_paths(run_root, prompt_ids, steps)
    )


def _human_static_repeat_compatible_scores(
    run_root: Path, rows: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Backfill deterministic repeat scores for early Human-GT R0 shards only."""

    manifest_path = run_root / "score-proxy-minimum" / "manifest.json"
    if not manifest_path.is_file() or read_json(manifest_path).get("static_r0") != "human_gt":
        return list(rows)
    compatible = []
    for row in rows:
        if "judge_repeat_score" in row:
            compatible.append(row)
            continue
        compatible.append(
            {
                **row,
                "judge_repeat_score": float(row["score"]),
                "judge_repeat_method": "deterministic_temperature_zero_cache_identity",
            }
        )
    return compatible


def analyze_interim(
    run_root: Path, prompt_ids: Sequence[str], steps: Sequence[int] = FOCAL_STEPS
) -> dict[str, Any]:
    """Create paired N-curves for the same prompt subset at every checkpoint."""

    if len(set(prompt_ids)) != len(prompt_ids):
        raise MinimumExperimentError("interim prompt subset contains duplicates")
    policy_steps = _normalize_steps(steps)
    selections: list[dict[str, Any]] = []
    scores: dict[str, list[dict[str, Any]]] = defaultdict(list)
    gold: dict[tuple[str, str], float] = {}
    for policy_id, prompt_id, selection_path, score_path, gold_path in _target_paths(
        run_root, prompt_ids, policy_steps
    ):
        for path in (selection_path, score_path, gold_path):
            if not path.is_file():
                raise MinimumExperimentError(f"missing interim artifact: {path}")
        selection_rows = list(_jsonl(selection_path))
        if len(selection_rows) != 2 * len(N_GRID) * PERMUTATIONS:
            raise MinimumExperimentError(f"interim selection inventory mismatch: {selection_path}")
        selections.extend(selection_rows)
        scores[policy_id].extend(_jsonl(score_path))
        for row in _jsonl(gold_path):
            key = prompt_id, str(row["response_id"])
            value = float(row["gold_score"])
            previous = gold.setdefault(key, value)
            if previous != value:
                raise MinimumExperimentError(f"interim gold score drift: {key}")

    output_name = f"interim-bon-{len(prompt_ids)}prompt"
    if policy_steps != tuple(FOCAL_STEPS):
        suffix = "-".join(str(step) for step in policy_steps)
        output_name = f"{output_name}-pi-{suffix}-v3"
    output_root = run_root / output_name
    curve_rows: list[dict[str, Any]] = []
    aligned_by_policy: dict[str, Any] = {}
    for step in policy_steps:
        policy_id = f"pi_{step}"
        policy_selections = [row for row in selections if str(row["policy_id"]) == policy_id]
        aligned = analyze_aligned(
            policy_selections,
            gold,
            _human_static_repeat_compatible_scores(run_root, scores[policy_id]),
            iterations=10_000,
            seed=20_250_807 + step,
            n_grid=N_GRID,
            permutations=PERMUTATIONS,
        )
        aligned.pop("bootstrap_results", None)
        aligned_by_policy[policy_id] = aligned
        values: dict[tuple[str, int, str], list[float]] = defaultdict(list)
        for row in policy_selections:
            key = str(row["mode"]), int(row["n"]), str(row["prompt_id"])
            values[key].append(gold[(str(row["prompt_id"]), str(row["response_id"]))])
        for n in N_GRID:
            pairs = []
            static_prompt_means = []
            dynamic_prompt_means = []
            for prompt_id in prompt_ids:
                static_values = values[("static", n, prompt_id)]
                dynamic_values = values[(MODE, n, prompt_id)]
                if len(static_values) != PERMUTATIONS or len(dynamic_values) != PERMUTATIONS:
                    raise MinimumExperimentError(
                        f"interim permutation inventory mismatch: {(policy_id, prompt_id, n)}"
                    )
                static_mean = math.fsum(static_values) / len(static_values)
                dynamic_mean = math.fsum(dynamic_values) / len(dynamic_values)
                static_prompt_means.append(static_mean)
                dynamic_prompt_means.append(dynamic_mean)
                pairs.append((prompt_id, dynamic_mean, static_mean))
            bootstrap = paired_prompt_bootstrap(
                pairs,
                iterations=10_000,
                seed=20_250_807 + step * 2_048 + n,
            )
            curve_rows.append(
                {
                    "policy_id": policy_id,
                    "policy_step": step,
                    "n": n,
                    "prompts": len(prompt_ids),
                    "permutations": PERMUTATIONS,
                    "static_mean_gold": math.fsum(static_prompt_means) / len(static_prompt_means),
                    "dynamic_mean_gold": math.fsum(dynamic_prompt_means)
                    / len(dynamic_prompt_means),
                    "dynamic_minus_static": bootstrap.point_estimate,
                    "ci_low": bootstrap.ci_low,
                    "ci_high": bootstrap.ci_high,
                    "classification": bootstrap.classification,
                }
            )

    csv_stream = io.StringIO(newline="")
    writer = csv.DictWriter(csv_stream, fieldnames=list(curve_rows[0]))
    writer.writeheader()
    writer.writerows(curve_rows)
    csv_path = output_root / "n_curves.csv"
    write_text_atomic(csv_path, csv_stream.getvalue())
    figure_paths = _plot_curves(output_root, curve_rows, len(prompt_ids), policy_steps)
    report = {
        "schema_version": 1,
        "diagnostic_only": True,
        "prompt_selection": "first-N-in-immutable-pi_3-BoN-order",
        "prompt_ids": list(prompt_ids),
        "focal_steps": list(policy_steps),
        "n_grid": list(N_GRID),
        "permutations": PERMUTATIONS,
        "curves": curve_rows,
        "aligned_analysis": aligned_by_policy,
        "inputs": {
            "bon_pool_sha256": sha256_file(run_root / "generate-bon" / "bon_pool.jsonl"),
            "score_manifest_sha256": sha256_file(
                run_root / "score-proxy-minimum" / "manifest.json"
            ),
            "selection_manifest_sha256": sha256_file(
                run_root / "select-bon-minimum" / "manifest.json"
            ),
        },
        "csv": artifact_record(csv_path),
        "figures": [artifact_record(path) for path in figure_paths],
    }
    write_json_atomic(output_root / "result.json", report)
    return report


def _plot_curves(
    output_root: Path,
    rows: Sequence[dict[str, Any]],
    prompt_count: int,
    steps: Sequence[int],
) -> tuple[Path, Path]:
    import matplotlib  # pyright: ignore[reportMissingImports]

    matplotlib.use("Agg")
    matplotlib.rcParams["svg.hashsalt"] = "dynamic-rubric-interim-v1"
    import matplotlib.pyplot as plt  # pyright: ignore[reportMissingImports]

    output_root.mkdir(parents=True, exist_ok=True)
    plot_rows = math.ceil(len(steps) / 2)
    figure, axes = plt.subplots(
        plot_rows,
        2,
        figsize=(12, 4.8 * plot_rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    flat_axes = list(axes.flat)
    for axis in flat_axes[len(steps) :]:
        axis.set_visible(False)
    for axis, step in zip(flat_axes, steps):
        current = [row for row in rows if int(row["policy_step"]) == step]
        x = [math.log2(int(row["n"])) for row in current]
        axis.plot(x, [row["static_mean_gold"] for row in current], marker="o", label="Static R0")
        axis.plot(
            x,
            [row["dynamic_mean_gold"] for row in current],
            marker="o",
            label=f"Dynamic R{step}",
        )
        axis.set_title(f"pi_{step}: R0 vs R{step}")
        axis.set_ylim(0.0, 1.0)
        axis.grid(alpha=0.25)
        axis.set_xticks(x, [str(row["n"]) for row in current], rotation=45)
    axes[0, 0].legend(loc="best")
    figure.supxlabel("BoN size N (log2 spacing)", y=0.08)
    figure.supylabel("Mean hidden-gold score")
    figure.suptitle(
        f"Interim static vs current-aligned dynamic BoN curves ({prompt_count} prompts)"
    )
    figure.text(
        0.5,
        0.02,
        "Diagnostic subset; 5 fixed permutations per prompt. Final claims require the 96-prompt run.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0.03, 0.14, 1.0, 0.95))
    png_path = output_root / "n_curves.png"
    svg_path = output_root / "n_curves.svg"
    for output_path in (png_path, svg_path):
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{output_path.name}.", suffix=output_path.suffix, dir=output_root
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            figure.savefig(temporary, dpi=180, metadata={"Date": None})
            os.replace(temporary, output_path)
        finally:
            temporary.unlink(missing_ok=True)
    plt.close(figure)
    return png_path, svg_path
