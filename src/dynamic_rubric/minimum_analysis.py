"""Inventory-gated analysis for the minimum rubric-staleness experiment."""

from __future__ import annotations

import csv
import io
import math
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .artifacts import artifact_record, write_json_atomic, write_text_atomic
from .hashing import sha256_file
from .minimum_staleness import FOCAL_STEPS, MODE, N_GRID, PERMUTATIONS, _audit_prompt_ids, _jsonl
from .reporting.aligned_analysis import analyze_aligned


BOOTSTRAP_ITERATIONS = 10_000
BOOTSTRAP_SEED = 20_250_807
EXPECTED_AUDIT_PROMPTS = 96
EXPECTED_POOL_SIZE = 1024


class MinimumAnalysisError(RuntimeError):
    """Raised when the minimum experiment cannot support a causal analysis."""


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise MinimumAnalysisError("cannot average an empty sequence")
    return math.fsum(values) / len(values)


def _probe_summary(run_root: Path, step: int, audit_prompts: set[str]) -> dict[str, float | int]:
    rows = [
        row
        for row in _jsonl(run_root / "train-static" / "verl-run" / "probes" / f"{step}.jsonl")
        if str(row["prompt_id"]) in audit_prompts
    ]
    if not rows:
        raise MinimumAnalysisError(f"no audit probes at policy step {step}")
    kl = [float(row["kl_from_pi0"]) for row in rows]
    rewards = [float(row["static_reward"]) for row in rows]
    lengths = [len(str(row["output"])) for row in rows]
    return {
        "probe_rows": len(rows),
        "mean_kl_from_pi0": _mean(kl),
        "median_kl_from_pi0": statistics.median(kl),
        "mean_abs_kl_from_pi0": _mean([abs(value) for value in kl]),
        "mean_static_reward": _mean(rewards),
        "mean_output_characters": _mean([float(value) for value in lengths]),
    }


def _replay_summary(
    replay_rows: Sequence[Mapping[str, Any]],
    *,
    step: int,
    previous_step: int | None,
) -> dict[str, float | int | None]:
    current = [row for row in replay_rows if int(row["policy_step"]) == step]
    if len(current) != EXPECTED_AUDIT_PROMPTS:
        raise MinimumAnalysisError(
            f"expected 96 current rubrics at step {step}, got {len(current)}"
        )
    current_dynamic = {
        str(row["prompt_id"]): {
            str(item["criterion_id"]) for item in row["criteria"] if item["source"] == "dynamic"
        }
        for row in current
    }
    retained = previous_total = union_total = 0
    if previous_step is not None:
        previous = {
            str(row["prompt_id"]): {
                str(item["criterion_id"]) for item in row["criteria"] if item["source"] == "dynamic"
            }
            for row in replay_rows
            if int(row["policy_step"]) == previous_step
        }
        if set(previous) != set(current_dynamic):
            raise MinimumAnalysisError("focal replay prompt sets do not align")
        for prompt_id, current_ids in current_dynamic.items():
            previous_ids = previous[prompt_id]
            retained += len(current_ids & previous_ids)
            previous_total += len(previous_ids)
            union_total += len(current_ids | previous_ids)
    admissions = []
    similarities = []
    for row in replay_rows:
        if int(row["policy_step"]) > step or row.get("admitted_id") is None:
            continue
        admissions.append(str(row["admitted_id"]))
        for candidate in row["candidate_evidence"]:
            if candidate["criterion_id"] == row["admitted_id"] and "evidence" in candidate:
                similarities.append(float(candidate["evidence"]["max_active_similarity"]))
                break
    dynamic_counts = [len(values) for values in current_dynamic.values()]
    survival = retained / previous_total if previous_total else None
    return {
        "cumulative_admissions": len(admissions),
        "mean_criterion_count": _mean([float(row["criterion_count"]) for row in current]),
        "mean_dynamic_criterion_count": _mean([float(value) for value in dynamic_counts]),
        "dynamic_criterion_survival_from_previous_focal": survival,
        "dynamic_criterion_turnover_from_previous_focal": (
            1.0 - survival if survival is not None else None
        ),
        "dynamic_criterion_jaccard_from_previous_focal": (
            retained / union_total if union_total else None
        ),
        "mean_admitted_max_active_similarity": (_mean(similarities) if similarities else None),
    }


def _validate_inventory(
    *,
    audit_prompts: set[str],
    replay_rows: Sequence[Mapping[str, Any]],
    selections: Sequence[Mapping[str, Any]],
    gold: Mapping[tuple[str, str], float],
    scores: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    if len(audit_prompts) != EXPECTED_AUDIT_PROMPTS:
        raise MinimumAnalysisError(f"expected 96 audit prompts, got {len(audit_prompts)}")
    policies = {f"pi_{step}" for step in FOCAL_STEPS}
    expected_selection_count = (
        len(policies) * EXPECTED_AUDIT_PROMPTS * 2 * len(N_GRID) * PERMUTATIONS
    )
    if len(selections) != expected_selection_count:
        raise MinimumAnalysisError(
            f"selection count mismatch: {len(selections)} != {expected_selection_count}"
        )
    selection_keys = set()
    pools: dict[tuple[str, str, int, int], str] = {}
    selected_responses = set()
    for row in selections:
        policy_id = str(row["policy_id"])
        prompt_id = str(row["prompt_id"])
        mode = str(row["mode"])
        step = int(row["policy_step"])
        n = int(row["n"])
        permutation = int(row["permutation"])
        if (
            policy_id not in policies
            or policy_id != f"pi_{step}"
            or prompt_id not in audit_prompts
            or mode not in {"static", MODE}
            or n not in N_GRID
            or permutation not in range(PERMUTATIONS)
        ):
            raise MinimumAnalysisError(f"invalid selection cell: {row}")
        rubric_step = int(row["rubric_step"])
        if (mode == "static" and rubric_step != 0) or (mode == MODE and rubric_step != step):
            raise MinimumAnalysisError("selection is not static/current-aligned")
        key = policy_id, prompt_id, mode, n, permutation
        if key in selection_keys:
            raise MinimumAnalysisError(f"duplicate selection cell: {key}")
        selection_keys.add(key)
        pool_key = policy_id, prompt_id, n, permutation
        pool_hash = str(row["pool_hash"])
        previous = pools.setdefault(pool_key, pool_hash)
        if previous != pool_hash:
            raise MinimumAnalysisError(f"rubric modes used different candidate pools: {pool_key}")
        response_key = prompt_id, str(row["response_id"])
        selected_responses.add(response_key)
        if response_key not in gold:
            raise MinimumAnalysisError(
                f"selected response has no hidden-gold score: {response_key}"
            )
    if set(gold) != selected_responses:
        raise MinimumAnalysisError("hidden-gold inventory differs from deduplicated selections")

    expected_score_count = len(policies) * EXPECTED_AUDIT_PROMPTS * 2 * EXPECTED_POOL_SIZE
    if len(scores) != expected_score_count:
        raise MinimumAnalysisError(
            f"proxy score count mismatch: {len(scores)} != {expected_score_count}"
        )
    score_groups: dict[tuple[str, str, str], set[int]] = defaultdict(set)
    for row in scores:
        policy_id = str(row["policy_id"])
        prompt_id = str(row["prompt_id"])
        mode = str(row["mode"])
        step = int(row["policy_step"])
        rubric_step = int(row["rubric_step"])
        if policy_id != f"pi_{step}" or policy_id not in policies or prompt_id not in audit_prompts:
            raise MinimumAnalysisError("proxy score policy/prompt inventory drift")
        if (mode == "static" and rubric_step != 0) or (mode == MODE and rubric_step != step):
            raise MinimumAnalysisError("proxy scores are not static/current-aligned")
        group = policy_id, prompt_id, mode
        candidate_id = int(row["global_candidate_id"])
        if candidate_id in score_groups[group]:
            raise MinimumAnalysisError(
                f"duplicate candidate proxy score: {group + (candidate_id,)}"
            )
        score_groups[group].add(candidate_id)
    expected_groups = len(policies) * EXPECTED_AUDIT_PROMPTS * 2
    if len(score_groups) != expected_groups or any(
        len(values) != EXPECTED_POOL_SIZE for values in score_groups.values()
    ):
        raise MinimumAnalysisError("proxy score group inventory is incomplete")
    for policy_id in policies:
        for prompt_id in audit_prompts:
            static_ids = score_groups[(policy_id, prompt_id, "static")]
            dynamic_ids = score_groups[(policy_id, prompt_id, MODE)]
            if static_ids != dynamic_ids:
                raise MinimumAnalysisError(
                    f"proxy candidate IDs do not align: {(policy_id, prompt_id)}"
                )

    focal_replay = [
        row
        for row in replay_rows
        if str(row["prompt_id"]) in audit_prompts and int(row["policy_step"]) in FOCAL_STEPS
    ]
    if len(focal_replay) != len(FOCAL_STEPS) * EXPECTED_AUDIT_PROMPTS:
        raise MinimumAnalysisError("focal current-rubric inventory is incomplete")
    replay_keys = {(str(row["prompt_id"]), int(row["policy_step"])) for row in focal_replay}
    if len(replay_keys) != len(focal_replay):
        raise MinimumAnalysisError("duplicate focal replay snapshot")
    return {
        "audit_prompts": len(audit_prompts),
        "replay_focal_snapshots": len(focal_replay),
        "selection_rows": len(selections),
        "selected_unique_responses": len(gold),
        "proxy_score_rows": len(scores),
    }


def _n_curves(
    selections: Sequence[Mapping[str, Any]], gold: Mapping[tuple[str, str], float]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for row in selections:
        key = str(row["policy_id"]), str(row["mode"]), int(row["n"])
        grouped[key].append(gold[(str(row["prompt_id"]), str(row["response_id"]))])
    output = []
    for step in FOCAL_STEPS:
        policy_id = f"pi_{step}"
        for n in N_GRID:
            static = _mean(grouped[(policy_id, "static", n)])
            dynamic = _mean(grouped[(policy_id, MODE, n)])
            output.append(
                {
                    "policy_id": policy_id,
                    "policy_step": step,
                    "n": n,
                    "static_mean_gold": static,
                    "dynamic_mean_gold": dynamic,
                    "dynamic_minus_static": dynamic - static,
                }
            )
    return output


def _onset(checkpoints: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    significant = [row for row in checkpoints if float(row["paired_bootstrap"]["ci_low"]) > 0.0]
    meaningful = [
        row
        for row in checkpoints
        if row["paired_bootstrap"]["classification"] == "meaningful_difference"
        and float(row["paired_bootstrap"]["point_estimate"]) > 0.0
    ]
    first = significant[0] if significant else None
    first_meaningful = meaningful[0] if meaningful else None
    return {
        "statistically_positive_first_policy": first["policy_id"] if first else None,
        "statistically_positive_first_step": first["policy_step"] if first else None,
        "estimated_kl_at_statistical_onset": (
            first["policy_distance"]["mean_kl_from_pi0"] if first else None
        ),
        "meaningful_positive_first_policy": (
            first_meaningful["policy_id"] if first_meaningful else None
        ),
        "meaningful_positive_first_step": (
            first_meaningful["policy_step"] if first_meaningful else None
        ),
        "estimated_kl_at_meaningful_onset": (
            first_meaningful["policy_distance"]["mean_kl_from_pi0"] if first_meaningful else None
        ),
        "interpretation": (
            f"staleness_onset_{first['policy_id']}" if first else "not_observed_through_pi_50"
        ),
    }


def _csv_text(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Minimum rubric-staleness analysis",
        "",
        "Static R0 and current-aligned dynamic_fixed_budgeted Rt use the same 1,024-response pool, fixed permutations, and hidden-gold grader.",
        "",
        "| Policy | KL from pi0 | Static GT-AUC | Dynamic GT-AUC | Delta | 95% CI | Decision |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in report["checkpoints"]:
        bootstrap = row["paired_bootstrap"]
        lines.append(
            "| {policy_id} | {kl:.6f} | {static:.4f} | {dynamic:.4f} | {delta:+.4f} | [{low:+.4f}, {high:+.4f}] | {decision} |".format(
                policy_id=row["policy_id"],
                kl=row["policy_distance"]["mean_kl_from_pi0"],
                static=row["gt_auc"]["static"],
                dynamic=row["gt_auc"][MODE],
                delta=bootstrap["point_estimate"],
                low=bootstrap["ci_low"],
                high=bootstrap["ci_high"],
                decision=bootstrap["classification"],
            )
        )
    lines.extend(
        [
            "",
            "## Staleness onset",
            "",
            f"- Result: `{report['staleness_onset']['interpretation']}`",
            "- A positive onset requires the paired prompt-bootstrap 95% CI for dynamic minus static GT-AUC to be entirely above zero.",
            "- This locates onset only among the observed checkpoints (3, 10, 30, 50); it does not assert a continuous KL threshold between them.",
            "",
            "## Inventory",
            "",
        ]
    )
    lines.extend(f"- {key}: {value}" for key, value in report["inventory"].items())
    return "\n".join(lines) + "\n"


def analyze_minimum(run_root: Path) -> dict[str, Any]:
    """Validate exact joins, analyze each checkpoint, and publish aggregate-safe outputs."""

    stage_root = run_root / "analysis-minimum"
    selections_path = run_root / "select-bon-minimum" / "selections.jsonl"
    gold_path = run_root / "audit-gold-minimum-private" / "gold_scores.jsonl"
    scores_path = run_root / "score-proxy-minimum" / "rubric_scores.jsonl"
    replay_path = run_root / "replay-dynamic-minimum" / "replay_snapshots.jsonl"
    inputs = (selections_path, gold_path, scores_path, replay_path)
    for path in inputs:
        if not path.is_file():
            raise MinimumAnalysisError(f"missing analysis input: {path}")
    audit_prompts = _audit_prompt_ids(run_root)
    selections = list(_jsonl(selections_path))
    gold_rows = list(_jsonl(gold_path))
    gold: dict[tuple[str, str], float] = {}
    for row in gold_rows:
        key = str(row["prompt_id"]), str(row["response_id"])
        score = float(row["gold_score"])
        if key in gold or not 0.0 <= score <= 1.0:
            raise MinimumAnalysisError(f"invalid hidden-gold row: {key}")
        gold[key] = score
    scores = list(_jsonl(scores_path))
    replay_rows = [row for row in _jsonl(replay_path) if str(row["prompt_id"]) in audit_prompts]
    inventory = _validate_inventory(
        audit_prompts=audit_prompts,
        replay_rows=replay_rows,
        selections=selections,
        gold=gold,
        scores=scores,
    )
    selections_by_policy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    scores_by_policy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selections:
        selections_by_policy[str(row["policy_id"])].append(row)
    for row in scores:
        scores_by_policy[str(row["policy_id"])].append(row)

    checkpoints = []
    previous_step: int | None = None
    for step in FOCAL_STEPS:
        policy_id = f"pi_{step}"
        aligned = analyze_aligned(
            selections_by_policy[policy_id],
            gold,
            scores_by_policy[policy_id],
            iterations=BOOTSTRAP_ITERATIONS,
            seed=BOOTSTRAP_SEED + step,
            n_grid=N_GRID,
            permutations=PERMUTATIONS,
        )
        aligned.pop("bootstrap_results", None)
        n1024 = [row for row in selections_by_policy[policy_id] if int(row["n"]) == 1024]
        selected_by_cell = {
            (str(row["prompt_id"]), int(row["permutation"]), str(row["mode"])): str(
                row["response_id"]
            )
            for row in n1024
        }
        agreements = [
            selected_by_cell[(prompt_id, permutation, "static")]
            == selected_by_cell[(prompt_id, permutation, MODE)]
            for prompt_id in audit_prompts
            for permutation in range(PERMUTATIONS)
        ]
        checkpoints.append(
            {
                "policy_id": policy_id,
                "policy_step": step,
                "gt_auc": aligned["gt_auc"],
                "paired_bootstrap": aligned["paired_bootstrap"][MODE],
                "selection_agreement_all_n": aligned["top1_agreement"][MODE],
                "selection_agreement_n1024": sum(agreements) / len(agreements),
                "kendall_tau_b": aligned["kendall_tau_b"][MODE],
                "reward_resolution": aligned["reward_resolution"],
                "gt_auc_cross_matrix": aligned["gt_auc_cross_matrix"],
                "policy_distance": _probe_summary(run_root, step, audit_prompts),
                "rubric_dynamics": _replay_summary(
                    replay_rows, step=step, previous_step=previous_step
                ),
            }
        )
        previous_step = step
    n_curves = _n_curves(selections, gold)
    report = {
        "schema_version": 1,
        "run_id": run_root.name,
        "comparison": ["static", MODE],
        "focal_steps": list(FOCAL_STEPS),
        "n_grid": list(N_GRID),
        "permutations": PERMUTATIONS,
        "pool_size": EXPECTED_POOL_SIZE,
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "inventory": inventory,
        "checkpoints": checkpoints,
        "staleness_onset": _onset(checkpoints),
        "n_curve": n_curves,
    }
    manifest = {
        "schema_version": 1,
        "analysis": "static-vs-current-aligned-dynamic-fixed-budgeted",
        "input_sha256": {str(path): sha256_file(path) for path in inputs},
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }
    topology_path = run_root / "train-static" / "judge-topology-minimum-v2.json"
    if not topology_path.is_file():
        topology_path = run_root / "train-static" / "judge-topology-minimum.json"
    if topology_path.is_file():
        manifest["judge_topology_sha256"] = sha256_file(topology_path)
    routing_path = run_root / "train-static" / "judge-routing-minimum-v3.json"
    if not routing_path.is_file():
        routing_path = run_root / "train-static" / "judge-routing-minimum-v2.json"
    if routing_path.is_file():
        manifest["judge_routing_sha256"] = sha256_file(routing_path)
    write_json_atomic(stage_root / "manifest.json", manifest)
    report_path = stage_root / "report.json"
    write_json_atomic(report_path, report)
    checkpoint_rows = []
    for row in checkpoints:
        bootstrap = row["paired_bootstrap"]
        checkpoint_rows.append(
            {
                "policy_id": row["policy_id"],
                "policy_step": row["policy_step"],
                "mean_kl_from_pi0": row["policy_distance"]["mean_kl_from_pi0"],
                "static_gt_auc": row["gt_auc"]["static"],
                "dynamic_gt_auc": row["gt_auc"][MODE],
                "dynamic_minus_static": bootstrap["point_estimate"],
                "ci_low": bootstrap["ci_low"],
                "ci_high": bootstrap["ci_high"],
                "classification": bootstrap["classification"],
                "selection_agreement_n1024": row["selection_agreement_n1024"],
                "kendall_tau_b": row["kendall_tau_b"],
                "cumulative_admissions": row["rubric_dynamics"]["cumulative_admissions"],
                "mean_criterion_count": row["rubric_dynamics"]["mean_criterion_count"],
            }
        )
    checkpoint_path = stage_root / "checkpoint_summary.csv"
    write_text_atomic(
        checkpoint_path,
        _csv_text(checkpoint_rows, tuple(checkpoint_rows[0])),
    )
    n_curve_path = stage_root / "n_curve.csv"
    write_text_atomic(n_curve_path, _csv_text(n_curves, tuple(n_curves[0])))
    markdown_path = stage_root / "report.md"
    write_text_atomic(markdown_path, _markdown(report))
    result = {
        "staleness_onset": report["staleness_onset"],
        "report": artifact_record(report_path),
        "checkpoint_summary": artifact_record(checkpoint_path),
        "n_curve": artifact_record(n_curve_path),
        "markdown": artifact_record(markdown_path),
    }
    write_json_atomic(stage_root / "result.json", result)
    return result
