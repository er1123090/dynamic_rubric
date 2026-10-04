from __future__ import annotations

import hashlib
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from .artifacts import (
    artifact_record,
    read_jsonl,
    validate_artifact_record,
    write_json_atomic,
    write_jsonl_atomic,
    write_text_atomic,
)
from .data.healthbench import load_jsonl, scan_public_outputs_for_gold
from .fake_gold_audit import evaluate_fake_gold
from .hashing import sha256_file, sha256_json
from .inventory_validation import SemanticInventoryError, validate_semantic_inventory
from .pipeline import PipelineContext, StageError, _configured_splits
from .providers.fake import FakeEmbeddingProvider
from .reporting.tables import REQUIRED_COMPARISONS, build_report, render_markdown


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def run_export_audit_package(context: PipelineContext) -> dict[str, Any]:
    selections_path = context.run_root / "select-bon" / "selections.jsonl"
    context.begin_stage(inputs=(selections_path,), metadata={"selected_unique_only": True})
    selections = read_jsonl(selections_path)
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    for row in selections:
        key = str(row["prompt_id"]), str(row["response_id"])
        if key not in unique:
            unique[key] = {
                "prompt_id": row["prompt_id"],
                "response_id": row["response_id"],
                "response_text": row["response_text"],
                "response_text_hash": _hash_text(str(row["response_text"])),
                "selection_references": [],
            }
        unique[key]["selection_references"].append(
            {
                "policy_id": row["policy_id"],
                "rubric_id": row["rubric_id"],
                "n": row["n"],
                "permutation": row["permutation"],
            }
        )
    package = [unique[key] for key in sorted(unique)]
    write_jsonl_atomic(context.stage_root() / "audit_package.jsonl", package)
    result = {
        "selection_rows": len(selections),
        "selected_unique_responses": len(package),
        "deduplicated": len(package) <= len(selections),
        "contains_unselected_responses": False,
    }
    write_json_atomic(context.stage_root() / "result.json", result)
    return result


def run_audit_gold(context: PipelineContext, private_gt: Path) -> dict[str, Any]:
    package_path = context.run_root / "export-audit-package" / "audit_package.jsonl"
    if not private_gt.is_file():
        raise StageError(f"private physician-rubric file is missing: {private_gt}")
    context.begin_stage(inputs=(package_path, private_gt), metadata={"private_process": True})
    if context.mode != "fake":
        raise StageError("live hidden-GT scoring requires the calibrated gpt-5 grader preflight")
    public_scan_paths = [
        *sorted(context.public_root.glob("*.json*")),
        context.run_root / "generate-static" / "static_candidates.jsonl",
        context.run_root / "generate-static" / "static_rubrics.jsonl",
        context.run_root / "train-static" / "trajectory_development.jsonl",
        context.run_root / "trajectory" / "final_sealed" / "responses.jsonl",
        context.run_root / "replay-dynamic-final" / "replay_snapshots.jsonl",
        context.run_root / "generate-bon" / "bon_pool.jsonl",
        context.run_root / "score-proxy" / "rubric_scores.jsonl",
        context.run_root / "select-bon" / "selections.jsonl",
        context.run_root / "export-audit-package" / "audit_package.jsonl",
    ]
    scan_public_outputs_for_gold(public_scan_paths, private_gt)
    gold_by_prompt = {str(row["prompt_id"]): row["gold_rubric"] for row in load_jsonl(private_gt)}
    package = read_jsonl(package_path)
    scores: list[dict[str, Any]] = []
    cache: dict[str, float] = {}
    cache_hits = 0
    for row in package:
        prompt_id = str(row["prompt_id"])
        if prompt_id not in gold_by_prompt:
            raise StageError(f"private GT has no rubric for selected prompt: {prompt_id}")
        gold_hash = sha256_json(gold_by_prompt[prompt_id])
        key = sha256_json(
            [
                prompt_id,
                row["response_text_hash"],
                gold_hash,
                "gpt-5",
                "fake/gpt-5-v1",
                _hash_text("hidden-gold-grader-v1"),
                sha256_file(context.root / "configs" / "schemas" / "hidden_gold_grader_v1.json"),
                context.config.models.get("hidden_gt_grader", {}).get("reasoning_effort", "medium"),
            ]
        )
        evaluated_score, criterion_judgments = evaluate_fake_gold(
            prompt_id, str(row["response_text"]), gold_by_prompt[prompt_id]
        )
        if key in cache:
            cache_hits += 1
            score = cache[key]
        else:
            score = evaluated_score
            cache[key] = score
        scores.append(
            {
                "prompt_id": prompt_id,
                "response_id": row["response_id"],
                "gold_score": score,
                "criterion_judgments": criterion_judgments,
                "cache_key": key,
                "requested_model": "gpt-5",
                "returned_model": "fake/gpt-5-v1",
                "reasoning_effort": context.config.models.get("hidden_gt_grader", {}).get(
                    "reasoning_effort", "medium"
                ),
            }
        )
    write_jsonl_atomic(context.stage_root() / "gold_scores.jsonl", scores)
    result = {
        "selected_unique_responses": len(package),
        "grader_calls": len(cache),
        "cache_hits": cache_hits,
        "cache_misses": len(cache),
        "unselected_calls": 0,
        "public_artifact_leakage_scan": "passed",
    }
    write_json_atomic(context.stage_root() / "result.json", result)
    return result


def _metric_inputs(
    context: PipelineContext,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], float]]:
    selections = read_jsonl(context.run_root / "select-bon" / "selections.jsonl")
    gold_rows = read_jsonl(context.run_root / "audit-gold" / "gold_scores.jsonl")
    gold = {
        (str(row["prompt_id"]), str(row["response_id"])): float(row["gold_score"])
        for row in gold_rows
    }
    return selections, gold


def _cosine(left: Any, right: Any) -> float:
    return sum(float(a) * float(b) for a, b in zip(left, right))


def _semantic_overlap(
    previous: list[dict[str, Any]],
    current: list[dict[str, Any]],
    embedder: FakeEmbeddingProvider,
) -> float:
    if not previous and not current:
        return 1.0
    if not previous or not current:
        return 0.0
    previous_vectors = embedder.embed([str(item["text"]) for item in previous])
    current_vectors = embedder.embed([str(item["text"]) for item in current])

    def directed(
        source: list[dict[str, Any]],
        source_vectors: Any,
        target: list[dict[str, Any]],
        target_vectors: Any,
    ) -> float:
        target_ids = {str(item["criterion_id"]) for item in target}
        similarities = []
        for item, vector in zip(source, source_vectors):
            if str(item["criterion_id"]) in target_ids:
                similarities.append(1.0)
            else:
                similarities.append(
                    max(0.0, max(_cosine(vector, other) for other in target_vectors))
                )
        return sum(similarities) / len(similarities)

    return (
        directed(previous, previous_vectors, current, current_vectors)
        + directed(current, current_vectors, previous, previous_vectors)
    ) / 2


def _rubric_churn(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["prompt_id"]), str(row["mode"]))].append(row)
    embedder = FakeEmbeddingProvider()
    totals: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "admissions": 0,
            "deletions": 0,
            "textual_changes": 0,
            "semantic_overlaps": [],
            "survival_steps": [],
            "censored_survivors": 0,
        }
    )
    for (_prompt_id, mode), snapshots in grouped.items():
        ordered = sorted(snapshots, key=lambda item: int(item["policy_step"]))
        previous = [dict(item) for item in ordered[0]["criteria"] if item["source"] != "dynamic"]
        created: dict[str, int] = {}
        last_seen: dict[str, int] = {}
        for row in ordered:
            criteria = [dict(item) for item in row["criteria"]]
            admitted = row["admitted_id"] is not None
            totals[mode]["admissions"] += int(admitted)
            totals[mode]["deletions"] += int(admitted and row["evicted_id"] is not None)
            totals[mode]["textual_changes"] += int(
                [item["criterion_id"] for item in previous]
                != [item["criterion_id"] for item in criteria]
            )
            totals[mode]["semantic_overlaps"].append(
                _semantic_overlap(previous, criteria, embedder)
            )
            for criterion in criteria:
                if criterion["source"] != "dynamic":
                    continue
                criterion_id = str(criterion["criterion_id"])
                created.setdefault(criterion_id, int(criterion["created_step"]))
                last_seen[criterion_id] = int(row["policy_step"])
            previous = criteria
        final_ids = {
            str(item["criterion_id"])
            for item in ordered[-1]["criteria"]
            if item["source"] == "dynamic"
        }
        totals[mode]["censored_survivors"] += len(final_ids)
        totals[mode]["survival_steps"].extend(
            last_seen[criterion_id] - created_step + 1
            for criterion_id, created_step in created.items()
        )

    result: dict[str, dict[str, Any]] = {}
    for mode, values in totals.items():
        overlaps = values.pop("semantic_overlaps")
        survival = values.pop("survival_steps")
        result[mode] = {
            **values,
            "semantic_overlap_mean": sum(overlaps) / len(overlaps) if overlaps else 1.0,
            "semantic_overlap_min": min(overlaps) if overlaps else 1.0,
            "mean_survival_steps": sum(survival) / len(survival) if survival else None,
            "median_survival_steps": statistics.median(survival) if survival else None,
            "max_survival_steps": max(survival) if survival else None,
            "survival_observations": len(survival),
            "semantic_embedding": dict(embedder.identity),
        }
    return result


def run_analyze(context: PipelineContext) -> dict[str, Any]:
    from .reporting.aligned_analysis import AlignmentError, analyze_aligned
    from .reporting.interpretation import classify_interpretation

    input_paths = (
        context.run_root / "select-bon" / "selections.jsonl",
        context.run_root / "audit-gold" / "gold_scores.jsonl",
        context.run_root / "score-proxy" / "rubric_scores.jsonl",
        context.run_root / "replay-dynamic-final" / "replay_snapshots.jsonl",
        context.run_root / "train-static" / "trajectory_development.jsonl",
    )
    context.begin_stage(inputs=input_paths)
    selections, gold = _metric_inputs(context)
    bootstrap_iterations = int(context.raw.get("bootstrap", {}).get("iterations", 10_000))
    try:
        aligned = analyze_aligned(
            selections,
            gold,
            read_jsonl(input_paths[2]),
            iterations=bootstrap_iterations,
            seed=context.config.bootstrap_seed,
            n_grid=[int(value) for value in context.raw.get("bon", {}).get("sizes", [])],
            permutations=int(context.raw.get("bon", {}).get("permutations", 5)),
        )
    except AlignmentError as exc:
        raise StageError(f"analysis alignment failed: {exc}") from exc

    churn = _rubric_churn(read_jsonl(input_paths[3]))
    train_by_step: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(input_paths[4]):
        train_by_step[int(row["policy_step"])].append(row)
    policy_distance = {
        str(step): {
            "kl_from_pi0": sum(float(row["kl_from_pi0"]) for row in rows) / len(rows),
            "static_reward": sum(float(row["static_proxy_reward"]) for row in rows) / len(rows),
            "embedding_distance": sum(float(row["response_embedding_distance"]) for row in rows)
            / len(rows),
            "mean_length": sum(float(row["mean_response_length"]) for row in rows) / len(rows),
        }
        for step, rows in sorted(train_by_step.items())
    }
    primary = "dynamic_fixed_budgeted"
    previous = "dynamic_prev_budgeted"
    refresh = "refresh_only_budgeted"
    bootstraps = aligned.pop("bootstrap_results")
    missing = {primary, previous, refresh} - set(bootstraps)
    if missing:
        raise StageError(f"analysis is missing current-rubric comparisons: {sorted(missing)}")
    max_kl = max((value["kl_from_pi0"] for value in policy_distance.values()), default=0.0)
    decision = classify_interpretation(
        max_policy_kl=max_kl,
        admissions=churn[primary]["admissions"],
        primary=bootstraps[primary],
        previous=bootstraps[previous],
        refresh=bootstraps[refresh],
    )
    gt_auc = aligned["gt_auc"]
    regret = aligned["stale_rubric_regret"]
    comparisons = {
        name: {
            "gt_auc": gt_auc.get(name),
            "delta_vs_static": 0.0 if name == "static" else regret.get(name),
            "paired_bootstrap": aligned["paired_bootstrap"].get(name),
        }
        for name in REQUIRED_COMPARISONS
    }
    metrics = {
        "gt_auc": gt_auc,
        "gt_auc_cross_matrix": aligned["gt_auc_cross_matrix"],
        "stale_rubric_regret": regret,
        "top1_agreement": aligned["top1_agreement"],
        "kendall_tau_b": aligned["kendall_tau_b"],
        "reward_resolution": aligned["reward_resolution"],
        "rubric_churn": dict(churn),
        "policy_distance": policy_distance,
    }
    evidence = [
        *decision.evidence,
        "all functional comparisons use exact policy/prompt/N/permutation/pool joins",
        f"max_kl_from_pi0={max_kl}",
    ]
    report = build_report(
        context.run_id,
        sha256_file(context.stage_root() / "manifest.json"),
        comparisons,
        metrics,
        decision.label,
        evidence,
        "run causal schedule follow-up"
        if decision.label in {"policy_adaptive_gain", "frequent_update_need"}
        else "do not expand until the diagnosed limitation is addressed",
        bootstrap_samples=bootstrap_iterations,
        confidence=float(context.raw.get("bootstrap", {}).get("confidence", 0.95)),
    )
    report["paired_bootstrap"] = aligned["paired_bootstrap"][primary]
    report["paired_bootstrap_by_mode"] = aligned["paired_bootstrap"]
    report["textual_vs_functional_change"] = {
        "textual": dict(churn),
        "functional": {
            "top1_agreement": aligned["top1_agreement"],
            "gt_auc_delta": regret,
        },
    }
    report_path = context.results_root / context.run_id / "pilot_report.json"
    write_json_atomic(report_path, report)
    markdown_path = context.results_root / context.run_id / "pilot_report.md"
    markdown = render_markdown(report)
    write_text_atomic(markdown_path, markdown)
    result = {
        "report": str(report_path.relative_to(context.root)),
        "interpretation": decision.label,
        "paired_bootstrap": aligned["paired_bootstrap"][primary],
    }
    write_json_atomic(context.stage_root() / "result.json", result)
    return result


def expected_inventory(context: PipelineContext) -> dict[str, int]:
    specs = {spec.name: spec.count for spec in _configured_splits(context.raw)}
    probe_count = sum(count for name, count in specs.items() if "probe" in name)
    audit_count = sum(count for name, count in specs.items() if "audit" in name)
    steps = context.config.training.max_steps
    probe_cfg = context.raw.get("probe", {})
    bon_cfg = context.raw.get("bon", {})
    return {
        "prompts": sum(specs.values()),
        "static_candidates": sum(specs.values()) * 12,
        "development_discovery": steps
        * probe_count
        * int(probe_cfg.get("development_samples_per_family", 4)),
        "development_validation": steps
        * probe_count
        * int(probe_cfg.get("development_samples_per_family", 4)),
        "final_discovery": steps * audit_count * int(probe_cfg.get("final_samples_per_family", 4)),
        "final_validation": steps * audit_count * int(probe_cfg.get("final_samples_per_family", 4)),
        "bon_candidates": len(bon_cfg.get("focal_steps", [3, 10, 30, 100]))
        * audit_count
        * int(bon_cfg.get("pool_size", 64)),
    }


def run_validate_inventory(context: PipelineContext) -> dict[str, Any]:
    paths = {
        "static": context.run_root / "generate-static" / "static_rubrics.jsonl",
        "candidates": context.run_root / "generate-static" / "static_candidates.jsonl",
        "references": context.run_root / "train-static" / "reference_responses.jsonl",
        "development": context.run_root / "train-static" / "trajectory_development.jsonl",
        "final": context.run_root / "trajectory" / "final_sealed" / "responses.jsonl",
        "replay_final": context.run_root / "replay-dynamic-final" / "replay_snapshots.jsonl",
        "bon": context.run_root / "generate-bon" / "bon_pool.jsonl",
        "scores": context.run_root / "score-proxy" / "rubric_scores.jsonl",
        "selections": context.run_root / "select-bon" / "selections.jsonl",
        "audit_package": context.run_root / "export-audit-package" / "audit_package.jsonl",
        "gold": context.run_root / "audit-gold" / "gold_scores.jsonl",
    }
    split_paths = tuple(
        context.public_root / f"{spec.name}.jsonl" for spec in _configured_splits(context.raw)
    )
    context.begin_stage(
        inputs=(*paths.values(), context.public_root / "split_manifest.json", *split_paths)
    )
    expected = expected_inventory(context)
    static = read_jsonl(paths["static"])
    candidates = read_jsonl(paths["candidates"])
    references = read_jsonl(paths["references"])
    development = read_jsonl(paths["development"])
    final = read_jsonl(paths["final"])
    replay = read_jsonl(paths["replay_final"])
    bon = read_jsonl(paths["bon"])
    actual = {
        "prompts": len(static),
        "static_candidates": len(candidates),
        "development_discovery": sum(
            row["family"] == "trajectory_discovery" for row in development
        ),
        "development_validation": sum(
            row["family"] == "trajectory_validation" for row in development
        ),
        "final_discovery": sum(row["family"] == "trajectory_discovery" for row in final),
        "final_validation": sum(row["family"] == "trajectory_validation" for row in final),
        "bon_candidates": len(bon),
    }
    if actual != expected:
        raise StageError(f"inventory mismatch: expected={expected}, actual={actual}")
    if any(len(row["criteria"]) != 8 for row in static):
        raise StageError("static rubric inventory violated exact 8-criterion contract")
    if any(
        row["criterion_count"] > 12 for row in replay if row["mode"] != "dynamic_fixed_cumulative"
    ):
        raise StageError("budgeted rubric exceeded 12 criteria")
    if any(sum(item["source"] != "dynamic" for item in row["criteria"]) != 8 for row in replay):
        raise StageError("a replay snapshot did not preserve all 8 static criteria")
    package = read_jsonl(paths["audit_package"])
    gold = read_jsonl(paths["gold"])
    if len(package) != len(gold):
        raise StageError("hidden grader did not score exactly the selected unique response package")
    seed_values = [row["seed"] for row in candidates + references + development + final + bon]
    response_values = [
        row["response_id"] for row in candidates + references + development + final + bon
    ]
    if len(seed_values) != len(set(seed_values)) or len(response_values) != len(
        set(response_values)
    ):
        raise StageError("seed or response ID collision detected")
    try:
        semantic_checks = validate_semantic_inventory(context, paths)
    except SemanticInventoryError as error:
        raise StageError(f"semantic inventory mismatch: {error}") from error
    records = [artifact_record(path) for path in paths.values()]
    for record in records:
        validate_artifact_record(record)
    result = {
        "status": "passed",
        "expected": expected,
        "actual": actual,
        "artifact_records": records,
        "seed_collisions": 0,
        "response_id_reuse": 0,
        "semantic_checks": semantic_checks,
        "public_private_boundary": "passed_by_stage_config_and_audit_only_cli",
    }
    write_json_atomic(context.stage_root() / "inventory.json", result)
    return result


def estimate_cost_counts(context: PipelineContext) -> dict[str, int]:
    inventory = expected_inventory(context)
    specs = {spec.name: spec.count for spec in _configured_splits(context.raw)}
    audit_prompts = sum(count for name, count in specs.items() if "audit" in name)
    probe_prompts = sum(count for name, count in specs.items() if "probe" in name)
    steps = context.config.training.max_steps
    bon = context.raw.get("bon", {})
    replay_modes = len(context.raw.get("replay", {}).get("modes", REQUIRED_COMPARISONS)) - 1
    return {
        **inventory,
        "static_rubric_generator_calls": inventory["prompts"],
        "dynamic_generator_call_upper_bound": steps
        * (probe_prompts + audit_prompts)
        * replay_modes,
        "bon_selection_rows_upper_bound": len(bon.get("focal_steps", [3, 10, 30, 100]))
        * audit_prompts
        * len(bon.get("sizes", [1, 2, 4, 8, 16, 32, 64]))
        * int(bon.get("permutations", 5))
        * max(1, replay_modes + 1),
        "hidden_gt_call_upper_bound": inventory["bon_candidates"],
    }
