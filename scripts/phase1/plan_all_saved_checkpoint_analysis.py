#!/usr/bin/env python3
"""Write an analysis-only coverage plan for every committed saved checkpoint."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from collections.abc import Sequence
from typing import Any

from dynamic_rubric.artifacts import read_json, write_json_atomic
from dynamic_rubric.hashing import sha256_file
from dynamic_rubric.phase1.checkpoint_inventory import (
    comparison_plan,
    discover_committed_policy_checkpoints,
)


def _read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _probe_coverage(audit_roots: Sequence[Path], step: int) -> dict[str, Any]:
    candidates = [root / "responses" / f"checkpoint-{step:06d}" for root in audit_roots]
    required: dict[str, Path] = {}
    for pool in ("probe_A", "probe_B"):
        paths = [directory / f"{pool}.jsonl" for directory in candidates]
        required[pool] = next(
            (path for path in paths if path.is_file() and path.stat().st_size > 0), paths[-1]
        )
    present = {pool: path.is_file() and path.stat().st_size > 0 for pool, path in required.items()}
    validation_errors: list[str] = []
    rows_by_pool: dict[str, list[dict[str, Any]]] = {}
    expected_counts = {"probe_A": 800, "probe_B": 1600}
    for pool, path in required.items():
        if not present[pool]:
            validation_errors.append(f"{pool}:missing")
            continue
        try:
            rows = _read_jsonl_rows(path)
        except (OSError, json.JSONDecodeError):
            validation_errors.append(f"{pool}:invalid_jsonl")
            continue
        rows_by_pool[pool] = rows
        response_ids = [str(row.get("response_id", "")) for row in rows]
        prompt_ids = {str(row.get("prompt_id", "")) for row in rows}
        if len(rows) != expected_counts[pool]:
            validation_errors.append(f"{pool}:expected_{expected_counts[pool]}_rows_got_{len(rows)}")
        if not all(response_ids) or len(response_ids) != len(set(response_ids)):
            validation_errors.append(f"{pool}:response_ids_not_unique_nonempty")
        if "" in prompt_ids or len(prompt_ids) != 100:
            validation_errors.append(f"{pool}:expected_100_prompt_ids_got_{len(prompt_ids - {''})}")
        if any(str(row.get("pool", "")) != pool for row in rows):
            validation_errors.append(f"{pool}:pool_label_mismatch")
        if any(int(row.get("policy_step", -1)) != step for row in rows):
            validation_errors.append(f"{pool}:policy_step_mismatch")
    if set(rows_by_pool) == {"probe_A", "probe_B"}:
        ids_a = {str(row.get("response_id", "")) for row in rows_by_pool["probe_A"]}
        ids_b = {str(row.get("response_id", "")) for row in rows_by_pool["probe_B"]}
        if ids_a & ids_b:
            validation_errors.append("probe_A_probe_B_response_id_overlap")
        prompts_a = [str(row.get("prompt_id", "")) for row in rows_by_pool["probe_A"]]
        prompts_b = [str(row.get("prompt_id", "")) for row in rows_by_pool["probe_B"]]
        if set(prompts_a) != set(prompts_b):
            validation_errors.append("probe_A_probe_B_prompt_inventory_mismatch")
        if set(Counter(prompts_a).values()) != {8}:
            validation_errors.append("probe_A_expected_8_responses_per_prompt")
        if set(Counter(prompts_b).values()) != {16}:
            validation_errors.append("probe_B_expected_16_responses_per_prompt")
        manifest_candidates = [
            root / "manifests" / "fixed_train_probe_prompts.jsonl" for root in audit_roots
        ]
        manifest = next((path for path in manifest_candidates if path.is_file()), None)
        if manifest is None:
            validation_errors.append("fixed_train_probe_manifest_missing")
        else:
            try:
                manifest_rows = _read_jsonl_rows(manifest)
                manifest_ids = {str(row.get("prompt_id", "")) for row in manifest_rows}
            except (OSError, json.JSONDecodeError):
                validation_errors.append("fixed_train_probe_manifest_invalid_jsonl")
            else:
                if "" in manifest_ids or len(manifest_ids) != 100:
                    validation_errors.append("fixed_train_probe_manifest_expected_100_ids")
                elif set(prompts_a) != manifest_ids or set(prompts_b) != manifest_ids:
                    validation_errors.append("probe_prompt_inventory_differs_from_manifest")
    if not all(present.values()):
        generation_status = "planned_not_completed"
    elif validation_errors:
        generation_status = "files_present_unvalidated"
    else:
        generation_status = "completed_validated"
    return {
        "step": step,
        "probe_responses": {pool: str(path.resolve()) for pool, path in required.items()},
        "probe_response_pool_present": present,
        "response_generation_status": generation_status,
        "response_validation_errors": validation_errors,
        "fresh_rubric_and_cross_grading_status": "planned_not_completed",
    }


def build_plan(
    run_root: Path,
    audit_roots: Path | Sequence[Path],
    *,
    plan_config: Path | None = None,
) -> dict[str, Any]:
    inventory = discover_committed_policy_checkpoints(run_root)
    config = read_json(plan_config) if plan_config is not None else {}
    horizon_anchors = config.get("reuse_horizon_anchors")
    if (
        config.get("reuse_horizon_matrix") == "selected_anchor_checkpoints_only"
        and horizon_anchors is None
    ):
        raise ValueError("selected anchor analysis requires reuse_horizon_anchors")
    comparisons = comparison_plan(inventory["steps"], reuse_horizon_anchors=horizon_anchors)
    roots = [audit_roots] if isinstance(audit_roots, Path) else list(audit_roots)
    roots = [root.resolve() for root in roots]
    if not roots:
        raise ValueError("at least one audit root is required")
    coverage = [_probe_coverage(roots, step) for step in inventory["steps"]]
    return {
        "schema_version": 1,
        "analysis": "phase1_all_saved_checkpoint_probe_analysis",
        "primary_dataset": "fixed_train_probe_100",
        "analysis_plan_config": (
            {"path": str(plan_config.resolve()), "sha256": sha256_file(plan_config)}
            if plan_config is not None
            else None
        ),
        "checkpoint_inventory": inventory,
        "audit_roots": [str(root) for root in roots],
        "comparison_plan": comparisons,
        "coverage": coverage,
        "coverage_policy": (
            "Every committed saved checkpoint is required. Missing pools, rubrics, grades, or "
            "matrix cells remain explicitly planned_not_completed and are never silently omitted."
        ),
        "policy_outcome_evaluation": {
            "checkpoints": list(inventory["steps"]),
            "in_domain": "RaR-Medicine held-out 300",
            "external": "HealthBench",
            "status": "planned_not_completed",
            "timing_analysis_role": "forbidden_policy_outcomes_only",
        },
        "policy_audit_commands": {
            "export_all_discovered_checkpoints": [
                "python",
                "-m",
                "dynamic_rubric.phase1.audit_policy",
                "export",
                "--run-dir",
                str(run_root.resolve()),
                "--output-root",
                str(roots[-1]),
                "--steps",
                *[str(step) for step in inventory["steps"]],
            ],
            "generate_probe_pools_per_step_template": [
                "python",
                "-m",
                "dynamic_rubric.phase1.audit_policy",
                "generate",
                "--run-dir",
                str(run_root.resolve()),
                "--output-root",
                str(roots[-1]),
                "--step",
                "<each checkpoint_inventory.steps value>",
                "--base-url",
                "<policy-vllm-base-url>",
            ],
        },
        "analysis_command_after_coverage_is_complete": [
            "python",
            "-m",
            "dynamic_rubric.phase1.audit_analysis",
            "--all-saved-checkpoints-from",
            str(run_root.resolve()),
            "--through-step",
            str(max(inventory["steps"])),
            *(
                ["--reuse-horizon-anchors", *[str(step) for step in horizon_anchors]]
                if horizon_anchors is not None
                else []
            ),
            "--scores",
            "<all-probe-B-score-jsonl-paths>",
            "--policy-state",
            "<all-policy-state-jsonl>",
            "--policy-distance",
            "<all-policy-distance-summary-paths>",
            "--config",
            str((run_root / "config.resolved.json").resolve()),
            "--output",
            "<analysis-output-directory>",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--audit-root", action="append", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = read_json(args.config) if args.config is not None else {}
    run_root = args.run_root or (Path(config["run_root"]) if "run_root" in config else None)
    audit_roots = args.audit_root or [Path(item) for item in config.get("audit_roots", ())]
    if run_root is None or not audit_roots:
        parser.error("provide --config or both --run-root and at least one --audit-root")
    plan = build_plan(run_root, audit_roots, plan_config=args.config)
    write_json_atomic(args.output, plan)
    print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
