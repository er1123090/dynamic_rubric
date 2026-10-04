#!/usr/bin/env python3
"""Certify one audit judge against frozen, completed Inference B audit receipts."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
from typing import Any, Sequence

from dynamic_rubric.artifacts import (
    read_json,
    write_json_atomic,
    write_jsonl_atomic,
)
from dynamic_rubric.hashing import sha256_file, sha256_json
from dynamic_rubric.phase1.audit_run import AuditTask, build_tasks, endpoint_identity
from dynamic_rubric.phase1.audit_scoring import AuditScoreConfig, score_pool
from dynamic_rubric.providers.vllm_chat import VLLMChatAdapter


def select_groups(candidates: Sequence[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    """Evenly sample an ordered step/rubric-size inventory without outcome access."""
    ordered = sorted(
        candidates, key=lambda row: (row["step"], row["rubric_size"], row["prompt_id"])
    )
    if count <= 0 or len(ordered) < count:
        raise ValueError(f"need at least {count} completed groups, found {len(ordered)}")
    if count == 1:
        return [ordered[len(ordered) // 2]]
    indices = [round(i * (len(ordered) - 1) / (count - 1)) for i in range(count)]
    return [ordered[index] for index in indices]


def _completed_inventory(tasks: Sequence[AuditTask], audit: Path) -> list[dict[str, Any]]:
    inventory = []
    for task in tasks:
        path = audit / "groups" / f"step-{task.step:06d}" / f"{task.prompt_id}.json"
        seal = audit / "group_hashes" / f"step-{task.step:06d}" / f"{task.prompt_id}.json"
        if not path.is_file():
            continue
        digest = sha256_file(path)
        if seal.is_file() and read_json(seal).get("sha256") != digest:
            raise ValueError(f"completed audit group seal mismatch: {path}")
        group = read_json(path)
        expected_ids = [row["response_id"] for row in task.responses]
        if [row["response_id"] for row in group["stale"]] != expected_ids:
            raise ValueError(f"completed audit response IDs changed: {path}")
        inventory.append(
            {
                "step": task.step,
                "prompt_id": task.prompt_id,
                "rubric_size": len(task.stale_rubric),
                "group_path": str(path),
                "group_sha256": digest,
                "fresh_rubric_hash": task.fresh_rubric_hash,
                "stale_rubric_hash": task.stale_rubric_hash,
            }
        )
    return inventory


def _compare(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    exact = {
        field: reference[field] == candidate[field]
        for field in ("grades", "numerator", "denominator", "reward")
    }
    reward_close = math.isclose(
        float(reference["reward"]), float(candidate["reward"]), rel_tol=1e-14, abs_tol=1e-15
    )
    return {
        "response_id": reference["response_id"],
        "exact": exact,
        "reward_close_1e-14": reward_close,
        "tolerance_only": reward_close and not exact["reward"],
        "passed": all(exact.values()),
        "reference": {key: reference[key] for key in exact},
        "candidate": {key: candidate[key] for key in exact},
    }


def certify(args: argparse.Namespace) -> dict[str, Any]:
    run, audit, output = args.run.resolve(), args.audit.resolve(), args.output.resolve()
    if len({run, audit, output}) != 3 or run in output.parents or audit in output.parents:
        raise ValueError("run, existing audit, and canary output must be separate")
    tasks, source_inventory = build_tasks(run, args.through)
    by_key = {(task.step, task.prompt_id): task for task in tasks}
    completed = _completed_inventory(tasks, audit)
    source_hashes_sha256 = sha256_json(source_inventory["source_hashes"])
    manifest_path = output / "manifest.json"
    if manifest_path.is_file():
        manifest = read_json(manifest_path)
        if (
            manifest.get("requested_groups") != args.groups
            or manifest.get("build_tasks_source_hashes_sha256") != source_hashes_sha256
        ):
            raise ValueError("existing canary manifest is incompatible with this run")
        current = {(row["step"], row["prompt_id"]): row for row in completed}
        selected = manifest["selected"]
        for frozen in selected:
            actual = current.get((frozen["step"], frozen["prompt_id"]))
            if actual is None or actual["group_sha256"] != frozen["group_sha256"]:
                raise ValueError("frozen canary group is missing or changed")
    else:
        selected = select_groups(completed, args.groups)
        manifest = {
            "schema_version": 1,
            "selection_policy": "even indices after ordering by (step, rubric_size, prompt_id)",
            "requested_groups": args.groups,
            "eligible_groups": len(completed),
            "eligible_inventory_sha256": sha256_json(completed),
            "build_tasks_source_hashes_sha256": source_hashes_sha256,
            "selected": selected,
        }
        # Freeze selection and hashes before the first candidate-endpoint call.
        write_json_atomic(manifest_path, manifest, immutable=True)

    source_cfg = read_json(run / "config.resolved.json")
    judge_cfg = source_cfg["models"]["judge"]
    candidate_identity = endpoint_identity(
        args.candidate_url, judge_cfg["model"], judge_cfg["revision"]
    )
    existing_identity = read_json(audit / "judge_identity.json")
    candidate_common = {
        "model": judge_cfg["model"],
        "revision": judge_cfg["revision"],
        "vllm_version": candidate_identity["version"],
    }
    existing_common = {key: existing_identity[key] for key in candidate_common}
    if candidate_common != existing_common:
        raise ValueError("candidate judge identity differs from canonical Inference B audit judge")
    write_json_atomic(output / "candidate_identity.json", candidate_common, immutable=True)
    write_json_atomic(
        output / "endpoint_observations" / f"{sha256_json(candidate_identity)}.json",
        candidate_identity,
        immutable=True,
    )

    config = AuditScoreConfig(
        domain=source_cfg["domain"],
        seed=source_cfg["seed"],
        judge_revision=judge_cfg["revision"],
        concurrency=8,
    )
    grader = VLLMChatAdapter(
        args.candidate_url,
        config.judge_model,
        output / "provider_cache",
        timeout_seconds=600,
        max_retries=4,
    )

    def score(selected_group: dict[str, Any]) -> tuple[list[dict], list[dict]]:
        task = by_key[(selected_group["step"], selected_group["prompt_id"])]
        reference = read_json(Path(selected_group["group_path"]))["stale"]
        candidate = score_pool(
            task.responses,
            {task.prompt_id: task.stale_rubric},
            evaluator_checkpoint=str(task.prior_step),
            policy_checkpoint=str(task.step - 1),
            pool="train_batch",
            config=config,
            grader=grader,
            cache_dir=output / "grade_cache" / f"step-{task.step:06d}" / task.prompt_id,
        )
        if [row["response_id"] for row in candidate] != [row["response_id"] for row in reference]:
            raise ValueError("candidate response IDs differ from the frozen Inference B reference")

        comparisons = [_compare(left, right) for left, right in zip(reference, candidate)]
        write_json_atomic(
            output / "candidate_groups" / f"step-{task.step:06d}" / f"{task.prompt_id}.json",
            {"candidate": candidate, "comparisons": comparisons},
            immutable=True,
        )
        return candidate, comparisons

    with ThreadPoolExecutor(max_workers=args.group_workers) as executor:
        scored = list(executor.map(score, selected))
    comparisons = [row for _candidate, group_rows in scored for row in group_rows]
    mismatches = [row for row in comparisons if not row["passed"]]
    write_jsonl_atomic(output / "comparisons.jsonl", comparisons, immutable=True)
    write_jsonl_atomic(output / "mismatches.jsonl", mismatches, immutable=True)
    result = {
        "passed": not mismatches,
        "selected_groups": len(selected),
        "compared_responses": len(comparisons),
        "exact_mismatches": len(mismatches),
        "tolerance_only_rewards": sum(row["tolerance_only"] for row in comparisons),
        "candidate_url": args.candidate_url,
        "reference": "existing immutable Inference B stale receipts",
    }
    write_json_atomic(output / "report.json", result, immutable=True)
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--run", type=Path, required=True)
    value.add_argument("--audit", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--candidate-url", default="http://127.0.0.1:28007")
    value.add_argument("--through", type=int, default=34)
    value.add_argument("--groups", type=int, default=8)
    value.add_argument("--group-workers", type=int, default=8)
    return value


def main(argv: Sequence[str] | None = None) -> None:
    print(json.dumps(certify(parser().parse_args(argv)), sort_keys=True))


if __name__ == "__main__":
    main()
