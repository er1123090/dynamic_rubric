#!/usr/bin/env python3
"""Run a separate same-H200 fresh/stale paired sensitivity audit."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import fcntl
import json
from pathlib import Path
import time
from typing import Any, Sequence

from dynamic_rubric.artifacts import read_json, read_jsonl, write_json_atomic
from dynamic_rubric.hashing import sha256_file, sha256_json
from dynamic_rubric.phase1.audit_run import (
    AuditTask,
    build_tasks,
    endpoint_identity,
    equivalent_metrics,
    score_group,
)
from dynamic_rubric.phase1.audit_scoring import AuditScoreConfig, score_pool
from dynamic_rubric.phase1.metrics import compare_fresh_stale
from dynamic_rubric.providers.vllm_chat import VLLMChatAdapter


INTERPRETATION = (
    "separate same-H200 paired sensitivity; not canonical training rewards, "
    "not fixed probe, and canonical fresh grades are reference-only (not GT)"
)


def load_fresh_rubrics(run: Path, tasks: Sequence[AuditTask]) -> dict[tuple[int, str], list[dict]]:
    needed: dict[int, set[str]] = {}
    for task in tasks:
        needed.setdefault(task.step, set()).add(task.occurrence_id)
    resolved: dict[tuple[int, str], list[dict]] = {}
    for step, occurrence_ids in needed.items():
        path = run / "verl-run" / "online_steps" / f"step-{step:06d}" / "rubric_unions.jsonl"
        for union in read_jsonl(path):
            occurrence_id = union["prompt_occurrence_id"]
            if occurrence_id not in occurrence_ids:
                continue
            resolved[(step, occurrence_id)] = list(union["offline_criteria"]) + list(
                union["online_criteria"]
            )
            task = next(
                item for item in tasks if item.step == step and item.occurrence_id == occurrence_id
            )
            if union["content_hash"] != task.fresh_rubric_hash:
                raise ValueError("fresh rubric hash differs from build_tasks provenance")
    if len(resolved) != len(tasks):
        raise ValueError("not every paired task has its canonical fresh rubric")
    return resolved


def _label(rows: list[dict], task: AuditTask, *, fresh: bool) -> list[dict]:
    expected_ids = [row["response_id"] for row in task.responses]
    if [row["response_id"] for row in rows] != expected_ids:
        raise ValueError("paired score response IDs differ from canonical saved responses")
    for row, response in zip(rows, task.responses):
        row.update(
            fresh_or_stale="fresh" if fresh else "stale",
            evaluator_step=(task.step if fresh else task.prior_step) - 1,
            evaluator_checkpoint=f"visit-{task.step if fresh else task.prior_step}",
            rubric_hash=task.fresh_rubric_hash if fresh else task.stale_rubric_hash,
            rollout_index=response["rollout_index"],
        )
    return rows


def run_group(
    task: AuditTask,
    fresh_rubric: list[dict],
    output: Path,
    config: AuditScoreConfig,
    grader: Any,
    epsilon_z: float,
    epsilon_t: float,
    endpoint_hash: str,
) -> dict:
    destination = output / "groups" / f"step-{task.step:06d}" / f"{task.prompt_id}.json"
    identity = sha256_json(
        {
            "task": asdict(task),
            "fresh_rubric": fresh_rubric,
            "score_config": asdict(config),
            "endpoint_hash": endpoint_hash,
            "epsilon_z": epsilon_z,
            "epsilon_t": epsilon_t,
        }
    )
    seal = output / "group_hashes" / f"step-{task.step:06d}" / f"{task.prompt_id}.json"
    if destination.is_file():
        value = read_json(destination)
        if value.get("identity") != identity:
            raise ValueError("completed paired group identity changed")
        digest = sha256_file(destination)
        if not seal.is_file() or read_json(seal).get("sha256") != digest:
            raise ValueError("completed paired group seal changed or is missing")
        expected = compare_fresh_stale(
            score_group(value["stale"], task, f"visit-{task.prior_step}"),
            score_group(value["fresh"], task, f"visit-{task.step}"),
            epsilon_z=epsilon_z,
            epsilon_t=epsilon_t,
        )
        expected.update(same_pool_b=False, same_response_pool=True)
        if not equivalent_metrics(expected, value["comparison"]):
            raise ValueError("cached paired metrics differ from underlying grades")
        return value
    common = dict(
        evaluator_checkpoint=str(task.step),
        policy_checkpoint=str(task.step - 1),
        pool="train_batch",
        config=config,
        grader=grader,
    )
    fresh = score_pool(
        task.responses,
        {task.prompt_id: fresh_rubric},
        cache_dir=output / "grade_cache" / "fresh" / f"step-{task.step:06d}" / task.prompt_id,
        **common,
    )
    stale = score_pool(
        task.responses,
        {task.prompt_id: task.stale_rubric},
        evaluator_checkpoint=str(task.prior_step),
        policy_checkpoint=str(task.step - 1),
        pool="train_batch",
        config=config,
        grader=grader,
        cache_dir=output / "grade_cache" / "stale" / f"step-{task.step:06d}" / task.prompt_id,
    )
    fresh = _label(fresh, task, fresh=True)
    stale = _label(stale, task, fresh=False)
    comparison = compare_fresh_stale(
        score_group(stale, task, f"visit-{task.prior_step}"),
        score_group(fresh, task, f"visit-{task.step}"),
        epsilon_z=epsilon_z,
        epsilon_t=epsilon_t,
    )
    comparison.update(same_pool_b=False, same_response_pool=True)
    value = {
        "schema_version": 1,
        "identity": identity,
        "domain": config.domain,
        "method": config.method,
        "seed": config.seed,
        "global_step": task.step,
        "prompt_id": task.prompt_id,
        "policy_step": task.step - 1,
        "fresh_creation_update": task.step,
        "stale_creation_update": task.prior_step,
        "evaluator_age_steps": task.step - task.prior_step,
        "clock": task.clock,
        "pool": "train_batch",
        "fresh": fresh,
        "stale": stale,
        "comparison": comparison,
        "canonical_fresh_reference": task.fresh,
        "canonical_fresh_reference_role": "reference-only; not GT and not used in paired metrics",
        "interpretation": INTERPRETATION,
    }
    write_json_atomic(destination, value, immutable=True)
    write_json_atomic(seal, {"sha256": sha256_file(destination)}, immutable=True)
    return value


def run(args: argparse.Namespace) -> dict:
    run_root, output = args.run.resolve(), args.output.resolve()
    if output == run_root or run_root in output.parents:
        raise ValueError("paired output must be outside the immutable canonical run")
    if args.group_workers <= 0 or args.requests_per_group <= 0:
        raise ValueError("worker counts must be positive")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "paired.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        tasks, inventory = build_tasks(run_root, args.through)
        selected = tasks[: args.limit_groups] if args.limit_groups else tasks
        fresh_rubrics = load_fresh_rubrics(run_root, selected)
        source_cfg = read_json(run_root / "config.resolved.json")
        judge_cfg = source_cfg["models"]["judge"]
        endpoint = endpoint_identity(args.candidate_url, judge_cfg["model"], judge_cfg["revision"])
        logical_endpoint = {
            "url": args.candidate_url,
            "model_id": endpoint["model"]["id"],
            "model_root": endpoint["model"]["root"],
            "revision": judge_cfg["revision"],
            "vllm_version": endpoint["version"],
        }
        endpoint_hash = sha256_json(logical_endpoint)
        write_json_atomic(output / "endpoint_identity.json", logical_endpoint, immutable=True)
        write_json_atomic(
            output / "endpoint_observations" / f"{sha256_json(endpoint)}.json",
            endpoint,
            immutable=True,
        )
        write_json_atomic(
            output / "config.json",
            {
                "run": str(run_root),
                "through": args.through,
                "candidate_url": args.candidate_url,
                "source_hashes_sha256": sha256_json(inventory["source_hashes"]),
                "requests_per_group": args.requests_per_group,
                "epsilon_z": source_cfg["analysis"]["epsilon_z"],
                "epsilon_t": source_cfg["analysis"]["epsilon_t"],
                "interpretation": INTERPRETATION,
            },
            immutable=True,
        )
        config = AuditScoreConfig(
            domain=source_cfg["domain"],
            seed=source_cfg["seed"],
            judge_revision=judge_cfg["revision"],
            concurrency=args.requests_per_group,
        )
        grader = VLLMChatAdapter(
            args.candidate_url,
            config.judge_model,
            output / "provider_cache",
            timeout_seconds=600,
            max_retries=4,
        )
        start = time.monotonic()
        errors, completed = [], 0
        with ThreadPoolExecutor(max_workers=args.group_workers) as executor:
            futures = {
                executor.submit(
                    run_group,
                    task,
                    fresh_rubrics[(task.step, task.occurrence_id)],
                    output,
                    config,
                    grader,
                    source_cfg["analysis"]["epsilon_z"],
                    source_cfg["analysis"]["epsilon_t"],
                    endpoint_hash,
                ): task
                for task in selected
            }
            for future in as_completed(futures):
                task = futures[future]
                try:
                    future.result()
                    completed += 1
                except Exception as error:
                    errors.append(
                        {"step": task.step, "prompt_id": task.prompt_id, "error": repr(error)}
                    )
                status = {
                    "state": "running",
                    "selected_groups": len(selected),
                    "total_groups": len(tasks),
                    "completed_groups": completed,
                    "errors": errors,
                    "elapsed_seconds": time.monotonic() - start,
                    "training_resumed": False,
                    "interpretation": INTERPRETATION,
                }
                write_json_atomic(output / "status.json", status, immutable=False)
        status["state"] = (
            "failed" if errors else "smoke_complete" if args.limit_groups else "complete"
        )
        write_json_atomic(output / "status.json", status, immutable=False)
        if errors:
            raise RuntimeError(f"paired sensitivity failed for {len(errors)} groups")
        return status


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--run", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--candidate-url", default="http://127.0.0.1:28007")
    value.add_argument("--through", type=int, default=34)
    value.add_argument("--group-workers", type=int, default=8)
    value.add_argument("--requests-per-group", type=int, default=8)
    value.add_argument("--limit-groups", type=int)
    return value


def main(argv: Sequence[str] | None = None) -> None:
    print(json.dumps(run(parser().parse_args(argv)), sort_keys=True))


if __name__ == "__main__":
    main()
