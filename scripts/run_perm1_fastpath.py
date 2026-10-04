#!/usr/bin/env python3
"""Publish permutation-0 GPT/graphs before the full five-permutation audit."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

from dynamic_rubric.artifacts import write_json_atomic, write_jsonl_atomic
from dynamic_rubric.evaluation.bon import BON_SIZES
from dynamic_rubric.minimum_gold_streaming import sync_streaming_gold
from dynamic_rubric.minimum_interim import ordered_prompt_subset
from dynamic_rubric.minimum_staleness import FOCAL_STEPS, _jsonl, _shard_name
from dynamic_rubric.online_rubric_bon import prepare_or_submit_available_gold


def fast_root(main_run_root: Path) -> Path:
    return main_run_root.with_name(main_run_root.name + "-perm1-fast")


def _safe_link(source: Path, target: Path) -> None:
    if target.is_symlink():
        if target.resolve() != source.resolve():
            raise RuntimeError(f"fast-path source link drift: {target}")
        return
    if target.exists():
        raise RuntimeError(f"fast-path link target exists: {target}")
    target.symlink_to(source.resolve(), target_is_directory=source.is_dir())


def permutation_zero_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    selected = [dict(row) for row in rows if int(row["permutation"]) == 0]
    if not selected:
        raise RuntimeError("selection shard has no permutation 0 rows")
    if {int(row["permutation"]) for row in selected} != {0}:
        raise RuntimeError("permutation-0 filter failed")
    return selected


def prepare(main_run_root: Path, prompt_count: int) -> dict[str, Any]:
    root = fast_root(main_run_root)
    root.mkdir(parents=True, exist_ok=True)
    for name in ("generate-bon", "score-proxy-minimum", "paper-judge-gold-egress-approval.json"):
        source = main_run_root / name
        if not source.exists():
            raise RuntimeError(f"missing main-run fast-path input: {source}")
        _safe_link(source, root / name)
    prompt_ids = ordered_prompt_subset(root, prompt_count)
    write_json_atomic(
        root / "perm1-fast-source.json",
        {
            "schema_version": 1,
            "main_run_root": str(main_run_root.resolve()),
            "prompt_ids": list(prompt_ids),
            "permutations": 1,
            "selected_permutation": 0,
            "n_grid": list(BON_SIZES),
            "purpose": "early policy-by-policy graph before full five-permutation audit",
        },
    )
    return {"fast_run_root": str(root), "prompt_ids": list(prompt_ids)}


def sync_selections(main_run_root: Path, prompt_count: int) -> dict[str, Any]:
    prepared = prepare(main_run_root, prompt_count)
    root = Path(prepared["fast_run_root"])
    source_root = main_run_root / "select-bon-minimum" / "shards"
    output_root = root / "select-bon-minimum" / "shards"
    published = 0
    for source_path in sorted(source_root.glob("*.jsonl")):
        output_path = output_root / source_path.name
        rows = permutation_zero_rows(_jsonl(source_path))
        expected = 2 * len(BON_SIZES)
        if len(rows) != expected:
            raise RuntimeError(
                f"permutation-0 selection inventory mismatch: {source_path}={len(rows)}"
            )
        write_jsonl_atomic(output_path, rows)
        published += 1
    write_json_atomic(
        root / "select-bon-minimum" / "progress.json",
        {"completed_prompt_policy_shards": published, "expected_prompt_policy_shards": 20},
        immutable=False,
    )
    return {"fast_run_root": str(root), "selection_shards": published}


def completed_steps(root: Path, prompt_count: int) -> tuple[int, ...]:
    prompt_ids = ordered_prompt_subset(root, prompt_count)
    completed = []
    for step in FOCAL_STEPS:
        policy_id = f"pi_{step}"
        if all(
            (
                root
                / "audit-gold-streaming-private"
                / "groups"
                / f"{policy_id}-{_shard_name(policy_id, prompt_id)}"
                / "gold_scores.jsonl"
            ).is_file()
            for prompt_id in prompt_ids
        ):
            completed.append(step)
    return tuple(completed)


def plot(main_run_root: Path, prompt_count: int) -> dict[str, Any]:
    root = fast_root(main_run_root)
    steps = completed_steps(root, prompt_count)
    if not steps:
        raise RuntimeError("no complete policy is ready for a permutation-0 graph")
    prompt_ids = ordered_prompt_subset(root, prompt_count)
    output_root = root / f"interim-bon-{prompt_count}prompt"
    write_json_atomic(
        output_root / "manifest.json",
        {
            "schema_version": 1,
            "diagnostic_only": True,
            "prompt_ids": list(prompt_ids),
            "focal_steps": list(steps),
            "n_grid": list(BON_SIZES),
            "permutations": 1,
            "selected_permutation": 0,
        },
        immutable=False,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/plot_paper_proxy_gt.py",
            "--run-root",
            str(root),
            "--interim-root",
            str(output_root),
            "--steps",
            *(str(step) for step in steps),
        ],
        check=True,
    )
    return {
        "fast_run_root": str(root),
        "completed_steps": list(steps),
        "graph": str(output_root / "combined_proxy_gt_curves.svg"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        required=True,
        choices=("prepare", "sync-selections", "submit", "sync", "plot", "status"),
    )
    parser.add_argument("--main-run-root", type=Path, required=True)
    parser.add_argument("--prompt-count", type=int, default=5)
    parser.add_argument(
        "--private-gt",
        type=Path,
        default=Path("data/private_gt/healthbench_gold_rubrics.jsonl"),
    )
    parser.add_argument(
        "--gold-schema",
        type=Path,
        default=Path("configs/schemas/paper_gold_grader_v1.json"),
    )
    args = parser.parse_args()
    if args.phase == "prepare":
        result = prepare(args.main_run_root, args.prompt_count)
    elif args.phase == "sync-selections":
        result = sync_selections(args.main_run_root, args.prompt_count)
    elif args.phase == "submit":
        sync_selections(args.main_run_root, args.prompt_count)
        result = prepare_or_submit_available_gold(
            fast_root(args.main_run_root),
            args.private_gt,
            args.gold_schema,
            prompt_count=args.prompt_count,
            submit=True,
        )
    elif args.phase == "sync":
        result = sync_streaming_gold(fast_root(args.main_run_root), args.gold_schema)
    elif args.phase == "plot":
        result = plot(args.main_run_root, args.prompt_count)
    else:
        root = fast_root(args.main_run_root)
        result = {
            "fast_run_root": str(root),
            "selection_shards": len(list((root / "select-bon-minimum" / "shards").glob("*.jsonl"))),
            "submitted_groups": len(
                list((root / "audit-gold-streaming-private" / "groups").glob("*/submission.json"))
            ),
            "completed_groups": len(
                list((root / "audit-gold-streaming-private" / "groups").glob("*/result.json"))
            ),
            "completed_steps": list(completed_steps(root, args.prompt_count)),
        }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
