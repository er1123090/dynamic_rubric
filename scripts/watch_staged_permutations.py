#!/usr/bin/env python3
"""Run permutation 0 first, then the existing five-permutation GPT audit."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from dynamic_rubric.minimum_gold_streaming import sync_streaming_gold
from dynamic_rubric.paper_judge_experiment import prepare_or_submit_gold_subset

from run_perm1_fastpath import (
    completed_steps,
    fast_root,
    plot,
    sync_selections,
)


DEFAULT_SOURCE = Path("artifacts/runs/pilot-static-r0-100step-20260821")
DEFAULT_RUNS = {
    "pi_ref": Path(
        "artifacts/runs/pilot-static-r0-100step-20260821-onlinerubric-piref-paper-judge-v1"
    ),
    "pi_old": Path(
        "artifacts/runs/pilot-static-r0-100step-20260821-onlinerubric-piold-paper-judge-v1"
    ),
}


def _emit(event: str, **details: Any) -> None:
    print(
        json.dumps(
            {"event": event, "time_unix": time.time(), **details},
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )


def _count(root: Path, pattern: str) -> int:
    return len(list(root.glob(pattern)))


def _install_api_key(key_file: Path) -> None:
    value = key_file.read_text(encoding="utf-8").strip()
    if not value:
        raise RuntimeError(f"empty OpenAI API key file: {key_file}")
    os.environ["OPENAI_API_KEY"] = value


def _sync_fast(
    main_root: Path,
    private_gt: Path,
    schema: Path,
    *,
    prompt_count: int,
) -> dict[str, Any]:
    selection = sync_selections(main_root, prompt_count)
    root = fast_root(main_root)
    selection_count = int(selection["selection_shards"])
    submitted = _count(root, "audit-gold-streaming-private/groups/*/submission.json")
    if selection_count > submitted:
        from dynamic_rubric.online_rubric_bon import prepare_or_submit_available_gold

        prepare_or_submit_available_gold(
            root,
            private_gt,
            schema,
            prompt_count=prompt_count,
            submit=True,
        )
        submitted = _count(root, "audit-gold-streaming-private/groups/*/submission.json")
        _emit(
            "perm1-submitted",
            run_root=str(main_root),
            selections=selection_count,
            submitted=submitted,
        )
    completed = _count(root, "audit-gold-streaming-private/groups/*/result.json")
    if submitted > completed:
        result = sync_streaming_gold(root, schema)
        completed = int(result["completed_groups"])
    steps = completed_steps(root, prompt_count)
    graph = root / f"interim-bon-{prompt_count}prompt" / "combined_proxy_gt_curves.svg"
    marker = root / f"interim-bon-{prompt_count}prompt" / "plotted-steps.json"
    previous: tuple[int, ...] = ()
    if marker.is_file():
        previous = tuple(json.loads(marker.read_text(encoding="utf-8"))["steps"])
    if steps and steps != previous:
        plotted = plot(main_root, prompt_count)
        marker.write_text(
            json.dumps({"steps": list(steps)}, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _emit("perm1-graph", run_root=str(main_root), **plotted)
    return {
        "selection": selection_count,
        "submitted": submitted,
        "completed": completed,
        "steps": list(steps),
        "graph": str(graph) if graph.is_file() else None,
    }


def _sync_full(
    control: str,
    main_root: Path,
    source_root: Path,
    private_gt: Path,
    schema: Path,
    *,
    prompt_count: int,
) -> dict[str, Any]:
    expected = 4 * prompt_count
    submitted = _count(main_root, "audit-gold-streaming-private/groups/*/submission.json")
    if submitted < expected:
        prepare_or_submit_gold_subset(
            main_root,
            source_root,
            private_gt,
            schema,
            prompt_count=prompt_count,
            submit=True,
        )
        submitted = _count(main_root, "audit-gold-streaming-private/groups/*/submission.json")
        _emit("perm5-submitted", control=control, submitted=submitted)
    completed = _count(main_root, "audit-gold-streaming-private/groups/*/result.json")
    if submitted > completed:
        result = sync_streaming_gold(main_root, schema)
        completed = int(result["completed_groups"])
    graph = main_root / f"interim-bon-{prompt_count}prompt" / "combined_proxy_gt_curves.svg"
    if completed == expected and not graph.is_file():
        subprocess.run(
            [
                sys.executable,
                "scripts/run_static_online_bon.py",
                "--phase",
                "analyze",
                "--online-control",
                control,
                "--run-root",
                str(main_root),
                "--source-run-root",
                str(source_root),
                "--prompt-count",
                str(prompt_count),
            ],
            check=True,
        )
        _emit("perm5-graph", control=control, graph=str(graph))
    return {
        "submitted": submitted,
        "completed": completed,
        "graph": str(graph) if graph.is_file() else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-count", type=int, default=5)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--source-run-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--api-key-file", type=Path, default=Path("/tmp/openai_auto_rubric_key"))
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
    if args.prompt_count < 1 or args.poll_seconds < 1:
        parser.error("prompt-count and poll-seconds must be positive")
    _install_api_key(args.api_key_file)
    expected = 4 * args.prompt_count
    while True:
        all_final = True
        for control, main_root in DEFAULT_RUNS.items():
            try:
                fast = _sync_fast(
                    main_root,
                    args.private_gt,
                    args.gold_schema,
                    prompt_count=args.prompt_count,
                )
                full: dict[str, Any] | None = None
                if int(fast["completed"]) == expected:
                    full = _sync_full(
                        control,
                        main_root,
                        args.source_run_root,
                        args.private_gt,
                        args.gold_schema,
                        prompt_count=args.prompt_count,
                    )
                if full is None or full["graph"] is None:
                    all_final = False
                _emit("status", control=control, fast=fast, full=full)
            except Exception as error:  # keep polling through transient provider failures
                all_final = False
                _emit(
                    "poll-error",
                    control=control,
                    error_type=type(error).__name__,
                    error=str(error),
                )
        if all_final:
            _emit("complete")
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
