#!/usr/bin/env python3
"""Stream cached/missing GPT gold and plot Human-R0 OnlineRubric runs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from dynamic_rubric.human_static_gold_reuse import (
    prepare_or_submit_reused_gold,
    sync_reused_gold,
)


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


def _poll_run(
    control: str,
    run_root: Path,
    source_root: Path,
    private_gt: Path,
    schema: Path,
    reuse_roots: list[Path],
    *,
    prompt_count: int,
    allow_gold_prepare: bool,
) -> dict[str, Any]:
    expected = 4 * prompt_count
    selections = _count(run_root, "select-bon-minimum/shards/*.jsonl")
    if selections and allow_gold_prepare:
        prepared = _count(run_root, "audit-gold-streaming-private/groups/*/manifest.json")
        if selections > prepared:
            result = prepare_or_submit_reused_gold(
                run_root,
                private_gt,
                schema,
                prompt_count=prompt_count,
                reuse_roots=reuse_roots,
                submit=True,
            )
            _emit(
                "gold-ready",
                control=control,
                ready_groups=result["ready_groups"],
                selected_responses=result["selected_responses"],
                cached_responses=result["cached_responses"],
                new_gpt_requests=result["requests"],
            )

    submitted = _count(run_root, "audit-gold-streaming-private/groups/*/submission.json")
    completed = _count(run_root, "audit-gold-streaming-private/groups/*/result.json")
    prepared = _count(run_root, "audit-gold-streaming-private/groups/*/manifest.json")
    if submitted and completed < prepared:
        synced = sync_reused_gold(run_root, schema)
        completed = int(synced["completed_groups"])

    graph = run_root / f"interim-bon-{prompt_count}prompt" / "combined_proxy_gt_curves.svg"
    if completed == expected and not graph.is_file():
        subprocess.run(
            [
                sys.executable,
                "scripts/run_human_static_online_bon.py",
                "--phase",
                "analyze",
                "--online-control",
                control,
                "--run-root",
                str(run_root),
                "--source-run-root",
                str(source_root),
                "--private-gt",
                str(private_gt),
                "--gold-schema",
                str(schema),
                "--prompt-count",
                str(prompt_count),
            ],
            check=True,
        )
        _emit("graph", control=control, graph=str(graph))

    return {
        "score_groups": _count(run_root, "score-proxy-minimum/shards/*.jsonl"),
        "selection_groups": selections,
        "prepared_gold_groups": prepared,
        "submitted_gold_groups": submitted,
        "completed_gold_groups": completed,
        "expected_groups": expected,
        "graph": str(graph) if graph.is_file() else None,
        "gold_prepare_enabled": allow_gold_prepare,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run-root", type=Path, required=True)
    parser.add_argument("--pi-ref-run-root", type=Path, required=True)
    parser.add_argument("--pi-old-run-root", type=Path, required=True)
    parser.add_argument("--prompt-count", type=int, default=5)
    parser.add_argument("--poll-seconds", type=int, default=30)
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
    parser.add_argument("--reuse-gold-run-root", type=Path, action="append", default=[])
    args = parser.parse_args()
    os.environ["OPENAI_API_KEY"] = args.api_key_file.read_text().strip()
    expected = 4 * args.prompt_count
    while True:
        ref_status: dict[str, Any]
        old_status: dict[str, Any]
        try:
            ref_status = _poll_run(
                "pi_ref",
                args.pi_ref_run_root,
                args.source_run_root,
                args.private_gt,
                args.gold_schema,
                list(args.reuse_gold_run_root),
                prompt_count=args.prompt_count,
                allow_gold_prepare=True,
            )
            _emit("status", control="pi_ref", **ref_status)
        except Exception as error:
            ref_status = {"completed_gold_groups": 0, "graph": None}
            _emit("poll-error", control="pi_ref", error_type=type(error).__name__, error=str(error))
        ref_gold_complete = int(ref_status.get("completed_gold_groups", 0)) == expected
        try:
            old_status = _poll_run(
                "pi_old",
                args.pi_old_run_root,
                args.source_run_root,
                args.private_gt,
                args.gold_schema,
                [*args.reuse_gold_run_root, args.pi_ref_run_root],
                prompt_count=args.prompt_count,
                allow_gold_prepare=ref_gold_complete,
            )
            _emit("status", control="pi_old", **old_status)
        except Exception as error:
            old_status = {"graph": None}
            _emit("poll-error", control="pi_old", error_type=type(error).__name__, error=str(error))
        if ref_status.get("graph") and old_status.get("graph"):
            _emit("complete")
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
