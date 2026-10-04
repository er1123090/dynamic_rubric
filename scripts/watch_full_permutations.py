#!/usr/bin/env python3
"""Stream all five-permutation selections to GPT-5 as Qwen shards finish."""

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
from dynamic_rubric.online_rubric_bon import prepare_or_submit_available_gold


DEFAULT_RUNS = {
    "pi_ref": Path(
        "artifacts/runs/pilot-static-r0-100step-20260821-onlinerubric-piref-paper-judge-v1"
    ),
    "pi_old": Path(
        "artifacts/runs/pilot-static-r0-100step-20260821-onlinerubric-piold-paper-judge-v1"
    ),
}
DEFAULT_SOURCE = Path("artifacts/runs/pilot-static-r0-100step-20260821")


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


def _install_api_key(path: Path) -> None:
    value = path.read_text(encoding="utf-8").strip()
    if not value:
        raise RuntimeError(f"empty OpenAI API key file: {path}")
    os.environ["OPENAI_API_KEY"] = value


def _poll_run(
    control: str,
    run_root: Path,
    source_root: Path,
    private_gt: Path,
    schema: Path,
    *,
    prompt_count: int,
) -> dict[str, Any]:
    expected = 4 * prompt_count
    selections = _count(run_root, "select-bon-minimum/shards/*.jsonl")
    submitted = _count(run_root, "audit-gold-streaming-private/groups/*/submission.json")
    if selections > submitted:
        result = prepare_or_submit_available_gold(
            run_root,
            private_gt,
            schema,
            prompt_count=prompt_count,
            submit=True,
        )
        submitted = _count(run_root, "audit-gold-streaming-private/groups/*/submission.json")
        _emit(
            "perm5-submitted",
            control=control,
            ready_groups=result["ready_groups"],
            requests=result["requests"],
            submitted_groups=submitted,
        )

    completed = _count(run_root, "audit-gold-streaming-private/groups/*/result.json")
    if submitted > completed:
        result = sync_streaming_gold(run_root, schema)
        completed = int(result["completed_groups"])

    graph = run_root / f"interim-bon-{prompt_count}prompt" / "combined_proxy_gt_curves.svg"
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
                str(run_root),
                "--source-run-root",
                str(source_root),
                "--prompt-count",
                str(prompt_count),
            ],
            check=True,
        )
        _emit("perm5-graph", control=control, graph=str(graph))

    return {
        "selection_groups": selections,
        "submitted_groups": submitted,
        "completed_groups": completed,
        "expected_groups": expected,
        "graph": str(graph) if graph.is_file() else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-count", type=int, default=5)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--source-run-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--pi-ref-run-root", type=Path, default=DEFAULT_RUNS["pi_ref"]
    )
    parser.add_argument(
        "--pi-old-run-root", type=Path, default=DEFAULT_RUNS["pi_old"]
    )
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

    runs = {"pi_ref": args.pi_ref_run_root, "pi_old": args.pi_old_run_root}
    while True:
        all_complete = True
        for control, run_root in runs.items():
            try:
                status = _poll_run(
                    control,
                    run_root,
                    args.source_run_root,
                    args.private_gt,
                    args.gold_schema,
                    prompt_count=args.prompt_count,
                )
                if status["graph"] is None:
                    all_complete = False
                _emit("status", control=control, **status)
            except Exception as error:  # keep polling through transient provider failures
                all_complete = False
                _emit(
                    "poll-error",
                    control=control,
                    error_type=type(error).__name__,
                    error=str(error),
                )
        if all_complete:
            _emit("complete")
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
