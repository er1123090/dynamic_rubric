#!/usr/bin/env python3
"""Resume-safe entry point for the Human-R0 gold watcher."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import watch_human_static_online as watcher
from dynamic_rubric.human_static_gold_reuse import prepare_or_submit_reused_gold


_original_poll_run = watcher._poll_run


def _resume_safe_poll_run(
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
    selections = list(run_root.glob("select-bon-minimum/shards/*.jsonl"))
    if selections and allow_gold_prepare:
        handled = {
            path.parent.name
            for pattern in (
                "audit-gold-streaming-private/groups/*/submission.json",
                "audit-gold-streaming-private/groups/*/result.json",
            )
            for path in run_root.glob(pattern)
        }
        if len(selections) > len(handled):
            result = prepare_or_submit_reused_gold(
                run_root,
                private_gt,
                schema,
                prompt_count=prompt_count,
                reuse_roots=reuse_roots,
                submit=True,
            )
            watcher._emit(
                "gold-resumed",
                control=control,
                ready_groups=result["ready_groups"],
                cached_responses=result["cached_responses"],
                new_gpt_requests=result["requests"],
            )
    return _original_poll_run(
        control,
        run_root,
        source_root,
        private_gt,
        schema,
        reuse_roots,
        prompt_count=prompt_count,
        allow_gold_prepare=allow_gold_prepare,
    )


watcher._poll_run = _resume_safe_poll_run


if __name__ == "__main__":
    watcher.main()
