#!/usr/bin/env python3
"""Prefetch non-primary Qwen route chunks onto trainer through the shared cache."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Sequence

from dynamic_rubric.judge_prompts import PAPER_JUDGE_PROMPT_VERSION
from dynamic_rubric.minimum_interim import ordered_prompt_subset, target_groups
from dynamic_rubric.minimum_staleness import (
    TargetScoreClient,
    _audit_conversations,
    _balanced_routes,
    _bon_groups,
    _focal_rubrics,
    _score_prompt,
    _shard_name,
)


Task = tuple[tuple[str, str], str]


def _current_tasks(run_root: Path, prompt_count: int) -> tuple[str, str, list[Task]]:
    prompt_ids = ordered_prompt_subset(run_root, prompt_count)
    targets = target_groups(prompt_ids)
    score_root = run_root / "score-proxy-minimum"
    current: tuple[str, str, list[dict[str, Any]]] | None = None
    for (policy_id, prompt_id), candidates in _bon_groups(
        run_root / "generate-bon" / "bon_pool.jsonl"
    ):
        if (policy_id, prompt_id) not in targets:
            continue
        shard = _shard_name(policy_id, prompt_id)
        if (score_root / "shards" / f"{policy_id}-{shard}.jsonl").is_file() and (
            score_root / "criterion-shards" / f"{policy_id}-{shard}.jsonl.gz"
        ).is_file():
            continue
        current = policy_id, prompt_id, candidates
        break
    if current is None:
        raise RuntimeError("no incomplete Qwen score group")

    policy_id, prompt_id, candidates = current
    step = int(policy_id[3:])
    static, dynamic = _focal_rubrics(run_root)
    criteria: dict[str, dict[str, Any]] = {}
    for criterion in (*static[prompt_id]["criteria"], *dynamic[(prompt_id, step)]["criteria"]):
        criterion_id = str(criterion["criterion_id"])
        previous = criteria.setdefault(criterion_id, dict(criterion))
        if previous["text"] != criterion["text"]:
            raise RuntimeError(f"criterion collision: {criterion_id}")
    conversation = _audit_conversations(run_root)[prompt_id]
    tasks = [
        (
            (criterion_id, str(candidate["response_id"])),
            _score_prompt(
                str(criterion["text"]),
                str(candidate["response_text"]),
                conversation=conversation,
                prompt_version=PAPER_JUDGE_PROMPT_VERSION,
            ),
        )
        for candidate in candidates
        for criterion_id, criterion in criteria.items()
    ]
    return policy_id, prompt_id, tasks


def _prefetch_chunks(
    client: TargetScoreClient,
    chunks: Sequence[Sequence[Task]],
    *,
    workers: int,
) -> int:
    completed = 0

    def prefetch(chunk: Sequence[Task]) -> int:
        client._post([prompt for _, prompt in chunk], upstream_index=0)
        return len(chunk)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(prefetch, chunk) for chunk in chunks]
        for future in as_completed(futures):
            completed += future.result()
            print(json.dumps({"prefetched_tasks": completed}), flush=True)
    return completed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--score-endpoint", default="http://127.0.0.1:8104")
    parser.add_argument("--prompt-count", type=int, default=5)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.prompt_count < 1 or args.workers < 1:
        parser.error("prompt-count and workers must be positive")

    policy_id, prompt_id, tasks = _current_tasks(args.run_root, args.prompt_count)
    client = TargetScoreClient(args.score_endpoint)
    chunks = [tasks[start : start + client.batch_size] for start in range(0, len(tasks), 32)]
    routes = _balanced_routes(chunks, client.routing_weights())
    by_route: dict[int, list[Sequence[Task]]] = defaultdict(list)
    for route, chunk in zip(routes, chunks):
        by_route[route].append(chunk)
    # The first chunk on each inference_a route is already in flight. Prefetch only
    # later chunks to avoid duplicating the requests currently running there.
    prefetch = [chunk for route in sorted(by_route) if route != 0 for chunk in by_route[route][1:]]
    completed = _prefetch_chunks(client, prefetch, workers=args.workers)
    print(
        json.dumps(
            {
                "policy_id": policy_id,
                "prompt_id": prompt_id,
                "source_tasks": len(tasks),
                "prefetch_chunks": len(prefetch),
                "prefetched_tasks": completed,
                "target_upstream_index": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
