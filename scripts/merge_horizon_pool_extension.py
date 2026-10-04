from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dynamic_rubric.artifacts import read_jsonl, write_jsonl_atomic
from dynamic_rubric.horizon.pools import validate_pool_rows


def _by_prompt(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["prompt_id"]), []).append(row)
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge prefix and delta horizon pool shards.")
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--prefix-prompts", type=int, required=True)
    parser.add_argument("--existing-pool", type=Path, required=True)
    parser.add_argument("--delta-pool", type=Path, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    prompt_ids = [str(row["prompt_id"]) for row in read_jsonl(args.prompts)]
    if len(prompt_ids) != len(set(prompt_ids)):
        raise ValueError("prompt IDs must be unique")
    existing = read_jsonl(args.existing_pool)
    delta = read_jsonl(args.delta_pool)
    existing_by_prompt = _by_prompt(existing)
    delta_by_prompt = _by_prompt(delta)
    if set(existing_by_prompt) != set(prompt_ids[: args.prefix_prompts]):
        raise ValueError("existing pool does not match the preserved prompt prefix")
    if set(delta_by_prompt) != set(prompt_ids[args.prefix_prompts :]):
        raise ValueError("delta pool does not match the added prompts")

    rows = existing + delta
    if {str(row["pool_family"]) for row in rows} == {"sham_control"}:
        raise ValueError("sham pools are forbidden in the eval300 no-sham run")
    validate_pool_rows(rows, expected_count=args.count)
    response_ids = [str(row["response_id"]) for row in rows]
    if len(response_ids) != len(set(response_ids)):
        raise ValueError("response IDs overlap between prefix and delta shards")
    for field in (
        "suite_id",
        "domain",
        "training_seed",
        "policy_step",
        "checkpoint_hash",
        "pool_family",
        "model_revision",
        "tokenizer_revision",
        "generation_config_hash",
    ):
        if len({json.dumps(row.get(field), sort_keys=True) for row in rows}) != 1:
            raise ValueError(f"pool metadata drifted for {field}")

    merged_by_prompt = _by_prompt(rows)
    ordered = [
        row
        for prompt_id in prompt_ids
        for row in sorted(
            merged_by_prompt[prompt_id],
            key=lambda item: int(item["sample_index"]),
        )
    ]
    write_jsonl_atomic(args.output, ordered)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "prompt_count": len(prompt_ids),
                "row_count": len(ordered),
                "pool_family": ordered[0]["pool_family"],
                "policy_step": ordered[0]["policy_step"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
