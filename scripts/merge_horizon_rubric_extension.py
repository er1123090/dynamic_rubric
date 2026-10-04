from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dynamic_rubric.artifacts import read_jsonl, write_jsonl_atomic


def _index(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for raw_row in rows:
        row = dict(raw_row)
        prompt_id = str(row["prompt_id"])
        if prompt_id in indexed:
            raise ValueError(f"{label} contains duplicate prompt ID {prompt_id}")
        row["control_extension"] = None
        row["control_match"] = None
        indexed[prompt_id] = row
    return indexed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge no-sham prefix and delta horizon rubric artifacts."
    )
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--prefix-prompts", type=int, required=True)
    parser.add_argument("--existing-rubric", type=Path, required=True)
    parser.add_argument("--delta-rubric", type=Path, required=True)
    parser.add_argument("--checkpoint-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    prompt_ids = [str(row["prompt_id"]) for row in read_jsonl(args.prompts)]
    existing = _index(read_jsonl(args.existing_rubric), "existing rubric")
    delta = _index(read_jsonl(args.delta_rubric), "delta rubric")
    if set(existing) != set(prompt_ids[: args.prefix_prompts]):
        raise ValueError("existing rubric does not match the preserved prompt prefix")
    if set(delta) != set(prompt_ids[args.prefix_prompts :]):
        raise ValueError("delta rubric does not match the added prompts")

    rows = [existing.get(prompt_id, delta.get(prompt_id)) for prompt_id in prompt_ids]
    if any(row is None for row in rows):
        raise ValueError("rubric merge omitted one or more prompts")
    merged = [dict(row) for row in rows if row is not None]
    if {str(row["checkpoint_id"]) for row in merged} != {args.checkpoint_id}:
        raise ValueError("rubric checkpoint ID drifted")
    if any(row["control_extension"] is not None for row in merged):
        raise ValueError("control extensions must be absent from a no-sham rubric")
    if any(row["control_match"] is not None for row in merged):
        raise ValueError("control matches must be absent from a no-sham rubric")

    write_jsonl_atomic(args.output, merged)
    print(
        json.dumps(
            {
                "checkpoint_id": args.checkpoint_id,
                "output": str(args.output),
                "prompt_count": len(merged),
                "sham_included": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
