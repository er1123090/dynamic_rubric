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
        indexed[prompt_id] = row
    return indexed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select a prompt-aligned no-sham subset from a horizon rubric."
    )
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--rubric", type=Path, required=True)
    parser.add_argument("--checkpoint-id", required=True)
    parser.add_argument("--expected-source-count", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    prompt_rows = read_jsonl(args.prompts)
    prompt_ids = [str(row["prompt_id"]) for row in prompt_rows]
    if len(set(prompt_ids)) != len(prompt_ids):
        raise ValueError("prompts contain duplicate prompt IDs")

    rubric_rows = read_jsonl(args.rubric)
    if (
        args.expected_source_count is not None
        and len(rubric_rows) != args.expected_source_count
    ):
        raise ValueError(
            "source rubric row count drifted: "
            f"expected={args.expected_source_count}, actual={len(rubric_rows)}"
        )
    rubric_by_id = _index(rubric_rows, "source rubric")
    missing = [prompt_id for prompt_id in prompt_ids if prompt_id not in rubric_by_id]
    if missing:
        raise ValueError(f"source rubric is missing {len(missing)} requested prompts")

    selected = [dict(rubric_by_id[prompt_id]) for prompt_id in prompt_ids]
    if {str(row.get("checkpoint_id")) for row in selected} != {args.checkpoint_id}:
        raise ValueError("rubric checkpoint ID drifted")
    if any(row.get("control_extension") is not None for row in selected):
        raise ValueError("control extensions must be absent from a no-sham rubric")
    if any(row.get("control_match") is not None for row in selected):
        raise ValueError("control matches must be absent from a no-sham rubric")

    write_jsonl_atomic(args.output, selected)
    print(
        json.dumps(
            {
                "checkpoint_id": args.checkpoint_id,
                "output": str(args.output),
                "prompt_count": len(selected),
                "sham_included": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
