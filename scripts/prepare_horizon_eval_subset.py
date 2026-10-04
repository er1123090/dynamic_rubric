#!/usr/bin/env python3
"""Create a deterministic evaluation subset without mutating the full pool."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


STATIC_POOL_FILES = (
    "fixed.jsonl",
    "sham.jsonl",
    "seed-11-step-0-pool-b.jsonl",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def selected_prompt_ids(manifest_path: Path, count: int) -> list[str]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    final = manifest.get("splits", {}).get("final", {})
    ids = [str(prompt_id) for prompt_id in final.get("prompt_ids", [])]
    if len(ids) < count:
        raise ValueError(
            f"split manifest has only {len(ids)} final prompt IDs; requested {count}"
        )
    selected = ids[:count]
    if len(set(selected)) != count:
        raise ValueError("selected final prompt IDs are not unique")
    return selected


def filter_pool(
    source: Path,
    output: Path,
    selected: set[str],
    expected_per_prompt: int,
) -> dict[str, Any]:
    source_rows = read_jsonl(source)
    rows = [row for row in source_rows if str(row.get("prompt_id")) in selected]
    counts = Counter(str(row.get("prompt_id")) for row in rows)
    missing = sorted(selected - counts.keys())
    wrong = sorted(
        (prompt_id, count)
        for prompt_id, count in counts.items()
        if count != expected_per_prompt
    )
    if missing or wrong or len(rows) != len(selected) * expected_per_prompt:
        raise ValueError(
            f"invalid filtered pool {source}: missing={missing[:5]}, "
            f"wrong_counts={wrong[:5]}, rows={len(rows)}"
        )
    write_jsonl_atomic(output, rows)
    return {
        "source": str(source),
        "source_sha256": sha256(source),
        "output": str(output),
        "output_sha256": sha256(output),
        "rows": len(rows),
        "responses_per_prompt": expected_per_prompt,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--source-pool-root", type=Path, required=True)
    parser.add_argument("--output-prompts", type=Path, required=True)
    parser.add_argument("--output-pool-root", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--training-seed", type=int, default=11)
    args = parser.parse_args()

    if args.count <= 0:
        raise ValueError("count must be positive")

    prompt_ids = selected_prompt_ids(args.split_manifest, args.count)
    selected = set(prompt_ids)
    prompt_by_id = {
        str(row["prompt_id"]): row for row in read_jsonl(args.prompts)
    }
    missing_prompts = [prompt_id for prompt_id in prompt_ids if prompt_id not in prompt_by_id]
    if missing_prompts:
        raise ValueError(f"full final file is missing prompt IDs: {missing_prompts[:5]}")

    prompt_rows = [prompt_by_id[prompt_id] for prompt_id in prompt_ids]
    write_jsonl_atomic(args.output_prompts, prompt_rows)

    pool_specs = {
        "fixed.jsonl": 8,
        "sham.jsonl": 8,
        f"seed-{args.training_seed}-step-0-pool-b.jsonl": 16,
    }
    pools: dict[str, Any] = {}
    for filename, expected_per_prompt in pool_specs.items():
        pools[filename] = filter_pool(
            args.source_pool_root / filename,
            args.output_pool_root / filename,
            selected,
            expected_per_prompt,
        )

    manifest = {
        "schema_version": 1,
        "selection": "split_manifest.final.prompt_ids prefix",
        "count": args.count,
        "training_seed": args.training_seed,
        "prompt_ids": prompt_ids,
        "prompts": {
            "source": str(args.prompts),
            "source_sha256": sha256(args.prompts),
            "output": str(args.output_prompts),
            "output_sha256": sha256(args.output_prompts),
            "rows": len(prompt_rows),
        },
        "split_manifest": {
            "path": str(args.split_manifest),
            "sha256": sha256(args.split_manifest),
        },
        "pools": pools,
    }
    write_json_atomic(args.output_pool_root / "eval_subset_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
