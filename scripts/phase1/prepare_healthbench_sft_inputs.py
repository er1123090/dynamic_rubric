#!/usr/bin/env python3
"""Convert the canonical EvoRubrics HealthBench JSON splits to prompt-only JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dynamic_rubric.artifacts import write_json_atomic, write_jsonl_atomic
from dynamic_rubric.hashing import sha256_file, sha256_json


class PreparationError(RuntimeError):
    """Raised when a source split cannot satisfy the SFT input contract."""


def _load_split(path: Path, *, expected: int, split: str) -> list[dict[str, Any]]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PreparationError(f"cannot read {split} split {path}: {exc}") from exc
    if not isinstance(raw, list) or len(raw) != expected:
        count = len(raw) if isinstance(raw, list) else type(raw).__name__
        raise PreparationError(f"{split} must contain {expected} rows, got {count}")

    rows: list[dict[str, Any]] = []
    for index, source in enumerate(raw, start=1):
        if not isinstance(source, dict):
            raise PreparationError(f"{split} row {index} is not an object")
        prompt_id = source.get("prompt_id")
        prompt = source.get("prompt")
        if not isinstance(prompt_id, str) or not prompt_id:
            raise PreparationError(f"{split} row {index} has invalid prompt_id")
        if not isinstance(prompt, list) or not prompt:
            raise PreparationError(f"{split} row {index} has invalid prompt")
        messages: list[dict[str, str]] = []
        for message_index, message in enumerate(prompt):
            if not isinstance(message, dict):
                raise PreparationError(
                    f"{split} row {index} message {message_index} is not an object"
                )
            role, content = message.get("role"), message.get("content")
            if role not in {"system", "user", "assistant"}:
                raise PreparationError(
                    f"{split} row {index} message {message_index} has invalid role"
                )
            if not isinstance(content, str) or not content.strip():
                raise PreparationError(
                    f"{split} row {index} message {message_index} has invalid content"
                )
            messages.append({"role": role, "content": content})
        rows.append(
            {
                "prompt_id": prompt_id,
                "prompt_hash": sha256_json(messages),
                "messages": messages,
                "source_row_sha256": sha256_json(source),
            }
        )

    ids = [row["prompt_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise PreparationError(f"{split} contains duplicate prompt_id values")
    return rows


def prepare(
    *, train_path: Path, heldout_path: Path, output_dir: Path, train_count: int, heldout_count: int
) -> dict[str, Any]:
    train = _load_split(train_path, expected=train_count, split="train")
    heldout = _load_split(heldout_path, expected=heldout_count, split="heldout")
    overlap = {row["prompt_id"] for row in train} & {row["prompt_id"] for row in heldout}
    if overlap:
        raise PreparationError(f"train and heldout overlap by {len(overlap)} prompt IDs")

    output_dir.mkdir(parents=True, exist_ok=True)
    train_output = output_dir / "train.jsonl"
    heldout_output = output_dir / "heldout.jsonl"
    write_jsonl_atomic(train_output, train)
    write_jsonl_atomic(heldout_output, heldout)
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "rubrics_or_reference_answers_exported": False,
        "sources": {
            "train": {"path": str(train_path.resolve()), "sha256": sha256_file(train_path)},
            "heldout": {
                "path": str(heldout_path.resolve()),
                "sha256": sha256_file(heldout_path),
            },
        },
        "outputs": {
            "train": {
                "path": str(train_output.resolve()),
                "count": len(train),
                "sha256": sha256_file(train_output),
            },
            "heldout": {
                "path": str(heldout_output.resolve()),
                "count": len(heldout),
                "sha256": sha256_file(heldout_output),
            },
        },
        "train_prompt_ids_sha256": sha256_json([row["prompt_id"] for row in train]),
        "heldout_prompt_ids_sha256": sha256_json([row["prompt_id"] for row in heldout]),
    }
    write_json_atomic(output_dir / "manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--heldout", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-count", type=int, default=4000)
    parser.add_argument("--heldout-count", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.train_count < 1 or args.heldout_count < 1:
        raise PreparationError("split sizes must be positive")
    result = prepare(
        train_path=args.train,
        heldout_path=args.heldout,
        output_dir=args.output_dir,
        train_count=args.train_count,
        heldout_count=args.heldout_count,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
