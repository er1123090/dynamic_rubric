from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from dynamic_rubric.artifacts import (
    read_jsonl,
    write_json_atomic,
    write_jsonl_atomic,
)
from dynamic_rubric.data.splits import SplitSpec, assign_splits, validate_disjoint, write_splits
from dynamic_rubric.hashing import sha256_file


def _require_equal(
    label: str,
    actual: Sequence[dict[str, Any]],
    expected: Sequence[dict[str, Any]],
) -> None:
    if list(actual) != list(expected):
        raise ValueError(f"{label} changed while extending the evaluation split")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extend a deterministic horizon final split while preserving its prefix."
    )
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-count", type=int, default=1500)
    parser.add_argument("--development-count", type=int, default=150)
    parser.add_argument("--final-count", type=int, default=300)
    args = parser.parse_args()

    baseline_dir = args.baseline_dir.resolve()
    normalized_path = baseline_dir / "normalized.jsonl"
    manifest_path = baseline_dir / "rar_manifest.json"
    baseline_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    seed = int(baseline_manifest["seed"])
    normalized = read_jsonl(normalized_path)
    splits = assign_splits(
        normalized,
        (
            SplitSpec("train", args.train_count),
            SplitSpec("development", args.development_count),
            SplitSpec("final", args.final_count),
        ),
        seed,
    )
    validate_disjoint(splits)

    baseline_train = read_jsonl(baseline_dir / "train.jsonl")
    baseline_development = read_jsonl(baseline_dir / "development.jsonl")
    baseline_final = read_jsonl(baseline_dir / "final.jsonl")
    _require_equal("train split", splits["train"], baseline_train)
    _require_equal("development split", splits["development"], baseline_development)
    _require_equal(
        "final split prefix",
        splits["final"][: len(baseline_final)],
        baseline_final,
    )
    if args.final_count <= len(baseline_final):
        raise ValueError("final-count must be larger than the baseline final split")

    output_dir = args.output_dir.resolve()
    manifest = write_splits(splits, output_dir, seed)
    delta_rows = splits["final"][len(baseline_final) :]
    delta_path = output_dir / f"final_delta{len(delta_rows)}.jsonl"
    write_jsonl_atomic(delta_path, delta_rows)
    manifest.update(
        {
            "schema_version": 3,
            "domain": baseline_manifest["domain"],
            "extension": {
                "baseline_dir": str(baseline_dir),
                "baseline_manifest_sha256": sha256_file(manifest_path),
                "baseline_final_count": len(baseline_final),
                "final_count": args.final_count,
                "delta_count": len(delta_rows),
                "prefix_preserved": True,
                "delta_path": str(delta_path),
                "delta_sha256": sha256_file(delta_path),
            },
            "normalized_source_path": str(normalized_path),
            "normalized_source_sha256": sha256_file(normalized_path),
        }
    )
    write_json_atomic(output_dir / "rar_manifest.json", manifest)
    print(json.dumps(manifest["extension"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
