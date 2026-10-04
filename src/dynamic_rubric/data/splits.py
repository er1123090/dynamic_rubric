from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..artifacts import write_bytes_atomic


@dataclass(frozen=True)
class SplitSpec:
    name: str
    count: int


PILOT_SPLITS = (
    SplitSpec("pilot_train", 256),
    SplitSpec("pilot_probe", 48),
    SplitSpec("pilot_audit", 96),
)
MAIN_SPLITS = (
    SplitSpec("main_train", 512),
    SplitSpec("main_probe", 64),
    SplitSpec("main_audit", 128),
)


def _order_key(prompt_id: str, seed: int) -> str:
    return hashlib.sha256(f"split-v1:{seed}:{prompt_id}".encode()).hexdigest()


def assign_splits(
    rows: Iterable[Mapping[str, Any]],
    specs: Iterable[SplitSpec],
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    materialized = [dict(row) for row in rows]
    ids = [str(row["prompt_id"]) for row in materialized]
    if len(ids) != len(set(ids)):
        raise ValueError("prompt_id values must be unique")
    specs = tuple(specs)
    requested = sum(spec.count for spec in specs)
    if len(materialized) < requested:
        raise ValueError(f"need {requested} prompts, received {len(materialized)}")
    ordered = sorted(materialized, key=lambda row: _order_key(str(row["prompt_id"]), seed))
    result: dict[str, list[dict[str, Any]]] = {}
    offset = 0
    for spec in specs:
        result[spec.name] = ordered[offset : offset + spec.count]
        offset += spec.count
    return result


def assign_pilot_and_main(
    rows: Iterable[Mapping[str, Any]], pilot_seed: int, main_seed: int
) -> dict[str, list[dict[str, Any]]]:
    """Allocate all pilot rows first, then allocate main from the remainder."""

    materialized = [dict(row) for row in rows]
    pilot = assign_splits(materialized, PILOT_SPLITS, pilot_seed)
    used = {str(row["prompt_id"]) for values in pilot.values() for row in values}
    remaining = [row for row in materialized if str(row["prompt_id"]) not in used]
    main = assign_splits(remaining, MAIN_SPLITS, main_seed)
    return {**pilot, **main}


def validate_disjoint(splits: Mapping[str, Iterable[Mapping[str, Any]]]) -> None:
    ownership: dict[str, str] = {}
    for name, rows in splits.items():
        for row in rows:
            prompt_id = str(row["prompt_id"])
            if prompt_id in ownership:
                raise ValueError(f"prompt {prompt_id} appears in {ownership[prompt_id]} and {name}")
            ownership[prompt_id] = name


def _canonical_line(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        + b"\n"
    )


def _atomic_write(path: Path, payload: bytes) -> None:
    write_bytes_atomic(path, payload)


def write_splits(
    splits: Mapping[str, Iterable[Mapping[str, Any]]], output_dir: Path, seed: int
) -> dict[str, Any]:
    materialized = {name: [dict(row) for row in rows] for name, rows in splits.items()}
    validate_disjoint(materialized)
    inventory: dict[str, Any] = {}
    for name, rows in materialized.items():
        path = output_dir / f"{name}.jsonl"
        payload = b"".join(_canonical_line(row) for row in rows)
        _atomic_write(path, payload)
        inventory[name] = {
            "count": len(rows),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "prompt_ids": [str(row["prompt_id"]) for row in rows],
        }
    manifest = {
        "schema_version": 1,
        "algorithm": "sha256(split-v1:seed:prompt_id)",
        "seed": seed,
        "specs": [{"name": name, "count": len(rows)} for name, rows in materialized.items()],
        "splits": inventory,
    }
    _atomic_write(
        output_dir / "split_manifest.json",
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2).encode() + b"\n",
    )
    return manifest
