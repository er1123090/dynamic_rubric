"""Attach count/weight-matched stale controls to independently built rubrics."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..artifacts import read_jsonl, write_jsonl_atomic
from .controls import match_control_extension
from .live_rubrics import criterion_from_artifact


def _index_unique(
    rows: Sequence[Mapping[str, Any]], *, label: str
) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        prompt_id = str(row["prompt_id"])
        if prompt_id in indexed:
            raise ValueError(f"duplicate {label} prompt_id: {prompt_id}")
        indexed[prompt_id] = row
    return indexed


def attach_control_extensions(
    current_rows: Sequence[Mapping[str, Any]],
    control_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return current rubric rows with stale extensions matched per prompt.

    Rubric extraction remains checkpoint-independent. This local deterministic
    pass supplies the previous-checkpoint control only after both artifacts exist.
    """

    controls = _index_unique(control_rows, label="control rubric")
    current_prompt_ids = {str(row["prompt_id"]) for row in current_rows}
    if current_prompt_ids != set(controls):
        raise ValueError("current and control rubrics must have the same prompt grid")

    attached: list[dict[str, Any]] = []
    for current_row in current_rows:
        prompt_id = str(current_row["prompt_id"])
        current = tuple(
            criterion_from_artifact(item) for item in current_row.get("extension", ())
        )
        available = tuple(
            criterion_from_artifact(item)
            for item in controls[prompt_id].get("extension", ())
        )
        control_match = match_control_extension(current, available)
        row = dict(current_row)
        row["control_extension"] = (
            [asdict(item) for item in control_match.selected]
            if control_match.eligible
            else None
        )
        row["control_match"] = asdict(control_match)
        attached.append(row)
    return attached


def attach_control_extensions_from_files(
    *, current_path: Path, control_path: Path, output_path: Path
) -> dict[str, Any]:
    rows = attach_control_extensions(read_jsonl(current_path), read_jsonl(control_path))
    write_jsonl_atomic(output_path, rows)
    return {
        "rubrics": str(output_path),
        "prompt_count": len(rows),
        "eligible_prompt_count": sum(
            bool(row["control_match"]["eligible"]) for row in rows
        ),
    }
