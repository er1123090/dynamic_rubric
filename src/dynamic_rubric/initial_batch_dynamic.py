"""Explicitly named wrapper for the experiment's initial dynamic-rubric Batch path."""

from __future__ import annotations

from typing import Any

from .artifacts import artifact_record, write_json_atomic

from .batch_dynamic import (
    collect_dynamic_batch,
    dynamic_batch_status,
    prepare_dynamic_batch,
    submit_dynamic_batch,
)
from .pipeline import PipelineContext, StageError
from .rubrics.replay import ReplayMode


def initial_rubric_batch_stage(mode: str) -> str:
    parsed = ReplayMode(mode)
    if parsed is ReplayMode.DYNAMIC_FIXED_BUDGETED:
        return "initial_dynamic_fixed_batch"
    if parsed is ReplayMode.DYNAMIC_PREV_BUDGETED:
        return "initial_dynamic_prev_batch"
    raise ValueError(f"unsupported initial rubric mode: {mode}")


def _validate_stage(context: PipelineContext, mode: str) -> None:
    expected = initial_rubric_batch_stage(mode)
    if context.stage != expected:
        raise StageError(f"initial rubric Batch must use stage {expected!r}")


def prepare_initial_rubric_batch(
    context: PipelineContext,
    *,
    max_step: int = 50,
    mode: str = ReplayMode.DYNAMIC_PREV_BUDGETED.value,
) -> dict[str, Any]:
    _validate_stage(context, mode)
    manifest = prepare_dynamic_batch(context, max_step=max_step, mode=mode)
    marker_path = context.stage_root() / "generation_method.json"
    write_json_atomic(
        marker_path,
        {
            "schema_version": 1,
            "generation_method": "initial",
            "stage": context.stage,
            "mode": mode,
            "note": "Original post-hoc dynamic-rubric prompt; not OnlineRubrics.",
        },
    )
    return {
        **manifest,
        "generation_method": "initial",
        "generation_method_marker": artifact_record(marker_path),
    }


def submit_initial_rubric_batch(context: PipelineContext) -> dict[str, Any]:
    return submit_dynamic_batch(context)


def initial_rubric_batch_status(context: PipelineContext) -> dict[str, Any]:
    return dynamic_batch_status(context)


def collect_initial_rubric_batch(context: PipelineContext) -> dict[str, Any]:
    return collect_dynamic_batch(context)
