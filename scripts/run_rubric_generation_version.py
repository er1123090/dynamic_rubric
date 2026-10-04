#!/usr/bin/env python3
"""Run explicitly versioned initial or paper-style OnlineRubrics Batch stages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

from dynamic_rubric.initial_batch_dynamic import (
    collect_initial_rubric_batch,
    initial_rubric_batch_stage,
    initial_rubric_batch_status,
    prepare_initial_rubric_batch,
    submit_initial_rubric_batch,
)
from dynamic_rubric.onlinerubric_batch import (
    SUPPORTED_ONLINERUBRIC_MODES,
    collect_onlinerubric_dedup_batch,
    collect_onlinerubric_extraction_batch,
    onlinerubric_batch_status,
    onlinerubric_dedup_stage,
    onlinerubric_extraction_stage,
    prepare_onlinerubric_dedup_batch,
    prepare_onlinerubric_extraction_batch,
    submit_onlinerubric_batch,
)
from dynamic_rubric.onlinerubric_bootstrap import (
    ONLINERUBRIC_R0_DEDUP_STAGE,
    ONLINERUBRIC_R0_EXTRACTION_STAGE,
    collect_onlinerubric_r0_dedup_batch,
    prepare_onlinerubric_r0_dedup_batch,
    prepare_onlinerubric_r0_extraction_batch,
)
from dynamic_rubric.pipeline import PipelineContext

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Versioned rubric generation: initial_* preserves the original implementation; "
            "onlinerubric_* uses the paper's pair-level extraction and LLM dedup prompts."
        )
    )
    parser.add_argument(
        "command",
        choices=(
            "prepare-initial",
            "submit-initial",
            "status-initial",
            "collect-initial",
            "prepare-onlinerubric-extraction",
            "submit-onlinerubric-extraction",
            "status-onlinerubric-extraction",
            "collect-onlinerubric-extraction",
            "prepare-onlinerubric-dedup",
            "submit-onlinerubric-dedup",
            "status-onlinerubric-dedup",
            "collect-onlinerubric-dedup",
            "prepare-onlinerubric-r0-extraction",
            "submit-onlinerubric-r0-extraction",
            "status-onlinerubric-r0-extraction",
            "collect-onlinerubric-r0-extraction",
            "prepare-onlinerubric-r0-dedup",
            "submit-onlinerubric-r0-dedup",
            "status-onlinerubric-r0-dedup",
            "collect-onlinerubric-r0-dedup",
        ),
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/pilot.yaml"))
    parser.add_argument(
        "--mode",
        choices=SUPPORTED_ONLINERUBRIC_MODES,
        default="dynamic_prev_budgeted",
    )
    parser.add_argument("--max-step", type=int, default=50)
    parser.add_argument(
        "--policy-steps",
        help="Comma-separated OnlineRubrics policy steps, for example 3,10,30,50.",
    )
    parser.add_argument(
        "--prompt-manifest",
        type=Path,
        help="JSON manifest containing the exact prompt_ids to generate.",
    )
    return parser


def _policy_steps(value: str | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    try:
        steps = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise SystemExit("--policy-steps must contain comma-separated integers") from error
    if not steps:
        raise SystemExit("--policy-steps cannot be empty")
    return steps


def _selected_prompt_ids(path: Path | None) -> tuple[str, ...] | None:
    if path is None:
        return None
    resolved = path if path.is_absolute() else PROJECT_ROOT / path
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    prompt_ids = payload.get("prompt_ids") if isinstance(payload, dict) else None
    if not isinstance(prompt_ids, list) or not all(isinstance(item, str) for item in prompt_ids):
        raise SystemExit("--prompt-manifest must be a JSON object with a string prompt_ids list")
    return tuple(prompt_ids)


def _context(args: argparse.Namespace, stage: str) -> PipelineContext:
    return PipelineContext.create(PROJECT_ROOT, args.config, stage, args.run_id)


def _call(args: argparse.Namespace) -> Any:
    command = str(args.command)
    if "onlinerubric-r0" in command:
        extraction = command.endswith("r0-extraction")
        stage = ONLINERUBRIC_R0_EXTRACTION_STAGE if extraction else ONLINERUBRIC_R0_DEDUP_STAGE
        context = _context(args, stage)
        if command == "prepare-onlinerubric-r0-extraction":
            return prepare_onlinerubric_r0_extraction_batch(
                context,
                selected_prompt_ids=_selected_prompt_ids(args.prompt_manifest),
            )
        if command == "prepare-onlinerubric-r0-dedup":
            return prepare_onlinerubric_r0_dedup_batch(context)
        if command.startswith("submit-"):
            return submit_onlinerubric_batch(context)
        if command.startswith("status-"):
            return onlinerubric_batch_status(context)
        if command == "collect-onlinerubric-r0-extraction":
            return collect_onlinerubric_extraction_batch(context)
        if command == "collect-onlinerubric-r0-dedup":
            return collect_onlinerubric_r0_dedup_batch(context)
        raise AssertionError(f"unhandled OnlineRubrics R0 command: {command}")

    if command.endswith("-initial"):
        stage = initial_rubric_batch_stage(args.mode)
        context = _context(args, stage)
        operations: dict[str, Callable[[PipelineContext], Any]] = {
            "submit-initial": submit_initial_rubric_batch,
            "status-initial": initial_rubric_batch_status,
            "collect-initial": collect_initial_rubric_batch,
        }
        if command == "prepare-initial":
            return prepare_initial_rubric_batch(
                context,
                max_step=args.max_step,
                mode=args.mode,
            )
        return operations[command](context)

    extraction = "onlinerubric-extraction" in command
    stage = (
        onlinerubric_extraction_stage(args.mode)
        if extraction
        else onlinerubric_dedup_stage(args.mode)
    )
    context = _context(args, stage)
    if command == "prepare-onlinerubric-extraction":
        return prepare_onlinerubric_extraction_batch(
            context,
            max_step=args.max_step,
            policy_steps=_policy_steps(args.policy_steps),
            selected_prompt_ids=_selected_prompt_ids(args.prompt_manifest),
            mode=args.mode,
        )
    if command == "prepare-onlinerubric-dedup":
        return prepare_onlinerubric_dedup_batch(context, mode=args.mode)
    if command.startswith("submit-"):
        return submit_onlinerubric_batch(context)
    if command.startswith("status-"):
        return onlinerubric_batch_status(context)
    if command == "collect-onlinerubric-extraction":
        return collect_onlinerubric_extraction_batch(context)
    if command == "collect-onlinerubric-dedup":
        return collect_onlinerubric_dedup_batch(context)
    raise AssertionError(f"unhandled command: {command}")


def main() -> None:
    args = _parser().parse_args()
    print(json.dumps(_call(args), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
