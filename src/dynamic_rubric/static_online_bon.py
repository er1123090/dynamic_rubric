"""Prepare Static-R0 versus one OnlineRubric variant for paper-style BoN."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .artifacts import write_json_atomic
from .hashing import canonical_json_bytes, sha256_file
from .judge_prompts import PAPER_JUDGE_PROMPT_VERSION, PAPER_JUDGE_SYSTEM_PROMPT
from .minimum_gold import PAPER_APPROVED_PAYLOAD_CATEGORIES
from .minimum_staleness import (
    FOCAL_STEPS,
    MinimumExperimentError,
    _jsonl,
    _publish_jsonl,
)


CONTROL_STAGES = {
    "pi_ref": "onlinerubric_dedup_fixed_batch",
    "pi_old": "onlinerubric_dedup_prev_batch",
}
LINKED_BASE_STAGES = ("generate-bon", "generate-static", "train-static")


def _safe_link(source: Path, target: Path) -> None:
    if target.is_symlink():
        if target.resolve() != source.resolve():
            raise MinimumExperimentError(f"static/OnlineRubric source link drift: {target}")
        return
    if target.exists():
        raise MinimumExperimentError(f"static/OnlineRubric link target exists: {target}")
    target.symlink_to(source.resolve(), target_is_directory=True)


def _online_rows(source_run_root: Path, control: str) -> dict[tuple[str, int], dict[str, Any]]:
    try:
        stage = CONTROL_STAGES[control]
    except KeyError as error:
        raise MinimumExperimentError(f"unsupported OnlineRubric control: {control}") from error
    path = source_run_root / stage / "onlinerubric_rubrics.jsonl"
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    for row in _jsonl(path):
        key = str(row["prompt_id"]), int(row["policy_step"])
        if key in rows:
            raise MinimumExperimentError(f"duplicate OnlineRubric row: {key}")
        rows[key] = row
    if len(rows) != 40 or {step for _, step in rows} != set(FOCAL_STEPS):
        raise MinimumExperimentError(
            f"OnlineRubric inventory must be 10 prompts x 4 focal steps: {len(rows)}"
        )
    return rows


def weighted_expanded_criteria(
    prompt_id: str,
    step: int,
    control: str,
    criteria: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Encode integer criterion weights as repeated IDs without repeated Qwen calls."""

    expanded: list[dict[str, Any]] = []
    for index, item in enumerate(criteria):
        weight = item.get("weight")
        if isinstance(weight, bool) or not isinstance(weight, int) or weight < 1:
            raise MinimumExperimentError(f"invalid OnlineRubric weight: {weight!r}")
        text = str(item["text"])
        criterion_id = (
            f"onlinerubric-{control}-{hashlib.sha256(prompt_id.encode()).hexdigest()[:10]}-"
            f"{step:03d}-{index:03d}"
        )
        criterion = {
            "criterion_id": criterion_id,
            "text": text,
            "source": "onlinerubric_pairwise",
            "created_step": step,
            "online_control": control,
            "source_weight": weight,
        }
        expanded.extend(dict(criterion) for _ in range(weight))
    if not expanded:
        raise MinimumExperimentError("OnlineRubric has no criteria")
    return expanded


def _replay_rows(source_run_root: Path, control: str) -> list[dict[str, Any]]:
    online = _online_rows(source_run_root, control)
    source_path = source_run_root / "replay-dynamic-minimum" / "replay_snapshots.jsonl"
    rows = []
    replaced: set[tuple[str, int]] = set()
    for source in _jsonl(source_path):
        step = int(source["policy_step"])
        if step not in FOCAL_STEPS:
            continue
        row = dict(source)
        key = str(row["prompt_id"]), step
        rubric = online.get(key)
        if rubric is not None:
            criteria = weighted_expanded_criteria(key[0], step, control, tuple(rubric["criteria"]))
            row.update(
                {
                    "mode": "dynamic_fixed_budgeted",
                    "rubric_step": step,
                    "criteria": criteria,
                    "criterion_count": len(criteria),
                    "content_hash": hashlib.sha256(canonical_json_bytes(criteria)).hexdigest(),
                    "online_control": control,
                    "online_source_stage": CONTROL_STAGES[control],
                    "online_rubric_id": rubric["rubric_id"],
                    "online_weight_encoding": "repeated-criterion-id",
                }
            )
            replaced.add(key)
        rows.append(row)
    if replaced != set(online):
        raise MinimumExperimentError(
            f"OnlineRubric rows absent from replay: {sorted(set(online) - replaced)}"
        )
    return rows


def prepare_static_online_run(
    source_run_root: Path,
    run_root: Path,
    control: str,
) -> dict[str, Any]:
    """Create one isolated Static-R0-vs-OnlineRubric derived experiment."""

    if control not in CONTROL_STAGES:
        raise MinimumExperimentError(f"unsupported OnlineRubric control: {control}")
    source_run_root = source_run_root.resolve()
    run_root = run_root.absolute()
    if source_run_root == run_root or source_run_root.parent != run_root.parent.resolve():
        raise MinimumExperimentError("derived run must be a sibling distinct from its source")
    run_root.mkdir(parents=True, exist_ok=True)
    linked_stages = (*LINKED_BASE_STAGES, CONTROL_STAGES[control])
    for stage in linked_stages:
        source = source_run_root / stage
        if not source.exists():
            raise MinimumExperimentError(f"missing source stage: {source}")
        _safe_link(source, run_root / stage)

    replay_path = run_root / "replay-dynamic-minimum" / "replay_snapshots.jsonl"
    replay_rows = _replay_rows(source_run_root, control)
    _publish_jsonl(replay_path, replay_rows)
    write_json_atomic(
        run_root / "replay-dynamic-minimum" / "result.json",
        {
            "schema_version": 1,
            "comparison": ["static", f"onlinerubric_{control}"],
            "score_mode": "dynamic_fixed_budgeted",
            "online_control": control,
            "online_source_stage": CONTROL_STAGES[control],
            "weighted_aggregation": "integer-weight-expanded-criterion-ids",
            "snapshots": len(replay_rows),
            "output": str(replay_path),
            "output_sha256": sha256_file(replay_path),
        },
    )
    source_record = {
        "schema_version": 1,
        "experiment": "static-r0-vs-dynamic-onlinerubric-paper-judge-bon",
        "online_control": control,
        "source_run_root": str(source_run_root),
        "linked_stages": {
            stage: str((source_run_root / stage).resolve()) for stage in linked_stages
        },
        "judge_prompt_version": PAPER_JUDGE_PROMPT_VERSION,
        "dynamic_rubric_aggregation": "positive-integer-weighted-mean",
        "weight_encoding": "repeated criterion IDs in aggregation; one Qwen call per unique ID",
    }
    write_json_atomic(run_root / "static-online-rubric-source.json", source_record)
    write_json_atomic(
        run_root / "paper-judge-prompt-contract.json",
        {
            "schema_version": 1,
            "prompt_version": PAPER_JUDGE_PROMPT_VERSION,
            "system_prompt": PAPER_JUDGE_SYSTEM_PROMPT,
            "system_prompt_sha256": hashlib.sha256(PAPER_JUDGE_SYSTEM_PROMPT.encode()).hexdigest(),
            "qwen_contract": "one criterion; target likelihood over exact YES/NO",
            "gpt5_contract": "all physician criteria; structured integer 1/0",
            "shared_inputs": ["user conversation", "assistant response", "criterion text"],
        },
    )
    approval_path = run_root / "paper-judge-gold-egress-approval.json"
    if not approval_path.is_file():
        write_json_atomic(
            approval_path,
            {
                "schema_version": 1,
                "approved": True,
                "approval_source": "explicit user request in active conversation",
                "approved_at": datetime.now(timezone.utc).isoformat(),
                "destination": "OpenAI GPT-5 Batch API",
                "endpoint": "/v1/responses",
                "purpose": "hidden-gold-evaluation",
                "requested_model": "gpt-5",
                "payload_categories": list(PAPER_APPROVED_PAYLOAD_CATEGORIES),
            },
        )
    return {
        "run_root": str(run_root),
        "source_run_root": str(source_run_root),
        "online_control": control,
        "replay_snapshots": len(replay_rows),
    }
