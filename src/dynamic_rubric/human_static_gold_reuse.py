"""Reuse compatible GPT-5 gold rows and submit only missing selected responses."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .artifacts import artifact_record, read_json, write_bytes_atomic, write_json_atomic
from .batch_dynamic import _as_dict, _download_file
from .hashing import sha256_file
from .judge_prompts import PAPER_JUDGE_PROMPT_VERSION
from .minimum_gold import (
    GOLD_MAX_OUTPUT_TOKENS,
    PAPER_APPROVED_PAYLOAD_CATEGORIES,
    REASONING_EFFORT,
    REQUESTED_MODEL,
    _request,
)
from .minimum_gold_streaming import (
    BATCH_COMPLETION_WINDOW,
    BATCH_ENDPOINT,
    TERMINAL_BATCH_FAILURES,
    _normalize_payload,
    _openai_client,
    _selection_group,
    _stream_root,
    submit_gold_selection_shard,
)
from .minimum_interim import ordered_prompt_subset, target_groups
from .minimum_staleness import (
    MinimumExperimentError,
    _audit_conversations,
    _jsonl,
    _publish_jsonl,
    _shard_name,
)


def _compatible_reuse_rows(
    reuse_roots: Sequence[Path],
    *,
    private_gt_sha256: str,
    schema_sha256: str,
) -> dict[tuple[str, str], dict[str, Any]]:
    cached: dict[tuple[str, str], dict[str, Any]] = {}
    for reuse_root in reuse_roots:
        for score_path in sorted(
            reuse_root.glob("audit-gold-streaming-private/groups/*/gold_scores.jsonl")
        ):
            manifest_path = score_path.parent / "manifest.json"
            if not manifest_path.is_file():
                continue
            manifest = read_json(manifest_path)
            if (
                manifest.get("prompt_version") != PAPER_JUDGE_PROMPT_VERSION
                or manifest.get("private_gt_sha256") != private_gt_sha256
                or manifest.get("schema_sha256") != schema_sha256
                or manifest.get("requested_model") != REQUESTED_MODEL
                or manifest.get("reasoning_effort") != REASONING_EFFORT
            ):
                continue
            for row in _jsonl(score_path):
                key = str(row["prompt_id"]), str(row["response_id"])
                cached.setdefault(key, dict(row))
    return cached


def prepare_reused_gold_selection_shard(
    run_root: Path,
    selection_path: Path,
    private_gt: Path,
    schema_path: Path,
    *,
    reuse_roots: Sequence[Path],
) -> dict[str, Any]:
    """Prepare a paper-style Batch containing only selections absent from prior gold."""

    group_id, group_root = _selection_group(run_root, selection_path)
    manifest_path = group_root / "manifest.json"
    if manifest_path.is_file():
        return read_json(manifest_path)
    for path in (selection_path, private_gt, schema_path):
        if not path.is_file():
            raise MinimumExperimentError(f"missing reused-gold input: {path}")
    schema = read_json(schema_path)
    private_gt_sha256 = sha256_file(private_gt)
    schema_sha256 = sha256_file(schema_path)
    gold_by_prompt = {str(row["prompt_id"]): row["gold_rubric"] for row in _jsonl(private_gt)}
    conversations = _audit_conversations(run_root)
    selected: dict[tuple[str, str], dict[str, str]] = {}
    for row in _jsonl(selection_path):
        key = str(row["prompt_id"]), str(row["response_id"])
        value = {
            "prompt_id": key[0],
            "response_id": key[1],
            "response_text": str(row["response_text"]),
        }
        previous = selected.setdefault(key, value)
        if previous["response_text"] != value["response_text"]:
            raise MinimumExperimentError(f"response text drift in selection: {key}")
    reusable = _compatible_reuse_rows(
        reuse_roots,
        private_gt_sha256=private_gt_sha256,
        schema_sha256=schema_sha256,
    )
    cached = [reusable[key] for key in sorted(selected) if key in reusable]
    missing = [key for key in sorted(selected) if key not in reusable]
    cached_path = group_root / "cached_gold_scores.jsonl"
    if cached:
        _publish_jsonl(cached_path, cached)

    lines: list[dict[str, Any]] = []
    mapping: list[dict[str, Any]] = []
    for prompt_id, response_id in missing:
        row = selected[(prompt_id, response_id)]
        if prompt_id not in gold_by_prompt:
            raise MinimumExperimentError(f"private GT has no prompt: {prompt_id}")
        line, identity = _request(
            prompt_id,
            response_id,
            row["response_text"],
            gold_by_prompt[prompt_id],
            schema,
            conversation=conversations.get(prompt_id),
            prompt_version=PAPER_JUDGE_PROMPT_VERSION,
        )
        lines.append(line)
        mapping.append(identity)
    input_path = group_root / "input.jsonl"
    map_path = group_root / "request_map.jsonl"
    _publish_jsonl(input_path, lines)
    _publish_jsonl(map_path, mapping)
    manifest = {
        "schema_version": 1,
        "private_process": True,
        "streaming_group": group_id,
        "requested_model": REQUESTED_MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "prompt_version": PAPER_JUDGE_PROMPT_VERSION,
        "payload_categories": list(PAPER_APPROVED_PAYLOAD_CATEGORIES),
        "max_output_tokens": GOLD_MAX_OUTPUT_TOKENS,
        "endpoint": BATCH_ENDPOINT,
        "completion_window": BATCH_COMPLETION_WINDOW,
        "selected_responses": len(selected),
        "cached_responses": len(cached),
        "requests": len(lines),
        "input_file": artifact_record(input_path),
        "request_map": artifact_record(map_path),
        "cached_gold_scores": artifact_record(cached_path) if cached else None,
        "selection_shard": artifact_record(selection_path),
        "selection_shards": [artifact_record(selection_path)],
        "private_gt_sha256": private_gt_sha256,
        "schema_sha256": schema_sha256,
        "reuse_roots": [str(path.resolve()) for path in reuse_roots],
    }
    write_json_atomic(manifest_path, manifest)
    if not lines:
        returned_models = {str(row["returned_model"]) for row in cached}
        if len(returned_models) != 1 or len(cached) != len(selected):
            raise MinimumExperimentError(f"invalid cache-only gold group: {group_id}")
        output_path = group_root / "gold_scores.jsonl"
        _publish_jsonl(output_path, cached)
        write_json_atomic(
            group_root / "result.json",
            {
                "grader_calls": 0,
                "cached_responses": len(cached),
                "total_responses": len(cached),
                "returned_model": next(iter(returned_models)),
                "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                "output": artifact_record(output_path),
            },
        )
    return manifest


def prepare_or_submit_reused_gold(
    run_root: Path,
    private_gt: Path,
    schema_path: Path,
    *,
    prompt_count: int,
    reuse_roots: Sequence[Path],
    submit: bool,
) -> dict[str, Any]:
    prompt_ids = ordered_prompt_subset(run_root, prompt_count)
    groups = []
    for policy_id, prompt_id in sorted(target_groups(prompt_ids)):
        stem = f"{policy_id}-{_shard_name(policy_id, prompt_id)}.jsonl"
        selection_path = run_root / "select-bon-minimum" / "shards" / stem
        if not selection_path.is_file():
            continue
        manifest = prepare_reused_gold_selection_shard(
            run_root,
            selection_path,
            private_gt,
            schema_path,
            reuse_roots=reuse_roots,
        )
        row: dict[str, Any] = {
            "policy_id": policy_id,
            "prompt_id": prompt_id,
            "selected_responses": int(manifest["selected_responses"]),
            "cached_responses": int(manifest["cached_responses"]),
            "requests": int(manifest["requests"]),
        }
        if submit and int(manifest["requests"]) > 0:
            receipt = submit_gold_selection_shard(
                run_root,
                selection_path,
                approval_path=run_root / "paper-judge-gold-egress-approval.json",
            )
            row["batch_id"] = receipt["batch_id"]
        groups.append(row)
    return {
        "prompt_ids": list(prompt_ids),
        "groups": groups,
        "ready_groups": len(groups),
        "expected_groups": len(target_groups(prompt_ids)),
        "selected_responses": sum(int(row["selected_responses"]) for row in groups),
        "cached_responses": sum(int(row["cached_responses"]) for row in groups),
        "requests": sum(int(row["requests"]) for row in groups),
        "submitted": submit,
    }


def sync_reused_gold(run_root: Path, schema_path: Path) -> dict[str, Any]:
    """Collect submitted missing rows and merge immutable reused rows before publication."""

    client = _openai_client()
    statuses: list[dict[str, Any]] = []
    group_roots = sorted((_stream_root(run_root) / "groups").glob("*"))
    completed_groups = 0
    submitted_groups = 0
    for group_root in group_roots:
        manifest_path = group_root / "manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = read_json(manifest_path)
        result_path = group_root / "result.json"
        submission_path = group_root / "submission.json"
        if result_path.is_file():
            result = read_json(result_path)
            statuses.append(
                {
                    "group": manifest["streaming_group"],
                    "status": "completed",
                    "requests": result["grader_calls"],
                    "cached_responses": result.get("cached_responses", 0),
                }
            )
            completed_groups += 1
            submitted_groups += int(submission_path.is_file())
            continue
        if not submission_path.is_file():
            statuses.append(
                {"group": manifest["streaming_group"], "status": "prepared"}
            )
            continue
        submitted_groups += 1
        receipt = read_json(submission_path)
        current = _as_dict(client.batches.retrieve(str(receipt["batch_id"])))
        status = str(current.get("status", ""))
        statuses.append(
            {
                "group": receipt["group"],
                "batch_id": receipt["batch_id"],
                "status": status,
                "request_counts": current.get("request_counts"),
            }
        )
        if status in TERMINAL_BATCH_FAILURES:
            raise MinimumExperimentError(
                f"reused hidden-gold Batch failed: {receipt['batch_id']}={status}"
            )
        if status != "completed" or not current.get("output_file_id"):
            continue
        identities = {
            str(row["custom_id"]): row
            for row in _jsonl(Path(str(manifest["request_map"]["path"])))
        }
        payload = _download_file(client, str(current["output_file_id"]))
        write_bytes_atomic(group_root / "output.raw.jsonl", payload)
        normalized, returned_models, usage = _normalize_payload(
            payload,
            identities,
            schema_path,
            prompt_version=PAPER_JUDGE_PROMPT_VERSION,
        )
        cached_record = manifest.get("cached_gold_scores")
        cached = (
            list(_jsonl(Path(str(cached_record["path"]))))
            if isinstance(cached_record, Mapping)
            else []
        )
        merged: dict[tuple[str, str], dict[str, Any]] = {}
        for row in (*cached, *normalized):
            key = str(row["prompt_id"]), str(row["response_id"])
            if key in merged:
                raise MinimumExperimentError(f"duplicate reused gold row: {key}")
            merged[key] = dict(row)
        if len(merged) != int(manifest["selected_responses"]):
            raise MinimumExperimentError(
                f"reused gold inventory mismatch: {manifest['streaming_group']}"
            )
        returned_models.update(str(row["returned_model"]) for row in cached)
        if len(returned_models) != 1:
            raise MinimumExperimentError(
                f"reused gold returned-model drift: {sorted(returned_models)}"
            )
        output_rows = [merged[key] for key in sorted(merged)]
        output_path = group_root / "gold_scores.jsonl"
        _publish_jsonl(output_path, output_rows)
        error_file_id = current.get("error_file_id")
        if error_file_id:
            error_payload = _download_file(client, str(error_file_id))
            write_bytes_atomic(group_root / "errors.jsonl", error_payload)
            if error_payload.strip():
                raise MinimumExperimentError(
                    f"reused hidden-gold Batch has errors: {receipt['group']}"
                )
        write_json_atomic(
            result_path,
            {
                "grader_calls": len(normalized),
                "cached_responses": len(cached),
                "total_responses": len(output_rows),
                "returned_model": next(iter(returned_models)),
                "usage": usage,
                "output": artifact_record(output_path),
            },
        )
        completed_groups += 1
    result = {
        "prepared_groups": len(statuses),
        "submitted_groups": submitted_groups,
        "completed_groups": completed_groups,
        "all_completed": bool(statuses) and completed_groups == len(statuses),
        "jobs": statuses,
    }
    write_json_atomic(_stream_root(run_root) / "status.json", result, immutable=False)
    return result
