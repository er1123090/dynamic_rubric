"""Incremental GPT-5 Batch audit while Qwen BoN scoring is still running."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from .artifacts import artifact_record, read_json, write_bytes_atomic, write_json_atomic
from .batch_dynamic import (
    BATCH_COMPLETION_WINDOW,
    BATCH_ENDPOINT,
    _as_dict,
    _download_file,
)
from .evaluation.gold_score import gold_cache_key, weighted_gold_score
from .hashing import sha256_file
from .minimum_gold import (
    GOLD_MAX_OUTPUT_TOKENS,
    APPROVED_PAYLOAD_CATEGORIES,
    PAPER_APPROVED_PAYLOAD_CATEGORIES,
    PROMPT_VERSION,
    REASONING_EFFORT,
    REQUESTED_MODEL,
    MinimumGoldError,
    _egress_approval_sha256,
    _request,
)
from .judge_prompts import PAPER_JUDGE_PROMPT_VERSION
from .minimum_staleness import _audit_conversations, _jsonl, _publish_jsonl
from .providers.openai_responses import _output_text


TERMINAL_BATCH_FAILURES = frozenset({"cancelled", "expired", "failed"})


def _openai_client() -> Any:
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise MinimumGoldError("OPENAI_API_KEY is not set")
    try:
        from openai import OpenAI  # pyright: ignore[reportMissingImports]
    except ImportError as error:
        raise MinimumGoldError("OpenAI Python SDK is required") from error
    return OpenAI(api_key=api_key)


def _stream_root(run_root: Path) -> Path:
    return run_root / "audit-gold-streaming-private"


def _selection_group(run_root: Path, selection_path: Path) -> tuple[str, Path]:
    expected_parent = (run_root / "select-bon-minimum" / "shards").resolve()
    resolved = selection_path.resolve()
    if resolved.parent != expected_parent or resolved.suffix != ".jsonl":
        raise MinimumGoldError(f"selection shard is outside the streaming source: {selection_path}")
    return resolved.stem, _stream_root(run_root) / "groups" / resolved.stem


def prepare_gold_selection_shard(
    run_root: Path,
    selection_path: Path,
    private_gt: Path,
    schema_path: Path,
    *,
    additional_selection_paths: tuple[Path, ...] = (),
    prompt_version: str = PROMPT_VERSION,
) -> dict[str, Any]:
    """Prepare one immutable Batch input from one completed selection shard."""

    group_id, group_root = _selection_group(run_root, selection_path)
    selection_paths = (selection_path, *additional_selection_paths)
    for path in (*selection_paths, private_gt, schema_path):
        if not path.is_file():
            raise MinimumGoldError(f"missing streaming hidden-gold input: {path}")
    schema = read_json(schema_path)
    gold_by_prompt = {str(row["prompt_id"]): row["gold_rubric"] for row in _jsonl(private_gt)}
    if prompt_version not in {PROMPT_VERSION, PAPER_JUDGE_PROMPT_VERSION}:
        raise MinimumGoldError(f"unsupported streaming prompt version: {prompt_version}")
    conversations = (
        _audit_conversations(run_root) if prompt_version == PAPER_JUDGE_PROMPT_VERSION else {}
    )
    selected: dict[tuple[str, str], dict[str, Any]] = {}
    for row in (item for path in selection_paths for item in _jsonl(path)):
        prompt_id = str(row["prompt_id"])
        response_id = str(row["response_id"])
        response_text = str(row["response_text"])
        key = prompt_id, response_id
        previous = selected.setdefault(
            key,
            {
                "prompt_id": prompt_id,
                "response_id": response_id,
                "response_text": response_text,
            },
        )
        if previous["response_text"] != response_text:
            raise MinimumGoldError(f"response text drift within selection shard: {response_id}")
    lines: list[dict[str, Any]] = []
    mapping: list[dict[str, Any]] = []
    for prompt_id, response_id in sorted(selected):
        row = selected[(prompt_id, response_id)]
        if prompt_id not in gold_by_prompt:
            raise MinimumGoldError(f"private GT has no prompt: {prompt_id}")
        line, identity = _request(
            prompt_id,
            response_id,
            str(row["response_text"]),
            gold_by_prompt[prompt_id],
            schema,
            conversation=conversations.get(prompt_id),
            prompt_version=prompt_version,
        )
        lines.append(line)
        mapping.append(identity)
    if not lines:
        raise MinimumGoldError(f"selection shard has no responses: {selection_path}")
    if len({str(row["custom_id"]) for row in lines}) != len(lines):
        raise MinimumGoldError("streaming hidden-gold Batch custom_id collision")
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
        "prompt_version": prompt_version,
        "payload_categories": list(
            PAPER_APPROVED_PAYLOAD_CATEGORIES
            if prompt_version == PAPER_JUDGE_PROMPT_VERSION
            else APPROVED_PAYLOAD_CATEGORIES
        ),
        "max_output_tokens": GOLD_MAX_OUTPUT_TOKENS,
        "endpoint": BATCH_ENDPOINT,
        "completion_window": BATCH_COMPLETION_WINDOW,
        "requests": len(lines),
        "input_file": artifact_record(input_path),
        "request_map": artifact_record(map_path),
        "selection_shard": artifact_record(selection_path),
        "selection_shards": [artifact_record(path) for path in selection_paths],
        "private_gt_sha256": sha256_file(private_gt),
        "schema_sha256": sha256_file(schema_path),
    }
    write_json_atomic(group_root / "manifest.json", manifest)
    return manifest


def submit_gold_selection_shard(
    run_root: Path,
    selection_path: Path,
    *,
    approval_path: Path | None = None,
) -> dict[str, Any]:
    """Submit one prepared streaming group, with an immutable local receipt."""

    group_id, group_root = _selection_group(run_root, selection_path)
    manifest_path = group_root / "manifest.json"
    manifest = read_json(manifest_path)
    payload_categories = tuple(manifest.get("payload_categories", ()))
    approval_sha256 = _egress_approval_sha256(
        run_root,
        approval_path=approval_path,
        payload_categories=payload_categories,
    )
    receipt_path = group_root / "submission.json"
    if receipt_path.is_file():
        receipt = read_json(receipt_path)
        if (
            receipt.get("manifest_sha256") != sha256_file(manifest_path)
            or receipt.get("egress_approval_sha256") != approval_sha256
            or not receipt.get("batch_id")
        ):
            raise MinimumGoldError(f"streaming Batch receipt drift: {receipt_path}")
        return receipt
    input_record = manifest["input_file"]
    input_path = Path(str(input_record["path"]))
    if sha256_file(input_path) != input_record["sha256"]:
        raise MinimumGoldError(f"streaming Batch input changed: {input_path}")
    client = _openai_client()
    with input_path.open("rb") as stream:
        uploaded = _as_dict(client.files.create(file=stream, purpose="batch"))
    created = _as_dict(
        client.batches.create(
            input_file_id=str(uploaded["id"]),
            endpoint=BATCH_ENDPOINT,
            completion_window=BATCH_COMPLETION_WINDOW,
            metadata={"experiment": "minimum-staleness-stream", "group": group_id},
        )
    )
    receipt = {
        "schema_version": 1,
        "group": group_id,
        "manifest_sha256": sha256_file(manifest_path),
        "egress_approval_sha256": approval_sha256,
        "input_file_id": str(uploaded["id"]),
        "batch_id": str(created["id"]),
        "status": str(created.get("status", "validating")),
        "created_at": created.get("created_at"),
    }
    write_json_atomic(receipt_path, receipt)
    return receipt


def stream_gold_selection_shard(
    run_root: Path,
    selection_path: Path,
    private_gt: Path,
    schema_path: Path,
) -> dict[str, Any]:
    """Prepare and immediately submit one completed Qwen selection shard."""

    _, group_root = _selection_group(run_root, selection_path)
    manifest_path = group_root / "manifest.json"
    submission_path = group_root / "submission.json"
    if manifest_path.is_file() and submission_path.is_file():
        manifest = read_json(manifest_path)
    else:
        manifest = prepare_gold_selection_shard(run_root, selection_path, private_gt, schema_path)
    receipt = submit_gold_selection_shard(run_root, selection_path)
    groups = len(list((_stream_root(run_root) / "groups").glob("*/submission.json")))
    write_json_atomic(
        _stream_root(run_root) / "progress.json",
        {
            "submitted_groups": groups,
            "last_group": receipt["group"],
            "last_requests": manifest["requests"],
        },
        immutable=False,
    )
    return {"manifest": manifest, "submission": receipt}


def submit_available_gold_shards(
    run_root: Path,
    private_gt: Path,
    schema_path: Path,
) -> dict[str, Any]:
    """Catch up any completed selections that were not submitted before a restart."""

    submitted = []
    for selection_path in sorted((run_root / "select-bon-minimum" / "shards").glob("*.jsonl")):
        result = stream_gold_selection_shard(run_root, selection_path, private_gt, schema_path)
        submitted.append(str(result["submission"]["batch_id"]))
    return {"submitted_groups": len(submitted), "batch_ids": submitted}


def _normalize_payload(
    payload: bytes,
    identities: Mapping[str, Mapping[str, Any]],
    schema_path: Path,
    *,
    prompt_version: str = PROMPT_VERSION,
) -> tuple[list[dict[str, Any]], set[str], dict[str, int]]:
    rows = [json.loads(line) for line in payload.splitlines() if line.strip()]
    actual = {str(row.get("custom_id", "")) for row in rows}
    if actual != set(identities):
        raise MinimumGoldError("streaming hidden-gold Batch output inventory mismatch")
    normalized: list[dict[str, Any]] = []
    returned_models: set[str] = set()
    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for row in rows:
        custom_id = str(row["custom_id"])
        if row.get("error") is not None:
            raise MinimumGoldError(
                f"streaming hidden-gold request failed: {custom_id}: {row['error']}"
            )
        response = row.get("response")
        if not isinstance(response, Mapping) or int(response.get("status_code", 0)) != 200:
            raise MinimumGoldError(f"streaming hidden-gold response is not HTTP 200: {custom_id}")
        body = response.get("body")
        if not isinstance(body, Mapping):
            raise MinimumGoldError(f"streaming hidden-gold body is absent: {custom_id}")
        returned_model = str(body.get("model", ""))
        if not returned_model:
            raise MinimumGoldError(f"streaming hidden-gold model is absent: {custom_id}")
        returned_models.add(returned_model)
        try:
            parsed = json.loads(_output_text(body))
        except json.JSONDecodeError as error:
            raise MinimumGoldError(
                f"streaming hidden-gold output is not JSON: {custom_id}"
            ) from error
        criterion_rows = parsed.get("criterion_scores") if isinstance(parsed, Mapping) else None
        if not isinstance(criterion_rows, list):
            raise MinimumGoldError(f"streaming hidden-gold criterion_scores absent: {custom_id}")
        score_map: dict[str, float] = {}
        for item in criterion_rows:
            if not isinstance(item, Mapping):
                raise MinimumGoldError(f"streaming hidden-gold criterion malformed: {custom_id}")
            criterion_id = str(item.get("criterion_id", ""))
            raw_score = item.get("score")
            if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
                raise MinimumGoldError(f"streaming hidden-gold score is non-numeric: {custom_id}")
            if prompt_version == PAPER_JUDGE_PROMPT_VERSION and raw_score not in {0, 1}:
                raise MinimumGoldError(f"paper hidden-gold score is not binary: {custom_id}")
            if criterion_id in score_map:
                raise MinimumGoldError(f"duplicate streaming gold criterion: {custom_id}")
            score_map[criterion_id] = float(raw_score)
        identity = identities[custom_id]
        weights = {str(key): float(value) for key, value in identity["criterion_weights"].items()}
        body_usage = body.get("usage", {})
        if isinstance(body_usage, Mapping):
            for key in usage:
                usage[key] += int(body_usage.get(key, 0) or 0)
        normalized.append(
            {
                "prompt_id": identity["prompt_id"],
                "response_id": identity["response_id"],
                "gold_score": weighted_gold_score(score_map, weights),
                "criterion_scores": score_map,
                "cache_key": gold_cache_key(
                    prompt_id=str(identity["prompt_id"]),
                    response_text_hash=str(identity["response_text_hash"]),
                    gold_rubric_hash=str(identity["gold_rubric_hash"]),
                    requested_model=REQUESTED_MODEL,
                    returned_model=returned_model,
                    grader_prompt_hash=hashlib.sha256(prompt_version.encode()).hexdigest(),
                    schema_hash=sha256_file(schema_path),
                    reasoning_effort=REASONING_EFFORT,
                ),
                "requested_model": REQUESTED_MODEL,
                "returned_model": returned_model,
                "reasoning_effort": REASONING_EFFORT,
                "provider_request_id": str(body.get("id") or response.get("request_id") or ""),
            }
        )
    return normalized, returned_models, usage


def sync_streaming_gold(run_root: Path, schema_path: Path) -> dict[str, Any]:
    """Poll all pending groups and collect every group that has completed."""

    client = _openai_client()
    statuses: list[dict[str, Any]] = []
    completed_groups = 0
    for receipt_path in sorted((_stream_root(run_root) / "groups").glob("*/submission.json")):
        group_root = receipt_path.parent
        receipt = read_json(receipt_path)
        result_path = group_root / "result.json"
        if result_path.is_file():
            result = read_json(result_path)
            statuses.append(
                {
                    "group": receipt["group"],
                    "batch_id": receipt["batch_id"],
                    "status": "completed",
                    "requests": result["grader_calls"],
                }
            )
            completed_groups += 1
            continue
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
            raise MinimumGoldError(
                f"streaming hidden-gold Batch failed: {receipt['batch_id']}={status}"
            )
        if status != "completed" or not current.get("output_file_id"):
            continue
        manifest = read_json(group_root / "manifest.json")
        identities = {
            str(row["custom_id"]): row for row in _jsonl(Path(str(manifest["request_map"]["path"])))
        }
        payload = _download_file(client, str(current["output_file_id"]))
        write_bytes_atomic(group_root / "output.raw.jsonl", payload)
        normalized, returned_models, usage = _normalize_payload(
            payload,
            identities,
            schema_path,
            prompt_version=str(manifest.get("prompt_version", PROMPT_VERSION)),
        )
        if len(returned_models) != 1:
            raise MinimumGoldError(
                f"streaming hidden-gold returned-model drift: {sorted(returned_models)}"
            )
        normalized.sort(key=lambda row: (str(row["prompt_id"]), str(row["response_id"])))
        output_path = group_root / "gold_scores.jsonl"
        _publish_jsonl(output_path, normalized)
        error_file_id = current.get("error_file_id")
        if error_file_id:
            error_payload = _download_file(client, str(error_file_id))
            write_bytes_atomic(group_root / "errors.jsonl", error_payload)
            if error_payload.strip():
                raise MinimumGoldError(
                    f"streaming hidden-gold Batch group has errors: {receipt['group']}"
                )
        write_json_atomic(
            result_path,
            {
                "grader_calls": len(normalized),
                "returned_model": next(iter(returned_models)),
                "usage": usage,
                "output": artifact_record(output_path),
            },
        )
        completed_groups += 1
    result = {
        "submitted_groups": len(statuses),
        "completed_groups": completed_groups,
        "all_completed": bool(statuses) and completed_groups == len(statuses),
        "jobs": statuses,
    }
    write_json_atomic(_stream_root(run_root) / "status.json", result, immutable=False)
    return result


def finalize_streaming_gold(run_root: Path, schema_path: Path) -> dict[str, Any]:
    """Merge completed streaming outputs into the canonical analysis artifact."""

    package_path = run_root / "export-audit-package-minimum" / "audit_package.jsonl"
    if not package_path.is_file():
        raise MinimumGoldError("full audit package is not ready")
    expected = {(str(row["prompt_id"]), str(row["response_id"])) for row in _jsonl(package_path)}
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    returned_models: set[str] = set()
    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    group_results = sorted((_stream_root(run_root) / "groups").glob("*/result.json"))
    for result_path in group_results:
        result = read_json(result_path)
        returned_models.add(str(result["returned_model"]))
        for key in usage:
            usage[key] += int(result["usage"].get(key, 0) or 0)
        for row in _jsonl(Path(str(result["output"]["path"]))):
            key = str(row["prompt_id"]), str(row["response_id"])
            previous = merged.setdefault(key, row)
            if previous != row:
                raise MinimumGoldError(f"conflicting streaming gold result: {key}")
    if set(merged) != expected:
        missing = len(expected - set(merged))
        extra = len(set(merged) - expected)
        raise MinimumGoldError(
            f"streaming gold inventory differs from audit package: missing={missing} extra={extra}"
        )
    if len(returned_models) != 1:
        raise MinimumGoldError(
            f"streaming hidden-gold returned-model drift: {sorted(returned_models)}"
        )
    stage_root = run_root / "audit-gold-minimum-private"
    output_path = stage_root / "gold_scores.jsonl"
    _publish_jsonl(output_path, (merged[key] for key in sorted(merged)))
    manifest = {
        "schema_version": 2,
        "private_process": True,
        "streaming": True,
        "requested_model": REQUESTED_MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "prompt_version": PROMPT_VERSION,
        "max_output_tokens": GOLD_MAX_OUTPUT_TOKENS,
        "endpoint": BATCH_ENDPOINT,
        "egress_approval_sha256": _egress_approval_sha256(run_root),
        "audit_package": artifact_record(package_path),
        "schema_sha256": sha256_file(schema_path),
        "groups": len(group_results),
        "requests": len(merged),
    }
    write_json_atomic(stage_root / "manifest.json", manifest)
    result = {
        "grader_calls": len(merged),
        "returned_model": next(iter(returned_models)),
        "usage": usage,
        "streaming_groups": len(group_results),
        "output": artifact_record(output_path),
    }
    write_json_atomic(stage_root / "result.json", result)
    return result
