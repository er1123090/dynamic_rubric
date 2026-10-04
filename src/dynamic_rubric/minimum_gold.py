"""Private GPT-5 Batch audit for the minimum staleness experiment."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from .artifacts import (
    artifact_record,
    read_json,
    write_bytes_atomic,
    write_json_atomic,
)
from .batch_dynamic import (
    BATCH_COMPLETION_WINDOW,
    BATCH_ENDPOINT,
    _as_dict,
    _download_file,
    _responses_payload,
    _shard,
)
from .judge_prompts import (
    PAPER_JUDGE_PROMPT_VERSION,
    PAPER_JUDGE_SYSTEM_PROMPT,
    gold_judge_user_prompt,
)
from .evaluation.gold_score import gold_cache_key, weighted_gold_score
from .hashing import canonical_json_bytes, sha256_file, sha256_json
from .minimum_staleness import _jsonl, _publish_jsonl
from .providers.base import GenerationRequest
from .providers.openai_responses import _output_text


REQUESTED_MODEL = "gpt-5"
REASONING_EFFORT = "medium"
PROMPT_VERSION = "hidden-gold-grader-v2"
GOLD_MAX_OUTPUT_TOKENS = 8192
APPROVED_PAYLOAD_CATEGORIES = (
    "selected_medical_response_texts",
    "private_physician_rubrics",
    "criterion_weights",
)
PAPER_APPROVED_PAYLOAD_CATEGORIES = (
    "selected_medical_response_texts",
    "public_healthbench_conversations",
    "private_physician_rubrics",
    "criterion_weights",
)


class MinimumGoldError(RuntimeError):
    """Raised when a private hidden-gold Batch invariant is violated."""


def _egress_approval_sha256(
    run_root: Path,
    *,
    approval_path: Path | None = None,
    payload_categories: tuple[str, ...] = APPROVED_PAYLOAD_CATEGORIES,
) -> str:
    path = approval_path or run_root / "train-static" / "gold-egress-approval.json"
    if not path.is_file():
        raise MinimumGoldError("explicit hidden-gold egress approval artifact is missing")
    value = read_json(path)
    if (
        value.get("approved") is not True
        or value.get("destination") != "OpenAI GPT-5 Batch API"
        or value.get("endpoint") != BATCH_ENDPOINT
        or value.get("purpose") != "hidden-gold-evaluation"
        or value.get("requested_model") != REQUESTED_MODEL
        or tuple(value.get("payload_categories", ())) != payload_categories
    ):
        raise MinimumGoldError("hidden-gold egress approval scope does not match submission")
    return sha256_file(path)


def _criterion_inventory(gold_rubric: Any) -> tuple[list[dict[str, str]], dict[str, float]]:
    if not isinstance(gold_rubric, list) or not gold_rubric:
        raise MinimumGoldError("gold_rubric must be a non-empty list")
    criteria: list[dict[str, str]] = []
    weights: dict[str, float] = {}
    for index, item in enumerate(gold_rubric):
        if not isinstance(item, Mapping):
            raise MinimumGoldError("gold criterion must be an object")
        text = item.get("criterion")
        points = item.get("points")
        if (
            not isinstance(text, str)
            or not text.strip()
            or isinstance(points, bool)
            or not isinstance(points, (int, float))
        ):
            raise MinimumGoldError("gold criterion text/points are malformed")
        criterion_id = f"gold-{index:03d}"
        criteria.append({"criterion_id": criterion_id, "criterion": text})
        weights[criterion_id] = float(points)
    return criteria, weights


def _request(
    prompt_id: str,
    response_id: str,
    response_text: str,
    gold_rubric: Any,
    schema: Mapping[str, Any],
    *,
    conversation: tuple[dict[str, Any], ...] | None = None,
    prompt_version: str = PROMPT_VERSION,
) -> tuple[dict[str, Any], dict[str, Any]]:
    criteria, weights = _criterion_inventory(gold_rubric)
    rubric_hash = sha256_json(gold_rubric)
    response_hash = hashlib.sha256(response_text.encode()).hexdigest()
    identity = {
        "prompt_id": prompt_id,
        "response_id": response_id,
        "response_text_hash": response_hash,
        "gold_rubric_hash": rubric_hash,
        "criterion_weights": weights,
    }
    if prompt_version == PROMPT_VERSION:
        if conversation is not None:
            raise MinimumGoldError("legacy hidden-gold prompt does not accept a conversation")
        request_messages = (
            {
                "role": "user",
                "content": (
                    "Judge the assistant response independently against every physician "
                    "criterion. Return each supplied criterion_id exactly once. A score of 1 "
                    "means fully satisfied, 0 means not satisfied, and intermediate values "
                    "represent partial satisfaction. Do not add or omit criteria.\n\n"
                    "Physician criteria:\n"
                    + json.dumps(criteria, ensure_ascii=False, sort_keys=True)
                    + "\n\nAssistant response:\n"
                    + response_text
                ),
            },
        )
    elif prompt_version == PAPER_JUDGE_PROMPT_VERSION:
        if conversation is None:
            raise MinimumGoldError("paper hidden-gold prompt requires the user conversation")
        identity["conversation_hash"] = hashlib.sha256(
            canonical_json_bytes(conversation)
        ).hexdigest()
        identity["grader_prompt_version"] = prompt_version
        request_messages = (
            {"role": "system", "content": PAPER_JUDGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": gold_judge_user_prompt(conversation, response_text, criteria),
            },
        )
    else:
        raise MinimumGoldError(f"unsupported hidden-gold prompt version: {prompt_version}")
    identity_hash = hashlib.sha256(canonical_json_bytes(identity)).hexdigest()
    custom_id = f"gold-{identity_hash[:32]}"
    request = GenerationRequest(
        prompt_id=f"{prompt_id}-hidden-gold",
        messages=request_messages,
        family="hidden_gold_audit",
        seed=0,
        json_schema=schema,
        schema_name=(
            "paper_gold_grader_v1"
            if prompt_version == PAPER_JUDGE_PROMPT_VERSION
            else "hidden_gold_grader_v1"
        ),
        reasoning_effort=REASONING_EFFORT,
        max_output_tokens=GOLD_MAX_OUTPUT_TOKENS,
        metadata={
            "prompt_id": prompt_id,
            "response_id": response_id,
            "prompt_version": prompt_version,
        },
    )
    line = {
        "custom_id": custom_id,
        "method": "POST",
        "url": BATCH_ENDPOINT,
        "body": _responses_payload(REQUESTED_MODEL, request),
    }
    return line, {"custom_id": custom_id, **identity}


def prepare_gold_batch(run_root: Path, private_gt: Path, schema_path: Path) -> dict[str, Any]:
    """Prepare private Batch inputs only for unique selected responses."""

    stage_root = run_root / "audit-gold-minimum-private"
    package_path = run_root / "export-audit-package-minimum" / "audit_package.jsonl"
    for path in (package_path, private_gt, schema_path):
        if not path.is_file():
            raise MinimumGoldError(f"missing hidden-gold input: {path}")
    schema = read_json(schema_path)
    gold_by_prompt = {str(row["prompt_id"]): row["gold_rubric"] for row in _jsonl(private_gt)}
    lines: list[dict[str, Any]] = []
    mapping: list[dict[str, Any]] = []
    for row in _jsonl(package_path):
        prompt_id = str(row["prompt_id"])
        response_id = str(row["response_id"])
        response_text = str(row["response_text"])
        if hashlib.sha256(response_text.encode()).hexdigest() != row["response_text_hash"]:
            raise MinimumGoldError(f"audit package response hash mismatch: {response_id}")
        if prompt_id not in gold_by_prompt:
            raise MinimumGoldError(f"private GT has no prompt: {prompt_id}")
        line, identity = _request(
            prompt_id,
            response_id,
            response_text,
            gold_by_prompt[prompt_id],
            schema,
        )
        lines.append(line)
        mapping.append(identity)
    if len({row["custom_id"] for row in lines}) != len(lines):
        raise MinimumGoldError("hidden-gold Batch custom_id collision")
    input_records = []
    for index, rows in enumerate(_shard(lines), 1):
        path = stage_root / "inputs" / f"hidden-gold-{index:03d}.jsonl"
        _publish_jsonl(path, rows)
        input_records.append(artifact_record(path))
    mapping_path = stage_root / "request_map.jsonl"
    _publish_jsonl(mapping_path, mapping)
    manifest = {
        "schema_version": 1,
        "private_process": True,
        "requested_model": REQUESTED_MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "prompt_version": PROMPT_VERSION,
        "max_output_tokens": GOLD_MAX_OUTPUT_TOKENS,
        "endpoint": BATCH_ENDPOINT,
        "completion_window": BATCH_COMPLETION_WINDOW,
        "requests": len(lines),
        "input_files": input_records,
        "request_map": artifact_record(mapping_path),
        "audit_package": artifact_record(package_path),
        "private_gt_sha256": sha256_file(private_gt),
        "schema_sha256": sha256_file(schema_path),
    }
    write_json_atomic(stage_root / "manifest.json", manifest)
    return manifest


def submit_gold_batch(run_root: Path) -> dict[str, Any]:
    """Upload prepared private shards and create 24-hour GPT-5 Batch jobs."""

    stage_root = run_root / "audit-gold-minimum-private"
    manifest_path = stage_root / "manifest.json"
    manifest = read_json(manifest_path)
    approval_sha256 = _egress_approval_sha256(run_root)
    receipt_path = stage_root / "submission.json"
    if receipt_path.is_file():
        receipt = read_json(receipt_path)
        if receipt.get("egress_approval_sha256") != approval_sha256:
            raise MinimumGoldError("Batch receipt is not bound to current egress approval")
        return receipt
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise MinimumGoldError("OPENAI_API_KEY is not set")
    try:
        from openai import OpenAI  # pyright: ignore[reportMissingImports]
    except ImportError as error:
        raise MinimumGoldError("OpenAI Python SDK is required") from error
    client = OpenAI(api_key=api_key)
    jobs = []
    for index, record in enumerate(manifest["input_files"], 1):
        input_path = Path(str(record["path"]))
        if sha256_file(input_path) != record["sha256"]:
            raise MinimumGoldError(f"prepared Batch input changed: {input_path}")
        shard_receipt = stage_root / f"submission-{index:03d}.json"
        if shard_receipt.is_file():
            job = read_json(shard_receipt)
            if (
                int(job.get("shard", -1)) != index
                or job.get("input_path") != str(input_path)
                or job.get("input_sha256") != record["sha256"]
                or not job.get("batch_id")
            ):
                raise MinimumGoldError(
                    f"existing Batch shard receipt does not match input: {shard_receipt}"
                )
            jobs.append(job)
            continue
        with input_path.open("rb") as stream:
            uploaded = _as_dict(client.files.create(file=stream, purpose="batch"))
        created = _as_dict(
            client.batches.create(
                input_file_id=str(uploaded["id"]),
                endpoint=BATCH_ENDPOINT,
                completion_window=BATCH_COMPLETION_WINDOW,
                metadata={"experiment": "minimum-staleness", "shard": str(index)},
            )
        )
        job = {
            "shard": index,
            "input_path": str(input_path),
            "input_sha256": record["sha256"],
            "input_file_id": str(uploaded["id"]),
            "batch_id": str(created["id"]),
            "status": str(created.get("status", "validating")),
            "created_at": created.get("created_at"),
        }
        write_json_atomic(shard_receipt, job)
        jobs.append(job)
    receipt = {
        "schema_version": 1,
        "manifest_sha256": sha256_file(manifest_path),
        "egress_approval_sha256": approval_sha256,
        "jobs": jobs,
    }
    write_json_atomic(receipt_path, receipt)
    return receipt


def gold_batch_status(run_root: Path) -> dict[str, Any]:
    stage_root = run_root / "audit-gold-minimum-private"
    receipt = read_json(stage_root / "submission.json")
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise MinimumGoldError("OPENAI_API_KEY is not set")
    try:
        from openai import OpenAI  # pyright: ignore[reportMissingImports]
    except ImportError as error:
        raise MinimumGoldError("OpenAI Python SDK is required") from error
    client = OpenAI(api_key=api_key)
    jobs = []
    for job in receipt["jobs"]:
        current = _as_dict(client.batches.retrieve(str(job["batch_id"])))
        jobs.append(
            {
                "batch_id": str(current["id"]),
                "status": str(current["status"]),
                "request_counts": current.get("request_counts"),
                "output_file_id": current.get("output_file_id"),
                "error_file_id": current.get("error_file_id"),
                "expires_at": current.get("expires_at"),
            }
        )
    status = {"jobs": jobs}
    write_json_atomic(stage_root / "status.json", status, immutable=False)
    return status


def collect_gold_batch(run_root: Path, schema_path: Path) -> dict[str, Any]:
    """Collect completed GPT-5 jobs and compute signed HealthBench scores privately."""

    stage_root = run_root / "audit-gold-minimum-private"
    manifest = read_json(stage_root / "manifest.json")
    receipt = read_json(stage_root / "submission.json")
    identities = {
        str(row["custom_id"]): row for row in _jsonl(Path(manifest["request_map"]["path"]))
    }
    if len(identities) != int(manifest["requests"]):
        raise MinimumGoldError("hidden-gold request map inventory mismatch")
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise MinimumGoldError("OPENAI_API_KEY is not set")
    try:
        from openai import OpenAI  # pyright: ignore[reportMissingImports]
    except ImportError as error:
        raise MinimumGoldError("OpenAI Python SDK is required") from error
    client = OpenAI(api_key=api_key)
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    returned_models: set[str] = set()
    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for job in receipt["jobs"]:
        current = _as_dict(client.batches.retrieve(str(job["batch_id"])))
        if current.get("status") != "completed" or not current.get("output_file_id"):
            raise MinimumGoldError(
                f"hidden-gold Batch is not ready: {current.get('id')}={current.get('status')}"
            )
        shard = int(job["shard"])
        payload = _download_file(client, str(current["output_file_id"]))
        raw_path = stage_root / "outputs" / f"hidden-gold-{shard:03d}.raw.jsonl"
        write_bytes_atomic(raw_path, payload)
        expected = {str(row["custom_id"]) for row in _jsonl(Path(str(job["input_path"])))}
        shard_rows = [json.loads(line) for line in payload.splitlines() if line.strip()]
        actual = {str(row.get("custom_id", "")) for row in shard_rows}
        if actual != expected:
            raise MinimumGoldError(f"hidden-gold Batch output inventory mismatch: {shard}")
        current_normalized = []
        for row in shard_rows:
            custom_id = str(row["custom_id"])
            if custom_id in seen:
                raise MinimumGoldError(f"duplicate hidden-gold output: {custom_id}")
            seen.add(custom_id)
            if row.get("error") is not None:
                raise MinimumGoldError(f"hidden-gold request failed: {custom_id}: {row['error']}")
            response = row.get("response")
            if not isinstance(response, Mapping) or int(response.get("status_code", 0)) != 200:
                raise MinimumGoldError(f"hidden-gold response is not HTTP 200: {custom_id}")
            body = response.get("body")
            if not isinstance(body, Mapping):
                raise MinimumGoldError(f"hidden-gold response body is absent: {custom_id}")
            returned_model = str(body.get("model", ""))
            if not returned_model:
                raise MinimumGoldError(f"hidden-gold returned model is absent: {custom_id}")
            returned_models.add(returned_model)
            try:
                parsed = json.loads(_output_text(body))
            except json.JSONDecodeError as error:
                raise MinimumGoldError(f"hidden-gold output is not JSON: {custom_id}") from error
            criterion_rows = parsed.get("criterion_scores") if isinstance(parsed, Mapping) else None
            if not isinstance(criterion_rows, list):
                raise MinimumGoldError(f"hidden-gold criterion_scores absent: {custom_id}")
            score_map: dict[str, float] = {}
            for item in criterion_rows:
                if not isinstance(item, Mapping):
                    raise MinimumGoldError(f"hidden-gold criterion score malformed: {custom_id}")
                criterion_id = str(item.get("criterion_id", ""))
                raw_score = item.get("score")
                if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
                    raise MinimumGoldError(
                        f"hidden-gold criterion score is non-numeric: {custom_id}"
                    )
                score = float(raw_score)
                if criterion_id in score_map:
                    raise MinimumGoldError(f"duplicate gold criterion score: {custom_id}")
                score_map[criterion_id] = score
            identity = identities[custom_id]
            weights = {
                str(key): float(value) for key, value in identity["criterion_weights"].items()
            }
            gold_score = weighted_gold_score(score_map, weights)
            body_usage = body.get("usage", {})
            if isinstance(body_usage, Mapping):
                for key in usage:
                    usage[key] += int(body_usage.get(key, 0) or 0)
            current_normalized.append(
                {
                    "prompt_id": identity["prompt_id"],
                    "response_id": identity["response_id"],
                    "gold_score": gold_score,
                    "criterion_scores": score_map,
                    "cache_key": gold_cache_key(
                        prompt_id=str(identity["prompt_id"]),
                        response_text_hash=str(identity["response_text_hash"]),
                        gold_rubric_hash=str(identity["gold_rubric_hash"]),
                        requested_model=REQUESTED_MODEL,
                        returned_model=returned_model,
                        grader_prompt_hash=hashlib.sha256(PROMPT_VERSION.encode()).hexdigest(),
                        schema_hash=sha256_file(schema_path),
                        reasoning_effort=REASONING_EFFORT,
                    ),
                    "requested_model": REQUESTED_MODEL,
                    "returned_model": returned_model,
                    "reasoning_effort": REASONING_EFFORT,
                    "provider_request_id": str(body.get("id") or response.get("request_id") or ""),
                }
            )
        normalized_path = stage_root / "outputs" / f"hidden-gold-{shard:03d}.jsonl"
        _publish_jsonl(normalized_path, current_normalized)
        normalized.extend(current_normalized)
        error_file_id = current.get("error_file_id")
        if error_file_id:
            error_payload = _download_file(client, str(error_file_id))
            error_path = stage_root / "outputs" / f"hidden-gold-{shard:03d}.errors.jsonl"
            write_bytes_atomic(error_path, error_payload)
            if error_payload.strip():
                raise MinimumGoldError(f"hidden-gold Batch shard {shard} has errors")
    if seen != set(identities):
        raise MinimumGoldError("combined hidden-gold output inventory is incomplete")
    if len(returned_models) != 1:
        raise MinimumGoldError(f"hidden-gold returned-model drift: {sorted(returned_models)}")
    normalized.sort(key=lambda row: (str(row["prompt_id"]), str(row["response_id"])))
    output_path = stage_root / "gold_scores.jsonl"
    _publish_jsonl(output_path, normalized)
    result = {
        "grader_calls": len(normalized),
        "returned_model": next(iter(returned_models)),
        "usage": usage,
        "output": artifact_record(output_path),
    }
    write_json_atomic(stage_root / "result.json", result)
    return result
