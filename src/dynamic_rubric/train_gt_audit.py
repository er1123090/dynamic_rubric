"""Post-hoc GPT-5 mini hidden-GT audit for on-policy training rollouts."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .artifacts import (
    artifact_record,
    read_json,
    read_jsonl,
    validate_artifact_record,
    write_bytes_atomic,
    write_json_atomic,
    write_jsonl_atomic,
    write_text_atomic,
)
from .batch_dynamic import (
    BATCH_COMPLETION_WINDOW,
    BATCH_ENDPOINT,
    _as_dict,
    _download_file,
    _responses_payload,
    _shard,
)
from .evaluation.gold_score import gold_cache_key, weighted_gold_score
from .hashing import canonical_json_bytes, sha256_file, sha256_json
from .judge_prompts import (
    PAPER_JUDGE_PROMPT_VERSION,
    PAPER_JUDGE_SYSTEM_PROMPT,
    gold_judge_user_prompt,
)
from .minimum_gold import _criterion_inventory
from .providers.base import GenerationRequest
from .providers.openai_responses import _output_text


DEFAULT_MODEL = "gpt-5-mini"
DEFAULT_REASONING_EFFORT = "medium"
DEFAULT_MAX_OUTPUT_TOKENS = 8192
APPROVED_PAYLOAD_CATEGORIES = (
    "training_medical_response_texts",
    "public_healthbench_conversations",
    "private_physician_rubrics",
    "criterion_weights",
)
APPROVAL_DESTINATION = "OpenAI Batch API"
APPROVAL_PURPOSE = "train-reward-vs-hidden-gt-evaluation"


class TrainGoldAuditError(RuntimeError):
    """Raised when train/GT audit provenance or inventory is incomplete."""


def stage_root(run_root: Path, model: str = DEFAULT_MODEL) -> Path:
    normalized = "".join(character if character.isalnum() else "-" for character in model)
    return run_root / f"audit-train-gt-{normalized}-private"


def _project_root(run_root: Path) -> Path:
    resolved = run_root.resolve()
    try:
        return resolved.parents[2]
    except IndexError as error:
        raise TrainGoldAuditError(f"run root is too shallow: {run_root}") from error


def _records(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise TrainGoldAuditError(f"JSONL row is not an object: {path}")
                yield row


def _train_conversations(project_root: Path) -> dict[str, tuple[dict[str, Any], ...]]:
    path = project_root / "data" / "public" / "pilot_train.jsonl"
    if not path.is_file():
        raise TrainGoldAuditError(f"missing public train conversations: {path}")
    conversations: dict[str, tuple[dict[str, Any], ...]] = {}
    for row in _records(path):
        prompt_id = str(row["prompt_id"])
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages:
            raise TrainGoldAuditError(f"malformed train conversation: {prompt_id}")
        normalized = tuple(dict(message) for message in messages if isinstance(message, Mapping))
        if len(normalized) != len(messages):
            raise TrainGoldAuditError(f"non-object train message: {prompt_id}")
        if prompt_id in conversations:
            raise TrainGoldAuditError(f"duplicate train conversation: {prompt_id}")
        conversations[prompt_id] = normalized
    return conversations


def _approval_sha256(approval_path: Path, model: str) -> str:
    if not approval_path.is_file():
        raise TrainGoldAuditError(f"missing explicit egress approval: {approval_path}")
    approval = read_json(approval_path)
    expected = {
        "approved": True,
        "destination": APPROVAL_DESTINATION,
        "endpoint": BATCH_ENDPOINT,
        "purpose": APPROVAL_PURPOSE,
        "requested_model": model,
        "payload_categories": list(APPROVED_PAYLOAD_CATEGORIES),
    }
    if any(approval.get(key) != value for key, value in expected.items()):
        raise TrainGoldAuditError("train hidden-GT egress approval scope does not match submission")
    return sha256_file(approval_path)


def _evaluation_response_id(
    *,
    step: int,
    prompt_id: str,
    replicate_index: int,
    response_text: str,
) -> str:
    identity = {
        "policy_step": step,
        "prompt_id": prompt_id,
        "replicate_index": replicate_index,
        "response_text_sha256": hashlib.sha256(response_text.encode()).hexdigest(),
    }
    return f"train-gt-{sha256_json(identity)}"


def _request(
    *,
    model: str,
    reasoning_effort: str,
    schema: Mapping[str, Any],
    conversation: Sequence[Mapping[str, Any]],
    gold_rubric: Any,
    prompt_id: str,
    policy_step: int,
    replicate_index: int,
    response_text: str,
    proxy_reward: float,
    source_response_id: str,
    source_sample_index: int,
    source_logical_seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    criteria, weights = _criterion_inventory(gold_rubric)
    evaluation_response_id = _evaluation_response_id(
        step=policy_step,
        prompt_id=prompt_id,
        replicate_index=replicate_index,
        response_text=response_text,
    )
    identity = {
        "prompt_id": prompt_id,
        "policy_step": policy_step,
        "replicate_index": replicate_index,
        "evaluation_response_id": evaluation_response_id,
        "source_response_id": source_response_id,
        "source_sample_index": source_sample_index,
        "source_logical_seed": source_logical_seed,
        "response_text_hash": hashlib.sha256(response_text.encode()).hexdigest(),
        "conversation_hash": hashlib.sha256(canonical_json_bytes(conversation)).hexdigest(),
        "gold_rubric_hash": sha256_json(gold_rubric),
        "criterion_weights": weights,
        "proxy_reward": proxy_reward,
        "grader_prompt_version": PAPER_JUDGE_PROMPT_VERSION,
    }
    custom_id = f"train-gold-{hashlib.sha256(canonical_json_bytes(identity)).hexdigest()[:32]}"
    request = GenerationRequest(
        prompt_id=f"{prompt_id}-train-hidden-gold-s{policy_step}",
        messages=(
            {"role": "system", "content": PAPER_JUDGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": gold_judge_user_prompt(conversation, response_text, criteria),
            },
        ),
        family="train_hidden_gold_audit",
        seed=0,
        json_schema=schema,
        schema_name="paper_gold_grader_v1",
        reasoning_effort=reasoning_effort,
        max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
        metadata={
            "prompt_id": prompt_id,
            "policy_step": policy_step,
            "evaluation_response_id": evaluation_response_id,
        },
    )
    line = {
        "custom_id": custom_id,
        "method": "POST",
        "url": BATCH_ENDPOINT,
        "body": _responses_payload(model, request),
    }
    return line, {"custom_id": custom_id, **identity}


def prepare_train_gold_batch(
    run_root: Path,
    private_gt: Path,
    schema_path: Path,
    approval_path: Path,
    *,
    model: str = DEFAULT_MODEL,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    max_step: int = 50,
    expected_rows_per_step: int = 384,
    expected_outputs_per_prompt: int = 8,
) -> dict[str, Any]:
    """Prepare immutable Batch inputs for every exported on-policy train response."""

    if max_step < 1 or expected_rows_per_step < 1 or expected_outputs_per_prompt < 1:
        raise ValueError("train GT inventory limits must be positive")
    run_root = run_root.resolve()
    private_gt = private_gt.resolve()
    schema_path = schema_path.resolve()
    approval_path = approval_path.resolve()
    for path in (private_gt, schema_path, approval_path):
        if not path.is_file():
            raise TrainGoldAuditError(f"missing train GT input: {path}")
    approval_sha256 = _approval_sha256(approval_path, model)
    schema = read_json(schema_path)
    gold_by_prompt = {
        str(row["prompt_id"]): row["gold_rubric"] for row in _records(private_gt)
    }
    conversations = _train_conversations(_project_root(run_root))
    rollout_root = run_root / "train-static" / "verl-run" / "rollouts"
    lines: list[dict[str, Any]] = []
    mapping: list[dict[str, Any]] = []
    source_files: list[dict[str, Any]] = []
    source_response_ids: set[str] = set()

    for step in range(1, max_step + 1):
        source_path = rollout_root / f"{step}.jsonl"
        if not source_path.is_file():
            raise TrainGoldAuditError(f"missing train rollout export: {source_path}")
        prompt_counts: dict[str, int] = defaultdict(int)
        step_rows = 0
        for source_line, row in enumerate(_records(source_path), start=1):
            if int(row.get("policy_step", -1)) != step:
                raise TrainGoldAuditError(f"policy step mismatch: {source_path}:{source_line}")
            prompt_id = str(row["prompt_id"])
            response_text = str(row.get("output", ""))
            if not response_text.strip():
                raise TrainGoldAuditError(f"empty train response: {source_path}:{source_line}")
            if prompt_id not in conversations or prompt_id not in gold_by_prompt:
                raise TrainGoldAuditError(f"train prompt lacks conversation or GT rubric: {prompt_id}")
            replicate_index = prompt_counts[prompt_id]
            prompt_counts[prompt_id] += 1
            source_response_id = str(row["response_id"])
            source_response_ids.add(source_response_id)
            line, identity = _request(
                model=model,
                reasoning_effort=reasoning_effort,
                schema=schema,
                conversation=conversations[prompt_id],
                gold_rubric=gold_by_prompt[prompt_id],
                prompt_id=prompt_id,
                policy_step=step,
                replicate_index=replicate_index,
                response_text=response_text,
                proxy_reward=float(row["static_reward"]),
                source_response_id=source_response_id,
                source_sample_index=int(row["sample_index"]),
                source_logical_seed=int(row["logical_seed"]),
            )
            identity["source_file"] = str(source_path)
            identity["source_line"] = source_line
            lines.append(line)
            mapping.append(identity)
            step_rows += 1
        if step_rows != expected_rows_per_step:
            raise TrainGoldAuditError(
                f"step {step} has {step_rows} train rows, expected {expected_rows_per_step}"
            )
        wrong = {
            prompt_id: count
            for prompt_id, count in prompt_counts.items()
            if count != expected_outputs_per_prompt
        }
        if wrong:
            raise TrainGoldAuditError(f"step {step} has unexpected prompt multiplicity: {wrong}")
        source_files.append(artifact_record(source_path))

    if len({str(row["custom_id"]) for row in lines}) != len(lines):
        raise TrainGoldAuditError("train hidden-GT Batch custom_id collision")
    if len({str(row["evaluation_response_id"]) for row in mapping}) != len(mapping):
        raise TrainGoldAuditError("train hidden-GT evaluation response ID collision")

    output_root = stage_root(run_root, model)
    input_files: list[dict[str, Any]] = []
    for index, rows in enumerate(_shard(lines), start=1):
        input_path = output_root / "inputs" / f"train-gold-{index:03d}.jsonl"
        write_jsonl_atomic(input_path, rows)
        input_files.append(artifact_record(input_path))
    request_map = output_root / "request_map.jsonl"
    write_jsonl_atomic(request_map, mapping)
    manifest = {
        "schema_version": 1,
        "private_process": True,
        "requested_model": model,
        "reasoning_effort": reasoning_effort,
        "prompt_version": PAPER_JUDGE_PROMPT_VERSION,
        "max_output_tokens": DEFAULT_MAX_OUTPUT_TOKENS,
        "endpoint": BATCH_ENDPOINT,
        "completion_window": BATCH_COMPLETION_WINDOW,
        "steps": max_step,
        "requests": len(lines),
        "unique_source_response_ids": len(source_response_ids),
        "evaluation_response_ids": len(mapping),
        "input_files": input_files,
        "request_map": artifact_record(request_map),
        "source_files": source_files,
        "private_gt": artifact_record(private_gt),
        "schema": artifact_record(schema_path),
        "egress_approval": artifact_record(approval_path),
        "egress_approval_sha256": approval_sha256,
    }
    write_json_atomic(output_root / "manifest.json", manifest)
    return manifest


def _client():
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise TrainGoldAuditError("OPENAI_API_KEY is not set")
    try:
        from openai import OpenAI  # pyright: ignore[reportMissingImports]
    except ImportError as error:
        raise TrainGoldAuditError("OpenAI Python SDK is required") from error
    return OpenAI(api_key=api_key)


def submit_train_gold_batch(
    run_root: Path, approval_path: Path, *, model: str = DEFAULT_MODEL
) -> dict[str, Any]:
    output_root = stage_root(run_root.resolve(), model)
    manifest_path = output_root / "manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("requested_model") != model:
        raise TrainGoldAuditError("prepared train GT model differs from requested submission")
    approval_sha256 = _approval_sha256(approval_path.resolve(), model)
    if manifest.get("egress_approval_sha256") != approval_sha256:
        raise TrainGoldAuditError("prepared train GT inputs are not bound to current approval")
    receipt_path = output_root / "submission.json"
    if receipt_path.is_file():
        receipt = read_json(receipt_path)
        if receipt.get("manifest_sha256") != sha256_file(manifest_path):
            raise TrainGoldAuditError("existing train GT receipt has manifest drift")
        return receipt

    client = _client()
    jobs: list[dict[str, Any]] = []
    for index, record in enumerate(manifest["input_files"], start=1):
        validate_artifact_record(record)
        input_path = Path(str(record["path"]))
        shard_receipt = output_root / f"submission-{index:03d}.json"
        if shard_receipt.is_file():
            jobs.append(read_json(shard_receipt))
            continue
        with input_path.open("rb") as stream:
            uploaded = _as_dict(client.files.create(file=stream, purpose="batch"))
        created = _as_dict(
            client.batches.create(
                input_file_id=str(uploaded["id"]),
                endpoint=BATCH_ENDPOINT,
                completion_window=BATCH_COMPLETION_WINDOW,
                metadata={"experiment": "train-reward-vs-gt", "shard": str(index)},
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


def train_gold_batch_status(run_root: Path, *, model: str = DEFAULT_MODEL) -> dict[str, Any]:
    output_root = stage_root(run_root.resolve(), model)
    receipt = read_json(output_root / "submission.json")
    client = _client()
    jobs = []
    for job in receipt["jobs"]:
        current = _as_dict(client.batches.retrieve(str(job["batch_id"])))
        jobs.append(
            {
                "shard": int(job["shard"]),
                "batch_id": str(current["id"]),
                "status": str(current["status"]),
                "request_counts": current.get("request_counts"),
                "output_file_id": current.get("output_file_id"),
                "error_file_id": current.get("error_file_id"),
                "expires_at": current.get("expires_at"),
            }
        )
    status = {"jobs": jobs}
    write_json_atomic(output_root / "status.json", status, immutable=False)
    return status


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        raise TrainGoldAuditError("correlation requires paired non-trivial samples")
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    centered_x = [value - mean_x for value in xs]
    centered_y = [value - mean_y for value in ys]
    denominator = math.sqrt(
        math.fsum(value * value for value in centered_x)
        * math.fsum(value * value for value in centered_y)
    )
    if denominator == 0.0:
        raise TrainGoldAuditError("correlation is undefined for a constant series")
    return math.fsum(x * y for x, y in zip(centered_x, centered_y, strict=True)) / denominator


def summarize_train_gold_scores(
    rows: Sequence[Mapping[str, Any]], output_path: Path
) -> dict[str, Any]:
    grouped: dict[int, dict[str, list[Mapping[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        grouped[int(row["policy_step"])][str(row["prompt_id"])].append(row)
    summaries: list[dict[str, Any]] = []
    for step, prompt_rows in sorted(grouped.items()):
        proxy_prompt_means = [
            statistics.fmean(float(row["proxy_reward"]) for row in values)
            for values in prompt_rows.values()
        ]
        gold_prompt_means = [
            statistics.fmean(float(row["gold_score"]) for row in values)
            for values in prompt_rows.values()
        ]
        sample_count = sum(len(values) for values in prompt_rows.values())
        prompt_count = len(prompt_rows)
        if prompt_count < 2:
            raise TrainGoldAuditError(f"step {step} needs at least two prompts for a CI")
        summaries.append(
            {
                "step": step,
                "train_proxy_mean": statistics.fmean(proxy_prompt_means),
                "train_proxy_ci95": 1.96
                * statistics.stdev(proxy_prompt_means)
                / math.sqrt(prompt_count),
                "train_gt_mean": statistics.fmean(gold_prompt_means),
                "train_gt_ci95": 1.96
                * statistics.stdev(gold_prompt_means)
                / math.sqrt(prompt_count),
                "sample_count": sample_count,
                "prompt_count": prompt_count,
            }
        )
    if not summaries:
        raise TrainGoldAuditError("no train GT scores to summarize")
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(summaries[0]))
    writer.writeheader()
    writer.writerows(summaries)
    write_text_atomic(output_path, buffer.getvalue())
    proxy_rows = [float(row["proxy_reward"]) for row in rows]
    gold_rows = [float(row["gold_score"]) for row in rows]
    proxy_steps = [float(row["train_proxy_mean"]) for row in summaries]
    gold_steps = [float(row["train_gt_mean"]) for row in summaries]
    return {
        "rows": len(rows),
        "steps": len(summaries),
        "response_level_pearson": _pearson(proxy_rows, gold_rows),
        "step_mean_pearson": _pearson(proxy_steps, gold_steps),
        "first_step": summaries[0],
        "last_step": summaries[-1],
        "output": artifact_record(output_path),
    }


def collect_train_gold_batch(
    run_root: Path, schema_path: Path, *, model: str = DEFAULT_MODEL
) -> dict[str, Any]:
    output_root = stage_root(run_root.resolve(), model)
    manifest = read_json(output_root / "manifest.json")
    receipt = read_json(output_root / "submission.json")
    if manifest.get("requested_model") != model:
        raise TrainGoldAuditError("train GT manifest model drift")
    identities = {
        str(row["custom_id"]): row
        for row in read_jsonl(Path(str(manifest["request_map"]["path"])))
    }
    if len(identities) != int(manifest["requests"]):
        raise TrainGoldAuditError("train GT request map inventory mismatch")
    client = _client()
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    returned_models: set[str] = set()
    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for job in receipt["jobs"]:
        current = _as_dict(client.batches.retrieve(str(job["batch_id"])))
        if current.get("status") != "completed" or not current.get("output_file_id"):
            raise TrainGoldAuditError(
                f"train GT Batch is not ready: {current.get('id')}={current.get('status')}"
            )
        shard = int(job["shard"])
        payload = _download_file(client, str(current["output_file_id"]))
        raw_path = output_root / "outputs" / f"train-gold-{shard:03d}.raw.jsonl"
        write_bytes_atomic(raw_path, payload)
        expected = {
            str(row["custom_id"]) for row in _records(Path(str(job["input_path"])))
        }
        shard_rows = [json.loads(line) for line in payload.splitlines() if line.strip()]
        actual = {str(row.get("custom_id", "")) for row in shard_rows}
        if actual != expected:
            raise TrainGoldAuditError(f"train GT Batch output inventory mismatch: {shard}")
        current_normalized: list[dict[str, Any]] = []
        for row in shard_rows:
            custom_id = str(row["custom_id"])
            if custom_id in seen:
                raise TrainGoldAuditError(f"duplicate train GT output: {custom_id}")
            seen.add(custom_id)
            if row.get("error") is not None:
                raise TrainGoldAuditError(f"train GT request failed: {custom_id}: {row['error']}")
            response = row.get("response")
            if not isinstance(response, Mapping) or int(response.get("status_code", 0)) != 200:
                raise TrainGoldAuditError(f"train GT response is not HTTP 200: {custom_id}")
            body = response.get("body")
            if not isinstance(body, Mapping):
                raise TrainGoldAuditError(f"train GT response body is absent: {custom_id}")
            returned_model = str(body.get("model", ""))
            if not returned_model:
                raise TrainGoldAuditError(f"train GT returned model is absent: {custom_id}")
            returned_models.add(returned_model)
            try:
                parsed = json.loads(_output_text(body))
            except json.JSONDecodeError as error:
                raise TrainGoldAuditError(f"train GT output is not JSON: {custom_id}") from error
            criterion_rows = parsed.get("criterion_scores") if isinstance(parsed, Mapping) else None
            if not isinstance(criterion_rows, list):
                raise TrainGoldAuditError(f"train GT criterion_scores absent: {custom_id}")
            score_map: dict[str, float] = {}
            for item in criterion_rows:
                if not isinstance(item, Mapping):
                    raise TrainGoldAuditError(f"malformed train GT criterion: {custom_id}")
                criterion_id = str(item.get("criterion_id", ""))
                raw_score = item.get("score")
                if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
                    raise TrainGoldAuditError(f"non-numeric train GT score: {custom_id}")
                if criterion_id in score_map:
                    raise TrainGoldAuditError(f"duplicate train GT criterion: {custom_id}")
                score_map[criterion_id] = float(raw_score)
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
                    "policy_step": int(identity["policy_step"]),
                    "prompt_id": str(identity["prompt_id"]),
                    "evaluation_response_id": str(identity["evaluation_response_id"]),
                    "source_response_id": str(identity["source_response_id"]),
                    "replicate_index": int(identity["replicate_index"]),
                    "proxy_reward": float(identity["proxy_reward"]),
                    "gold_score": gold_score,
                    "criterion_scores": score_map,
                    "cache_key": gold_cache_key(
                        prompt_id=str(identity["prompt_id"]),
                        response_text_hash=str(identity["response_text_hash"]),
                        gold_rubric_hash=str(identity["gold_rubric_hash"]),
                        requested_model=model,
                        returned_model=returned_model,
                        grader_prompt_hash=hashlib.sha256(
                            PAPER_JUDGE_PROMPT_VERSION.encode()
                        ).hexdigest(),
                        schema_hash=sha256_file(schema_path),
                        reasoning_effort=str(manifest["reasoning_effort"]),
                    ),
                    "requested_model": model,
                    "returned_model": returned_model,
                    "reasoning_effort": str(manifest["reasoning_effort"]),
                    "provider_request_id": str(
                        body.get("id") or response.get("request_id") or ""
                    ),
                }
            )
        normalized_path = output_root / "outputs" / f"train-gold-{shard:03d}.jsonl"
        write_jsonl_atomic(normalized_path, current_normalized)
        normalized.extend(current_normalized)
        error_file_id = current.get("error_file_id")
        if error_file_id:
            error_payload = _download_file(client, str(error_file_id))
            error_path = output_root / "outputs" / f"train-gold-{shard:03d}.errors.jsonl"
            write_bytes_atomic(error_path, error_payload)
            if error_payload.strip():
                raise TrainGoldAuditError(f"train GT Batch shard {shard} has errors")
    if seen != set(identities):
        raise TrainGoldAuditError("combined train GT inventory is incomplete")
    if len(returned_models) != 1:
        raise TrainGoldAuditError(f"train GT returned-model drift: {sorted(returned_models)}")
    normalized.sort(
        key=lambda row: (
            int(row["policy_step"]),
            str(row["prompt_id"]),
            int(row["replicate_index"]),
        )
    )
    scores_path = output_root / "gold_scores.jsonl"
    write_jsonl_atomic(scores_path, normalized)
    summary_path = output_root / "train_proxy_vs_gt_summary.csv"
    summary = summarize_train_gold_scores(normalized, summary_path)
    result = {
        "grader_calls": len(normalized),
        "returned_model": next(iter(returned_models)),
        "usage": usage,
        "scores": artifact_record(scores_path),
        "summary": summary,
    }
    write_json_atomic(output_root / "result.json", result)
    return result
