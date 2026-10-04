"""Crash-safe two-stage OpenAI Batch runner for horizon rubric construction.

The dependency boundary is deliberately small: request JSONL follows the official
Batch shape and callers may inject an SDK-compatible client for tests.  No API key
or client configuration is ever persisted in artifacts.
"""

from __future__ import annotations

from dataclasses import asdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Callable, Mapping, Sequence

from ..artifacts import (
    artifact_record,
    read_json,
    read_jsonl,
    validate_artifact_record,
    write_bytes_atomic,
    write_json_atomic,
    write_jsonl_atomic,
)
from ..batch_dynamic import (
    BATCH_COMPLETION_WINDOW,
    BATCH_ENDPOINT,
    _as_dict,
    _download_file,
    _responses_payload,
    _shard,
)
from ..hashing import canonical_json_bytes, sha256_file
from ..onlinerubric_batch import _openai_client
from ..pipeline import StageError
from ..providers.base import GenerationRequest
from ..providers.openai_responses import _output_text
from .contracts import CriterionType, ImportanceClass
from .controls import match_control_extension
from .extraction import prepare_dedup_request, prepare_extraction_requests
from .live_rubrics import _groups, criterion_from_artifact
from .rubric_refresh import (
    DedupCluster,
    ExtractionCandidate,
    Resolution,
    build_current_extension,
    resolve_dedup_cluster,
)


TERMINAL_FAILURE_STATUSES = frozenset({"failed", "expired", "cancelled"})
HORIZON_BATCH_SCHEMA_VERSION = 1
MAX_OUTPUT_TOKENS = 8192
DEDUP_MAX_OUTPUT_TOKENS = 16384
MAX_OUTPUT_RETRY_TOKENS = (8192, 16384)
FILE_PROCESSING_POLL_INTERVAL_SECONDS = 1.0
FILE_PROCESSING_TIMEOUT_SECONDS = 300.0
MAX_BATCH_SUBMISSION_ATTEMPTS = 3
HORIZON_MAX_BATCH_FILE_BYTES = 300_000
HORIZON_MAX_DEDUP_BATCH_FILE_BYTES = 2_000_000
HORIZON_MAX_BATCH_REQUESTS = 24


def _throttle_openai_request() -> None:
    """Apply an optional process-shared request budget before an OpenAI API call."""

    raw_limit = os.environ.get("DYNAMIC_RUBRIC_OPENAI_REQUESTS_PER_MINUTE", "")
    if not raw_limit:
        return
    try:
        requests_per_minute = float(raw_limit)
    except ValueError as error:
        raise StageError("DYNAMIC_RUBRIC_OPENAI_REQUESTS_PER_MINUTE must be numeric") from error
    if requests_per_minute <= 0:
        raise StageError("DYNAMIC_RUBRIC_OPENAI_REQUESTS_PER_MINUTE must be positive")
    interval_seconds = 60.0 / requests_per_minute
    lock_path = Path(
        os.environ.get(
            "DYNAMIC_RUBRIC_OPENAI_RATE_LIMIT_PATH",
            "/tmp/dynamic-rubric-openai-rate-limit.lock",
        )
    )
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        stream.seek(0)
        raw_last_request = stream.read().strip()
        last_request = float(raw_last_request) if raw_last_request else 0.0
        delay = max(0.0, last_request + interval_seconds - time.time())
        if delay:
            time.sleep(delay)
        stream.seek(0)
        stream.truncate()
        stream.write(f"{time.time():.9f}\n")
        stream.flush()
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _openai_call(operation: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    _throttle_openai_request()
    return operation(*args, **kwargs)


def _request_line(
    *,
    custom_id: str,
    model: str,
    messages: Sequence[Mapping[str, str]],
    family: str,
    seed: int,
    schema: Mapping[str, Any],
    schema_name: str,
    reasoning_effort: str,
    max_output_tokens: int | None = None,
) -> dict[str, Any]:
    if max_output_tokens is None:
        max_output_tokens = MAX_OUTPUT_TOKENS
    request = GenerationRequest(
        prompt_id=custom_id,
        messages=tuple(messages),
        family=family,
        seed=seed,
        max_output_tokens=max_output_tokens,
        json_schema=schema,
        schema_name=schema_name,
        reasoning_effort=reasoning_effort,
    )
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": BATCH_ENDPOINT,
        "body": _responses_payload(model, request),
    }


def _write_prepared_stage(
    stage_root: Path,
    *,
    run_id: str,
    checkpoint_id: str,
    phase: str,
    model: str,
    lines: Sequence[Mapping[str, Any]],
    identities: Sequence[Mapping[str, Any]],
    batch_sharding: bool = True,
    max_file_bytes: int = HORIZON_MAX_BATCH_FILE_BYTES,
) -> Mapping[str, Any]:
    if len(lines) != len(identities) or not lines:
        raise StageError(f"horizon {phase} Batch request inventory is empty or inconsistent")
    custom_ids = [str(line["custom_id"]) for line in lines]
    if len(set(custom_ids)) != len(custom_ids):
        raise StageError(f"horizon {phase} Batch custom_id collision")
    manifest_path = stage_root / "manifest.json"
    if manifest_path.is_file():
        manifest = read_json(manifest_path)
        expected = {
            "schema_version": HORIZON_BATCH_SCHEMA_VERSION,
            "run_id": run_id,
            "checkpoint_id": checkpoint_id,
            "phase": phase,
            "model": model,
            "endpoint": BATCH_ENDPOINT,
            "completion_window": BATCH_COMPLETION_WINDOW,
            "requests": len(lines),
        }
        if any(manifest.get(key) != value for key, value in expected.items()):
            raise StageError(f"existing horizon {phase} manifest is incompatible")
        for record in manifest.get("input_files", ()):
            validate_artifact_record(record)
        request_map = manifest.get("request_map")
        if not isinstance(request_map, Mapping):
            raise StageError(f"existing horizon {phase} request map is missing")
        validate_artifact_record(request_map)
        existing_custom_ids = [
            str(row["custom_id"]) for row in read_jsonl(Path(str(request_map["path"])))
        ]
        if existing_custom_ids != custom_ids:
            raise StageError(f"existing horizon {phase} request inventory is incompatible")
        return manifest
    input_paths = []
    shards = (tuple(lines),)
    if batch_sharding:
        shards = _shard(
            lines,
            max_file_bytes=max_file_bytes,
            max_requests=HORIZON_MAX_BATCH_REQUESTS,
        )
    for index, rows in enumerate(shards, start=1):
        path = stage_root / "inputs" / f"{phase}-{index:03d}.jsonl"
        write_jsonl_atomic(path, rows)
        input_paths.append(path)
    request_map_path = stage_root / "request_map.jsonl"
    write_jsonl_atomic(request_map_path, identities)
    manifest = {
        "schema_version": HORIZON_BATCH_SCHEMA_VERSION,
        "run_id": run_id,
        "checkpoint_id": checkpoint_id,
        "phase": phase,
        "model": model,
        "endpoint": BATCH_ENDPOINT,
        "completion_window": BATCH_COMPLETION_WINDOW,
        "requests": len(lines),
        "input_files": [artifact_record(path) for path in input_paths],
        "request_map": artifact_record(request_map_path),
    }
    write_json_atomic(manifest_path, manifest)
    return manifest


def _wait_for_uploaded_file(
    client: Any,
    file_id: str,
    *,
    sleep: Callable[[float], None],
    poll_interval_seconds: float = FILE_PROCESSING_POLL_INTERVAL_SECONDS,
    timeout_seconds: float = FILE_PROCESSING_TIMEOUT_SECONDS,
) -> None:
    """Wait until an uploaded Batch input is visible and fully processed."""

    if poll_interval_seconds < 0 or timeout_seconds <= 0:
        raise ValueError("invalid Batch file-processing wait configuration")
    deadline = time.monotonic() + timeout_seconds
    while True:
        uploaded = _as_dict(_openai_call(client.files.retrieve, file_id))
        status = str(uploaded.get("status", ""))
        if status == "processed":
            return
        if status in {"error", "deleted"}:
            raise StageError(f"horizon Batch input file entered terminal state: {status}")
        if time.monotonic() >= deadline:
            raise StageError(
                f"horizon Batch input file was not processed within {timeout_seconds:g}s"
            )
        sleep(poll_interval_seconds)


def _retryable_file_visibility_failure(job: Mapping[str, Any]) -> bool:
    if str(job.get("status")) != "failed":
        return False
    errors = job.get("errors")
    data = errors.get("data") if isinstance(errors, Mapping) else None
    if not isinstance(data, list) or not data:
        return False
    return all(
        isinstance(item, Mapping)
        and item.get("code") == "invalid_request"
        and item.get("param") == "file_id"
        and str(item.get("message", "")).startswith("Cannot find file ")
        for item in data
    )


def _submit_stage(
    client: Any, stage_root: Path, *, sleep: Callable[[float], None] = time.sleep
) -> Mapping[str, Any]:
    manifest_path = stage_root / "manifest.json"
    manifest = read_json(manifest_path)
    receipt_path = stage_root / "submission.json"
    if receipt_path.is_file():
        receipt = read_json(receipt_path)
        if receipt.get("manifest_sha256") != sha256_file(manifest_path):
            raise StageError("horizon Batch submission is not bound to the prepared manifest")
        current_jobs = [
            _as_dict(_openai_call(client.batches.retrieve, str(job["batch_id"])))
            for job in receipt["jobs"]
        ]
        if current_jobs and all(_retryable_file_visibility_failure(job) for job in current_jobs):
            attempt = int(receipt.get("submission_attempt", 1)) + 1
            if attempt <= MAX_BATCH_SUBMISSION_ATTEMPTS:
                jobs = []
                for prior_job in receipt["jobs"]:
                    shard = int(prior_job["shard"])
                    record = manifest["input_files"][shard - 1]
                    if prior_job.get("input_sha256") != record["sha256"]:
                        raise StageError("retrying horizon Batch submission input hash mismatch")
                    input_file_id = str(prior_job["input_file_id"])
                    _wait_for_uploaded_file(client, input_file_id, sleep=sleep)
                    created = _as_dict(
                        _openai_call(
                            client.batches.create,
                            input_file_id=input_file_id,
                            endpoint=BATCH_ENDPOINT,
                            completion_window=BATCH_COMPLETION_WINDOW,
                            metadata={
                                "run_id": str(manifest["run_id"]),
                                "checkpoint_id": str(manifest["checkpoint_id"]),
                                "phase": str(manifest["phase"]),
                                "shard": str(shard),
                            },
                        )
                    )
                    job = {
                        "shard": shard,
                        "input_path": str(prior_job["input_path"]),
                        "input_sha256": str(prior_job["input_sha256"]),
                        "input_file_id": input_file_id,
                        "batch_id": str(created["id"]),
                        "status": str(created.get("status", "validating")),
                    }
                    write_json_atomic(
                        stage_root / f"submission-attempt-{attempt:03d}-shard-{shard:03d}.json",
                        job,
                    )
                    jobs.append(job)
                receipt = {
                    **receipt,
                    "submission_attempt": attempt,
                    "jobs": jobs,
                }
                write_json_atomic(receipt_path, receipt, immutable=False)
        return receipt
    jobs = []
    for shard, record in enumerate(manifest["input_files"], start=1):
        shard_path = stage_root / f"submission-{shard:03d}.json"
        if shard_path.is_file():
            job = read_json(shard_path)
            if job.get("input_sha256") != record["sha256"]:
                raise StageError("existing horizon Batch submission input hash mismatch")
            jobs.append(job)
            continue
        input_path = Path(str(record["path"]))
        if sha256_file(input_path) != record["sha256"]:
            raise StageError(f"horizon Batch input changed after preparation: {input_path}")
        with input_path.open("rb") as stream:
            uploaded = _as_dict(_openai_call(client.files.create, file=stream, purpose="batch"))
        input_file_id = str(uploaded["id"])
        _wait_for_uploaded_file(client, input_file_id, sleep=sleep)
        created = _as_dict(
            _openai_call(
                client.batches.create,
                input_file_id=input_file_id,
                endpoint=BATCH_ENDPOINT,
                completion_window=BATCH_COMPLETION_WINDOW,
                metadata={
                    "run_id": str(manifest["run_id"]),
                    "checkpoint_id": str(manifest["checkpoint_id"]),
                    "phase": str(manifest["phase"]),
                    "shard": str(shard),
                },
            )
        )
        job = {
            "shard": shard,
            "input_path": str(input_path),
            "input_sha256": str(record["sha256"]),
            "input_file_id": input_file_id,
            "batch_id": str(created["id"]),
            "status": str(created.get("status", "validating")),
        }
        write_json_atomic(shard_path, job)
        jobs.append(job)
    receipt = {
        "schema_version": HORIZON_BATCH_SCHEMA_VERSION,
        "submission_attempt": 1,
        "run_id": manifest["run_id"],
        "checkpoint_id": manifest["checkpoint_id"],
        "phase": manifest["phase"],
        "manifest_sha256": sha256_file(manifest_path),
        "jobs": jobs,
    }
    write_json_atomic(receipt_path, receipt)
    return receipt


def _wait_for_stage(
    client: Any,
    stage_root: Path,
    *,
    poll_interval_seconds: float,
    sleep: Callable[[float], None],
) -> tuple[Mapping[str, Any], ...]:
    if poll_interval_seconds < 0:
        raise ValueError("poll_interval_seconds must be non-negative")
    receipt = read_json(stage_root / "submission.json")
    while True:
        current_jobs = tuple(
            _as_dict(_openai_call(client.batches.retrieve, str(job["batch_id"])))
            for job in receipt["jobs"]
        )
        status = {
            "phase": receipt["phase"],
            "jobs": [
                {
                    "batch_id": str(job["id"]),
                    "status": str(job["status"]),
                    "request_counts": job.get("request_counts"),
                    "output_file_id": job.get("output_file_id"),
                    "error_file_id": job.get("error_file_id"),
                    "expires_at": job.get("expires_at"),
                }
                for job in current_jobs
            ],
        }
        write_json_atomic(stage_root / "status.json", status, immutable=False)
        failed = [
            job for job in current_jobs if str(job.get("status")) in TERMINAL_FAILURE_STATUSES
        ]
        if failed:
            states = [(str(job.get("id")), str(job.get("status"))) for job in failed]
            raise StageError(f"horizon Batch entered terminal failure state: {states}")
        if all(
            job.get("status") == "completed" and job.get("output_file_id") for job in current_jobs
        ):
            return current_jobs
        sleep(poll_interval_seconds)


def _collect_stage(
    client: Any, stage_root: Path, jobs: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, Mapping[str, Any]], Mapping[str, Mapping[str, Any]]]:
    manifest = read_json(stage_root / "manifest.json")
    identities = {str(row["custom_id"]): row for row in read_jsonl(manifest["request_map"]["path"])}
    if len(identities) != int(manifest["requests"]):
        raise StageError("horizon Batch request identity inventory is incomplete")
    receipt = read_json(stage_root / "submission.json")
    by_batch_id = {str(job["id"]): job for job in jobs}
    outputs: dict[str, Mapping[str, Any]] = {}
    for submitted in receipt["jobs"]:
        current = by_batch_id[str(submitted["batch_id"])]
        shard = int(submitted["shard"])
        payload = _openai_call(_download_file, client, str(current["output_file_id"]))
        raw_path = stage_root / "outputs" / f"{manifest['phase']}-{shard:03d}.raw.jsonl"
        write_bytes_atomic(raw_path, payload)
        input_ids = {
            str(row["custom_id"]) for row in read_jsonl(Path(str(submitted["input_path"])))
        }
        rows = [json.loads(line) for line in payload.splitlines() if line.strip()]
        output_ids = {str(row.get("custom_id", "")) for row in rows}
        if output_ids != input_ids:
            raise StageError(f"horizon Batch output inventory mismatch for shard {shard}")
        for row in rows:
            custom_id = str(row["custom_id"])
            if custom_id in outputs:
                raise StageError(f"duplicate horizon Batch output: {custom_id}")
            outputs[custom_id] = row
        if current.get("error_file_id"):
            error_payload = _download_file(client, str(current["error_file_id"]))
            error_path = stage_root / "outputs" / f"{manifest['phase']}-{shard:03d}.errors.jsonl"
            write_bytes_atomic(error_path, error_payload)
            if error_payload.strip():
                raise StageError(f"horizon Batch shard {shard} produced errors")
    if set(outputs) != set(identities):
        raise StageError("combined horizon Batch output inventory is incomplete")
    return outputs, identities


def _max_output_incomplete_ids(
    outputs: Mapping[str, Mapping[str, Any]],
) -> tuple[str, ...]:
    pending = []
    for custom_id, row in outputs.items():
        response = row.get("response")
        body = response.get("body") if isinstance(response, Mapping) else None
        details = body.get("incomplete_details") if isinstance(body, Mapping) else None
        if (
            isinstance(body, Mapping)
            and body.get("status") == "incomplete"
            and isinstance(details, Mapping)
            and details.get("reason") == "max_output_tokens"
        ):
            pending.append(custom_id)
    return tuple(sorted(pending))


def _collect_stage_with_max_output_retries(
    client: Any,
    stage_root: Path,
    jobs: Sequence[Mapping[str, Any]],
    *,
    poll_interval_seconds: float,
    sleep: Callable[[float], None],
) -> tuple[dict[str, Mapping[str, Any]], Mapping[str, Mapping[str, Any]]]:
    outputs, identities = _collect_stage(client, stage_root, jobs)
    manifest = read_json(stage_root / "manifest.json")
    request_lines = {
        str(row["custom_id"]): row
        for record in manifest["input_files"]
        for row in read_jsonl(Path(str(record["path"])))
    }
    if set(request_lines) != set(identities):
        raise StageError("horizon Batch retry request inventory is incomplete")
    try:
        initial_limits = {int(row["body"]["max_output_tokens"]) for row in request_lines.values()}
    except (KeyError, TypeError, ValueError) as error:
        raise StageError("horizon Batch retry max_output_tokens is invalid") from error
    if len(initial_limits) != 1 or next(iter(initial_limits)) <= 0:
        raise StageError("horizon Batch retry initial token limit is inconsistent")
    initial_limit = next(iter(initial_limits))
    max_file_bytes = (
        HORIZON_MAX_DEDUP_BATCH_FILE_BYTES
        if str(manifest["phase"]).startswith("dedup")
        else HORIZON_MAX_BATCH_FILE_BYTES
    )
    for max_output_tokens in MAX_OUTPUT_RETRY_TOKENS:
        if max_output_tokens <= initial_limit:
            continue
        pending = _max_output_incomplete_ids(outputs)
        if not pending:
            break
        retry_lines = []
        retry_identities = []
        for custom_id in pending:
            line = json.loads(json.dumps(request_lines[custom_id]))
            body = line.get("body")
            if not isinstance(body, dict):
                raise StageError(f"horizon Batch retry body is invalid: {custom_id}")
            body["max_output_tokens"] = max_output_tokens
            retry_lines.append(line)
            retry_identities.append(
                {
                    "custom_id": custom_id,
                    "source_stage": str(stage_root),
                    "max_output_tokens": max_output_tokens,
                }
            )
        retry_root = stage_root / "retries" / f"max-output-{max_output_tokens}"
        _write_prepared_stage(
            retry_root,
            run_id=str(manifest["run_id"]),
            checkpoint_id=str(manifest["checkpoint_id"]),
            phase=f"{manifest['phase']}-max-output-{max_output_tokens}",
            model=str(manifest["model"]),
            lines=retry_lines,
            identities=retry_identities,
            max_file_bytes=max_file_bytes,
        )
        _submit_stage(client, retry_root, sleep=sleep)
        retry_jobs = _wait_for_stage(
            client,
            retry_root,
            poll_interval_seconds=poll_interval_seconds,
            sleep=sleep,
        )
        retry_outputs, _ = _collect_stage(client, retry_root, retry_jobs)
        outputs.update(retry_outputs)
    return outputs, identities


def _parsed_response(row: Mapping[str, Any], custom_id: str) -> tuple[Mapping[str, Any], Any]:
    if row.get("error") is not None:
        raise StageError(f"horizon Batch request failed: {custom_id}: {row['error']}")
    response = row.get("response")
    if not isinstance(response, Mapping) or int(response.get("status_code", 0)) != 200:
        raise StageError(f"horizon Batch response is not HTTP 200: {custom_id}")
    body = response.get("body")
    if not isinstance(body, Mapping) or body.get("status") not in (None, "completed"):
        raise StageError(f"horizon Batch response body is incomplete: {custom_id}")
    try:
        parsed = json.loads(_output_text(body))
    except json.JSONDecodeError as error:
        raise StageError(f"horizon Batch structured output is invalid JSON: {custom_id}") from error
    if not isinstance(parsed, Mapping):
        raise StageError(f"horizon Batch structured output is not an object: {custom_id}")
    return body, parsed


def _provider_call(body: Mapping[str, Any], model: str) -> dict[str, Any]:
    return {
        "requested_model": model,
        "returned_model": str(body.get("model", "")),
        "request_id": str(body.get("id", "")),
        "created_at": body.get("created_at", body.get("created")),
        "usage": dict(body.get("usage", {})),
        "raw_response_hash": hashlib.sha256(canonical_json_bytes(body)).hexdigest(),
    }


def _prepare_extraction(
    *,
    state_root: Path,
    run_id: str,
    prompts: Sequence[Mapping[str, Any]],
    current_rows: Sequence[Mapping[str, Any]],
    control_rows: Sequence[Mapping[str, Any]],
    checkpoint_id: str,
    pairing_seed: int,
    model: str,
    schema: Mapping[str, Any],
    reasoning_effort: str,
) -> tuple[Mapping[str, Any], ...]:
    current_by_prompt = _groups(current_rows)
    control_by_prompt = _groups(control_rows)
    lines = []
    identities = []
    requests = []
    for prompt in prompts:
        prompt_id = str(prompt["prompt_id"])
        prepared = prepare_extraction_requests(
            prompt_id=prompt_id,
            checkpoint_id=checkpoint_id,
            prompt=prompt["messages"],
            existing_r0=prompt["r0"]["criteria"],
            current_rows=current_by_prompt.get(prompt_id, ()),
            control_rows=control_by_prompt.get(prompt_id, ()),
            pairing_seed=pairing_seed,
        )
        for request_index, request in enumerate(prepared):
            custom_id = f"hex-{str(request['request_id']).removeprefix('hex_')[:48]}"
            lines.append(
                _request_line(
                    custom_id=custom_id,
                    model=model,
                    messages=request["messages"],
                    family="horizon_extraction",
                    seed=pairing_seed + request_index,
                    schema=schema,
                    schema_name="horizon_extraction_v1",
                    reasoning_effort=reasoning_effort,
                )
            )
            identities.append({"custom_id": custom_id, **request})
            requests.append(request)
    write_jsonl_atomic(state_root / "extraction_requests.jsonl", requests)
    _write_prepared_stage(
        state_root / "extraction",
        run_id=run_id,
        checkpoint_id=checkpoint_id,
        phase="extraction",
        model=model,
        lines=lines,
        identities=identities,
    )
    return tuple(identities)


def _parse_extraction_outputs(
    *,
    outputs: Mapping[str, Mapping[str, Any]],
    identities: Mapping[str, Mapping[str, Any]],
    state_root: Path,
    model: str,
) -> tuple[dict[str, list[ExtractionCandidate]], list[dict[str, Any]]]:
    if set(outputs) != set(identities):
        raise StageError("horizon extraction output inventory is incomplete")
    candidates_by_prompt: dict[str, list[ExtractionCandidate]] = {}
    candidate_rows = []
    for custom_id, identity in sorted(identities.items()):
        body, parsed = _parsed_response(outputs[custom_id], custom_id)
        criteria = parsed.get("new_criteria")
        if not isinstance(parsed.get("analysis"), str) or not isinstance(criteria, list):
            raise StageError(f"horizon extraction schema violation: {custom_id}")
        for criterion_index, item in enumerate(criteria):
            if not isinstance(item, Mapping):
                raise StageError(f"horizon extraction criterion is not an object: {custom_id}")
            extractor_candidate_id = str(item["candidate_id"])
            scoped_candidate_id = (
                "hc-"
                + hashlib.sha256(
                    canonical_json_bytes([custom_id, criterion_index, extractor_candidate_id])
                ).hexdigest()[:24]
            )
            candidate = ExtractionCandidate(
                candidate_id=scoped_candidate_id,
                criterion=str(item["criterion"]),
                evidence_quote=str(item["quote"]),
                source_pair_id=str(identity["source_pair_id"]),
                source_checkpoint=str(identity["checkpoint_id"]),
                raw_paper_weight=int(item["weight"]),
                importance_class=ImportanceClass(str(item["importance_class"])),
                criterion_type=CriterionType(str(item["criterion_type"])),
                response_a=str(identity["response_a"]),
                response_b=str(identity["response_b"]),
            )
            prompt_id = str(identity["prompt_id"])
            values = candidates_by_prompt.setdefault(prompt_id, [])
            if candidate.candidate_id in {value.candidate_id for value in values}:
                raise StageError("extractor candidate IDs must be unique per prompt/checkpoint")
            values.append(candidate)
            candidate_rows.append(
                {
                    "prompt_id": prompt_id,
                    **asdict(candidate),
                    "extractor_candidate_id": extractor_candidate_id,
                    "extractor_candidate_index": criterion_index,
                    "provider_call": _provider_call(body, model),
                }
            )
    write_jsonl_atomic(state_root / "extraction_candidates.jsonl", candidate_rows)
    return candidates_by_prompt, candidate_rows


def _collect_extraction(
    *,
    client: Any,
    state_root: Path,
    jobs: Sequence[Mapping[str, Any]],
    model: str,
    poll_interval_seconds: float,
    sleep: Callable[[float], None],
) -> tuple[dict[str, list[ExtractionCandidate]], list[dict[str, Any]]]:
    outputs, identities = _collect_stage_with_max_output_retries(
        client,
        state_root / "extraction",
        jobs,
        poll_interval_seconds=poll_interval_seconds,
        sleep=sleep,
    )
    return _parse_extraction_outputs(
        outputs=outputs, identities=identities, state_root=state_root, model=model
    )


def _prepare_dedup(
    *,
    state_root: Path,
    run_id: str,
    prompts: Sequence[Mapping[str, Any]],
    candidates_by_prompt: Mapping[str, Sequence[ExtractionCandidate]],
    checkpoint_id: str,
    pairing_seed: int,
    model: str,
    schema: Mapping[str, Any],
    reasoning_effort: str,
    batch_sharding: bool = True,
) -> None:
    lines = []
    identities = []
    for prompt in prompts:
        prompt_id = str(prompt["prompt_id"])
        candidates = tuple(candidates_by_prompt.get(prompt_id, ()))
        request = prepare_dedup_request(
            prompt_id=prompt_id,
            checkpoint_id=checkpoint_id,
            prompt=prompt["messages"],
            existing_r0=prompt["r0"]["criteria"],
            candidate_criteria=[asdict(item) for item in candidates],
        )
        digest = hashlib.sha256(
            canonical_json_bytes(
                [prompt_id, checkpoint_id, sorted(item.candidate_id for item in candidates)]
            )
        ).hexdigest()
        custom_id = f"hdd-{digest[:48]}"
        lines.append(
            _request_line(
                custom_id=custom_id,
                model=model,
                messages=request["messages"],
                family="horizon_dedup",
                seed=pairing_seed,
                schema=schema,
                schema_name="horizon_dedup_v1",
                reasoning_effort=reasoning_effort,
                max_output_tokens=DEDUP_MAX_OUTPUT_TOKENS,
            )
        )
        identities.append({"custom_id": custom_id, **request})
    _write_prepared_stage(
        state_root / "dedup",
        run_id=run_id,
        checkpoint_id=checkpoint_id,
        phase="dedup",
        model=model,
        lines=lines,
        identities=identities,
        batch_sharding=batch_sharding,
        max_file_bytes=HORIZON_MAX_DEDUP_BATCH_FILE_BYTES,
    )


def _finalize_outputs(
    *,
    outputs: Mapping[str, Mapping[str, Any]],
    identities: Mapping[str, Mapping[str, Any]],
    state_root: Path,
    prompts: Sequence[Mapping[str, Any]],
    candidates_by_prompt: Mapping[str, Sequence[ExtractionCandidate]],
    checkpoint_id: str,
    output_path: Path,
    max_online_criteria: int,
    control_rubric_rows: Sequence[Mapping[str, Any]],
    model: str,
    generation_method: str,
) -> Mapping[str, Any]:
    if set(outputs) != set(identities):
        raise StageError("horizon dedup output inventory is incomplete")
    prompt_index = {str(prompt["prompt_id"]): prompt for prompt in prompts}
    controls = {str(row["prompt_id"]): row for row in control_rubric_rows}
    rows = []
    returned_models = set()
    for custom_id, identity in sorted(identities.items()):
        body, parsed = _parsed_response(outputs[custom_id], custom_id)
        final_criteria = parsed.get("final_criteria")
        if not isinstance(parsed.get("analysis"), str) or not isinstance(final_criteria, list):
            raise StageError(f"horizon dedup schema violation: {custom_id}")
        prompt_id = str(identity["prompt_id"])
        candidates = {item.candidate_id: item for item in candidates_by_prompt.get(prompt_id, ())}
        resolutions = []
        for item in final_criteria:
            if not isinstance(item, Mapping):
                raise StageError(f"horizon dedup criterion is not an object: {custom_id}")
            source_candidate_ids = tuple(
                dict.fromkeys(str(value) for value in item["source_candidate_ids"])
            )
            known_source_candidate_ids = tuple(
                value for value in source_candidate_ids if value in candidates
            )
            if not known_source_candidate_ids:
                resolutions.append(Resolution(False, None, "unknown_source_candidate_ids"))
                continue
            resolutions.append(
                resolve_dedup_cluster(
                    DedupCluster(
                        str(item["criterion"]),
                        known_source_candidate_ids,
                    ),
                    candidates,
                    prompt_id=prompt_id,
                    checkpoint_id=checkpoint_id,
                )
            )
        prompt = prompt_index[prompt_id]
        extension, rejected = build_current_extension(
            resolutions,
            candidates,
            checkpoint_id=checkpoint_id,
            r0_texts=(str(item["criterion"]) for item in prompt["r0"]["criteria"]),
            max_count=max_online_criteria,
        )
        control_match = None
        if prompt_id in controls:
            available = tuple(
                criterion_from_artifact(item) for item in controls[prompt_id].get("extension", ())
            )
            control_match = match_control_extension(extension, available)
        provider_call = _provider_call(body, model)
        returned_models.add(str(provider_call["returned_model"]))
        rows.append(
            {
                "schema_version": 1,
                "prompt_id": prompt_id,
                "checkpoint_id": checkpoint_id,
                "r0": list(prompt["r0"]["criteria"]),
                "extension": [asdict(item) for item in extension],
                "control_extension": (
                    [asdict(item) for item in control_match.selected]
                    if control_match is not None and control_match.eligible
                    else None
                ),
                "control_match": asdict(control_match) if control_match is not None else None,
                "rejected": list(rejected),
                "pool_b_inputs": [],
                "dedup_provider_call": provider_call,
            }
        )
    if "" in returned_models or len(returned_models) != 1:
        raise StageError(f"horizon returned-model drift: {sorted(returned_models)}")
    write_jsonl_atomic(output_path, rows)
    result = {
        "generation_method": generation_method,
        "rubrics": str(output_path),
        "rubrics_artifact": artifact_record(output_path),
        "state_root": str(state_root),
        "prompt_count": len(rows),
        "request_count": len(read_jsonl(state_root / "extraction_requests.jsonl")),
        "candidate_count": sum(len(values) for values in candidates_by_prompt.values()),
        "returned_models": sorted(returned_models),
    }
    write_json_atomic(state_root / "result.json", result)
    return result


def _finalize(
    *,
    client: Any,
    state_root: Path,
    jobs: Sequence[Mapping[str, Any]],
    prompts: Sequence[Mapping[str, Any]],
    candidates_by_prompt: Mapping[str, Sequence[ExtractionCandidate]],
    checkpoint_id: str,
    output_path: Path,
    max_online_criteria: int,
    control_rubric_rows: Sequence[Mapping[str, Any]],
    model: str,
    poll_interval_seconds: float,
    sleep: Callable[[float], None],
) -> Mapping[str, Any]:
    outputs, identities = _collect_stage_with_max_output_retries(
        client,
        state_root / "dedup",
        jobs,
        poll_interval_seconds=poll_interval_seconds,
        sleep=sleep,
    )
    return _finalize_outputs(
        outputs=outputs,
        identities=identities,
        state_root=state_root,
        prompts=prompts,
        candidates_by_prompt=candidates_by_prompt,
        checkpoint_id=checkpoint_id,
        output_path=output_path,
        max_online_criteria=max_online_criteria,
        control_rubric_rows=control_rubric_rows,
        model=model,
        generation_method="openai_batch_responses_two_stage",
    )


def build_horizon_rubrics_batch(
    *,
    client: Any,
    run_id: str,
    prompts: Sequence[Mapping[str, Any]],
    current_rows: Sequence[Mapping[str, Any]],
    control_rows: Sequence[Mapping[str, Any]],
    checkpoint_id: str,
    pairing_seed: int,
    model: str,
    extraction_schema: Mapping[str, Any],
    dedup_schema: Mapping[str, Any],
    output_path: Path,
    state_root: Path,
    max_online_criteria: int = 8,
    reasoning_effort: str = "medium",
    control_rubric_rows: Sequence[Mapping[str, Any]] = (),
    poll_interval_seconds: float = 60.0,
    sleep: Callable[[float], None] = time.sleep,
) -> Mapping[str, Any]:
    """Run extraction Batch to completion before preparing the dependent dedup Batch."""

    if not re.fullmatch(r"step\d+", checkpoint_id):
        raise ValueError("checkpoint_id must use the form step<non-negative integer>")
    invocation = {
        "schema_version": HORIZON_BATCH_SCHEMA_VERSION,
        "run_id": run_id,
        "checkpoint_id": checkpoint_id,
        "pairing_seed": pairing_seed,
        "model": model,
        "reasoning_effort": reasoning_effort,
        "max_online_criteria": max_online_criteria,
        "output_path": str(output_path),
        "prompts_sha256": hashlib.sha256(canonical_json_bytes(prompts)).hexdigest(),
        "current_rows_sha256": hashlib.sha256(canonical_json_bytes(current_rows)).hexdigest(),
        "control_rows_sha256": hashlib.sha256(canonical_json_bytes(control_rows)).hexdigest(),
        "control_rubrics_sha256": hashlib.sha256(
            canonical_json_bytes(control_rubric_rows)
        ).hexdigest(),
        "extraction_schema_sha256": hashlib.sha256(
            canonical_json_bytes(extraction_schema)
        ).hexdigest(),
        "dedup_schema_sha256": hashlib.sha256(canonical_json_bytes(dedup_schema)).hexdigest(),
    }
    write_json_atomic(state_root / "invocation.json", invocation)
    result_path = state_root / "result.json"
    if result_path.is_file():
        result = read_json(result_path)
        validate_artifact_record(result["rubrics_artifact"])
        return result
    _prepare_extraction(
        state_root=state_root,
        run_id=run_id,
        prompts=prompts,
        current_rows=current_rows,
        control_rows=control_rows,
        checkpoint_id=checkpoint_id,
        pairing_seed=pairing_seed,
        model=model,
        schema=extraction_schema,
        reasoning_effort=reasoning_effort,
    )
    _submit_stage(client, state_root / "extraction", sleep=sleep)
    extraction_jobs = _wait_for_stage(
        client,
        state_root / "extraction",
        poll_interval_seconds=poll_interval_seconds,
        sleep=sleep,
    )
    candidates_by_prompt, _ = _collect_extraction(
        client=client,
        state_root=state_root,
        jobs=extraction_jobs,
        model=model,
        poll_interval_seconds=poll_interval_seconds,
        sleep=sleep,
    )
    _prepare_dedup(
        state_root=state_root,
        run_id=run_id,
        prompts=prompts,
        candidates_by_prompt=candidates_by_prompt,
        checkpoint_id=checkpoint_id,
        pairing_seed=pairing_seed,
        model=model,
        schema=dedup_schema,
        reasoning_effort=reasoning_effort,
    )
    _submit_stage(client, state_root / "dedup", sleep=sleep)
    dedup_jobs = _wait_for_stage(
        client,
        state_root / "dedup",
        poll_interval_seconds=poll_interval_seconds,
        sleep=sleep,
    )
    return _finalize(
        client=client,
        state_root=state_root,
        jobs=dedup_jobs,
        prompts=prompts,
        candidates_by_prompt=candidates_by_prompt,
        checkpoint_id=checkpoint_id,
        output_path=output_path,
        max_online_criteria=max_online_criteria,
        control_rubric_rows=control_rubric_rows,
        model=model,
        poll_interval_seconds=poll_interval_seconds,
        sleep=sleep,
    )


def build_horizon_rubrics_batch_from_files(
    *,
    run_id: str,
    prompts_path: Path,
    current_pool_path: Path,
    control_pool_path: Path,
    checkpoint_id: str,
    pairing_seed: int,
    model: str,
    extraction_schema_path: Path,
    dedup_schema_path: Path,
    output_path: Path,
    state_root: Path,
    max_online_criteria: int,
    reasoning_effort: str,
    control_rubrics_path: Path | None = None,
    poll_interval_seconds: float = 60.0,
    client: Any | None = None,
) -> Mapping[str, Any]:
    return build_horizon_rubrics_batch(
        client=client if client is not None else _openai_client(),
        run_id=run_id,
        prompts=read_jsonl(prompts_path),
        current_rows=read_jsonl(current_pool_path),
        control_rows=read_jsonl(control_pool_path),
        checkpoint_id=checkpoint_id,
        pairing_seed=pairing_seed,
        model=model,
        extraction_schema=read_json(extraction_schema_path),
        dedup_schema=read_json(dedup_schema_path),
        output_path=output_path,
        state_root=state_root,
        max_online_criteria=max_online_criteria,
        reasoning_effort=reasoning_effort,
        control_rubric_rows=(
            read_jsonl(control_rubrics_path) if control_rubrics_path is not None else ()
        ),
        poll_interval_seconds=poll_interval_seconds,
    )
