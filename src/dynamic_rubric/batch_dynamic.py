"""Crash-safe OpenAI Batch preparation and submission for fixed-control replay."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .artifacts import (
    artifact_record,
    read_json,
    read_jsonl,
    write_bytes_atomic,
    write_json_atomic,
    write_jsonl_atomic,
)
from .hashing import canonical_json_bytes, sha256_file
from .pipeline import PipelineContext, StageError
from .providers.openai_responses import _output_text
from .providers.base import GenerationRequest
from .prompt_versions.initial_rubric_prompt import INITIAL_RUBRIC_CRITERION_INSTRUCTIONS
from .rubrics.extractor import make_blind_pairing
from .rubrics.replay import ReplayMode
from .seeds import SeedFamily, derive_seed

BATCH_ENDPOINT = "/v1/responses"
BATCH_COMPLETION_WINDOW = "24h"
MAX_BATCH_FILE_BYTES = 180_000_000
MAX_BATCH_REQUESTS = 45_000
DYNAMIC_MAX_OUTPUT_TOKENS = 8192
# Backward-compatible alias. New code should use the explicit ``initial_*`` name so
# this legacy replay prompt is not confused with the paper-faithful OnlineRubrics prompt.
DYNAMIC_CRITERION_INSTRUCTIONS = INITIAL_RUBRIC_CRITERION_INSTRUCTIONS
SUPPORTED_BATCH_MODES = (
    ReplayMode.DYNAMIC_FIXED_BUDGETED.value,
    ReplayMode.DYNAMIC_PREV_BUDGETED.value,
)


def dynamic_batch_stage(mode: str) -> str:
    parsed = ReplayMode(mode)
    if parsed is ReplayMode.DYNAMIC_FIXED_BUDGETED:
        return "dynamic-batch"
    if parsed is ReplayMode.DYNAMIC_PREV_BUDGETED:
        return "dynamic-prev-batch"
    raise ValueError(f"unsupported live dynamic Batch mode: {mode}")


def dynamic_batch_prefix(mode: str) -> str:
    parsed = ReplayMode(mode)
    if parsed is ReplayMode.DYNAMIC_FIXED_BUDGETED:
        return "dynamic-fixed"
    if parsed is ReplayMode.DYNAMIC_PREV_BUDGETED:
        return "dynamic-prev"
    raise ValueError(f"unsupported live dynamic Batch mode: {mode}")


def control_policy_step(mode: str, policy_step: int) -> int:
    if policy_step < 1:
        raise ValueError("policy_step must be positive")
    parsed = ReplayMode(mode)
    if parsed is ReplayMode.DYNAMIC_FIXED_BUDGETED:
        return 0
    if parsed is ReplayMode.DYNAMIC_PREV_BUDGETED:
        return policy_step - 1
    raise ValueError(f"unsupported live dynamic Batch mode: {mode}")


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "model_dump"):
        return dict(value.model_dump())
    if hasattr(value, "to_dict"):
        return dict(value.to_dict())
    raise TypeError(f"unsupported OpenAI SDK response: {type(value).__name__}")


def _responses_payload(model: str, request: GenerationRequest) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "input": [dict(message) for message in request.messages],
        "max_output_tokens": request.max_output_tokens,
    }
    if request.reasoning_effort:
        payload["reasoning"] = {"effort": request.reasoning_effort}
    if request.json_schema:
        payload["text"] = {
            "format": {
                "type": "json_schema",
                "name": request.schema_name or "structured_response",
                "schema": dict(request.json_schema),
                "strict": True,
            }
        }
    if request.temperature != 0.0:
        payload["temperature"] = request.temperature
    if request.top_p != 1.0:
        payload["top_p"] = request.top_p
    return payload


def _index(
    rows: Iterable[Mapping[str, Any]], *, text_key: str
) -> dict[tuple[str, int, str], tuple[Mapping[str, Any], ...]]:
    grouped: dict[tuple[str, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for source in rows:
        row = dict(source)
        if text_key not in row:
            raise StageError(f"source response is missing {text_key!r}")
        row["response_text"] = str(row[text_key])
        key = (str(row["prompt_id"]), int(row["policy_step"]), str(row["family"]))
        grouped[key].append(row)
    return {
        key: tuple(sorted(values, key=lambda item: int(item["sample_index"])))
        for key, values in grouped.items()
    }


def _replicate_ids(prompt_id: str, fraction: float) -> tuple[str, ...]:
    subset = int(hashlib.sha256(prompt_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return ("A", "B") if subset < fraction else ("A",)


def _request(
    context: PipelineContext,
    prompt_id: str,
    step: int,
    replicate_id: str,
    current: Sequence[Mapping[str, Any]],
    control: Sequence[Mapping[str, Any]],
    mode: str = ReplayMode.DYNAMIC_FIXED_BUDGETED.value,
) -> tuple[dict[str, Any], dict[str, Any]]:
    parsed_mode = ReplayMode(mode)
    if parsed_mode.value not in SUPPORTED_BATCH_MODES:
        raise ValueError(f"unsupported live dynamic Batch mode: {mode}")
    pairing = make_blind_pairing(
        [str(row["response_text"]) for row in current],
        [str(row["response_text"]) for row in control],
        seed=derive_seed(context.run_id, SeedFamily.PAIRING, prompt_id, step, 0),
        prompt_id=prompt_id,
        step=step,
    )
    blinded_payload = [dataclasses.asdict(pair) for pair in pairing.generator_payload()]
    logical_seed = derive_seed(context.run_id, SeedFamily.PAIRING, prompt_id, step, 1)
    request = GenerationRequest(
        prompt_id=f"{prompt_id}-s{step}-{parsed_mode.value}",
        messages=(
            {
                "role": "user",
                "content": DYNAMIC_CRITERION_INSTRUCTIONS
                + json.dumps(blinded_payload, ensure_ascii=False, sort_keys=True),
            },
        ),
        family=f"dynamic_extraction:{parsed_mode.value}",
        seed=logical_seed,
        max_output_tokens=DYNAMIC_MAX_OUTPUT_TOKENS,
        json_schema=read_json(context.root / "configs" / "schemas" / "dynamic_candidate_v1.json"),
        schema_name="dynamic_candidate_v1",
        reasoning_effort=str(context.config.models["rubric_generator"]["reasoning_effort"]),
        metadata={
            "run_id": context.run_id,
            "prompt_id": prompt_id,
            "policy_step": step,
            "replicate_id": replicate_id,
        },
    )
    model = str(context.config.models["rubric_generator"]["requested_model"])
    identity = {
        "prompt_id": prompt_id,
        "policy_step": step,
        "replicate_id": replicate_id,
        "logical_seed": logical_seed,
        "mode": parsed_mode.value,
        "control_policy": f"pi_{control_policy_step(parsed_mode.value, step)}",
        "current_response_ids": [str(row["response_id"]) for row in current],
        "control_response_ids": [str(row["response_id"]) for row in control],
        "pairing_hash": hashlib.sha256(canonical_json_bytes(blinded_payload)).hexdigest(),
    }
    identity_hash = hashlib.sha256(canonical_json_bytes(identity)).hexdigest()
    line = {
        "custom_id": (
            f"{dynamic_batch_prefix(parsed_mode.value)}-{step:03d}-"
            f"{replicate_id.lower()}-{identity_hash[:24]}"
        ),
        "method": "POST",
        "url": BATCH_ENDPOINT,
        "body": _responses_payload(model, request),
    }
    return line, {"custom_id": line["custom_id"], **identity}


def _shard(
    lines: Sequence[Mapping[str, Any]],
    *,
    max_file_bytes: int | None = None,
    max_requests: int | None = None,
) -> list[list[Mapping[str, Any]]]:
    if max_file_bytes is None:
        max_file_bytes = MAX_BATCH_FILE_BYTES
    if max_requests is None:
        max_requests = MAX_BATCH_REQUESTS
    if max_file_bytes <= 0 or max_requests <= 0:
        raise ValueError("Batch shard limits must be positive")
    shards: list[list[Mapping[str, Any]]] = []
    current: list[Mapping[str, Any]] = []
    current_bytes = 0
    for line in lines:
        line_bytes = len(canonical_json_bytes(line)) + 1
        if line_bytes > max_file_bytes:
            raise StageError("one Batch request exceeds the safe input-file size")
        if current and (
            len(current) >= max_requests or current_bytes + line_bytes > max_file_bytes
        ):
            shards.append(current)
            current = []
            current_bytes = 0
        current.append(line)
        current_bytes += line_bytes
    if current:
        shards.append(current)
    return shards


def prepare_dynamic_batch(
    context: PipelineContext,
    max_step: int = 50,
    mode: str = ReplayMode.DYNAMIC_FIXED_BUDGETED.value,
) -> dict[str, Any]:
    if max_step <= 0:
        raise ValueError("max_step must be positive")
    parsed_mode = ReplayMode(mode)
    if parsed_mode.value not in SUPPORTED_BATCH_MODES:
        raise ValueError(f"unsupported live dynamic Batch mode: {mode}")
    stage_root = context.stage_root()
    live_root = context.run_root / "train-static" / "verl-run"
    reference_path = context.run_root / "train-static" / "reference_responses.jsonl"
    probe_paths = [live_root / "probes" / f"{step}.jsonl" for step in range(1, max_step + 1)]
    missing = [str(path) for path in (reference_path, *probe_paths) if not path.is_file()]
    if missing:
        raise StageError(f"dynamic Batch sources are incomplete: {missing[:3]}")

    reference_index = _index(read_jsonl(reference_path), text_key="response_text")
    prompt_ids = sorted(
        key[0]
        for key in reference_index
        if key[1] == 0 and key[2] == SeedFamily.REFERENCE_DISCOVERY.value
    )
    if len(prompt_ids) != len(set(prompt_ids)):
        raise StageError("reference discovery prompt index is not unique")
    replicate_fraction = float(context.raw.get("replay", {}).get("replicate_fraction", 0.20))
    lines: list[dict[str, Any]] = []
    mapping: list[dict[str, Any]] = []
    previous_index: dict[tuple[str, int, str], tuple[Mapping[str, Any], ...]] | None = None
    for step, path in enumerate(probe_paths, start=1):
        current_index = _index(read_jsonl(path), text_key="output")
        for prompt_id in prompt_ids:
            current = current_index.get(
                (prompt_id, step, SeedFamily.TRAJECTORY_DISCOVERY.value), ()
            )
            control_step = control_policy_step(parsed_mode.value, step)
            if control_step == 0:
                control = reference_index.get(
                    (prompt_id, 0, SeedFamily.REFERENCE_DISCOVERY.value), ()
                )
            else:
                assert previous_index is not None
                control = previous_index.get(
                    (prompt_id, control_step, SeedFamily.TRAJECTORY_DISCOVERY.value), ()
                )
            if len(current) < 4 or len(control) < 4:
                raise StageError(
                    "dynamic discovery requires four current/control responses: "
                    f"{prompt_id=} {step=} {control_step=}"
                )
            for replicate_id in _replicate_ids(prompt_id, replicate_fraction):
                line, identity = _request(
                    context,
                    prompt_id,
                    step,
                    replicate_id,
                    current[:4],
                    control[:4],
                    parsed_mode.value,
                )
                lines.append(line)
                mapping.append(identity)
        previous_index = current_index

    if len({line["custom_id"] for line in lines}) != len(lines):
        raise StageError("Batch custom_id collision")
    input_paths: list[Path] = []
    prefix = dynamic_batch_prefix(parsed_mode.value)
    for index, shard in enumerate(_shard(lines), start=1):
        path = stage_root / "inputs" / f"{prefix}-{index:03d}.jsonl"
        write_jsonl_atomic(path, shard)
        input_paths.append(path)
    mapping_path = stage_root / "request_map.jsonl"
    write_jsonl_atomic(mapping_path, mapping)
    manifest = {
        "schema_version": 1,
        "run_id": context.run_id,
        "config_hash": context.config.config_hash,
        "mode": parsed_mode.value,
        "max_step": max_step,
        "model": context.config.models["rubric_generator"]["requested_model"],
        "endpoint": BATCH_ENDPOINT,
        "completion_window": BATCH_COMPLETION_WINDOW,
        "requests": len(lines),
        "primary_requests": len(prompt_ids) * max_step,
        "replicate_requests": len(lines) - len(prompt_ids) * max_step,
        "prompt_count": len(prompt_ids),
        "input_files": [artifact_record(path) for path in input_paths],
        "request_map": artifact_record(mapping_path),
        "reference_source": artifact_record(reference_path),
        "probe_sources": [artifact_record(path) for path in probe_paths],
    }
    write_json_atomic(stage_root / "manifest.json", manifest)
    return manifest


def submit_dynamic_batch(context: PipelineContext) -> dict[str, Any]:
    stage_root = context.stage_root()
    manifest_path = stage_root / "manifest.json"
    if not manifest_path.is_file():
        raise StageError("prepare-dynamic-batch must complete before submission")
    manifest = read_json(manifest_path)
    receipt_path = stage_root / "submission.json"
    if receipt_path.is_file():
        return read_json(receipt_path)
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise StageError("OPENAI_API_KEY is not set")
    try:
        from openai import OpenAI  # type: ignore[import-not-found]
    except ImportError as error:
        raise StageError("the OpenAI Python SDK is required for Batch submission") from error

    client = OpenAI(api_key=api_key)
    jobs: list[dict[str, Any]] = []
    for index, record in enumerate(manifest["input_files"], start=1):
        shard_receipt = stage_root / f"submission-{index:03d}.json"
        if shard_receipt.is_file():
            job = read_json(shard_receipt)
            if job.get("input_sha256") != record["sha256"]:
                raise StageError("existing Batch submission does not match prepared input")
            jobs.append(job)
            continue
        input_path = Path(str(record["path"]))
        if sha256_file(input_path) != record["sha256"]:
            raise StageError(f"Batch input changed after preparation: {input_path}")
        with input_path.open("rb") as stream:
            uploaded = _as_dict(client.files.create(file=stream, purpose="batch"))
        input_file_id = str(uploaded["id"])
        created = _as_dict(
            client.batches.create(
                input_file_id=input_file_id,
                endpoint=BATCH_ENDPOINT,
                completion_window=BATCH_COMPLETION_WINDOW,
                metadata={
                    "run_id": context.run_id,
                    "mode": str(manifest["mode"]),
                    "shard": str(index),
                },
            )
        )
        job = {
            "shard": index,
            "input_path": str(input_path),
            "input_sha256": record["sha256"],
            "input_file_id": input_file_id,
            "batch_id": str(created["id"]),
            "status": str(created.get("status", "validating")),
            "created_at": created.get("created_at"),
        }
        write_json_atomic(shard_receipt, job)
        jobs.append(job)
    receipt = {
        "schema_version": 1,
        "run_id": context.run_id,
        "manifest_sha256": sha256_file(manifest_path),
        "jobs": jobs,
    }
    write_json_atomic(receipt_path, receipt)
    return receipt


def dynamic_batch_status(context: PipelineContext) -> dict[str, Any]:
    stage_root = context.stage_root()
    receipt = read_json(stage_root / "submission.json")
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise StageError("OPENAI_API_KEY is not set")
    try:
        from openai import OpenAI  # type: ignore[import-not-found]
    except ImportError as error:
        raise StageError("the OpenAI Python SDK is required for Batch status") from error
    client = OpenAI(api_key=api_key)
    jobs = []
    for job in receipt["jobs"]:
        current = _as_dict(client.batches.retrieve(str(job["batch_id"])))
        jobs.append(
            {
                "batch_id": current["id"],
                "status": current["status"],
                "request_counts": current.get("request_counts"),
                "output_file_id": current.get("output_file_id"),
                "error_file_id": current.get("error_file_id"),
                "expires_at": current.get("expires_at"),
            }
        )
    status = {"run_id": context.run_id, "jobs": jobs}
    write_json_atomic(stage_root / "status.json", status, immutable=False)
    return status


def _download_file(client: Any, file_id: str) -> bytes:
    response = client.files.content(file_id)
    content = getattr(response, "content", None)
    if isinstance(content, bytes):
        return content
    if isinstance(content, bytearray):
        return bytes(content)
    if hasattr(response, "read"):
        value = response.read()
        if isinstance(value, bytes):
            return value
    raise StageError("OpenAI file download did not return bytes")


def _normalize_batch_row(
    row: Mapping[str, Any],
    identity: Mapping[str, Any],
    requested_model: str,
) -> dict[str, Any]:
    if row.get("custom_id") != identity.get("custom_id"):
        raise StageError("Batch output custom_id does not match request identity")
    if row.get("error") is not None:
        raise StageError(f"Batch request failed: {row['custom_id']}: {row['error']}")
    response = row.get("response")
    if not isinstance(response, Mapping) or int(response.get("status_code", 0)) != 200:
        raise StageError(f"Batch response is not HTTP 200: {row['custom_id']}")
    body = response.get("body")
    if not isinstance(body, Mapping):
        raise StageError(f"Batch response body is missing: {row['custom_id']}")
    if body.get("status") not in (None, "completed"):
        reason = (body.get("incomplete_details") or {}).get("reason")
        raise StageError(f"Batch response is incomplete: {row['custom_id']}: {reason}")
    returned_model = str(body.get("model", ""))
    if not returned_model:
        raise StageError(f"Batch response model identity is missing: {row['custom_id']}")
    output_text = _output_text(body)
    try:
        parsed = json.loads(output_text)
    except json.JSONDecodeError as error:
        raise StageError(f"Batch structured output is invalid JSON: {row['custom_id']}") from error
    criteria = parsed.get("criteria") if isinstance(parsed, Mapping) else None
    if not isinstance(criteria, list) or len(criteria) > 3:
        raise StageError(f"Batch output violates the three-criterion bound: {row['custom_id']}")
    for item in criteria:
        if not isinstance(item, Mapping):
            raise StageError(f"Batch criterion is not an object: {row['custom_id']}")
        text = item.get("text")
        rationale = item.get("rationale")
        if (
            not isinstance(text, str)
            or not 12 <= len(text) <= 500
            or not isinstance(rationale, str)
            or len(rationale) > 1000
        ):
            raise StageError(f"Batch criterion violates the candidate schema: {row['custom_id']}")
    return {
        **dict(identity),
        "criteria": [dict(item) for item in criteria],
        "provider_call": {
            "requested_model": requested_model,
            "returned_model": returned_model,
            "request_id": str(body.get("id") or response.get("request_id") or ""),
            "batch_request_id": str(row.get("id", "")),
            "created_at": body.get("created_at", body.get("created")),
            "usage": dict(body.get("usage", {})),
            "raw_response_hash": hashlib.sha256(canonical_json_bytes(body)).hexdigest(),
        },
    }


def collect_dynamic_batch(context: PipelineContext) -> dict[str, Any]:
    stage_root = context.stage_root()
    manifest_path = stage_root / "manifest.json"
    manifest = read_json(manifest_path)
    receipt = read_json(stage_root / "submission.json")
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise StageError("OPENAI_API_KEY is not set")
    try:
        from openai import OpenAI  # type: ignore[import-not-found]
    except ImportError as error:
        raise StageError("the OpenAI Python SDK is required for Batch collection") from error
    identities = {str(row["custom_id"]): row for row in read_jsonl(manifest["request_map"]["path"])}
    if len(identities) != int(manifest["requests"]):
        raise StageError("Batch request identity inventory is incomplete")
    client = OpenAI(api_key=api_key)
    prefix = dynamic_batch_prefix(str(manifest["mode"]))

    def collect_jobs(
        jobs: Sequence[Mapping[str, Any]],
        *,
        output_root: Path,
        filename_prefix: str,
        kind: str,
    ) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
        rows_by_id: dict[str, dict[str, Any]] = {}
        records: list[dict[str, Any]] = []
        for job in jobs:
            current = _as_dict(client.batches.retrieve(str(job["batch_id"])))
            if current.get("status") != "completed" or not current.get("output_file_id"):
                raise StageError(
                    f"Batch is not ready for collection: "
                    f"{current.get('id')}={current.get('status')}"
                )
            shard = int(job["shard"])
            payload = _download_file(client, str(current["output_file_id"]))
            raw_path = output_root / f"{filename_prefix}-{shard:03d}.raw.jsonl"
            write_bytes_atomic(raw_path, payload)
            input_rows = read_jsonl(job["input_path"])
            expected = {str(row["custom_id"]) for row in input_rows}
            shard_rows = [json.loads(line) for line in payload.splitlines() if line.strip()]
            actual = {str(row.get("custom_id", "")) for row in shard_rows}
            if actual != expected:
                raise StageError(f"Batch output inventory mismatch for {kind} shard {shard}")
            for row in shard_rows:
                custom_id = str(row["custom_id"])
                if custom_id in rows_by_id:
                    raise StageError(f"duplicate {kind} Batch output custom_id: {custom_id}")
                rows_by_id[custom_id] = row
            record = {
                "kind": kind,
                "shard": shard,
                "batch_id": current["id"],
                "output_file_id": current["output_file_id"],
                "raw": artifact_record(raw_path),
            }
            error_file_id = current.get("error_file_id")
            if error_file_id:
                error_payload = _download_file(client, str(error_file_id))
                error_path = output_root / f"{filename_prefix}-{shard:03d}.errors.jsonl"
                write_bytes_atomic(error_path, error_payload)
                record["errors"] = artifact_record(error_path)
                if error_payload.strip():
                    raise StageError(f"Batch {kind} shard {shard} produced an error file")
            records.append(record)
        return rows_by_id, records

    base_rows, outputs = collect_jobs(
        receipt["jobs"],
        output_root=stage_root / "outputs",
        filename_prefix=prefix,
        kind="base",
    )
    if set(base_rows) != set(identities):
        raise StageError("combined base Batch output inventory is incomplete")

    final_rows = dict(base_rows)
    retried_ids: set[str] = set()
    retry_attempts = 0
    retry_records: list[dict[str, Any]] = []
    retry_number = 1
    previous_retry_manifest_path: Path | None = None
    while True:
        retry_name = "retry" if retry_number == 1 else f"retry-{retry_number}"
        retry_root = stage_root / retry_name
        retry_manifest_path = retry_root / "manifest.json"
        retry_receipt_path = retry_root / "submission.json"
        if not retry_manifest_path.is_file() and not retry_receipt_path.is_file():
            break
        if retry_manifest_path.is_file() != retry_receipt_path.is_file():
            raise StageError(
                f"dynamic Batch {retry_name} manifest/receipt inventory is inconsistent"
            )
        retry_manifest = read_json(retry_manifest_path)
        retry_receipt = read_json(retry_receipt_path)
        if retry_manifest.get("base_manifest_sha256") != sha256_file(manifest_path):
            raise StageError(f"dynamic Batch {retry_name} is not bound to the base manifest")
        if retry_manifest.get("mode") != manifest.get("mode"):
            raise StageError(f"dynamic Batch {retry_name} mode does not match the base manifest")
        if retry_manifest.get("model") != manifest.get("model"):
            raise StageError(f"dynamic Batch {retry_name} model does not match the base manifest")
        if previous_retry_manifest_path is not None and retry_manifest.get(
            "previous_retry_manifest_sha256"
        ) != sha256_file(previous_retry_manifest_path):
            raise StageError(f"dynamic Batch {retry_name} is not bound to the previous retry")
        retry_rows, retry_outputs = collect_jobs(
            retry_receipt["jobs"],
            output_root=retry_root / "outputs",
            filename_prefix=f"{prefix}-{retry_name}",
            kind=retry_name,
        )
        if not set(retry_rows).issubset(identities):
            raise StageError(f"dynamic Batch {retry_name} contains unknown custom IDs")
        if len(retry_rows) != int(retry_manifest["requests"]):
            raise StageError(f"dynamic Batch {retry_name} inventory is incomplete")
        final_rows.update(retry_rows)
        retried_ids.update(retry_rows)
        retry_attempts += len(retry_rows)
        retry_records.append(artifact_record(retry_manifest_path))
        outputs.extend(retry_outputs)
        previous_retry_manifest_path = retry_manifest_path
        retry_number += 1

    normalized_by_id: dict[str, dict[str, Any]] = {}
    returned_models: set[str] = set()
    for custom_id, identity in identities.items():
        item = _normalize_batch_row(
            final_rows[custom_id],
            identity,
            str(manifest["model"]),
        )
        returned_models.add(str(item["provider_call"]["returned_model"]))
        normalized_by_id[custom_id] = item
    if len(returned_models) != 1:
        raise StageError(f"Batch returned-model drift detected: {sorted(returned_models)}")

    for job in receipt["jobs"]:
        shard = int(job["shard"])
        shard_ids = [str(row["custom_id"]) for row in read_jsonl(job["input_path"])]
        normalized_path = stage_root / "outputs" / f"{prefix}-{shard:03d}.jsonl"
        write_jsonl_atomic(
            normalized_path,
            (normalized_by_id[custom_id] for custom_id in shard_ids),
        )
        for record in outputs:
            if record["kind"] == "base" and int(record["shard"]) == shard:
                record["normalized"] = artifact_record(normalized_path)
                break

    normalized = sorted(
        normalized_by_id.values(),
        key=lambda row: (
            int(row["policy_step"]),
            str(row["prompt_id"]),
            str(row["replicate_id"]),
        ),
    )
    output_path = stage_root / "dynamic_candidates.jsonl"
    write_jsonl_atomic(output_path, normalized)
    result = {
        "run_id": context.run_id,
        "mode": str(manifest["mode"]),
        "requests": len(normalized),
        "retried_requests": len(retried_ids),
        "returned_model": next(iter(returned_models)),
        "output": artifact_record(output_path),
        "shards": outputs,
    }
    if retry_records:
        result["retry_attempts"] = retry_attempts
        result["retry_manifests"] = retry_records
    write_json_atomic(stage_root / "collection.json", result)
    return result
