"""Crash-safe synchronous Responses API runner for horizon rubric construction."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from ..artifacts import (
    read_json,
    read_jsonl,
    validate_artifact_record,
    write_json_atomic,
)
from ..batch_dynamic import _as_dict
from ..hashing import canonical_json_bytes
from ..onlinerubric_batch import _require_api_key
from ..pipeline import StageError
from .batch_rubrics import (
    HORIZON_BATCH_SCHEMA_VERSION,
    _finalize_outputs,
    _parse_extraction_outputs,
    _parsed_response,
    _prepare_dedup,
    _prepare_extraction,
)

SYNC_GENERATION_METHOD = "openai_sync_responses_two_stage"
DEFAULT_SYNC_CONCURRENCY = 16
GPT5_MINI_PRICING_SOURCE = "https://developers.openai.com/api/docs/models/gpt-5-mini"
GPT5_MINI_INPUT_PER_MILLION_USD = 0.25
GPT5_MINI_CACHED_INPUT_PER_MILLION_USD = 0.025
GPT5_MINI_OUTPUT_PER_MILLION_USD = 2.0


def _openai_sync_client() -> Any:
    try:
        from openai import OpenAI  # type: ignore[import-not-found]
    except ImportError as error:
        raise StageError("the OpenAI Python SDK is required for sync Responses operations") from error
    return OpenAI(api_key=_require_api_key(), timeout=300.0, max_retries=6)


def _prepared_stage(
    stage_root: Path,
) -> tuple[Mapping[str, Any], list[Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    manifest = read_json(stage_root / "manifest.json")
    for record in manifest["input_files"]:
        validate_artifact_record(record)
    validate_artifact_record(manifest["request_map"])
    lines = [
        row
        for record in manifest["input_files"]
        for row in read_jsonl(Path(str(record["path"])))
    ]
    identities = {
        str(row["custom_id"]): row
        for row in read_jsonl(Path(str(manifest["request_map"]["path"])))
    }
    request_ids = [str(line.get("custom_id", "")) for line in lines]
    expected = int(manifest["requests"])
    if len(lines) != expected or len(identities) != expected:
        raise StageError("horizon sync request inventory is incomplete")
    if len(set(request_ids)) != expected or set(request_ids) != set(identities):
        raise StageError("horizon sync request identity inventory is inconsistent")
    return manifest, lines, identities


def _response_path(stage_root: Path, custom_id: str) -> Path:
    digest = hashlib.sha256(custom_id.encode("utf-8")).hexdigest()
    return stage_root / "responses" / f"{digest}.json"


def _response_artifact(
    *,
    custom_id: str,
    request_sha256: str,
    effective_request_sha256: str,
    effective_max_output_tokens: int,
    body: Mapping[str, Any],
) -> dict[str, Any]:
    row = {
        "custom_id": custom_id,
        "response": {"status_code": 200, "body": dict(body)},
        "error": None,
    }
    _parsed_response(row, custom_id)
    return {
        "schema_version": 1,
        "custom_id": custom_id,
        "request_sha256": request_sha256,
        "effective_request_sha256": effective_request_sha256,
        "effective_max_output_tokens": effective_max_output_tokens,
        "row": row,
    }


def _attempt_path(stage_root: Path, custom_id: str, effective_sha256: str) -> Path:
    digest = hashlib.sha256(custom_id.encode("utf-8")).hexdigest()
    return stage_root / "attempts" / f"{digest}-{effective_sha256[:16]}.json"


def _record_attempt(
    stage_root: Path,
    *,
    custom_id: str,
    request_sha256: str,
    effective_request_sha256: str,
    effective_max_output_tokens: int,
    body: Mapping[str, Any],
) -> None:
    write_json_atomic(
        _attempt_path(stage_root, custom_id, effective_request_sha256),
        {
            "schema_version": 1,
            "custom_id": custom_id,
            "request_sha256": request_sha256,
            "effective_request_sha256": effective_request_sha256,
            "effective_max_output_tokens": effective_max_output_tokens,
            "status": body.get("status"),
            "response": dict(body),
        },
    )


def _load_cached_response(
    stage_root: Path, *, custom_id: str, request_sha256: str
) -> Mapping[str, Any] | None:
    path = _response_path(stage_root, custom_id)
    if not path.is_file():
        return None
    artifact = read_json(path)
    if artifact.get("custom_id") != custom_id:
        raise StageError(f"horizon sync response custom_id mismatch: {custom_id}")
    if artifact.get("request_sha256") != request_sha256:
        raise StageError(f"horizon sync cached request drift: {custom_id}")
    row = artifact.get("row")
    if not isinstance(row, Mapping):
        raise StageError(f"horizon sync cached response is malformed: {custom_id}")
    _parsed_response(row, custom_id)
    return row


def _numeric_usage(value: Any) -> Any:
    if isinstance(value, Mapping):
        output = {
            str(key): parsed
            for key, item in value.items()
            if (parsed := _numeric_usage(item)) is not None
        }
        return output or None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    return None


def _add_usage(total: dict[str, Any], value: Mapping[str, Any]) -> None:
    for key, item in value.items():
        if isinstance(item, Mapping):
            nested = total.setdefault(str(key), {})
            if not isinstance(nested, dict):
                raise StageError(f"horizon sync usage shape drift at {key}")
            _add_usage(nested, item)
        elif isinstance(item, (int, float)) and not isinstance(item, bool):
            total[str(key)] = total.get(str(key), 0) + item


def _usage_cost(usage: Mapping[str, Any]) -> dict[str, Any]:
    input_tokens = int(usage.get("input_tokens", 0))
    details = usage.get("input_tokens_details", {})
    cached_tokens = int(details.get("cached_tokens", 0)) if isinstance(details, Mapping) else 0
    cached_tokens = min(max(cached_tokens, 0), input_tokens)
    uncached_tokens = input_tokens - cached_tokens
    output_tokens = int(usage.get("output_tokens", 0))
    input_cost = uncached_tokens * GPT5_MINI_INPUT_PER_MILLION_USD / 1_000_000
    cached_input_cost = (
        cached_tokens * GPT5_MINI_CACHED_INPUT_PER_MILLION_USD / 1_000_000
    )
    output_cost = output_tokens * GPT5_MINI_OUTPUT_PER_MILLION_USD / 1_000_000
    return {
        "uncached_input_tokens": uncached_tokens,
        "cached_input_tokens": cached_tokens,
        "output_tokens": output_tokens,
        "uncached_input_cost_usd": input_cost,
        "cached_input_cost_usd": cached_input_cost,
        "output_cost_usd": output_cost,
        "estimated_cost_usd": input_cost + cached_input_cost + output_cost,
    }


def _response_bodies(sync_root: Path) -> list[Mapping[str, Any]]:
    bodies: dict[str, Mapping[str, Any]] = {}
    for path in sorted(sync_root.rglob("attempts/*.json")):
        artifact = read_json(path)
        body = artifact.get("response")
        if isinstance(body, Mapping):
            identity = str(body.get("id", "")) or hashlib.sha256(
                canonical_json_bytes(body)
            ).hexdigest()
            bodies[identity] = body
    for path in sorted(sync_root.rglob("responses/*.json")):
        artifact = read_json(path)
        row = artifact.get("row", {})
        response = row.get("response", {}) if isinstance(row, Mapping) else {}
        body = response.get("body") if isinstance(response, Mapping) else None
        if isinstance(body, Mapping):
            identity = str(body.get("id", "")) or hashlib.sha256(
                canonical_json_bytes(body)
            ).hexdigest()
            bodies.setdefault(identity, body)
    return list(bodies.values())


def _write_cost_summary(sync_root: Path) -> Mapping[str, Any]:
    usage: dict[str, Any] = {}
    completed = 0
    incomplete = 0
    bodies = _response_bodies(sync_root)
    for body in bodies:
        status = body.get("status")
        completed += int(status == "completed")
        incomplete += int(status == "incomplete")
        parsed = _numeric_usage(body.get("usage", {}))
        if isinstance(parsed, Mapping):
            _add_usage(usage, parsed)
    summary = {
        "schema_version": 1,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "model": "gpt-5-mini",
        "api_mode": "sync",
        "pricing": {
            "source": GPT5_MINI_PRICING_SOURCE,
            "per_million_tokens_usd": {
                "input": GPT5_MINI_INPUT_PER_MILLION_USD,
                "cached_input": GPT5_MINI_CACHED_INPUT_PER_MILLION_USD,
                "output": GPT5_MINI_OUTPUT_PER_MILLION_USD,
            },
        },
        "attempt_count": len(bodies),
        "completed_attempts": completed,
        "incomplete_attempts": incomplete,
        "usage": usage,
        "cost": _usage_cost(usage),
    }
    write_json_atomic(sync_root / "cost_status.json", summary, immutable=False)
    return summary


def _stage_status(
    stage_root: Path,
    *,
    phase: str,
    total: int,
    outputs: Mapping[str, Mapping[str, Any]],
    failed: str | None = None,
) -> None:
    usage: dict[str, Any] = {}
    accounted_response_ids: set[str] = set()
    for path in sorted((stage_root / "attempts").glob("*.json")):
        attempt = read_json(path)
        body = attempt.get("response", {})
        if isinstance(body, Mapping):
            response_id = str(body.get("id", ""))
            if response_id:
                accounted_response_ids.add(response_id)
            parsed = _numeric_usage(body.get("usage", {}))
            if isinstance(parsed, Mapping):
                _add_usage(usage, parsed)
    for row in outputs.values():
        body = row.get("response", {}).get("body", {})
        if isinstance(body, Mapping) and str(body.get("id", "")) not in accounted_response_ids:
            parsed = _numeric_usage(body.get("usage", {}))
            if isinstance(parsed, Mapping):
                _add_usage(usage, parsed)
    stage_cost = _usage_cost(usage)
    write_json_atomic(
        stage_root / "status.json",
        {
            "schema_version": 1,
            "phase": phase,
            "status": "failed" if failed else ("completed" if len(outputs) == total else "running"),
            "total_requests": total,
            "completed_requests": len(outputs),
            "remaining_requests": total - len(outputs),
            "failed_request": failed,
            "usage": usage,
            "cost": stage_cost,
        },
        immutable=False,
    )
    run_root = stage_root.parent
    _write_cost_summary(run_root)
    if run_root.parent.name.startswith("seed-"):
        _write_cost_summary(run_root.parent)

def _run_one(client: Any, stage_root: Path, line: Mapping[str, Any]) -> Mapping[str, Any]:
    custom_id = str(line["custom_id"])
    request_sha256 = hashlib.sha256(canonical_json_bytes(line)).hexdigest()
    cached = _load_cached_response(
        stage_root, custom_id=custom_id, request_sha256=request_sha256
    )
    if cached is not None:
        return cached
    original_body = line.get("body")
    if not isinstance(original_body, Mapping):
        raise StageError(f"horizon sync request body is malformed: {custom_id}")
    initial_limit = int(original_body.get("max_output_tokens", 0))
    if initial_limit <= 0:
        raise StageError(f"horizon sync max_output_tokens is invalid: {custom_id}")
    limits = tuple(dict.fromkeys((initial_limit, max(initial_limit, 8192), 16384)))
    for limit in limits:
        body = {**dict(original_body), "max_output_tokens": limit}
        effective_line = {**dict(line), "body": body}
        effective_sha256 = hashlib.sha256(canonical_json_bytes(effective_line)).hexdigest()
        response = _as_dict(
            client.responses.create(
                **body,
                extra_headers={"Idempotency-Key": effective_sha256},
            )
        )
        _record_attempt(
            stage_root,
            custom_id=custom_id,
            request_sha256=request_sha256,
            effective_request_sha256=effective_sha256,
            effective_max_output_tokens=limit,
            body=response,
        )
        status = response.get("status")
        if status == "completed":
            if not str(response.get("model", "")):
                raise StageError(f"horizon sync response omitted returned model: {custom_id}")
            artifact = _response_artifact(
                custom_id=custom_id,
                request_sha256=request_sha256,
                effective_request_sha256=effective_sha256,
                effective_max_output_tokens=limit,
                body=response,
            )
            write_json_atomic(_response_path(stage_root, custom_id), artifact)
            return artifact["row"]
        details = response.get("incomplete_details")
        reason = details.get("reason") if isinstance(details, Mapping) else None
        if status != "incomplete" or reason != "max_output_tokens" or limit == limits[-1]:
            raise StageError(
                f"horizon sync response is incomplete: {custom_id}: "
                f"status={status}, reason={reason}"
            )
    raise AssertionError("horizon sync output-token retry loop exhausted unexpectedly")

def _run_sync_stage(
    client: Any, stage_root: Path, *, max_workers: int
) -> tuple[dict[str, Mapping[str, Any]], Mapping[str, Mapping[str, Any]]]:
    if max_workers <= 0:
        raise ValueError("sync concurrency must be positive")
    manifest, lines, identities = _prepared_stage(stage_root)
    phase = str(manifest["phase"])
    outputs: dict[str, Mapping[str, Any]] = {}
    pending = []
    for line in lines:
        custom_id = str(line["custom_id"])
        request_sha256 = hashlib.sha256(canonical_json_bytes(line)).hexdigest()
        cached = _load_cached_response(
            stage_root, custom_id=custom_id, request_sha256=request_sha256
        )
        if cached is None:
            pending.append(line)
        else:
            outputs[custom_id] = cached
    _stage_status(stage_root, phase=phase, total=len(lines), outputs=outputs)
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix=f"horizon-{phase}") as pool:
        futures = {pool.submit(_run_one, client, stage_root, line): line for line in pending}
        for future in as_completed(futures):
            custom_id = str(futures[future]["custom_id"])
            try:
                outputs[custom_id] = future.result()
            except BaseException:
                _stage_status(
                    stage_root,
                    phase=phase,
                    total=len(lines),
                    outputs=outputs,
                    failed=custom_id,
                )
                for other in futures:
                    other.cancel()
                raise
            _stage_status(stage_root, phase=phase, total=len(lines), outputs=outputs)
    if set(outputs) != set(identities):
        raise StageError("horizon sync output inventory is incomplete")
    returned_models = {
        str(row["response"]["body"].get("model", "")) for row in outputs.values()
    }
    if "" in returned_models or len(returned_models) != 1:
        raise StageError(f"horizon sync returned-model drift: {sorted(returned_models)}")
    return outputs, identities


def build_horizon_rubrics_sync(
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
    max_workers: int = DEFAULT_SYNC_CONCURRENCY,
) -> Mapping[str, Any]:
    """Run resumable extraction and dedup through synchronous Responses calls."""

    if not re.fullmatch(r"step\d+", checkpoint_id):
        raise ValueError("checkpoint_id must use the form step<non-negative integer>")
    invocation = {
        "schema_version": HORIZON_BATCH_SCHEMA_VERSION,
        "generation_method": SYNC_GENERATION_METHOD,
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
        "dedup_schema_sha256": hashlib.sha256(
            canonical_json_bytes(dedup_schema)
        ).hexdigest(),
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
    extraction_outputs, extraction_identities = _run_sync_stage(
        client, state_root / "extraction", max_workers=max_workers
    )
    candidates_by_prompt, _ = _parse_extraction_outputs(
        outputs=extraction_outputs,
        identities=extraction_identities,
        state_root=state_root,
        model=model,
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
        batch_sharding=False,
    )
    dedup_outputs, dedup_identities = _run_sync_stage(
        client, state_root / "dedup", max_workers=max_workers
    )
    return _finalize_outputs(
        outputs=dedup_outputs,
        identities=dedup_identities,
        state_root=state_root,
        prompts=prompts,
        candidates_by_prompt=candidates_by_prompt,
        checkpoint_id=checkpoint_id,
        output_path=output_path,
        max_online_criteria=max_online_criteria,
        control_rubric_rows=control_rubric_rows,
        model=model,
        generation_method=SYNC_GENERATION_METHOD,
    )


def build_horizon_rubrics_sync_from_files(
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
    max_workers: int = DEFAULT_SYNC_CONCURRENCY,
    client: Any | None = None,
) -> Mapping[str, Any]:
    return build_horizon_rubrics_sync(
        client=client if client is not None else _openai_sync_client(),
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
        max_workers=max_workers,
    )
