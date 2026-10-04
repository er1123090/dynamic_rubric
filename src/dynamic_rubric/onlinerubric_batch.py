"""Two-stage OpenAI Batch implementation of the paper-style OnlineRubrics prompts.

This module is deliberately separate from :mod:`dynamic_rubric.batch_dynamic`.
Artifacts use an ``onlinerubric_*`` namespace and never overwrite the original
``initial_*`` experiment artifacts.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .artifacts import (
    artifact_record,
    read_json,
    read_jsonl,
    write_bytes_atomic,
    write_json_atomic,
    write_jsonl_atomic,
)
from .batch_dynamic import (
    BATCH_COMPLETION_WINDOW,
    BATCH_ENDPOINT,
    _as_dict,
    _download_file,
    _index,
    _responses_payload,
    _shard,
    control_policy_step,
)
from .hashing import canonical_json_bytes, sha256_file
from .pipeline import PipelineContext, StageError
from .prompt_versions.onlinerubric_prompt import (
    build_onlinerubric_dedup_messages,
    build_onlinerubric_extractor_messages,
)
from .providers.base import GenerationRequest
from .providers.openai_responses import _output_text
from .rubrics.extractor import make_blind_pairing
from .rubrics.replay import ReplayMode
from .seeds import SeedFamily, derive_seed

ONLINERUBRIC_PAPER_ID = "arxiv-2510.07284v2"
ONLINERUBRIC_PROMPT_VERSION = "onlinerubric-figures-8-9-v1"
ONLINERUBRIC_PAIRWISE_COMPARISONS = 8
ONLINERUBRIC_MAX_OUTPUT_TOKENS = 8192
ONLINERUBRIC_EXTRACTION_SCHEMA = "configs/schemas/onlinerubric_extraction_v1.json"
ONLINERUBRIC_DEDUP_SCHEMA = "configs/schemas/onlinerubric_dedup_v1.json"
SUPPORTED_ONLINERUBRIC_MODES = (
    ReplayMode.DYNAMIC_FIXED_BUDGETED.value,
    ReplayMode.DYNAMIC_PREV_BUDGETED.value,
)
ONLINERUBRIC_BOOTSTRAP_R0_MODE = "onlinerubric_bootstrap_r0"


def _mode_tag(mode: str) -> str:
    if mode == ONLINERUBRIC_BOOTSTRAP_R0_MODE:
        return "r0"
    parsed = ReplayMode(mode)
    if parsed is ReplayMode.DYNAMIC_FIXED_BUDGETED:
        return "fixed"
    if parsed is ReplayMode.DYNAMIC_PREV_BUDGETED:
        return "prev"
    raise ValueError(f"unsupported OnlineRubrics control mode: {mode}")


def onlinerubric_extraction_stage(mode: str) -> str:
    return f"onlinerubric_extraction_{_mode_tag(mode)}_batch"


def onlinerubric_dedup_stage(mode: str) -> str:
    return f"onlinerubric_dedup_{_mode_tag(mode)}_batch"


def _prompt_index(public_root: Path) -> dict[str, tuple[Mapping[str, str], ...]]:
    prompts: dict[str, tuple[Mapping[str, str], ...]] = {}
    for path in sorted(public_root.glob("pilot_*.jsonl")):
        for row in read_jsonl(path):
            prompt_id = str(row["prompt_id"])
            raw_messages = row.get("messages")
            if not isinstance(raw_messages, list) or not raw_messages:
                raise StageError(f"public prompt has no messages: {prompt_id}")
            messages = tuple(
                {"role": str(item["role"]), "content": str(item["content"])}
                for item in raw_messages
            )
            previous = prompts.setdefault(prompt_id, messages)
            if previous != messages:
                raise StageError(f"prompt text drift across public splits: {prompt_id}")
    return prompts


def _integer_weighted_rubric(criteria: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    if not criteria:
        raise StageError("existing rubric is empty")
    magnitudes = [abs(float(item["weight"])) for item in criteria if float(item["weight"]) != 0]
    if not magnitudes:
        raise StageError("existing rubric has no non-zero weights")
    unit = min(magnitudes)
    output: list[dict[str, Any]] = []
    for item in criteria:
        raw_weight = float(item["weight"])
        if raw_weight == 0:
            weight = 0
        else:
            magnitude = max(1, round(abs(raw_weight) / unit))
            weight = magnitude if raw_weight > 0 else -magnitude
        output.append(
            {
                "criterion_id": str(item["criterion_id"]),
                "criterion": str(item["text"]),
                "weight": weight,
            }
        )
    return tuple(output)


def _rubric_index(run_root: Path) -> dict[str, tuple[dict[str, Any], ...]]:
    path = run_root / "generate-static" / "static_rubrics.jsonl"
    if not path.is_file():
        raise StageError(f"initial R0 artifact is missing: {path}")
    rubrics: dict[str, tuple[dict[str, Any], ...]] = {}
    for row in read_jsonl(path):
        prompt_id = str(row["prompt_id"])
        if prompt_id in rubrics:
            raise StageError(f"duplicate initial rubric: {prompt_id}")
        rubrics[prompt_id] = _integer_weighted_rubric(row["criteria"])
    return rubrics


def _trajectory_rows(
    index: Mapping[tuple[str, int, str], tuple[Mapping[str, Any], ...]],
    prompt_id: str,
    step: int,
) -> tuple[Mapping[str, Any], ...]:
    rows: list[Mapping[str, Any]] = []
    for family in (SeedFamily.TRAJECTORY_DISCOVERY, SeedFamily.TRAJECTORY_VALIDATION):
        family_rows = index.get((prompt_id, step, family.value), ())
        if len(family_rows) < 4:
            raise StageError(
                "OnlineRubrics requires four responses from each trajectory family: "
                f"{prompt_id=} {step=} {family.value=}"
            )
        rows.extend(family_rows[:4])
    if len(rows) != ONLINERUBRIC_PAIRWISE_COMPARISONS:
        raise AssertionError("trajectory response inventory must contain exactly eight rows")
    return tuple(rows)


def _reference_rows(
    index: Mapping[tuple[str, int, str], tuple[Mapping[str, Any], ...]],
    prompt_id: str,
) -> tuple[Mapping[str, Any], ...]:
    rows = index.get((prompt_id, 0, SeedFamily.REFERENCE_DISCOVERY.value), ())
    if len(rows) < ONLINERUBRIC_PAIRWISE_COMPARISONS:
        raise StageError(f"OnlineRubrics requires eight reference responses: {prompt_id}")
    return rows[:ONLINERUBRIC_PAIRWISE_COMPARISONS]


def _identity_hash(identity: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(identity)).hexdigest()


def _extraction_request(
    context: PipelineContext,
    *,
    prompt_id: str,
    prompt: Sequence[Mapping[str, str]],
    existing_rubric: Sequence[Mapping[str, Any]],
    policy_step: int,
    mode: str,
    control_policy: str,
    pair: Mapping[str, str],
    assignment: Mapping[str, Any],
    current_rows: Sequence[Mapping[str, Any]],
    control_rows: Sequence[Mapping[str, Any]],
    pair_index: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    messages = build_onlinerubric_extractor_messages(
        prompt=prompt,
        existing_rubric=existing_rubric,
        response_a=str(pair["response_a"]),
        response_b=str(pair["response_b"]),
    )
    logical_seed = derive_seed(
        context.run_id,
        SeedFamily.PAIRING,
        prompt_id,
        policy_step,
        pair_index,
    )
    request = GenerationRequest(
        prompt_id=f"{prompt_id}-s{policy_step}-pair-{pair_index}",
        messages=messages,
        family=f"onlinerubric_extraction:{mode}",
        seed=logical_seed,
        max_output_tokens=ONLINERUBRIC_MAX_OUTPUT_TOKENS,
        json_schema=read_json(context.root / ONLINERUBRIC_EXTRACTION_SCHEMA),
        schema_name="onlinerubric_extraction_v1",
        reasoning_effort=str(context.config.models["rubric_generator"]["reasoning_effort"]),
        metadata={
            "run_id": context.run_id,
            "prompt_id": prompt_id,
            "policy_step": policy_step,
            "pair_index": pair_index,
            "method": "onlinerubric",
        },
    )
    current_index = int(assignment["current_index"])
    control_index = int(assignment["control_index"])
    identity = {
        "generation_method": "onlinerubric",
        "paper_id": ONLINERUBRIC_PAPER_ID,
        "prompt_version": ONLINERUBRIC_PROMPT_VERSION,
        "prompt_id": prompt_id,
        "policy_step": policy_step,
        "mode": mode,
        "control_policy": control_policy,
        "pair_index": pair_index,
        "pair_id": str(pair["pair_id"]),
        "current_label": str(assignment["current_label"]),
        "control_label": str(assignment["control_label"]),
        "current_response_id": str(current_rows[current_index]["response_id"]),
        "control_response_id": str(control_rows[control_index]["response_id"]),
        "logical_seed": logical_seed,
    }
    digest = _identity_hash(identity)
    custom_id = f"onlinerubric-extract-{_mode_tag(mode)}-{policy_step:03d}-{digest[:24]}"
    model = str(context.config.models["rubric_generator"]["requested_model"])
    return (
        {
            "custom_id": custom_id,
            "method": "POST",
            "url": BATCH_ENDPOINT,
            "body": _responses_payload(model, request),
        },
        {"custom_id": custom_id, **identity},
    )


def prepare_onlinerubric_extraction_batch(
    context: PipelineContext,
    *,
    max_step: int = 50,
    policy_steps: Sequence[int] | None = None,
    selected_prompt_ids: Sequence[str] | None = None,
    mode: str = ReplayMode.DYNAMIC_PREV_BUDGETED.value,
) -> dict[str, Any]:
    """Prepare eight independent Figure-8 extraction calls per prompt and policy step."""

    if max_step <= 0:
        raise ValueError("max_step must be positive")
    if mode not in SUPPORTED_ONLINERUBRIC_MODES:
        raise ValueError(f"unsupported OnlineRubrics control mode: {mode}")
    steps = tuple(range(1, max_step + 1)) if policy_steps is None else tuple(policy_steps)
    if not steps or any(step <= 0 or step > max_step for step in steps):
        raise ValueError("policy_steps must be non-empty and within 1..max_step")
    if len(steps) != len(set(steps)) or tuple(sorted(steps)) != steps:
        raise ValueError("policy_steps must be unique and strictly increasing")
    requested_prompt_ids = (
        None if selected_prompt_ids is None else tuple(str(item) for item in selected_prompt_ids)
    )
    if requested_prompt_ids is not None and (
        not requested_prompt_ids or len(requested_prompt_ids) != len(set(requested_prompt_ids))
    ):
        raise ValueError("selected_prompt_ids must be non-empty and unique")
    expected_stage = onlinerubric_extraction_stage(mode)
    if context.stage != expected_stage:
        raise StageError(f"OnlineRubrics extraction must use stage {expected_stage!r}")

    run_root = context.run_root
    reference_path = run_root / "train-static" / "reference_responses.jsonl"
    probe_steps = set(steps)
    if mode == ReplayMode.DYNAMIC_PREV_BUDGETED.value:
        probe_steps.update(step - 1 for step in steps if step > 1)
    probe_paths = {
        step: run_root / "train-static" / "verl-run" / "probes" / f"{step}.jsonl"
        for step in sorted(probe_steps)
    }
    missing = [str(path) for path in (reference_path, *probe_paths.values()) if not path.is_file()]
    if missing:
        raise StageError(f"OnlineRubrics source artifacts are incomplete: {missing[:3]}")

    prompts = _prompt_index(context.public_root)
    rubrics = _rubric_index(run_root)
    references = _index(read_jsonl(reference_path), text_key="response_text")
    available_prompt_ids = sorted(
        key[0]
        for key in references
        if key[1] == 0 and key[2] == SeedFamily.REFERENCE_DISCOVERY.value
    )
    if len(available_prompt_ids) != len(set(available_prompt_ids)):
        raise StageError("reference prompt inventory is not unique")
    if requested_prompt_ids is None:
        prompt_ids = tuple(available_prompt_ids)
    else:
        missing_prompts = sorted(set(requested_prompt_ids) - set(available_prompt_ids))
        if missing_prompts:
            raise StageError(f"selected prompts are absent from references: {missing_prompts[:3]}")
        prompt_ids = requested_prompt_ids

    probe_indexes = {
        step: _index(read_jsonl(path), text_key="output") for step, path in probe_paths.items()
    }
    lines: list[dict[str, Any]] = []
    identities: list[dict[str, Any]] = []
    for step in steps:
        current_index = probe_indexes[step]
        for prompt_id in prompt_ids:
            if prompt_id not in prompts or prompt_id not in rubrics:
                raise StageError(f"prompt or initial rubric is missing: {prompt_id}")
            current_rows = _trajectory_rows(current_index, prompt_id, step)
            control_step = control_policy_step(mode, step)
            if control_step == 0:
                control_rows = _reference_rows(references, prompt_id)
            else:
                control_rows = _trajectory_rows(
                    probe_indexes[control_step], prompt_id, control_step
                )
            pairing = make_blind_pairing(
                [str(row["response_text"]) for row in current_rows],
                [str(row["response_text"]) for row in control_rows],
                seed=derive_seed(context.run_id, SeedFamily.PAIRING, prompt_id, step, 0),
                prompt_id=prompt_id,
                step=step,
            )
            for pair_index, (pair, assignment) in enumerate(
                zip(pairing.generator_payload(), pairing.assignments, strict=True)
            ):
                line, identity = _extraction_request(
                    context,
                    prompt_id=prompt_id,
                    prompt=prompts[prompt_id],
                    existing_rubric=rubrics[prompt_id],
                    policy_step=step,
                    mode=mode,
                    control_policy=f"pi_{control_step}",
                    pair={
                        "pair_id": pair.pair_id,
                        "response_a": pair.response_a,
                        "response_b": pair.response_b,
                    },
                    assignment={
                        "current_label": assignment.current_label,
                        "control_label": assignment.control_label,
                        "current_index": assignment.current_index,
                        "control_index": assignment.control_index,
                    },
                    current_rows=current_rows,
                    control_rows=control_rows,
                    pair_index=pair_index,
                )
                lines.append(line)
                identities.append(identity)

    if len({line["custom_id"] for line in lines}) != len(lines):
        raise StageError("OnlineRubrics extraction custom_id collision")
    stage_root = context.stage_root()
    input_paths: list[Path] = []
    for shard_index, shard in enumerate(_shard(lines), start=1):
        path = stage_root / "inputs" / f"onlinerubric_extraction_{shard_index:03d}.jsonl"
        write_jsonl_atomic(path, shard)
        input_paths.append(path)
    request_map = stage_root / "request_map.jsonl"
    write_jsonl_atomic(request_map, identities)
    manifest = {
        "schema_version": 1,
        "generation_method": "onlinerubric",
        "paper_id": ONLINERUBRIC_PAPER_ID,
        "prompt_version": ONLINERUBRIC_PROMPT_VERSION,
        "run_id": context.run_id,
        "config_hash": context.config.config_hash,
        "stage": context.stage,
        "mode": mode,
        "max_step": max(steps),
        "policy_steps": list(steps),
        "pairwise_comparisons_per_instance": ONLINERUBRIC_PAIRWISE_COMPARISONS,
        "prompt_count": len(prompt_ids),
        "prompt_ids": list(prompt_ids),
        "requests": len(lines),
        "max_output_tokens": ONLINERUBRIC_MAX_OUTPUT_TOKENS,
        "model": context.config.models["rubric_generator"]["requested_model"],
        "endpoint": BATCH_ENDPOINT,
        "completion_window": BATCH_COMPLETION_WINDOW,
        "input_files": [artifact_record(path) for path in input_paths],
        "request_map": artifact_record(request_map),
        "initial_rubrics": artifact_record(run_root / "generate-static" / "static_rubrics.jsonl"),
        "reference_source": artifact_record(reference_path),
        "probe_sources": [artifact_record(path) for path in probe_paths.values()],
    }
    expected = len(prompt_ids) * len(steps) * ONLINERUBRIC_PAIRWISE_COMPARISONS
    if len(lines) != expected:
        raise StageError(f"OnlineRubrics extraction inventory mismatch: {len(lines)} != {expected}")
    write_json_atomic(stage_root / "manifest.json", manifest)
    return manifest


def _require_api_key() -> str:
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise StageError("OPENAI_API_KEY is not set")
    return api_key


def _openai_client() -> Any:
    try:
        from openai import OpenAI  # type: ignore[import-not-found]
    except ImportError as error:
        raise StageError("the OpenAI Python SDK is required for Batch operations") from error
    return OpenAI(api_key=_require_api_key(), max_retries=8)


def submit_onlinerubric_batch(context: PipelineContext) -> dict[str, Any]:
    """Submit a prepared extraction or dedup stage without sharing initial artifacts."""

    stage_root = context.stage_root()
    manifest_path = stage_root / "manifest.json"
    if not manifest_path.is_file():
        raise StageError(f"prepare {context.stage} before Batch submission")
    manifest = read_json(manifest_path)
    if manifest.get("generation_method") != "onlinerubric":
        raise StageError("refusing to submit a non-OnlineRubrics manifest")
    receipt_path = stage_root / "submission.json"
    if receipt_path.is_file():
        return read_json(receipt_path)

    client = _openai_client()
    jobs: list[dict[str, Any]] = []
    for shard_index, record in enumerate(manifest["input_files"], start=1):
        shard_receipt = stage_root / f"submission-{shard_index:03d}.json"
        if shard_receipt.is_file():
            job = read_json(shard_receipt)
            if job.get("input_sha256") != record["sha256"]:
                raise StageError("existing submission does not match prepared OnlineRubrics input")
            jobs.append(job)
            continue
        input_path = Path(str(record["path"]))
        if sha256_file(input_path) != record["sha256"]:
            raise StageError(f"OnlineRubrics Batch input changed: {input_path}")
        with input_path.open("rb") as stream:
            uploaded = _as_dict(client.files.create(file=stream, purpose="batch"))
        created = _as_dict(
            client.batches.create(
                input_file_id=str(uploaded["id"]),
                endpoint=BATCH_ENDPOINT,
                completion_window=BATCH_COMPLETION_WINDOW,
                metadata={
                    "run_id": context.run_id,
                    "stage": context.stage,
                    "method": "onlinerubric",
                    "shard": str(shard_index),
                },
            )
        )
        job = {
            "shard": shard_index,
            "input_path": str(input_path),
            "input_sha256": record["sha256"],
            "input_file_id": str(uploaded["id"]),
            "batch_id": str(created["id"]),
            "status": str(created.get("status", "validating")),
        }
        write_json_atomic(shard_receipt, job)
        jobs.append(job)
    receipt = {
        "schema_version": 1,
        "generation_method": "onlinerubric",
        "run_id": context.run_id,
        "stage": context.stage,
        "manifest_sha256": sha256_file(manifest_path),
        "jobs": jobs,
    }
    write_json_atomic(receipt_path, receipt)
    return receipt


def onlinerubric_batch_status(context: PipelineContext) -> dict[str, Any]:
    stage_root = context.stage_root()
    receipt = read_json(stage_root / "submission.json")
    client = _openai_client()
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
    status = {
        "generation_method": "onlinerubric",
        "run_id": context.run_id,
        "stage": context.stage,
        "jobs": jobs,
    }
    write_json_atomic(stage_root / "status.json", status, immutable=False)
    return status


def _collect_rows(
    context: PipelineContext,
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]], Mapping[str, Any]]:
    stage_root = context.stage_root()
    manifest = read_json(stage_root / "manifest.json")
    receipt = read_json(stage_root / "submission.json")
    if receipt.get("manifest_sha256") != sha256_file(stage_root / "manifest.json"):
        raise StageError("OnlineRubrics submission is not bound to the current manifest")
    identities = {str(row["custom_id"]): row for row in read_jsonl(manifest["request_map"]["path"])}
    if len(identities) != int(manifest["requests"]):
        raise StageError("OnlineRubrics request identity inventory is incomplete")

    client = _openai_client()
    outputs: dict[str, Mapping[str, Any]] = {}
    for job in receipt["jobs"]:
        current = _as_dict(client.batches.retrieve(str(job["batch_id"])))
        if current.get("status") != "completed" or not current.get("output_file_id"):
            raise StageError(
                f"OnlineRubrics Batch is not complete: {current.get('id')}={current.get('status')}"
            )
        shard = int(job["shard"])
        payload = _download_file(client, str(current["output_file_id"]))
        raw_path = stage_root / "outputs" / f"onlinerubric_{shard:03d}.raw.jsonl"
        write_bytes_atomic(raw_path, payload)
        input_ids = {str(row["custom_id"]) for row in read_jsonl(Path(str(job["input_path"])))}
        rows = [json.loads(line) for line in payload.splitlines() if line.strip()]
        output_ids = {str(row.get("custom_id", "")) for row in rows}
        if output_ids != input_ids:
            raise StageError(f"OnlineRubrics Batch output inventory mismatch for shard {shard}")
        for row in rows:
            custom_id = str(row["custom_id"])
            if custom_id in outputs:
                raise StageError(f"duplicate OnlineRubrics output: {custom_id}")
            outputs[custom_id] = row
        error_file_id = current.get("error_file_id")
        if error_file_id:
            error_payload = _download_file(client, str(error_file_id))
            error_path = stage_root / "outputs" / f"onlinerubric_{shard:03d}.errors.jsonl"
            write_bytes_atomic(error_path, error_payload)
            if error_payload.strip():
                raise StageError(f"OnlineRubrics Batch shard {shard} produced errors")
    if set(outputs) != set(identities):
        raise StageError("combined OnlineRubrics Batch output inventory is incomplete")
    return outputs, identities, manifest


def _parsed_body(row: Mapping[str, Any], custom_id: str) -> tuple[Mapping[str, Any], Any]:
    if row.get("error") is not None:
        raise StageError(f"OnlineRubrics request failed: {custom_id}: {row['error']}")
    response = row.get("response")
    if not isinstance(response, Mapping) or int(response.get("status_code", 0)) != 200:
        raise StageError(f"OnlineRubrics response is not HTTP 200: {custom_id}")
    body = response.get("body")
    if not isinstance(body, Mapping):
        raise StageError(f"OnlineRubrics response body is missing: {custom_id}")
    if body.get("status") not in (None, "completed"):
        raise StageError(f"OnlineRubrics response is incomplete: {custom_id}")
    try:
        parsed = json.loads(_output_text(body))
    except json.JSONDecodeError as error:
        raise StageError(f"OnlineRubrics structured output is invalid JSON: {custom_id}") from error
    if not isinstance(parsed, Mapping):
        raise StageError(f"OnlineRubrics structured output is not an object: {custom_id}")
    return body, parsed


def _provider_call(body: Mapping[str, Any], requested_model: str) -> dict[str, Any]:
    return {
        "requested_model": requested_model,
        "returned_model": str(body.get("model", "")),
        "request_id": str(body.get("id", "")),
        "created_at": body.get("created_at", body.get("created")),
        "usage": dict(body.get("usage", {})),
        "raw_response_hash": hashlib.sha256(canonical_json_bytes(body)).hexdigest(),
    }


def collect_onlinerubric_extraction_batch(context: PipelineContext) -> dict[str, Any]:
    outputs, identities, manifest = _collect_rows(context)
    rows: list[dict[str, Any]] = []
    returned_models: set[str] = set()
    for custom_id, identity in identities.items():
        body, parsed = _parsed_body(outputs[custom_id], custom_id)
        analysis = parsed.get("analysis")
        criteria = parsed.get("new_criteria")
        if not isinstance(analysis, str) or not isinstance(criteria, list):
            raise StageError(f"OnlineRubrics extraction schema violation: {custom_id}")
        normalized: list[dict[str, Any]] = []
        for item in criteria:
            if (
                not isinstance(item, Mapping)
                or not isinstance(item.get("quote"), str)
                or not str(item["quote"]).strip()
                or not isinstance(item.get("criterion"), str)
                or not str(item["criterion"]).strip()
                or not isinstance(item.get("weight"), int)
                or int(item["weight"]) <= 0
            ):
                raise StageError(f"OnlineRubrics extracted criterion is invalid: {custom_id}")
            normalized.append(
                {
                    "quote": str(item["quote"]),
                    "criterion": str(item["criterion"]),
                    "weight": int(item["weight"]),
                }
            )
        provider_call = _provider_call(body, str(manifest["model"]))
        returned_models.add(str(provider_call["returned_model"]))
        rows.append(
            {
                **dict(identity),
                "analysis": analysis,
                "new_criteria": normalized,
                "provider_call": provider_call,
            }
        )
    if len(returned_models) != 1 or "" in returned_models:
        raise StageError(f"OnlineRubrics returned-model drift: {sorted(returned_models)}")
    rows.sort(key=lambda item: (item["prompt_id"], item["policy_step"], item["pair_index"]))
    output_path = context.stage_root() / "onlinerubric_extracted_criteria.jsonl"
    write_jsonl_atomic(output_path, rows)
    result = {
        "generation_method": "onlinerubric",
        "stage": context.stage,
        "requests": len(rows),
        "criteria": sum(len(row["new_criteria"]) for row in rows),
        "returned_model": next(iter(returned_models)),
        "output": artifact_record(output_path),
    }
    write_json_atomic(context.stage_root() / "collection.json", result)
    return result


def _dedup_request(
    context: PipelineContext,
    *,
    prompt_id: str,
    prompt: Sequence[Mapping[str, str]],
    existing_rubric: Sequence[Mapping[str, Any]],
    policy_step: int,
    mode: str,
    extraction_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for row in extraction_rows:
        for candidate_index, item in enumerate(row["new_criteria"]):
            candidates.append(
                {
                    "source_pair_id": str(row["pair_id"]),
                    "source_candidate_index": candidate_index,
                    "quote": str(item["quote"]),
                    "criterion": str(item["criterion"]),
                    "weight": int(item["weight"]),
                }
            )
    request = GenerationRequest(
        prompt_id=f"{prompt_id}-s{policy_step}-dedup",
        messages=build_onlinerubric_dedup_messages(
            prompt=prompt,
            existing_rubric=existing_rubric,
            candidate_criteria=candidates,
        ),
        family=f"onlinerubric_dedup:{mode}",
        seed=derive_seed(context.run_id, SeedFamily.PAIRING, prompt_id, policy_step, 99),
        max_output_tokens=ONLINERUBRIC_MAX_OUTPUT_TOKENS,
        json_schema=read_json(context.root / ONLINERUBRIC_DEDUP_SCHEMA),
        schema_name="onlinerubric_dedup_v1",
        reasoning_effort=str(context.config.models["rubric_generator"]["reasoning_effort"]),
        metadata={
            "run_id": context.run_id,
            "prompt_id": prompt_id,
            "policy_step": policy_step,
            "method": "onlinerubric",
            "stage": "dedup",
        },
    )
    identity = {
        "generation_method": "onlinerubric",
        "paper_id": ONLINERUBRIC_PAPER_ID,
        "prompt_version": ONLINERUBRIC_PROMPT_VERSION,
        "prompt_id": prompt_id,
        "policy_step": policy_step,
        "mode": mode,
        "control_policy": str(extraction_rows[0]["control_policy"]),
        "source_pair_ids": [str(row["pair_id"]) for row in extraction_rows],
        "source_request_ids": [str(row["custom_id"]) for row in extraction_rows],
        "candidate_count": len(candidates),
    }
    digest = _identity_hash(identity)
    custom_id = f"onlinerubric-dedup-{_mode_tag(mode)}-{policy_step:03d}-{digest[:24]}"
    model = str(context.config.models["rubric_generator"]["requested_model"])
    return (
        {
            "custom_id": custom_id,
            "method": "POST",
            "url": BATCH_ENDPOINT,
            "body": _responses_payload(model, request),
        },
        {"custom_id": custom_id, **identity},
    )


def prepare_onlinerubric_dedup_batch(
    context: PipelineContext,
    *,
    mode: str = ReplayMode.DYNAMIC_PREV_BUDGETED.value,
) -> dict[str, Any]:
    """Aggregate each set of eight pair-level extractions with the Figure 9 prompt."""

    expected_stage = onlinerubric_dedup_stage(mode)
    if context.stage != expected_stage:
        raise StageError(f"OnlineRubrics dedup must use stage {expected_stage!r}")
    extraction_root = context.run_root / onlinerubric_extraction_stage(mode)
    extraction_path = extraction_root / "onlinerubric_extracted_criteria.jsonl"
    extraction_manifest_path = extraction_root / "manifest.json"
    if not extraction_path.is_file() or not extraction_manifest_path.is_file():
        raise StageError("collect the OnlineRubrics extraction Batch before dedup preparation")
    extraction_manifest = read_json(extraction_manifest_path)
    if extraction_manifest.get("mode") != mode:
        raise StageError("OnlineRubrics extraction mode does not match dedup mode")

    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in read_jsonl(extraction_path):
        grouped[(str(row["prompt_id"]), int(row["policy_step"]))].append(row)
    prompts = _prompt_index(context.public_root)
    rubrics = _rubric_index(context.run_root)
    lines: list[dict[str, Any]] = []
    identities: list[dict[str, Any]] = []
    for (prompt_id, policy_step), rows in sorted(grouped.items()):
        rows.sort(key=lambda item: int(item["pair_index"]))
        if len(rows) != ONLINERUBRIC_PAIRWISE_COMPARISONS:
            raise StageError(
                "dedup requires eight pairwise extraction results: "
                f"{prompt_id=} {policy_step=} count={len(rows)}"
            )
        line, identity = _dedup_request(
            context,
            prompt_id=prompt_id,
            prompt=prompts[prompt_id],
            existing_rubric=rubrics[prompt_id],
            policy_step=policy_step,
            mode=mode,
            extraction_rows=rows,
        )
        lines.append(line)
        identities.append(identity)

    stage_root = context.stage_root()
    input_paths: list[Path] = []
    for shard_index, shard in enumerate(_shard(lines), start=1):
        path = stage_root / "inputs" / f"onlinerubric_dedup_{shard_index:03d}.jsonl"
        write_jsonl_atomic(path, shard)
        input_paths.append(path)
    request_map = stage_root / "request_map.jsonl"
    write_jsonl_atomic(request_map, identities)
    manifest = {
        "schema_version": 1,
        "generation_method": "onlinerubric",
        "paper_id": ONLINERUBRIC_PAPER_ID,
        "prompt_version": ONLINERUBRIC_PROMPT_VERSION,
        "run_id": context.run_id,
        "config_hash": context.config.config_hash,
        "stage": context.stage,
        "mode": mode,
        "policy_steps": list(extraction_manifest.get("policy_steps", [])),
        "prompt_ids": list(extraction_manifest.get("prompt_ids", [])),
        "requests": len(lines),
        "max_output_tokens": ONLINERUBRIC_MAX_OUTPUT_TOKENS,
        "model": context.config.models["rubric_generator"]["requested_model"],
        "endpoint": BATCH_ENDPOINT,
        "completion_window": BATCH_COMPLETION_WINDOW,
        "input_files": [artifact_record(path) for path in input_paths],
        "request_map": artifact_record(request_map),
        "extraction_manifest": artifact_record(extraction_manifest_path),
        "extraction_output": artifact_record(extraction_path),
    }
    policy_step_count = len(extraction_manifest.get("policy_steps", ()))
    if policy_step_count == 0:
        policy_step_count = int(extraction_manifest["max_step"])
    expected = int(extraction_manifest["prompt_count"]) * policy_step_count
    if len(lines) != expected:
        raise StageError(f"OnlineRubrics dedup inventory mismatch: {len(lines)} != {expected}")
    write_json_atomic(stage_root / "manifest.json", manifest)
    return manifest


def collect_onlinerubric_dedup_batch(context: PipelineContext) -> dict[str, Any]:
    outputs, identities, manifest = _collect_rows(context)
    rows: list[dict[str, Any]] = []
    returned_models: set[str] = set()
    for custom_id, identity in identities.items():
        body, parsed = _parsed_body(outputs[custom_id], custom_id)
        analysis = parsed.get("analysis")
        criteria = parsed.get("final_criteria")
        if not isinstance(analysis, str) or not isinstance(criteria, list):
            raise StageError(f"OnlineRubrics dedup schema violation: {custom_id}")
        normalized: list[dict[str, Any]] = []
        for criterion_index, item in enumerate(criteria):
            if (
                not isinstance(item, Mapping)
                or not isinstance(item.get("criterion"), str)
                or not str(item["criterion"]).strip()
                or not isinstance(item.get("weight"), int)
                or int(item["weight"]) <= 0
            ):
                raise StageError(f"OnlineRubrics deduplicated criterion is invalid: {custom_id}")
            normalized.append(
                {
                    "criterion_id": (
                        f"onlinerubric-{int(identity['policy_step']):03d}-{criterion_index:03d}"
                    ),
                    "text": str(item["criterion"]),
                    "weight": int(item["weight"]),
                    "source": "onlinerubric_pairwise",
                    "created_step": int(identity["policy_step"]),
                }
            )
        provider_call = _provider_call(body, str(manifest["model"]))
        returned_models.add(str(provider_call["returned_model"]))
        rows.append(
            {
                **dict(identity),
                "rubric_id": (
                    f"{identity['prompt_id']}:onlinerubric_R_{int(identity['policy_step'])}"
                ),
                "analysis": analysis,
                "criteria": normalized,
                "provider_call": provider_call,
            }
        )
    if len(returned_models) != 1 or "" in returned_models:
        raise StageError(f"OnlineRubrics returned-model drift: {sorted(returned_models)}")
    rows.sort(key=lambda item: (item["prompt_id"], item["policy_step"]))
    output_path = context.stage_root() / "onlinerubric_rubrics.jsonl"
    write_jsonl_atomic(output_path, rows)
    result = {
        "generation_method": "onlinerubric",
        "stage": context.stage,
        "rubrics": len(rows),
        "criteria": sum(len(row["criteria"]) for row in rows),
        "returned_model": next(iter(returned_models)),
        "output": artifact_record(output_path),
    }
    write_json_atomic(context.stage_root() / "collection.json", result)
    return result
