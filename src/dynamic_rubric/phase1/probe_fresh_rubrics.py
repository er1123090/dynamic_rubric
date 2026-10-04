"""Resumable fresh OnlineRubrics construction on the fixed train probe."""

from __future__ import annotations

import dataclasses
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from dynamic_rubric.artifacts import (
    read_json,
    read_jsonl,
    validate_artifact_record,
    write_json_atomic,
    write_jsonl_atomic,
)
from dynamic_rubric.hashing import sha256_file, sha256_json
from dynamic_rubric.phase1.pi0_cache import ImmutablePi0Cache
from dynamic_rubric.prompt_versions.onlinerubric_prompt import (
    build_onlinerubric_dedup_messages,
    build_onlinerubric_extractor_messages,
)
from dynamic_rubric.providers.base import GenerationRequest, RubricGenerator
from dynamic_rubric.rubrics.extractor import make_blind_pairing
from dynamic_rubric.training.online_contracts import (
    ELICITATION_PAIR_COUNT,
    PolicySnapshot,
    PromptOccurrence,
    ResponseRecord,
    RubricUnion,
    WeightedCriterion,
)
from dynamic_rubric.training.online_step import (
    DEDUP_SCHEMA,
    EXTRACTION_SCHEMA,
    OnlineStepError,
    _criterion_payload,
    _dedup_repair_schema,
    _parallel_generate,
    _parse_dedup,
    _parse_extraction,
    _receipt,
    _validate_model_identity,
)


class ProbeFreshRubricError(RuntimeError):
    pass


def _groups(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("prompt_id", "")), []).append(row)
    return grouped


def _load_inputs(
    *,
    train_path: Path,
    probe_manifest_path: Path,
    pool_a_path: Path,
    checkpoint_step: int,
    checkpoint_hash: str,
) -> tuple[list[Mapping[str, Any]], dict[str, list[Mapping[str, Any]]]]:
    manifest = read_json(probe_manifest_path)
    prompt_ids = [str(value) for value in manifest.get("prompt_ids", [])]
    if len(prompt_ids) != 100 or len(set(prompt_ids)) != 100:
        raise ProbeFreshRubricError("fixed train probe must contain exactly 100 unique prompts")
    if manifest.get("prompt_ids_sha256") != sha256_json(prompt_ids):
        raise ProbeFreshRubricError("fixed train probe prompt digest mismatch")
    if manifest.get("source_sha256") != sha256_file(train_path):
        raise ProbeFreshRubricError("fixed train probe does not bind the train source")
    train_rows = read_jsonl(train_path)
    by_prompt = {str(row.get("prompt_id", "")): row for row in train_rows}
    if len(by_prompt) != len(train_rows) or any(prompt_id not in by_prompt for prompt_id in prompt_ids):
        raise ProbeFreshRubricError("fixed train probe is not an exact train subset")
    prompts = [by_prompt[prompt_id] for prompt_id in prompt_ids]
    pool_groups = _groups(read_jsonl(pool_a_path))
    if set(pool_groups) != set(prompt_ids):
        raise ProbeFreshRubricError("Pool A prompt inventory differs from the fixed train probe")
    response_ids: set[str] = set()
    for prompt_id, rows in pool_groups.items():
        ordered = sorted(rows, key=lambda row: int(row.get("sample_index", -1)))
        if [int(row.get("sample_index", -1)) for row in ordered] != list(range(8)):
            raise ProbeFreshRubricError(f"Pool A must have sample indexes 0..7: {prompt_id}")
        if any(
            row.get("pool") != "probe_A"
            or int(row.get("policy_checkpoint", -1)) != checkpoint_step
            or row.get("checkpoint_hash") != checkpoint_hash
            or not str(row.get("response_text", "")).strip()
            for row in ordered
        ):
            raise ProbeFreshRubricError(f"Pool A provenance mismatch: {prompt_id}")
        ids = [str(row.get("response_id", "")) for row in ordered]
        if (
            any(not value for value in ids)
            or len(ids) != len(set(ids))
            or response_ids.intersection(ids)
        ):
            raise ProbeFreshRubricError("Pool A response IDs must be globally unique")
        response_ids.update(ids)
        pool_groups[prompt_id] = ordered
    return prompts, pool_groups


def _offline_criteria(prompt: Mapping[str, Any]) -> tuple[WeightedCriterion, ...]:
    criteria = prompt.get("r0", {}).get("criteria", [])
    try:
        result = tuple(
            WeightedCriterion(
                criterion_id=str(item["criterion_id"]),
                text=str(item["criterion"]),
                weight=int(item["weight_units"]),
                source="r0",
            )
            for item in criteria
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ProbeFreshRubricError("train prompt has malformed R0 criteria") from error
    if not result:
        raise ProbeFreshRubricError("train prompt has an empty R0 rubric")
    return result


def _construct_one(
    *,
    provider: RubricGenerator,
    prompt: Mapping[str, Any],
    current_rows: Sequence[Mapping[str, Any]],
    control_cache: ImmutablePi0Cache,
    run_id: str,
    checkpoint_step: int,
    checkpoint_hash: str,
    seed: int,
    extractor_model: str,
    extractor_returned_model: str,
    extractor_concurrency: int,
    output_dir: Path,
    domain: str,
    method: str,
) -> Mapping[str, Any]:
    prompt_id = str(prompt["prompt_id"])
    occurrence_id = f"fixed-train-probe:{checkpoint_step}:{prompt_id}"
    occurrence = PromptOccurrence(
        run_id=run_id,
        optimizer_update_index=max(1, checkpoint_step),
        batch_uid=f"fixed-train-probe-step-{checkpoint_step}",
        source_row_id=str(prompt.get("source_row_id", prompt_id)),
        prompt_id=prompt_id,
        prompt=tuple(dict(message) for message in prompt["messages"]),
        prompt_occurrence_id=occurrence_id,
    )
    controls, control_receipts = control_cache.bind(
        occurrence, step=max(1, checkpoint_step)
    )
    policy = PolicySnapshot(
        policy_version=checkpoint_step,
        content_hash=checkpoint_hash,
        model=str(current_rows[0]["model"]),
        revision=str(current_rows[0]["model_revision"]),
    )
    current = tuple(
        ResponseRecord(
            prompt_occurrence_id=occurrence_id,
            response_id=str(row["response_id"]),
            rollout_index=int(row["sample_index"]),
            text=str(row["response_text"]),
            policy=policy,
            family="current",
        )
        for row in current_rows
    )
    offline = _offline_criteria(prompt)
    invocation = {
        "schema_version": 1,
        "run_id": run_id,
        "checkpoint_step": checkpoint_step,
        "checkpoint_hash": checkpoint_hash,
        "prompt_id": prompt_id,
        "prompt_occurrence_id": occurrence_id,
        "seed": seed,
        "extractor_model": extractor_model,
        "pool_a_response_ids": [item.response_id for item in current],
        "pi0_control_response_ids": [item.response_id for item in controls],
        "r0_hash": sha256_json([dataclasses.asdict(item) for item in offline]),
    }
    invocation_hash = sha256_json(invocation)
    result_path = output_dir / "prompts" / f"{prompt_id}.json"
    if result_path.is_file():
        existing = read_json(result_path)
        if existing.get("invocation_hash") != invocation_hash:
            raise ProbeFreshRubricError(f"completed prompt invocation drift: {prompt_id}")
        return existing

    pairing = make_blind_pairing(
        [item.text for item in current],
        [item.text for item in controls],
        seed=seed,
        prompt_id=occurrence_id,
        step=checkpoint_step,
    )
    extraction_requests = tuple(
        GenerationRequest(
            prompt_id=prompt_id,
            messages=build_onlinerubric_extractor_messages(
                prompt=occurrence.prompt,
                existing_rubric=_criterion_payload(offline),
                response_a=pair.response_a,
                response_b=pair.response_b,
            ),
            family="online_rubric_extraction",
            seed=seed + pair_index,
            max_output_tokens=8192,
            json_schema=EXTRACTION_SCHEMA,
            schema_name="onlinerubric_extraction_v1",
            reasoning_effort="medium",
            metadata={
                "optimizer_update_index": checkpoint_step,
                "prompt_occurrence_id": occurrence_id,
                "pair_id": pair.pair_id,
                "pool": "probe_A",
            },
        )
        for pair_index, pair in enumerate(pairing.generator_payload())
    )
    extraction_results = _parallel_generate(
        provider, extraction_requests, max_concurrency=extractor_concurrency
    )
    _validate_model_identity(
        extraction_results,
        expected_requested_model=extractor_model,
        expected_returned_model=extractor_returned_model,
        family="extractor",
    )
    parsed = tuple(_parse_extraction(item) for item in extraction_results)
    candidates = tuple(candidate for group in parsed for candidate in group)
    dedup_request = GenerationRequest(
        prompt_id=prompt_id,
        messages=build_onlinerubric_dedup_messages(
            prompt=occurrence.prompt,
            existing_rubric=_criterion_payload(offline),
            candidate_criteria=candidates,
            exclude_existing=True,
        ),
        family="online_rubric_dedup",
        seed=seed,
        max_output_tokens=8192,
        json_schema=DEDUP_SCHEMA,
        schema_name="onlinerubric_dedup_v1",
        reasoning_effort="medium",
        metadata={
            "optimizer_update_index": checkpoint_step,
            "prompt_occurrence_id": occurrence_id,
            "pool": "probe_A",
        },
    )
    dedup_result = provider.generate(dedup_request)
    _validate_model_identity(
        (dedup_result,),
        expected_requested_model=extractor_model,
        expected_returned_model=extractor_returned_model,
        family="dedup",
    )
    repair = None
    try:
        online = _parse_dedup(
            dedup_result,
            occurrence_id=occurrence_id,
            offline=offline,
            candidates=candidates,
        )
    except OnlineStepError as error:
        if str(error) != "dedup introduced a criterion not grounded in candidates":
            raise
        repair_request = dataclasses.replace(
            dedup_request,
            messages=build_onlinerubric_dedup_messages(
                prompt=occurrence.prompt,
                existing_rubric=_criterion_payload(offline),
                candidate_criteria=candidates,
                exclude_existing=True,
                deterministic_provenance_repair=True,
            ),
            json_schema=_dedup_repair_schema(candidates),
            schema_name="onlinerubric_dedup_repair_v1",
            metadata={**dict(dedup_request.metadata), "dedup_attempt": 1,
                      "repair_reason": "candidate_grounding"},
        )
        repaired = provider.generate(repair_request)
        _validate_model_identity(
            (repaired,),
            expected_requested_model=extractor_model,
            expected_returned_model=extractor_returned_model,
            family="dedup repair",
        )
        online = _parse_dedup(
            repaired,
            occurrence_id=occurrence_id,
            offline=offline,
            candidates=candidates,
        )
        repair = {
            "validation_error": str(error),
            "rejected": _receipt(dedup_result, dedup_request),
            "accepted": _receipt(repaired, repair_request),
        }
        dedup_result, dedup_request = repaired, repair_request
    union = RubricUnion(occurrence_id, offline, online)
    analysis_identity = {
        "domain": domain,
        "method": method,
        "seed": seed,
        "global_step": checkpoint_step,
        "checkpoint_id": f"global_step_{checkpoint_step}",
        "prompt_id": prompt_id,
        "pool": "probe_A",
        "policy_checkpoint": checkpoint_step,
        "evaluator_checkpoint": checkpoint_step,
        "fresh_or_stale": "fresh",
    }
    value = {
        "schema_version": 1,
        "state": "verified_complete",
        **analysis_identity,
        "invocation": invocation,
        "invocation_hash": invocation_hash,
        "pool_a_provenance": [
            {
                **analysis_identity,
                "response_id": item.response_id,
                "rollout_index": item.rollout_index,
                "text_hash": item.text_hash,
            }
            for item in current
        ],
        "pi0_control_provenance": [
            {**analysis_identity, "pool": "probe_A_pi0_control", **dict(item)}
            for item in control_receipts
        ],
        "blind_pairs": [
            {"pair": dataclasses.asdict(pair), "assignment": dataclasses.asdict(assignment)}
            for pair, assignment in zip(pairing.blind_pairs, pairing.assignments)
        ],
        "extracted_criteria_before_dedup": list(candidates),
        "extraction_receipts": [
            _receipt(result, request)
            for request, result in zip(extraction_requests, extraction_results)
        ],
        "dedup_receipt": _receipt(dedup_result, dedup_request),
        "dedup_repair": repair,
        "fresh_rubric": dataclasses.asdict(union) | {"content_hash": union.content_hash},
    }
    write_json_atomic(result_path, value)
    return value


def build_probe_fresh_rubrics(
    provider: RubricGenerator,
    *,
    run_id: str,
    train_path: Path,
    probe_manifest_path: Path,
    pool_a_path: Path,
    pi0_manifest_path: Path,
    checkpoint_step: int,
    checkpoint_hash: str,
    seed: int,
    extractor_model: str,
    extractor_returned_model: str,
    output_root: Path,
    prompt_workers: int = 4,
    extractor_concurrency: int = 8,
    run_dir: Path | None = None,
    domain: str = "medicine",
    method: str = "online_rubrics",
) -> Mapping[str, Any]:
    if checkpoint_step < 0 or len(checkpoint_hash) != 64:
        raise ProbeFreshRubricError("checkpoint identity is invalid")
    if prompt_workers < 1 or extractor_concurrency < 1:
        raise ProbeFreshRubricError("concurrency must be positive")
    prompts, pool_groups = _load_inputs(
        train_path=train_path,
        probe_manifest_path=probe_manifest_path,
        pool_a_path=pool_a_path,
        checkpoint_step=checkpoint_step,
        checkpoint_hash=checkpoint_hash,
    )
    if run_dir is not None:
        from dynamic_rubric.phase1.audit_policy import (
            _validate_pool_rows,
            inspect_checkpoint,
            load_run_contract,
        )

        contract = load_run_contract(run_dir)
        checkpoint = inspect_checkpoint(contract, checkpoint_step)
        if (
            contract.run_id != run_id
            or contract.domain != domain
            or contract.method != method
            or contract.seed != seed
            or contract.train_path != train_path.resolve()
            or contract.probe_manifest_path != probe_manifest_path.resolve()
            or checkpoint.source_model_sha256 != checkpoint_hash
        ):
            raise ProbeFreshRubricError("Pool A inputs differ from the production run contract")
        pool_rows = [row for rows in pool_groups.values() for row in rows]
        _validate_pool_rows(
            pool_rows,
            contract,
            checkpoint,
            pool="probe_A",
            expected_prompt_ids={str(prompt["prompt_id"]) for prompt in prompts},
        )
        provenance_path = pool_a_path.parent / "provenance.json"
        if not provenance_path.is_file():
            raise ProbeFreshRubricError("Pool A has no adjacent provenance manifest")
        provenance = read_json(provenance_path)
        for record in provenance.get("artifacts", []):
            validate_artifact_record(record)
        expected_provenance = {
            "artifact_kind": "phase1_fixed_train_probe_policy_pools",
            "domain": domain,
            "method": method,
            "seed": seed,
            "run_id": run_id,
            "global_step": checkpoint_step,
            "checkpoint_id": f"global_step_{checkpoint_step}",
            "policy_checkpoint": checkpoint_step,
            "checkpoint_hash": checkpoint_hash,
            "config_sha256": contract.config_sha256,
            "launch_spec_sha256": contract.launch_spec_sha256,
            "probe_manifest_sha256": contract.probe_manifest_sha256,
            "train_sha256": contract.train_sha256,
        }
        if any(provenance.get(key) != value for key, value in expected_provenance.items()):
            raise ProbeFreshRubricError("Pool A provenance manifest identity mismatch")
        if (
            "probe_A" not in provenance.get("selected_pools", [])
            or provenance.get("pool_counts", {}).get("probe_A") != 800
            or not any(
                Path(str(record.get("path", ""))).resolve() == pool_a_path.resolve()
                and record.get("sha256") == sha256_file(pool_a_path)
                for record in provenance.get("artifacts", [])
            )
        ):
            raise ProbeFreshRubricError("Pool A provenance manifest artifact binding mismatch")
    control_cache = ImmutablePi0Cache(pi0_manifest_path, expected_prompt_count=1500)
    output_dir = output_root / f"checkpoint-{checkpoint_step:06d}"
    invocation = {
        "schema_version": 1,
        "run_id": run_id,
        "checkpoint_step": checkpoint_step,
        "checkpoint_hash": checkpoint_hash,
        "seed": seed,
        "extractor_model": extractor_model,
        "extractor_returned_model": extractor_returned_model,
        "train_sha256": sha256_file(train_path),
        "probe_manifest_sha256": sha256_file(probe_manifest_path),
        "pool_a_sha256": sha256_file(pool_a_path),
        "pi0_manifest_sha256": sha256_file(pi0_manifest_path),
        "prompt_count": 100,
        "pool_a_responses_per_prompt": 8,
        "pi0_controls_per_prompt": 8,
        "domain": domain,
        "method": method,
        "run_contract_dir": str(run_dir.resolve()) if run_dir is not None else None,
    }
    invocation_path = output_dir / "invocation.json"
    if invocation_path.is_file() and read_json(invocation_path) != invocation:
        raise ProbeFreshRubricError("checkpoint rubric invocation drift")
    write_json_atomic(invocation_path, invocation)
    status_path = output_dir / "status.json"
    aggregate_path = output_dir / "fresh_rubrics.jsonl"
    if status_path.is_file():
        existing_status = read_json(status_path)
        if existing_status.get("state") == "complete":
            if (
                existing_status.get("prompt_count") != 100
                or not aggregate_path.is_file()
                or existing_status.get("fresh_rubrics_sha256") != sha256_file(aggregate_path)
                or len(read_jsonl(aggregate_path)) != 100
            ):
                raise ProbeFreshRubricError("completed checkpoint rubric inventory is corrupt")
            return existing_status
    completed: list[Mapping[str, Any]] = []
    # Reused prompt artifacts are validated by _construct_one; do not count them
    # as newly generated throughput after a restart.
    existing_prompt_ids = {path.stem for path in (output_dir / "prompts").glob("*.json")}
    reused_completed = 0
    started = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    with ThreadPoolExecutor(max_workers=prompt_workers) as pool:
        futures = {
            pool.submit(
                _construct_one,
                provider=provider,
                prompt=prompt,
                current_rows=pool_groups[str(prompt["prompt_id"])],
                control_cache=control_cache,
                run_id=run_id,
                checkpoint_step=checkpoint_step,
                checkpoint_hash=checkpoint_hash,
                seed=seed,
                extractor_model=extractor_model,
                extractor_returned_model=extractor_returned_model,
                extractor_concurrency=extractor_concurrency,
                output_dir=output_dir,
                domain=domain,
                method=method,
            ): str(prompt["prompt_id"])
            for prompt in prompts
        }
        for future in as_completed(futures):
            completed.append(future.result())
            reused_completed += int(futures[future] in existing_prompt_ids)
            elapsed = time.monotonic() - started
            newly_completed = len(completed) - reused_completed
            rate = newly_completed / elapsed if elapsed > 0 and newly_completed else None
            write_json_atomic(
                status_path,
                {"schema_version": 1, "state": "running",
                 "checkpoint_step": checkpoint_step, "completed_prompts": len(completed),
                 "reused_prompts": reused_completed, "newly_completed_prompts": newly_completed,
                 "total_prompts": 100, "started_at": started_at,
                 "updated_at": datetime.now(timezone.utc).isoformat(),
                 "elapsed_seconds": elapsed, "prompts_per_second": rate,
                 "eta_seconds": ((100 - len(completed)) / rate if rate else None)},
                immutable=False,
            )
    ordered = sorted(completed, key=lambda item: str(item["invocation"]["prompt_id"]))
    write_jsonl_atomic(aggregate_path, ordered)
    summary = {
        "schema_version": 1,
        "state": "complete",
        "checkpoint_step": checkpoint_step,
        "checkpoint_hash": checkpoint_hash,
        "prompt_count": len(ordered),
        "reused_prompts": reused_completed,
        "newly_completed_prompts": len(ordered) - reused_completed,
        "extraction_request_count": len(ordered) * ELICITATION_PAIR_COUNT,
        "dedup_request_count": len(ordered),
        "fresh_rubrics_sha256": sha256_file(aggregate_path),
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json_atomic(status_path, summary, immutable=False)
    return summary
