"""Executable paper-style extraction/dedup/filter stage for horizon audits."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..artifacts import read_json, read_jsonl, write_jsonl_atomic
from ..providers.base import GenerationRequest, RubricGenerator
from .contracts import CriterionType, ImportanceClass, WeightedCriterion
from .controls import match_control_extension
from .extraction import prepare_dedup_request, prepare_extraction_requests
from .rubric_refresh import (
    DedupCluster,
    ExtractionCandidate,
    build_current_extension,
    resolve_dedup_cluster,
)


def _groups(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["prompt_id"]), []).append(row)
    return grouped


def criterion_from_artifact(item: Mapping[str, Any]) -> WeightedCriterion:
    importance = item["importance_class"]
    criterion_type = item["criterion_type"]
    return WeightedCriterion(
        criterion_instance_id=str(item["criterion_instance_id"]),
        canonical_criterion_hash=str(item["canonical_criterion_hash"]),
        text=str(item.get("text", item.get("criterion", ""))),
        importance_class=(
            importance if isinstance(importance, ImportanceClass) else ImportanceClass(str(importance))
        ),
        criterion_type=(
            criterion_type
            if isinstance(criterion_type, CriterionType)
            else CriterionType(str(criterion_type))
        ),
        weight_units=int(item["weight_units"]),
        source_candidate_ids=tuple(str(value) for value in item.get("source_candidate_ids", ())),
        source_checkpoint=item.get("source_checkpoint"),
        raw_paper_weight=item.get("raw_paper_weight"),
        distinct_source_pair_support=int(item.get("distinct_source_pair_support", 0)),
    )


def build_live_horizon_rubrics(
    provider: RubricGenerator,
    *,
    prompts: Sequence[Mapping[str, Any]],
    current_rows: Sequence[Mapping[str, Any]],
    control_rows: Sequence[Mapping[str, Any]],
    checkpoint_id: str,
    pairing_seed: int,
    extraction_schema: Mapping[str, Any],
    dedup_schema: Mapping[str, Any],
    output_path: Path,
    max_online_criteria: int = 8,
    reasoning_effort: str = "medium",
    control_rubric_rows: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run 8 independent extractions and one dedup per prompt, then publish E_t."""

    current_by_prompt = _groups(current_rows)
    control_by_prompt = _groups(control_rows)
    rubric_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    request_rows: list[dict[str, Any]] = []
    controls_by_prompt = {
        str(row["prompt_id"]): row for row in (control_rubric_rows or ())
    }
    for prompt in prompts:
        prompt_id = str(prompt["prompt_id"])
        r0 = list(prompt["r0"]["criteria"])
        requests = prepare_extraction_requests(
            prompt_id=prompt_id,
            checkpoint_id=checkpoint_id,
            prompt=prompt["messages"],
            existing_r0=r0,
            current_rows=current_by_prompt.get(prompt_id, ()),
            control_rows=control_by_prompt.get(prompt_id, ()),
            pairing_seed=pairing_seed,
        )
        candidates: dict[str, ExtractionCandidate] = {}
        for request_index, request in enumerate(requests):
            request_rows.append(request)
            result = provider.generate(
                GenerationRequest(
                    prompt_id=prompt_id,
                    messages=tuple(request["messages"]),
                    family="horizon_extraction",
                    seed=pairing_seed + request_index,
                    max_output_tokens=8192,
                    json_schema=extraction_schema,
                    schema_name="horizon_extraction_v1",
                    reasoning_effort=reasoning_effort,
                    metadata={
                        "checkpoint_id": checkpoint_id,
                        "source_pair_id": request["source_pair_id"],
                    },
                )
            )
            value = json.loads(result.text)
            for item in value["new_criteria"]:
                candidate = ExtractionCandidate(
                    candidate_id=str(item["candidate_id"]),
                    criterion=str(item["criterion"]),
                    evidence_quote=str(item["quote"]),
                    source_pair_id=str(request["source_pair_id"]),
                    source_checkpoint=checkpoint_id,
                    raw_paper_weight=int(item["weight"]),
                    importance_class=ImportanceClass(str(item["importance_class"])),
                    criterion_type=CriterionType(str(item["criterion_type"])),
                    response_a=str(request["response_a"]),
                    response_b=str(request["response_b"]),
                )
                if candidate.candidate_id in candidates:
                    raise ValueError("extractor candidate IDs must be unique per prompt/checkpoint")
                candidates[candidate.candidate_id] = candidate
                candidate_rows.append({"prompt_id": prompt_id, **asdict(candidate)})
        dedup = prepare_dedup_request(
            prompt_id=prompt_id,
            checkpoint_id=checkpoint_id,
            prompt=prompt["messages"],
            existing_r0=r0,
            candidate_criteria=[asdict(item) for item in candidates.values()],
        )
        result = provider.generate(
            GenerationRequest(
                prompt_id=prompt_id,
                messages=tuple(dedup["messages"]),
                family="horizon_dedup",
                seed=pairing_seed,
                max_output_tokens=8192,
                json_schema=dedup_schema,
                schema_name="horizon_dedup_v1",
                reasoning_effort=reasoning_effort,
                metadata={"checkpoint_id": checkpoint_id},
            )
        )
        dedup_value = json.loads(result.text)
        resolutions = [
            resolve_dedup_cluster(
                DedupCluster(
                    str(item["criterion"]), tuple(str(x) for x in item["source_candidate_ids"])
                ),
                candidates,
                prompt_id=prompt_id,
                checkpoint_id=checkpoint_id,
            )
            for item in dedup_value["final_criteria"]
        ]
        extension, rejected = build_current_extension(
            resolutions,
            candidates,
            checkpoint_id=checkpoint_id,
            r0_texts=(str(item["criterion"]) for item in r0),
            max_count=max_online_criteria,
        )
        control_match = None
        if prompt_id in controls_by_prompt:
            available = tuple(
                criterion_from_artifact(item)
                for item in controls_by_prompt[prompt_id].get("extension", ())
            )
            control_match = match_control_extension(extension, available)
        rubric_rows.append(
            {
                "schema_version": 1,
                "prompt_id": prompt_id,
                "checkpoint_id": checkpoint_id,
                "r0": r0,
                "extension": [asdict(item) for item in extension],
                "control_extension": (
                    [asdict(item) for item in control_match.selected]
                    if control_match is not None and control_match.eligible
                    else None
                ),
                "control_match": asdict(control_match) if control_match is not None else None,
                "rejected": list(rejected),
                "pool_b_inputs": [],
            }
        )
    write_jsonl_atomic(output_path, rubric_rows)
    write_jsonl_atomic(output_path.with_name("extraction_requests.jsonl"), request_rows)
    write_jsonl_atomic(output_path.with_name("extraction_candidates.jsonl"), candidate_rows)
    return {
        "rubrics": str(output_path),
        "prompt_count": len(rubric_rows),
        "request_count": len(request_rows),
        "candidate_count": len(candidate_rows),
    }


def build_live_horizon_rubrics_from_files(
    provider: RubricGenerator,
    *,
    prompts_path: Path,
    current_pool_path: Path,
    control_pool_path: Path,
    checkpoint_id: str,
    pairing_seed: int,
    extraction_schema_path: Path,
    dedup_schema_path: Path,
    output_path: Path,
    max_online_criteria: int,
    reasoning_effort: str,
    control_rubrics_path: Path | None = None,
) -> dict[str, Any]:
    return build_live_horizon_rubrics(
        provider,
        prompts=read_jsonl(prompts_path),
        current_rows=read_jsonl(current_pool_path),
        control_rows=read_jsonl(control_pool_path),
        checkpoint_id=checkpoint_id,
        pairing_seed=pairing_seed,
        extraction_schema=read_json(extraction_schema_path),
        dedup_schema=read_json(dedup_schema_path),
        output_path=output_path,
        max_online_criteria=max_online_criteria,
        reasoning_effort=reasoning_effort,
        control_rubric_rows=(
            read_jsonl(control_rubrics_path) if control_rubrics_path is not None else None
        ),
    )
