"""Pair-grounded OnlineRubrics-style bootstrap for a new prompt-specific R0.

The OnlineRubrics paper assumes an existing offline rubric and does not define a
procedure for creating that initial rubric. This experimental bootstrap keeps only
the two universal seed criteria, then applies the paper's Figure 8 pair extraction
and Figure 9 deduplication to two independent sets of pi0 responses. Artifacts are
kept separate from both the original synthetic R0 and online policy-step updates.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .artifacts import (
    artifact_record,
    read_json,
    read_jsonl,
    write_json_atomic,
    write_jsonl_atomic,
)
from .hashing import sha256_json
from .onlinerubric_batch import (
    BATCH_COMPLETION_WINDOW,
    BATCH_ENDPOINT,
    ONLINERUBRIC_BOOTSTRAP_R0_MODE,
    ONLINERUBRIC_MAX_OUTPUT_TOKENS,
    ONLINERUBRIC_PAIRWISE_COMPARISONS,
    ONLINERUBRIC_PAPER_ID,
    ONLINERUBRIC_PROMPT_VERSION,
    _dedup_request,
    _extraction_request,
    _index,
    _prompt_index,
    _reference_rows,
    _shard,
    collect_onlinerubric_dedup_batch,
)
from .pipeline import PipelineContext, StageError
from .rubrics.extractor import make_blind_pairing
from .rubrics.static import Criterion, StaticRubric, UNIVERSAL_CRITERIA
from .seeds import SeedFamily, derive_seed

ONLINERUBRIC_R0_EXTRACTION_STAGE = "onlinerubric_r0_extraction_batch"
ONLINERUBRIC_R0_DEDUP_STAGE = "onlinerubric_r0_dedup_batch"


def _universal_seed_rubric() -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "criterion_id": f"universal-{index:02d}",
            "criterion": text,
            "weight": 1,
        }
        for index, text in enumerate(UNIVERSAL_CRITERIA, start=1)
    )


def _candidate_index(path: Path) -> dict[str, tuple[Mapping[str, Any], ...]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in read_jsonl(path):
        if "response_text" not in row:
            raise StageError("pi0 static candidate is missing response_text")
        grouped[str(row["prompt_id"])].append(row)
    return {
        prompt_id: tuple(sorted(rows, key=lambda item: int(item["sample_index"])))
        for prompt_id, rows in grouped.items()
    }


def _candidate_rows(
    index: Mapping[str, tuple[Mapping[str, Any], ...]], prompt_id: str
) -> tuple[Mapping[str, Any], ...]:
    rows = index.get(prompt_id, ())
    if len(rows) < ONLINERUBRIC_PAIRWISE_COMPARISONS:
        raise StageError(f"OnlineRubrics R0 requires eight pi0 candidates: {prompt_id}")
    return rows[:ONLINERUBRIC_PAIRWISE_COMPARISONS]


def prepare_onlinerubric_r0_extraction_batch(
    context: PipelineContext,
    *,
    selected_prompt_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Prepare eight pair comparisons per prompt from independent pi0 samples."""

    if context.stage != ONLINERUBRIC_R0_EXTRACTION_STAGE:
        raise StageError(
            f"OnlineRubrics R0 extraction must use stage {ONLINERUBRIC_R0_EXTRACTION_STAGE!r}"
        )
    requested = None if selected_prompt_ids is None else tuple(map(str, selected_prompt_ids))
    if requested is not None and (not requested or len(requested) != len(set(requested))):
        raise ValueError("selected_prompt_ids must be non-empty and unique")

    run_root = context.run_root
    candidates_path = run_root / "generate-static" / "static_candidates.jsonl"
    references_path = run_root / "train-static" / "reference_responses.jsonl"
    missing = [str(path) for path in (candidates_path, references_path) if not path.is_file()]
    if missing:
        raise StageError(f"OnlineRubrics R0 source artifacts are incomplete: {missing}")

    prompts = _prompt_index(context.public_root)
    candidates = _candidate_index(candidates_path)
    references = _index(read_jsonl(references_path), text_key="response_text")
    available = tuple(sorted(set(prompts) & set(candidates)))
    if requested is None:
        prompt_ids = available
    else:
        absent = sorted(set(requested) - set(available))
        if absent:
            raise StageError(f"selected prompts are absent from pi0 candidates: {absent[:3]}")
        prompt_ids = requested

    seed_rubric = _universal_seed_rubric()
    lines: list[dict[str, Any]] = []
    identities: list[dict[str, Any]] = []
    for prompt_id in prompt_ids:
        current_rows = _candidate_rows(candidates, prompt_id)
        control_rows = _reference_rows(references, prompt_id)
        pairing = make_blind_pairing(
            [str(row["response_text"]) for row in current_rows],
            [str(row["response_text"]) for row in control_rows],
            seed=derive_seed(context.run_id, SeedFamily.PAIRING, prompt_id, 0, 0),
            prompt_id=prompt_id,
            step=0,
        )
        for pair_index, (pair, assignment) in enumerate(
            zip(pairing.generator_payload(), pairing.assignments, strict=True)
        ):
            line, identity = _extraction_request(
                context,
                prompt_id=prompt_id,
                prompt=prompts[prompt_id],
                existing_rubric=seed_rubric,
                policy_step=0,
                mode=ONLINERUBRIC_BOOTSTRAP_R0_MODE,
                control_policy="pi_0_independent_reference",
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
        raise StageError("OnlineRubrics R0 extraction custom_id collision")
    stage_root = context.stage_root()
    input_paths: list[Path] = []
    for shard_index, shard in enumerate(_shard(lines), start=1):
        path = stage_root / "inputs" / f"onlinerubric_r0_extraction_{shard_index:03d}.jsonl"
        write_jsonl_atomic(path, shard)
        input_paths.append(path)
    request_map = stage_root / "request_map.jsonl"
    write_jsonl_atomic(request_map, identities)
    manifest = {
        "schema_version": 1,
        "generation_method": "onlinerubric",
        "experiment_variant": ONLINERUBRIC_BOOTSTRAP_R0_MODE,
        "paper_id": ONLINERUBRIC_PAPER_ID,
        "paper_deviation": (
            "The paper assumes offline criteria; this bootstrap uses only two universal "
            "seed criteria and contrasts independent samples from the same pi0 policy."
        ),
        "prompt_version": ONLINERUBRIC_PROMPT_VERSION,
        "run_id": context.run_id,
        "config_hash": context.config.config_hash,
        "stage": context.stage,
        "mode": ONLINERUBRIC_BOOTSTRAP_R0_MODE,
        "policy_steps": [0],
        "prompt_count": len(prompt_ids),
        "prompt_ids": list(prompt_ids),
        "pairwise_comparisons_per_instance": ONLINERUBRIC_PAIRWISE_COMPARISONS,
        "seed_rubric": list(seed_rubric),
        "requests": len(lines),
        "max_output_tokens": ONLINERUBRIC_MAX_OUTPUT_TOKENS,
        "model": context.config.models["rubric_generator"]["requested_model"],
        "endpoint": BATCH_ENDPOINT,
        "completion_window": BATCH_COMPLETION_WINDOW,
        "input_files": [artifact_record(path) for path in input_paths],
        "request_map": artifact_record(request_map),
        "pi0_candidate_source": artifact_record(candidates_path),
        "pi0_reference_source": artifact_record(references_path),
        "gold_access": False,
        "original_prompt_specific_r0_access": False,
    }
    expected = len(prompt_ids) * ONLINERUBRIC_PAIRWISE_COMPARISONS
    if len(lines) != expected:
        raise StageError(
            f"OnlineRubrics R0 extraction inventory mismatch: {len(lines)} != {expected}"
        )
    write_json_atomic(stage_root / "manifest.json", manifest)
    return manifest


def prepare_onlinerubric_r0_dedup_batch(context: PipelineContext) -> dict[str, Any]:
    """Prepare one Figure 9 dedup request for each bootstrapped prompt rubric."""

    if context.stage != ONLINERUBRIC_R0_DEDUP_STAGE:
        raise StageError(f"OnlineRubrics R0 dedup must use stage {ONLINERUBRIC_R0_DEDUP_STAGE!r}")
    extraction_root = context.run_root / ONLINERUBRIC_R0_EXTRACTION_STAGE
    extraction_path = extraction_root / "onlinerubric_extracted_criteria.jsonl"
    extraction_manifest_path = extraction_root / "manifest.json"
    if not extraction_path.is_file() or not extraction_manifest_path.is_file():
        raise StageError("collect the OnlineRubrics R0 extraction Batch before dedup preparation")
    extraction_manifest = read_json(extraction_manifest_path)
    if extraction_manifest.get("experiment_variant") != ONLINERUBRIC_BOOTSTRAP_R0_MODE:
        raise StageError("OnlineRubrics R0 extraction manifest has the wrong variant")

    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in read_jsonl(extraction_path):
        grouped[(str(row["prompt_id"]), int(row["policy_step"]))].append(row)
    prompts = _prompt_index(context.public_root)
    seed_rubric = _universal_seed_rubric()
    lines: list[dict[str, Any]] = []
    identities: list[dict[str, Any]] = []
    for (prompt_id, policy_step), rows in sorted(grouped.items()):
        rows.sort(key=lambda item: int(item["pair_index"]))
        if policy_step != 0 or len(rows) != ONLINERUBRIC_PAIRWISE_COMPARISONS:
            raise StageError(f"invalid OnlineRubrics R0 extraction group: {prompt_id}")
        line, identity = _dedup_request(
            context,
            prompt_id=prompt_id,
            prompt=prompts[prompt_id],
            existing_rubric=seed_rubric,
            policy_step=0,
            mode=ONLINERUBRIC_BOOTSTRAP_R0_MODE,
            extraction_rows=rows,
        )
        lines.append(line)
        identities.append(identity)

    stage_root = context.stage_root()
    input_paths: list[Path] = []
    for shard_index, shard in enumerate(_shard(lines), start=1):
        path = stage_root / "inputs" / f"onlinerubric_r0_dedup_{shard_index:03d}.jsonl"
        write_jsonl_atomic(path, shard)
        input_paths.append(path)
    request_map = stage_root / "request_map.jsonl"
    write_jsonl_atomic(request_map, identities)
    manifest = {
        "schema_version": 1,
        "generation_method": "onlinerubric",
        "experiment_variant": ONLINERUBRIC_BOOTSTRAP_R0_MODE,
        "paper_id": ONLINERUBRIC_PAPER_ID,
        "prompt_version": ONLINERUBRIC_PROMPT_VERSION,
        "run_id": context.run_id,
        "config_hash": context.config.config_hash,
        "stage": context.stage,
        "mode": ONLINERUBRIC_BOOTSTRAP_R0_MODE,
        "policy_steps": [0],
        "prompt_count": len(extraction_manifest["prompt_ids"]),
        "prompt_ids": list(extraction_manifest["prompt_ids"]),
        "seed_rubric": list(seed_rubric),
        "requests": len(lines),
        "max_output_tokens": ONLINERUBRIC_MAX_OUTPUT_TOKENS,
        "model": context.config.models["rubric_generator"]["requested_model"],
        "endpoint": BATCH_ENDPOINT,
        "completion_window": BATCH_COMPLETION_WINDOW,
        "input_files": [artifact_record(path) for path in input_paths],
        "request_map": artifact_record(request_map),
        "extraction_manifest": artifact_record(extraction_manifest_path),
        "extraction_output": artifact_record(extraction_path),
        "gold_access": False,
        "original_prompt_specific_r0_access": False,
    }
    if len(lines) != int(extraction_manifest["prompt_count"]):
        raise StageError("OnlineRubrics R0 dedup inventory mismatch")
    write_json_atomic(stage_root / "manifest.json", manifest)
    return manifest


def _project_rl_compatible_r0(row: Mapping[str, Any]) -> dict[str, Any]:
    elicited = list(row["criteria"])
    if len(elicited) < 6:
        raise StageError(
            f"OnlineRubrics R0 has fewer than six task criteria: {row['prompt_id']}={len(elicited)}"
        )
    selected = sorted(
        elicited,
        key=lambda item: (
            -int(item["weight"]),
            str(item["text"]).casefold(),
            str(item["criterion_id"]),
        ),
    )[:6]
    criteria = tuple(
        Criterion(
            criterion_id=f"task-{index:02d}",
            text=str(item["text"]),
            source="task_specific",
            weight=0.125,
            created_step=0,
        )
        for index, item in enumerate(selected, start=1)
    ) + tuple(
        Criterion(
            criterion_id=f"universal-{index:02d}",
            text=text,
            source="universal",
            weight=0.125,
            created_step=0,
        )
        for index, text in enumerate(UNIVERSAL_CRITERIA, start=1)
    )
    rubric = StaticRubric(prompt_id=str(row["prompt_id"]), criteria=criteria)
    return {
        "prompt_id": rubric.prompt_id,
        "rubric_id": f"{rubric.prompt_id}:onlinerubric_bootstrap_R_0",
        "policy_step": 0,
        "trajectory": "onlinerubric_bootstrap_static",
        "criteria": [
            {
                "criterion_id": item.criterion_id,
                "text": item.text,
                "weight": item.weight,
                "source": item.source,
                "created_step": item.created_step,
            }
            for item in rubric.criteria
        ],
        "content_hash": rubric.content_hash,
        "provenance": {
            "generation_method": ONLINERUBRIC_BOOTSTRAP_R0_MODE,
            "paper_id": ONLINERUBRIC_PAPER_ID,
            "source_rubric_id": row["rubric_id"],
            "source_provider_call": row["provider_call"],
            "projection": "top_6_by_integer_weight_then_lexical_plus_2_universal",
            "gold_access": False,
            "original_prompt_specific_r0_access": False,
        },
    }


def collect_onlinerubric_r0_dedup_batch(context: PipelineContext) -> dict[str, Any]:
    """Collect full pair-grounded rubrics and emit an eight-criterion RL projection."""

    if context.stage != ONLINERUBRIC_R0_DEDUP_STAGE:
        raise StageError(
            f"OnlineRubrics R0 collection must use stage {ONLINERUBRIC_R0_DEDUP_STAGE!r}"
        )
    stage_root = context.stage_root()
    source_path = stage_root / "onlinerubric_rubrics.jsonl"
    collection_path = stage_root / "collection.json"
    if source_path.is_file() and collection_path.is_file():
        result = read_json(collection_path)
    else:
        result = collect_onlinerubric_dedup_batch(context)
    source_rows = read_jsonl(source_path)
    seed = _universal_seed_rubric()
    full_rows = [
        {
            **row,
            "generation_method": ONLINERUBRIC_BOOTSTRAP_R0_MODE,
            "rubric_id": f"{row['prompt_id']}:onlinerubric_bootstrap_full_R_0",
            "seed_criteria": list(seed),
            "elicited_criteria_count": len(row["criteria"]),
        }
        for row in source_rows
    ]
    full_path = context.stage_root() / "onlinerubric_r0_full.jsonl"
    write_jsonl_atomic(full_path, full_rows)
    projected = [_project_rl_compatible_r0(row) for row in full_rows]
    projected_path = context.stage_root() / "onlinerubric_r0_rl_compatible.jsonl"
    write_jsonl_atomic(projected_path, projected)
    enriched = {
        **result,
        "experiment_variant": ONLINERUBRIC_BOOTSTRAP_R0_MODE,
        "full_rubrics": artifact_record(full_path),
        "rl_compatible_rubrics": artifact_record(projected_path),
        "rl_projection_hash": sha256_json(projected),
    }
    write_json_atomic(context.stage_root() / "collection.json", enriched, immutable=False)
    return enriched
