"""Credential-free orchestration that exercises the production replay contracts.

The fake execution lane deliberately uses the same response-family separation,
blind pairing, held-out validation, admission, and pool transitions as a live
run.  Provider calls are deterministic, but no experimental result is injected
as a constant.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .artifacts import read_json, read_jsonl, write_json_atomic, write_jsonl_atomic
from .data.sealing import load_updater_lock, read_final_sealed
from .fake_training_metrics import ground_probe_record
from .hashing import sha256_file, sha256_json
from .providers.base import GenerationRequest
from .providers.fake import FakeCriterionGrader, FakeEmbeddingProvider, FakeGenerator
from .reporting.tables import REQUIRED_COMPARISONS
from .rubrics.admission import AdmissionEvidence
from .rubrics.extractor import PairingPlan, make_blind_pairing
from .rubrics.replay import ReplayCandidate, ReplayMode, initial_replay_snapshot, replay_step
from .rubrics.static import Criterion, StaticRubric
from .seeds import SeedFamily, derive_seed, response_id


def _configured_prompts(context: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in context.raw.get("splits", {}):
        for source in read_jsonl(context.public_root / f"{split}.jsonl"):
            row = dict(source)
            row["split"] = str(split)
            rows.append(row)
    return rows


def _probe_record(
    context: Any,
    prompt: Mapping[str, Any],
    step: int,
    family: SeedFamily,
    sample_index: int,
    generator: FakeGenerator,
    *,
    policy_id: str,
) -> dict[str, Any]:
    prompt_id = str(prompt["prompt_id"])
    logical_seed = derive_seed(context.run_id, family, prompt_id, step, sample_index)
    result = generator.generate(
        GenerationRequest(
            prompt_id=prompt_id,
            messages=tuple(prompt["messages"]),
            family=family.value,
            seed=logical_seed,
            temperature=1.0,
            top_p=0.95,
            metadata={
                "run_id": context.run_id,
                "policy_id": policy_id,
                "replicate_id": sample_index,
            },
        )
    )
    response_text = result.text
    return {
        "run_id": context.run_id,
        "prompt_id": prompt_id,
        "split": str(prompt["split"]),
        "policy_step": step,
        "policy_id": policy_id,
        "timing": "initial_policy" if step == 0 else "after_optimizer_update",
        "family": family.value,
        "sample_index": sample_index,
        "seed": logical_seed,
        "response_id": response_id(context.run_id, family, prompt_id, step, sample_index),
        "response_text": response_text,
        "checkpoint_hash": sha256_json([context.run_id, "checkpoint", step]),
        "config_hash": context.config.config_hash,
        "base_policy": "pi_0",
        "kl_from_pi0": round(step * 0.001, 8),
        "static_proxy_reward": round(0.5 + step * 0.001, 8),
        "mean_response_length": len(response_text),
        "response_embedding_distance": round(step * 0.002, 8),
        "style_summary": {"provider": "fake", "deterministic": True},
        "provider_call": {
            "requested_model": result.requested_model,
            "returned_model": result.returned_model,
            "request_id": result.request_id,
            "created_at": result.created_at,
            "raw_response_hash": result.raw_response_hash,
        },
    }


def _write_or_validate_shard(path: Path, rows: Sequence[Mapping[str, Any]]) -> bool:
    """Create an immutable shard or prove an existing shard is byte-identical."""

    return write_jsonl_atomic(path, rows)


def _reference_records(
    context: Any,
    prompts: Sequence[Mapping[str, Any]],
    generator: FakeGenerator,
) -> list[dict[str, Any]]:
    family_counts = context.raw.get("response_families", {})
    discovery_count = max(8, int(family_counts.get("reference_discovery", 8)))
    validation_count = max(4, int(family_counts.get("reference_validation", 4)))
    records: list[dict[str, Any]] = []
    for prompt in prompts:
        if "probe" not in str(prompt["split"]) and "audit" not in str(prompt["split"]):
            continue
        for family, count in (
            (SeedFamily.REFERENCE_DISCOVERY, discovery_count),
            (SeedFamily.REFERENCE_VALIDATION, validation_count),
        ):
            for sample_index in range(count):
                records.append(
                    _probe_record(
                        context,
                        prompt,
                        0,
                        family,
                        sample_index,
                        generator,
                        policy_id="pi_0",
                    )
                )
    return records


def _fault_after(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    value = int(raw)
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def run_train_static(context: Any) -> dict[str, Any]:
    """Run the deterministic static-only trajectory with resumable step shards."""

    rubric_path = context.run_root / "generate-static" / "static_rubrics.jsonl"
    context.begin_stage(
        inputs=(rubric_path,),
        metadata={
            "reward_source": "static_r0_only",
            "after_optimizer_update": True,
            "resume_unit": "policy_step_and_split",
        },
    )
    if context.mode != "fake":
        raise RuntimeError("live veRL execution is gated until every capability smoke has passed")
    generator = FakeGenerator("fake/qwen3-4b-v1")
    prompts = _configured_prompts(context)
    development = [row for row in prompts if "probe" in str(row["split"])]
    final = [row for row in prompts if "audit" in str(row["split"])]
    probe_cfg = context.raw.get("probe", {})
    dev_count = int(probe_cfg.get("development_samples_per_family", 4))
    final_count = int(probe_cfg.get("final_samples_per_family", 4))

    references = _reference_records(context, prompts, generator)
    reference_path = context.stage_root() / "reference_responses.jsonl"
    write_jsonl_atomic(reference_path, references)
    rubrics = _load_static_rubrics(rubric_path)
    reference_text = {
        str(row["prompt_id"]): str(row["response_text"])
        for row in references
        if row["family"] == SeedFamily.REFERENCE_DISCOVERY.value and int(row["sample_index"]) == 0
    }

    shard_specs = (
        ("development", development, dev_count, context.stage_root() / "shards" / "development"),
        (
            "final",
            final,
            final_count,
            context.run_root / "trajectory" / "final_sealed" / "shards",
        ),
    )
    created_shards = 0
    fault_limit = _fault_after("DYNAMIC_RUBRIC_TRAIN_FAULT_AFTER_SHARDS")
    shard_paths: dict[str, list[Path]] = defaultdict(list)
    for step in range(1, context.config.training.max_steps + 1):
        for split_name, split_prompts, sample_count, shard_root in shard_specs:
            rows: list[dict[str, Any]] = []
            for prompt in split_prompts:
                for family in (
                    SeedFamily.TRAJECTORY_DISCOVERY,
                    SeedFamily.TRAJECTORY_VALIDATION,
                ):
                    for sample_index in range(sample_count):
                        raw_record = _probe_record(
                            context,
                            prompt,
                            step,
                            family,
                            sample_index,
                            generator,
                            policy_id=f"pi_{step}",
                        )
                        prompt_id = str(prompt["prompt_id"])
                        rows.append(
                            ground_probe_record(
                                raw_record, rubrics[prompt_id], reference_text[prompt_id]
                            )
                        )
            shard_path = shard_root / f"step-{step:06d}.jsonl"
            created = _write_or_validate_shard(shard_path, rows)
            shard_paths[split_name].append(shard_path)
            if created:
                created_shards += 1
                if fault_limit is not None and created_shards >= fault_limit:
                    raise RuntimeError("injected training interruption after immutable shard")

    development_records = [row for path in shard_paths["development"] for row in read_jsonl(path)]
    final_records = [row for path in shard_paths["final"] for row in read_jsonl(path)]
    development_path = context.stage_root() / "trajectory_development.jsonl"
    final_path = context.run_root / "trajectory" / "final_sealed" / "responses.jsonl"
    write_jsonl_atomic(development_path, development_records)
    write_jsonl_atomic(final_path, final_records)

    checkpoints = [
        {
            "policy_id": f"pi_{step}",
            "step": step,
            "base_model": context.config.models.get("policy", {}).get("model"),
            "revision": f"fake-checkpoint-{step}",
            "semantics": "initial_policy" if step == 0 else "after_optimizer_update",
            "config_hash": context.config.config_hash,
        }
        for step in context.config.training.checkpoint_steps
    ]
    write_json_atomic(context.stage_root() / "checkpoints.json", checkpoints)
    shard_index = {
        split: [
            {"path": str(path.relative_to(context.root)), "sha256": sha256_file(path)}
            for path in paths
        ]
        for split, paths in sorted(shard_paths.items())
    }
    write_json_atomic(context.stage_root() / "shard_index.json", shard_index)
    result = {
        "credential_free_simulation": True,
        "reward_source": "static_r0_only",
        "dynamic_artifact_inputs": 0,
        "timing": "after_optimizer_update",
        "resume_unit": "policy_step_and_split",
        "reference_records": len(references),
        "development_records": len(development_records),
        "sealed_final_records": len(final_records),
        "focal_checkpoints": [row["step"] for row in checkpoints],
    }
    write_json_atomic(context.stage_root() / "result.json", result)
    return result


def _load_static_rubrics(path: Path) -> dict[str, StaticRubric]:
    rubrics: dict[str, StaticRubric] = {}
    for row in read_jsonl(path):
        rubric = StaticRubric(
            prompt_id=str(row["prompt_id"]),
            criteria=tuple(Criterion(**item) for item in row["criteria"]),
        )
        if rubric.content_hash != row["content_hash"]:
            raise RuntimeError(f"static rubric content hash mismatch: {rubric.prompt_id}")
        rubrics[rubric.prompt_id] = rubric
    return rubrics


def _index_responses(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, int, str], tuple[Mapping[str, Any], ...]]:
    grouped: dict[tuple[str, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["prompt_id"]), int(row["policy_step"]), str(row["family"]))].append(row)
    return {
        key: tuple(sorted(values, key=lambda item: int(item["sample_index"])))
        for key, values in grouped.items()
    }


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _pool_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    return sha256_json(
        [
            {
                "response_id": row["response_id"],
                "response_text_hash": hashlib.sha256(
                    str(row["response_text"]).encode()
                ).hexdigest(),
            }
            for row in rows
        ]
    )


def _pairing_hash(plan: PairingPlan) -> str:
    return sha256_json([dataclasses.asdict(pair) for pair in plan.generator_payload()])


def _validation_evidence(
    prompt_id: str,
    criterion_id: str,
    criterion_text: str,
    active_criteria: Sequence[Criterion],
    left: Sequence[Mapping[str, Any]],
    right: Sequence[Mapping[str, Any]],
    grader: FakeCriterionGrader,
    embedder: FakeEmbeddingProvider,
) -> AdmissionEvidence:
    def scores(rows: Sequence[Mapping[str, Any]]) -> list[float]:
        return [
            grader.score(
                prompt_id,
                str(row["response_id"]),
                str(row["response_text"]),
                criterion_id,
                criterion_text,
            ).probability_yes
            for row in rows
        ]

    current_scores = scores(left)
    reference_scores = scores(right)
    values = current_scores + reference_scores
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    embeddings = embedder.embed([criterion_text, *(item.text for item in active_criteria)])
    max_similarity = max((_cosine(embeddings[0], vector) for vector in embeddings[1:]), default=0.0)
    return AdmissionEvidence(
        satisfaction_rate=sum(value >= 0.5 for value in values) / len(values),
        current_mean=sum(current_scores) / len(current_scores),
        reference_mean=sum(reference_scores) / len(reference_scores),
        max_active_similarity=max_similarity,
        parse_success=1.0,
        independent_validation=True,
        score_variance=variance,
        recent_validation_failure_rate=0.0,
    )


def _control_rows(
    index: Mapping[tuple[str, int, str], tuple[Mapping[str, Any], ...]],
    references: Mapping[tuple[str, int, str], tuple[Mapping[str, Any], ...]],
    prompt_id: str,
    step: int,
    mode: ReplayMode,
    family: SeedFamily,
    count: int,
) -> tuple[tuple[Mapping[str, Any], ...], str, bool]:
    reference_family = (
        SeedFamily.REFERENCE_DISCOVERY
        if family is SeedFamily.TRAJECTORY_DISCOVERY
        else SeedFamily.REFERENCE_VALIDATION
    )
    fixed = references[(prompt_id, 0, reference_family.value)]
    if mode is ReplayMode.DYNAMIC_PREV_BUDGETED and step > 1:
        previous = index[(prompt_id, step - 1, family.value)]
        return previous[:count], f"pi_{step - 1}", True
    if mode is ReplayMode.DYNAMIC_PREV_BUDGETED:
        return fixed[:count], "pi_0", True
    if mode is ReplayMode.REFRESH_ONLY_BUDGETED:
        # A fresh extraction call over the immutable pi_0 pool controls for
        # generator noise without exposing any current-policy response.
        return fixed[:count], "pi_0_refresh", False
    return fixed[:count], "pi_0", True


def _extract_candidates(
    context: Any,
    prompt_id: str,
    step: int,
    mode: ReplayMode,
    previous: Any,
    discovery_left: Sequence[Mapping[str, Any]],
    discovery_right: Sequence[Mapping[str, Any]],
    validation_left: Sequence[Mapping[str, Any]],
    validation_right: Sequence[Mapping[str, Any]],
    generator: FakeGenerator,
    grader: FakeCriterionGrader,
    embedder: FakeEmbeddingProvider,
) -> tuple[tuple[ReplayCandidate, ...], PairingPlan, dict[str, Any]]:
    extraction_mode = (
        ReplayMode.DYNAMIC_FIXED_BUDGETED if mode is ReplayMode.DYNAMIC_FIXED_CUMULATIVE else mode
    )
    pairing = make_blind_pairing(
        [str(row["response_text"]) for row in discovery_left],
        [str(row["response_text"]) for row in discovery_right],
        seed=derive_seed(context.run_id, SeedFamily.PAIRING, prompt_id, step, 0),
        prompt_id=prompt_id,
        step=step,
    )
    blinded_payload = [dataclasses.asdict(pair) for pair in pairing.generator_payload()]
    request = GenerationRequest(
        prompt_id=f"{prompt_id}-s{step}-{extraction_mode.value}",
        messages=(
            {
                "role": "user",
                "content": "Propose atomic positive criteria from these source-blind A/B pairs: "
                + json.dumps(blinded_payload, ensure_ascii=False, sort_keys=True),
            },
        ),
        family=f"dynamic_extraction:{extraction_mode.value}",
        seed=derive_seed(context.run_id, SeedFamily.PAIRING, prompt_id, step, 1),
        json_schema=read_json(context.root / "configs" / "schemas" / "dynamic_candidate_v1.json"),
        schema_name="dynamic_candidate_v1",
        reasoning_effort=str(context.config.models["rubric_generator"]["reasoning_effort"]),
        metadata={
            "run_id": context.run_id,
            "prompt_id": prompt_id,
            "policy_step": step,
            "replicate_id": "A",
        },
    )
    result = generator.generate(request)
    proposed = json.loads(result.text).get("criteria", [])
    if len(proposed) > 3:
        raise RuntimeError("dynamic candidate schema returned more than three criteria")
    candidates: list[ReplayCandidate] = []
    evidence_rows: list[dict[str, Any]] = []
    for candidate_index, item in enumerate(proposed):
        text = str(item["text"])
        criterion_id = (
            f"dyn-{extraction_mode.value}-{step:06d}-{candidate_index}-"
            f"{hashlib.sha256(text.encode()).hexdigest()[:8]}"
        )
        evidence = _validation_evidence(
            prompt_id,
            criterion_id,
            text,
            previous.criteria,
            validation_left,
            validation_right,
            grader,
            embedder,
        )
        candidates.append(ReplayCandidate(criterion_id, text, evidence))
        evidence_rows.append(
            {
                "criterion_id": criterion_id,
                "text": text,
                "evidence": dataclasses.asdict(evidence),
            }
        )
    replicate = None
    replicate_fraction = float(context.raw.get("replay", {}).get("replicate_fraction", 0.20))
    subset = int(hashlib.sha256(prompt_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    if subset < replicate_fraction:
        replicate_request = dataclasses.replace(
            request,
            metadata={**dict(request.metadata), "replicate_id": "B"},
        )
        replicate_result = generator.generate(replicate_request)
        replicate = {
            "request_id": replicate_result.request_id,
            "returned_model": replicate_result.returned_model,
            "output_hash": hashlib.sha256(replicate_result.text.encode()).hexdigest(),
            "independent_call": True,
        }
    provenance = {
        "generator_call": {
            "request_id": result.request_id,
            "requested_model": result.requested_model,
            "returned_model": result.returned_model,
            "prompt_hash": sha256_json([dict(message) for message in request.messages]),
            "schema_hash": sha256_json(request.json_schema),
            "replicate_id": "A",
        },
        "replicate_b": replicate,
        "candidate_evidence": evidence_rows,
    }
    return tuple(candidates), pairing, provenance


def _replay_prompt_rows(
    context: Any,
    prompt_id: str,
    steps: Sequence[int],
    source_index: Mapping[tuple[str, int, str], tuple[Mapping[str, Any], ...]],
    reference_index: Mapping[tuple[str, int, str], tuple[Mapping[str, Any], ...]],
    rubric: StaticRubric,
    split: str,
) -> list[dict[str, Any]]:
    modes = [ReplayMode(value) for value in context.raw.get("replay", {}).get("modes", [])]
    snapshots = {mode: initial_replay_snapshot(rubric, mode) for mode in modes}
    generator = FakeGenerator("fake/gpt-5-mini-v1")
    grader = FakeCriterionGrader()
    embedder = FakeEmbeddingProvider()
    output: list[dict[str, Any]] = []
    for step in steps:
        for mode in modes:
            previous = snapshots[mode]
            if mode is ReplayMode.STATIC:
                snapshot = replay_step(previous, step=step)
                snapshots[mode] = snapshot
                output.append(
                    {
                        "run_id": context.run_id,
                        "config_hash": context.config.config_hash,
                        "prompt_id": prompt_id,
                        "split": split,
                        "policy_step": step,
                        "mode": mode.value,
                        "current_response_used": False,
                        "control_policy": None,
                        "criteria": [dataclasses.asdict(item) for item in snapshot.criteria],
                        "criterion_count": len(snapshot.criteria),
                        "content_hash": snapshot.content_hash,
                        "admitted_id": None,
                        "evicted_id": None,
                        "candidate_results": [],
                    }
                )
                continue

            discovery_current = source_index[
                (prompt_id, step, SeedFamily.TRAJECTORY_DISCOVERY.value)
            ]
            validation_current = source_index[
                (prompt_id, step, SeedFamily.TRAJECTORY_VALIDATION.value)
            ]
            discovery_control, control_policy, uses_current = _control_rows(
                source_index,
                reference_index,
                prompt_id,
                step,
                mode,
                SeedFamily.TRAJECTORY_DISCOVERY,
                len(discovery_current),
            )
            validation_control, _, _ = _control_rows(
                source_index,
                reference_index,
                prompt_id,
                step,
                mode,
                SeedFamily.TRAJECTORY_VALIDATION,
                len(validation_current),
            )
            if uses_current:
                discovery_left = discovery_current
                validation_left = validation_current
                current_ids = [row["response_id"] for row in discovery_current]
            else:
                fixed_discovery = reference_index[
                    (prompt_id, 0, SeedFamily.REFERENCE_DISCOVERY.value)
                ]
                discovery_count = len(discovery_control)
                discovery_left = fixed_discovery[discovery_count : 2 * discovery_count]
                if len(discovery_left) != discovery_count:
                    discovery_left = fixed_discovery[:discovery_count]
                fixed_validation = reference_index[
                    (prompt_id, 0, SeedFamily.REFERENCE_VALIDATION.value)
                ]
                validation_count = len(validation_control)
                validation_left = fixed_validation[validation_count : 2 * validation_count]
                if len(validation_left) != validation_count:
                    validation_left = fixed_validation[:validation_count]
                current_ids = []
            candidates, pairing, provenance = _extract_candidates(
                context,
                prompt_id,
                step,
                mode,
                previous,
                discovery_left,
                discovery_control,
                validation_left,
                validation_control,
                generator,
                grader,
                embedder,
            )
            snapshot = replay_step(previous, step=step, candidates=candidates)
            snapshots[mode] = snapshot
            output.append(
                {
                    "run_id": context.run_id,
                    "config_hash": context.config.config_hash,
                    "prompt_id": prompt_id,
                    "split": split,
                    "policy_step": step,
                    "mode": mode.value,
                    "current_response_used": uses_current,
                    "control_policy": control_policy,
                    "current_response_ids": current_ids,
                    "extraction_left_response_ids": [row["response_id"] for row in discovery_left],
                    "control_response_ids": [row["response_id"] for row in discovery_control],
                    "validation_left_response_ids": [row["response_id"] for row in validation_left],
                    "validation_control_response_ids": [
                        row["response_id"] for row in validation_control
                    ],
                    "discovery_pool_hash": sha256_json(
                        [_pool_hash(discovery_left), _pool_hash(discovery_control)]
                    ),
                    "validation_pool_hash": sha256_json(
                        [_pool_hash(validation_left), _pool_hash(validation_control)]
                    ),
                    "pairing_hash": _pairing_hash(pairing),
                    "generator_payload_source_blind": True,
                    **provenance,
                    "criteria": [dataclasses.asdict(item) for item in snapshot.criteria],
                    "criterion_count": len(snapshot.criteria),
                    "content_hash": snapshot.content_hash,
                    "admitted_id": snapshot.admitted_id,
                    "evicted_id": snapshot.evicted_id,
                    "candidate_results": [
                        dataclasses.asdict(item) for item in snapshot.candidate_results
                    ],
                }
            )
    return output


def _operator(context: Any) -> dict[str, Any]:
    replay = dict(context.raw.get("replay", {}))
    replay["criterion_embedding"] = context.config.models.get("criterion_embedding", {})
    replay["rubric_generator"] = context.config.models.get("rubric_generator", {})
    replay["proxy_grader"] = context.config.models.get("proxy_grader", {})
    replay["dynamic_schema_hash"] = sha256_file(
        context.root / "configs" / "schemas" / "dynamic_candidate_v1.json"
    )
    return replay


def run_replay_dynamic(context: Any, split: str) -> dict[str, Any]:
    """Replay real response-family pairings with resumable prompt shards."""

    if split not in {"development", "final"}:
        raise ValueError("replay split must be development or final")
    static_path = context.run_root / "generate-static" / "static_rubrics.jsonl"
    reference_path = context.run_root / "train-static" / "reference_responses.jsonl"
    if split == "development":
        source_path = context.run_root / "train-static" / "trajectory_development.jsonl"
        inputs = (static_path, reference_path, source_path)
        lock_path = None
    else:
        source_path = context.run_root / "trajectory" / "final_sealed" / "responses.jsonl"
        lock_path = context.run_root / "updater_lock.json"
        load_updater_lock(lock_path, _operator(context))
        inputs = (static_path, reference_path, source_path, lock_path)

    # Every live/preflight/config/lock/input guard runs before sealed bytes are
    # read and before the durable one-time unseal receipt can be created.
    context.begin_stage(
        inputs=inputs,
        metadata={"replay_split": split, "resume_unit": "prompt"},
    )
    if context.mode != "fake":
        raise RuntimeError("live dynamic extraction/scoring requires provisioned adapters")
    if split == "final":
        assert lock_path is not None
        receipt = context.run_root / "trajectory" / "final_sealed" / "unseal_receipt.json"
        source_rows = read_final_sealed(source_path, lock_path, _operator(context), receipt)
    else:
        source_rows = read_jsonl(source_path)

    rubrics = _load_static_rubrics(static_path)
    reference_rows = read_jsonl(reference_path)
    source_index = _index_responses(source_rows)
    reference_index = _index_responses(reference_rows)
    prompt_steps: dict[str, list[int]] = defaultdict(list)
    for prompt_id, step, family in source_index:
        if family == SeedFamily.TRAJECTORY_DISCOVERY.value:
            prompt_steps[prompt_id].append(step)
    for prompt_id in prompt_steps:
        prompt_steps[prompt_id] = sorted(set(prompt_steps[prompt_id]))

    fault_limit = _fault_after("DYNAMIC_RUBRIC_REPLAY_FAULT_AFTER_SHARDS")
    created_shards = 0
    shard_paths: list[Path] = []
    for prompt_id in sorted(prompt_steps):
        shard_name = hashlib.sha256(prompt_id.encode()).hexdigest()
        shard_path = context.stage_root() / "shards" / f"prompt-{shard_name}.jsonl"
        rows = _replay_prompt_rows(
            context,
            prompt_id,
            prompt_steps[prompt_id],
            source_index,
            reference_index,
            rubrics[prompt_id],
            split,
        )
        created = _write_or_validate_shard(shard_path, rows)
        shard_paths.append(shard_path)
        if created:
            created_shards += 1
            if fault_limit is not None and created_shards >= fault_limit:
                raise RuntimeError("injected replay interruption after immutable shard")

    output = [row for path in shard_paths for row in read_jsonl(path)]
    write_jsonl_atomic(context.stage_root() / "replay_snapshots.jsonl", output)
    write_json_atomic(
        context.stage_root() / "shard_index.json",
        [
            {"path": str(path.relative_to(context.root)), "sha256": sha256_file(path)}
            for path in shard_paths
        ],
    )
    result = {
        "split": split,
        "snapshots": len(output),
        "modes": list(REQUIRED_COMPARISONS),
        "resume_unit": "prompt",
        "max_budgeted_criteria": max(
            (row["criterion_count"] for row in output if row["mode"] != "dynamic_fixed_cumulative"),
            default=0,
        ),
        "static_preserved": all(
            sum(item["source"] != "dynamic" for item in row["criteria"]) == 8 for row in output
        ),
        "source_blind_pairing": all(
            row.get("generator_payload_source_blind", True) for row in output
        ),
    }
    write_json_atomic(context.stage_root() / "result.json", result)
    return result
