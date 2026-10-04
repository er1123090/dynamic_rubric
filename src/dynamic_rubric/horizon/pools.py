"""Deterministic, disjoint response-pool construction for horizon audits."""

from __future__ import annotations

import hashlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..artifacts import write_jsonl_atomic
from ..hashing import sha256_json
from ..providers.base import GenerationRequest, RubricGenerator
from ..seeds import SeedFamily, derive_seed, response_id, vllm_seed


POOL_SEED_FAMILY = {
    "fixed_control": SeedFamily.HORIZON_FIXED_CONTROL,
    "sham_control": SeedFamily.HORIZON_SHAM_CONTROL,
    "pool_a": SeedFamily.HORIZON_POOL_A,
    "pool_b": SeedFamily.HORIZON_POOL_B,
}


@dataclass(frozen=True, slots=True)
class PoolSpec:
    family: str
    count: int
    policy_step: int
    training_seed: int | None

    def __post_init__(self) -> None:
        if self.family not in POOL_SEED_FAMILY:
            raise ValueError(f"unknown horizon pool family: {self.family}")
        if self.count <= 0 or self.policy_step < 0:
            raise ValueError("pool count must be positive and policy_step non-negative")
        if self.family in {"fixed_control", "sham_control"} and self.training_seed is not None:
            raise ValueError("fixed/sham pools must be shared across training seeds")
        if self.family in {"pool_a", "pool_b"} and self.training_seed is None:
            raise ValueError("Pool A/B require a training seed")
        if self.family == "pool_a" and self.policy_step == 0:
            raise ValueError("Pool A is not generated at step zero")


def pool_run_namespace(suite_id: str, domain: str, training_seed: int | None) -> str:
    if domain not in {"medicine", "science"}:
        raise ValueError("domain must be medicine or science")
    suffix = "shared" if training_seed is None else f"seed-{training_seed}"
    return f"{suite_id}.{domain}.{suffix}"


def pool_identity(
    *,
    suite_id: str,
    domain: str,
    prompt_id: str,
    spec: PoolSpec,
    sample_index: int,
    checkpoint_hash: str,
    model_revision: str,
    tokenizer_revision: str,
    generation_config: Mapping[str, Any],
) -> dict[str, Any]:
    if sample_index < 0 or sample_index >= spec.count:
        raise ValueError("sample_index is outside the pool spec")
    namespace = pool_run_namespace(suite_id, domain, spec.training_seed)
    seed_family = POOL_SEED_FAMILY[spec.family]
    logical_seed = derive_seed(
        namespace, seed_family, prompt_id, spec.policy_step, sample_index
    )
    return {
        "schema_version": 1,
        "suite_id": suite_id,
        "domain": domain,
        "training_seed": spec.training_seed,
        "prompt_id": prompt_id,
        "policy_step": spec.policy_step,
        "checkpoint_hash": checkpoint_hash,
        "pool_family": spec.family,
        "sample_index": sample_index,
        "response_id": response_id(
            namespace, seed_family, prompt_id, spec.policy_step, sample_index
        ),
        "logical_seed": logical_seed,
        "vllm_seed": vllm_seed(logical_seed),
        "model_revision": model_revision,
        "tokenizer_revision": tokenizer_revision,
        "generation_config_hash": sha256_json(generation_config),
    }


def generate_pool_rows(
    provider: RubricGenerator,
    prompts: Iterable[Mapping[str, Any]],
    *,
    suite_id: str,
    domain: str,
    spec: PoolSpec,
    checkpoint_hash: str,
    model: str,
    model_revision: str,
    tokenizer_revision: str,
    temperature: float = 1.0,
    top_p: float = 0.95,
    max_output_tokens: int = 3584,
    concurrency: int = 1,
) -> list[dict[str, Any]]:
    if concurrency <= 0:
        raise ValueError("pool generation concurrency must be positive")
    generation_config = {
        "thinking": False,
        "temperature": temperature,
        "top_p": top_p,
        "max_output_tokens": max_output_tokens,
    }
    pending: list[tuple[dict[str, Any], GenerationRequest]] = []
    for prompt in prompts:
        prompt_id = str(prompt["prompt_id"])
        messages = tuple(dict(item) for item in prompt["messages"])
        for sample_index in range(spec.count):
            identity = pool_identity(
                suite_id=suite_id,
                domain=domain,
                prompt_id=prompt_id,
                spec=spec,
                sample_index=sample_index,
                checkpoint_hash=checkpoint_hash,
                model_revision=model_revision,
                tokenizer_revision=tokenizer_revision,
                generation_config=generation_config,
            )
            request = GenerationRequest(
                prompt_id=prompt_id,
                messages=messages,
                family=spec.family,
                seed=int(identity["logical_seed"]),
                temperature=temperature,
                top_p=top_p,
                max_output_tokens=max_output_tokens,
                metadata=identity,
            )
            pending.append((identity, request))
    requests = [request for _, request in pending]
    if concurrency == 1:
        results = [provider.generate(request) for request in requests]
    else:
        with ThreadPoolExecutor(max_workers=min(concurrency, len(requests))) as executor:
            results = list(executor.map(provider.generate, requests))
    rows: list[dict[str, Any]] = []
    for (identity, _), result in zip(pending, results):
        if result.requested_model != model or result.returned_model != model:
            raise RuntimeError("policy model identity drifted while generating horizon pool")
        rows.append(
            {
                **identity,
                "response_text": result.text,
                "request_id": result.request_id,
                "retry_count": result.retry_count,
                "raw_response_hash": result.raw_response_hash
                or hashlib.sha256(result.text.encode()).hexdigest(),
            }
        )
    validate_pool_rows(rows, expected_count=spec.count)
    return rows


def validate_pool_rows(rows: Sequence[Mapping[str, Any]], *, expected_count: int) -> None:
    if expected_count <= 0:
        raise ValueError("expected_count must be positive")
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    response_ids: set[str] = set()
    for row in rows:
        key = (
            row["domain"],
            row.get("training_seed"),
            row["prompt_id"],
            row["policy_step"],
            row["pool_family"],
        )
        groups.setdefault(key, []).append(row)
        response = str(row["response_id"])
        if response in response_ids:
            raise ValueError("horizon pool response IDs must be globally unique")
        response_ids.add(response)
    for key, values in groups.items():
        if len(values) != expected_count:
            raise ValueError(f"wrong pool count for {key}: {len(values)}")
        if {int(row["sample_index"]) for row in values} != set(range(expected_count)):
            raise ValueError(f"wrong sample slots for {key}")


def combine_pool_a_with_fixed_control(
    pool_a_rows: Sequence[Mapping[str, Any]],
    fixed_control_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build a 16-response Pool-A analysis shard without changing response identity."""

    if not pool_a_rows or not fixed_control_rows:
        raise ValueError("Pool A and fixed-control rows must not be empty")
    if {str(row.get("pool_family")) for row in pool_a_rows} != {"pool_a"}:
        raise ValueError("Pool A input must contain only pool_a rows")
    if {str(row.get("pool_family")) for row in fixed_control_rows} != {"fixed_control"}:
        raise ValueError("fixed-control input must contain only fixed_control rows")
    validate_pool_rows(pool_a_rows, expected_count=8)
    validate_pool_rows(fixed_control_rows, expected_count=8)

    training_seeds = {row.get("training_seed") for row in pool_a_rows}
    policy_steps = {int(row["policy_step"]) for row in pool_a_rows}
    if len(training_seeds) != 1 or None in training_seeds or len(policy_steps) != 1:
        raise ValueError("Pool A must bind exactly one training seed and policy step")
    training_seed = next(iter(training_seeds))
    policy_step = next(iter(policy_steps))
    if policy_step == 0:
        raise ValueError("combined Pool A is only defined after checkpoint zero")
    if {row.get("training_seed") for row in fixed_control_rows} != {None}:
        raise ValueError("fixed-control rows must be shared across training seeds")
    if {int(row["policy_step"]) for row in fixed_control_rows} != {0}:
        raise ValueError("fixed-control rows must come from policy step zero")

    def by_prompt(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
        grouped: dict[str, list[Mapping[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(str(row["prompt_id"]), []).append(row)
        return grouped

    current_by_prompt = by_prompt(pool_a_rows)
    fixed_by_prompt = by_prompt(fixed_control_rows)
    if set(current_by_prompt) != set(fixed_by_prompt):
        raise ValueError("Pool A and fixed control must have the same prompt grid")

    combined_rows: list[dict[str, Any]] = []
    for prompt_id in sorted(current_by_prompt):
        sources = (
            sorted(current_by_prompt[prompt_id], key=lambda row: int(row["sample_index"])),
            sorted(fixed_by_prompt[prompt_id], key=lambda row: int(row["sample_index"])),
        )
        for offset, rows in zip((0, 8), sources):
            for row in rows:
                combined = dict(row)
                combined.update(
                    {
                        "training_seed": training_seed,
                        "policy_step": policy_step,
                        "pool_family": "pool_a_combined",
                        "sample_index": offset + int(row["sample_index"]),
                        "source_pool_family": row["pool_family"],
                        "source_training_seed": row.get("training_seed"),
                        "source_policy_step": int(row["policy_step"]),
                        "source_sample_index": int(row["sample_index"]),
                    }
                )
                combined_rows.append(combined)
    validate_pool_rows(combined_rows, expected_count=16)
    return combined_rows


def publish_pool_shard(path: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    validate_pool_rows(rows, expected_count=len({int(row["sample_index"]) for row in rows}))
    write_jsonl_atomic(path, rows)
    return {
        "path": str(path),
        "rows": len(rows),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def validate_horizon_pool_inventory(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_counts: Mapping[str, int],
    expected_prompt_ids: Sequence[str] | None = None,
    training_seeds: Sequence[int] | None = None,
    checkpoint_steps: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Validate a combined inventory and report duplicate-text collisions."""

    if set(expected_counts) != set(POOL_SEED_FAMILY):
        raise ValueError("expected_counts must define every horizon pool family")
    if any(int(value) < 0 for value in expected_counts.values()):
        raise ValueError("expected pool counts must be non-negative")
    active_families = {
        family for family, count in expected_counts.items() if int(count) > 0
    }
    response_ids: set[str] = set()
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    text_hashes: Counter[str] = Counter()
    family_rows: Counter[str] = Counter()
    for row in rows:
        family = str(row["pool_family"])
        if family not in active_families:
            raise ValueError(f"unknown or disabled pool family in inventory: {family}")
        training_seed = row.get("training_seed")
        if family in {"fixed_control", "sham_control"} and training_seed is not None:
            raise ValueError("fixed and sham controls must be domain-shared")
        if family in {"pool_a", "pool_b"} and training_seed is None:
            raise ValueError("Pool A/B inventory rows must bind a training seed")
        response_id_value = str(row["response_id"])
        if response_id_value in response_ids:
            raise ValueError("pool-family response ID intersection is non-zero")
        response_ids.add(response_id_value)
        key = (
            row["domain"],
            training_seed,
            row["prompt_id"],
            row["policy_step"],
            family,
        )
        groups.setdefault(key, []).append(row)
        family_rows[family] += 1
        text_hashes[hashlib.sha256(str(row["response_text"]).encode()).hexdigest()] += 1
    for key, values in groups.items():
        expected = int(expected_counts[str(key[-1])])
        if len(values) != expected:
            raise ValueError(f"wrong combined inventory count for {key}: {len(values)}")
        if {int(row["sample_index"]) for row in values} != set(range(expected)):
            raise ValueError(f"wrong combined inventory sample slots for {key}")
    if any(value is not None for value in (expected_prompt_ids, training_seeds, checkpoint_steps)):
        if any(value is None for value in (expected_prompt_ids, training_seeds, checkpoint_steps)):
            raise ValueError("full-grid validation requires prompts, seeds, and checkpoints together")
        domains = {str(row["domain"]) for row in rows}
        if len(domains) != 1:
            raise ValueError("one horizon inventory must contain exactly one domain")
        domain = next(iter(domains))
        prompts = {str(value) for value in expected_prompt_ids or ()}
        seeds = {int(value) for value in training_seeds or ()}
        checkpoints = {int(value) for value in checkpoint_steps or ()}
        if not prompts or not seeds or 0 not in checkpoints:
            raise ValueError("full-grid expectations must be non-empty and include checkpoint zero")
        expected_groups = {
            (domain, None, prompt_id, 0, family)
            for prompt_id in prompts
            for family in ("fixed_control", "sham_control")
            if family in active_families
        }
        expected_groups.update(
            (domain, seed, prompt_id, checkpoint, "pool_b")
            for seed in seeds
            for prompt_id in prompts
            for checkpoint in checkpoints
        )
        expected_groups.update(
            (domain, seed, prompt_id, checkpoint, "pool_a")
            for seed in seeds
            for prompt_id in prompts
            for checkpoint in checkpoints
            if checkpoint != 0
        )
        actual_groups = set(groups)
        if actual_groups != expected_groups:
            missing = sorted(expected_groups - actual_groups, key=repr)
            extra = sorted(actual_groups - expected_groups, key=repr)
            raise ValueError(
                f"horizon inventory is not the complete preregistered grid; "
                f"missing={missing[:10]}, extra={extra[:10]}"
            )
    return {
        "schema_version": 1,
        "row_count": len(rows),
        "group_count": len(groups),
        "family_rows": {
            family: family_rows.get(family, 0) for family in sorted(expected_counts)
        },
        "unique_response_ids": len(response_ids),
        "duplicate_text_collision_groups": sum(count > 1 for count in text_hashes.values()),
        "full_grid_validated": expected_prompt_ids is not None,
        "valid": True,
    }
