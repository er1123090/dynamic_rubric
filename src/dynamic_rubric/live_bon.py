"""Resumable checkpoint-specific live BoN rollout generation."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

from .artifacts import read_jsonl, write_json_atomic, write_jsonl_atomic
from .hashing import sha256_file
from .pipeline import PipelineContext, StageError, _all_public_prompts
from .providers.base import GenerationRequest, GenerationResult
from .providers.vllm_generation import VLLMPolicyGenerator
from .seeds import SeedFamily, derive_seed, response_id


def _audit_prompts(context: PipelineContext) -> list[dict[str, Any]]:
    return [row for row in _all_public_prompts(context) if "audit" in str(row["split"])]


def _generate_one(
    generator: Any,
    request: GenerationRequest,
    *,
    attempts: int = 4,
) -> GenerationResult:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return generator.generate(request)
        except Exception as error:
            last_error = error
            if attempt + 1 < attempts:
                time.sleep(min(10.0, 0.5 * (2**attempt)))
    raise StageError(f"vLLM generation failed after {attempts} attempts: {last_error}")


def generate_prompt_rows(
    run_id: str,
    policy_step: int,
    prompt: Mapping[str, Any],
    prompt_index: int,
    pool_size: int,
    generator: Any,
    concurrency: int,
    max_output_tokens: int,
) -> list[dict[str, Any]]:
    return generate_sample_rows(
        run_id,
        policy_step,
        prompt,
        prompt_index,
        0,
        pool_size,
        pool_size,
        generator,
        concurrency,
        max_output_tokens,
    )


def generate_sample_rows(
    run_id: str,
    policy_step: int,
    prompt: Mapping[str, Any],
    prompt_index: int,
    sample_start: int,
    sample_stop: int,
    target_pool_size: int,
    generator: Any,
    concurrency: int,
    max_output_tokens: int,
) -> list[dict[str, Any]]:
    if not 0 <= sample_start < sample_stop <= target_pool_size:
        raise ValueError("sample range must be within the target pool")
    prompt_id = str(prompt["prompt_id"])

    def generate(sample_index: int) -> dict[str, Any]:
        logical_seed = derive_seed(
            run_id, SeedFamily.AUDIT_BON, prompt_id, policy_step, sample_index
        )
        result = _generate_one(
            generator,
            GenerationRequest(
                prompt_id=prompt_id,
                messages=tuple(prompt["messages"]),
                family=SeedFamily.AUDIT_BON.value,
                seed=logical_seed,
                temperature=1.0,
                top_p=0.95,
                max_output_tokens=max_output_tokens,
                metadata={
                    "run_id": run_id,
                    "policy_step": policy_step,
                    "replicate_id": sample_index,
                },
            ),
        )
        return {
            "policy_id": f"pi_{policy_step}",
            "policy_step": policy_step,
            "prompt_id": prompt_id,
            "sample_index": sample_index,
            "global_candidate_id": (
                policy_step * 10**9 + prompt_index * target_pool_size + sample_index
            ),
            "response_id": response_id(
                run_id,
                SeedFamily.AUDIT_BON,
                prompt_id,
                policy_step,
                sample_index,
            ),
            "seed": logical_seed,
            "response_text": result.text,
            "provider_call": {
                "requested_model": result.requested_model,
                "returned_model": result.returned_model,
                "request_id": result.request_id,
                "created_at": result.created_at,
                "retry_count": result.retry_count,
                "usage": dict(result.usage),
                "raw_response_hash": result.raw_response_hash,
            },
        }

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        return list(executor.map(generate, range(sample_start, sample_stop)))


def _validate_sample_range(
    rows: Sequence[Mapping[str, Any]],
    *,
    policy_step: int,
    prompt_id: str,
    sample_start: int,
    sample_stop: int,
) -> None:
    expected = list(range(sample_start, sample_stop))
    actual = [int(row["sample_index"]) for row in rows]
    if actual != expected:
        raise StageError(
            f"invalid BoN sample range for pi_{policy_step}/{prompt_id}: "
            f"expected {sample_start}:{sample_stop}"
        )
    if any(
        int(row["policy_step"]) != policy_step or str(row["prompt_id"]) != prompt_id for row in rows
    ):
        raise StageError(f"BoN shard identity mismatch for pi_{policy_step}/{prompt_id}")


def _normalized_candidate(
    row: Mapping[str, Any],
    *,
    policy_step: int,
    prompt_index: int,
    target_pool_size: int,
) -> dict[str, Any]:
    normalized = dict(row)
    candidate_id = policy_step * 10**9 + prompt_index * target_pool_size + int(row["sample_index"])
    source_id = int(row["global_candidate_id"])
    if source_id != candidate_id:
        normalized["source_global_candidate_id"] = source_id
    normalized["global_candidate_id"] = candidate_id
    return normalized


def run_live_bon_worker(
    context: PipelineContext,
    *,
    policy_step: int,
    base_url: str,
    model: str,
    concurrency: int = 64,
) -> dict[str, Any]:
    if policy_step <= 0 or concurrency <= 0:
        raise ValueError("policy_step and concurrency must be positive")
    prompts = _audit_prompts(context)
    pool_size = int(context.raw.get("bon", {}).get("pool_size", 64))
    max_output_tokens = int(context.raw.get("training", {}).get("max_response_length", 1536))
    stage_root = context.stage_root("generate-bon-live")
    worker_root = stage_root / f"pi_{policy_step}"
    generator = VLLMPolicyGenerator(
        base_url,
        model,
        revision=f"global_step_{policy_step}",
        tokenizer_revision=f"global_step_{policy_step}",
        timeout_seconds=900.0,
    )
    created = 0
    shard_paths: list[Path] = []
    for prompt_index, prompt in enumerate(prompts):
        prompt_id = str(prompt["prompt_id"])
        shard_path = worker_root / "shards" / f"{prompt_index:03d}-{prompt_id}.jsonl"
        if shard_path.is_file():
            rows = read_jsonl(shard_path)
            if len(rows) != pool_size:
                raise StageError(f"incomplete immutable BoN shard: {shard_path}")
        else:
            rows = generate_prompt_rows(
                context.run_id,
                policy_step,
                prompt,
                prompt_index,
                pool_size,
                generator,
                concurrency,
                max_output_tokens,
            )
            write_jsonl_atomic(shard_path, rows)
            created += 1
        shard_paths.append(shard_path)
    combined = [row for path in shard_paths for row in read_jsonl(path)]
    combined_path = worker_root / f"pi_{policy_step}.jsonl"
    write_jsonl_atomic(combined_path, combined)
    result = {
        "run_id": context.run_id,
        "policy_step": policy_step,
        "model": model,
        "base_url": base_url,
        "prompt_count": len(prompts),
        "pool_size": pool_size,
        "candidate_count": len(combined),
        "expected": len(prompts) * pool_size,
        "created_prompt_shards": created,
        "resume_unit": "prompt",
        "output_path": str(combined_path),
        "output_sha256": sha256_file(combined_path),
    }
    write_json_atomic(worker_root / "result.json", result)
    return result


def _prompt_is_assigned(
    prompt_index: int, prompt_shard_index: int, prompt_shard_count: int
) -> bool:
    return prompt_index % prompt_shard_count == prompt_shard_index


def _extension_chunk_path(
    extension_root: Path,
    prompt_index: int,
    prompt_id: str,
    sample_start: int,
    sample_stop: int,
) -> Path:
    prompt_root = extension_root / "shards" / f"{prompt_index:03d}-{prompt_id}"
    return prompt_root / f"samples-{sample_start:06d}-{sample_stop - 1:06d}.jsonl"


def _all_extension_chunks_exist(
    extension_root: Path,
    prompts: Sequence[Mapping[str, Any]],
    base_pool_size: int,
    target_pool_size: int,
    chunk_size: int,
) -> bool:
    return all(
        _extension_chunk_path(
            extension_root,
            prompt_index,
            str(prompt["prompt_id"]),
            sample_start,
            min(sample_start + chunk_size, target_pool_size),
        ).is_file()
        for prompt_index, prompt in enumerate(prompts)
        for sample_start in range(base_pool_size, target_pool_size, chunk_size)
    )


def _assemble_extended_pool(
    context: PipelineContext,
    *,
    policy_step: int,
    prompts: Sequence[Mapping[str, Any]],
    worker_root: Path,
    extension_root: Path,
    base_pool_size: int,
    target_pool_size: int,
    chunk_size: int,
    model: str,
    base_url: str,
) -> dict[str, Any]:
    combined: list[dict[str, Any]] = []
    for prompt_index, prompt in enumerate(prompts):
        prompt_id = str(prompt["prompt_id"])
        base_path = worker_root / "shards" / f"{prompt_index:03d}-{prompt_id}.jsonl"
        base_rows = read_jsonl(base_path)
        _validate_sample_range(
            base_rows,
            policy_step=policy_step,
            prompt_id=prompt_id,
            sample_start=0,
            sample_stop=base_pool_size,
        )
        combined.extend(
            _normalized_candidate(
                row,
                policy_step=policy_step,
                prompt_index=prompt_index,
                target_pool_size=target_pool_size,
            )
            for row in base_rows
        )
        for sample_start in range(base_pool_size, target_pool_size, chunk_size):
            sample_stop = min(sample_start + chunk_size, target_pool_size)
            shard_path = _extension_chunk_path(
                extension_root,
                prompt_index,
                prompt_id,
                sample_start,
                sample_stop,
            )
            rows = read_jsonl(shard_path)
            _validate_sample_range(
                rows,
                policy_step=policy_step,
                prompt_id=prompt_id,
                sample_start=sample_start,
                sample_stop=sample_stop,
            )
            combined.extend(
                _normalized_candidate(
                    row,
                    policy_step=policy_step,
                    prompt_index=prompt_index,
                    target_pool_size=target_pool_size,
                )
                for row in rows
            )
    combined_path = extension_root / f"pi_{policy_step}-pool-{target_pool_size}.jsonl"
    write_jsonl_atomic(combined_path, combined)
    result = {
        "run_id": context.run_id,
        "policy_step": policy_step,
        "model": model,
        "assembly_base_url": base_url,
        "base_pool_size": base_pool_size,
        "target_pool_size": target_pool_size,
        "prompt_count": len(prompts),
        "candidate_count": len(combined),
        "expected": len(prompts) * target_pool_size,
        "extension_chunk_count": len(prompts)
        * len(range(base_pool_size, target_pool_size, chunk_size)),
        "chunk_size": chunk_size,
        "resume_unit": "prompt_sample_chunk",
        "output_path": str(combined_path),
        "output_sha256": sha256_file(combined_path),
    }
    write_json_atomic(extension_root / "result.json", result)
    return result


def extend_live_bon_worker(
    context: PipelineContext,
    *,
    policy_step: int,
    base_url: str,
    model: str,
    target_pool_size: int,
    concurrency: int = 64,
    chunk_size: int = 64,
    prompt_shard_index: int = 0,
    prompt_shard_count: int = 1,
) -> dict[str, Any]:
    if policy_step <= 0 or concurrency <= 0 or chunk_size <= 0:
        raise ValueError("policy_step, concurrency and chunk_size must be positive")
    if prompt_shard_count <= 0 or not 0 <= prompt_shard_index < prompt_shard_count:
        raise ValueError("prompt shard index must be within a positive shard count")
    base_pool_size = int(context.raw.get("bon", {}).get("pool_size", 64))
    if target_pool_size <= base_pool_size:
        raise ValueError("target_pool_size must exceed the configured base pool size")
    prompts = _audit_prompts(context)
    max_output_tokens = int(context.raw.get("training", {}).get("max_response_length", 1536))
    worker_root = context.stage_root("generate-bon-live") / f"pi_{policy_step}"
    extension_root = worker_root / "extensions" / f"pool-{target_pool_size}"
    generator = VLLMPolicyGenerator(
        base_url,
        model,
        revision=f"global_step_{policy_step}",
        tokenizer_revision=f"global_step_{policy_step}",
        timeout_seconds=900.0,
    )
    created = 0
    assigned_prompt_count = 0
    for prompt_index, prompt in enumerate(prompts):
        if not _prompt_is_assigned(prompt_index, prompt_shard_index, prompt_shard_count):
            continue
        assigned_prompt_count += 1
        prompt_id = str(prompt["prompt_id"])
        base_path = worker_root / "shards" / f"{prompt_index:03d}-{prompt_id}.jsonl"
        if not base_path.is_file():
            raise StageError(f"base BoN shard is missing: {base_path}")
        base_rows = read_jsonl(base_path)
        _validate_sample_range(
            base_rows,
            policy_step=policy_step,
            prompt_id=prompt_id,
            sample_start=0,
            sample_stop=base_pool_size,
        )
        for sample_start in range(base_pool_size, target_pool_size, chunk_size):
            sample_stop = min(sample_start + chunk_size, target_pool_size)
            shard_path = _extension_chunk_path(
                extension_root,
                prompt_index,
                prompt_id,
                sample_start,
                sample_stop,
            )
            if shard_path.is_file():
                rows = read_jsonl(shard_path)
            else:
                rows = generate_sample_rows(
                    context.run_id,
                    policy_step,
                    prompt,
                    prompt_index,
                    sample_start,
                    sample_stop,
                    target_pool_size,
                    generator,
                    concurrency,
                    max_output_tokens,
                )
                write_jsonl_atomic(shard_path, rows)
                created += 1
            _validate_sample_range(
                rows,
                policy_step=policy_step,
                prompt_id=prompt_id,
                sample_start=sample_start,
                sample_stop=sample_stop,
            )
    shard_result = {
        "run_id": context.run_id,
        "policy_step": policy_step,
        "model": model,
        "base_url": base_url,
        "base_pool_size": base_pool_size,
        "target_pool_size": target_pool_size,
        "assigned_prompt_count": assigned_prompt_count,
        "assigned_candidate_count": assigned_prompt_count * target_pool_size,
        "created_chunks": created,
        "chunk_size": chunk_size,
        "prompt_shard_index": prompt_shard_index,
        "prompt_shard_count": prompt_shard_count,
        "resume_unit": "prompt_sample_chunk",
    }
    shard_result_path = (
        extension_root
        / "workers"
        / f"shard-{prompt_shard_index:02d}-of-{prompt_shard_count:02d}.json"
    )
    pool_result = None
    if _all_extension_chunks_exist(
        extension_root,
        prompts,
        base_pool_size,
        target_pool_size,
        chunk_size,
    ):
        pool_result = _assemble_extended_pool(
            context,
            policy_step=policy_step,
            prompts=prompts,
            worker_root=worker_root,
            extension_root=extension_root,
            base_pool_size=base_pool_size,
            target_pool_size=target_pool_size,
            chunk_size=chunk_size,
            model=model,
            base_url=base_url,
        )
    write_json_atomic(shard_result_path, shard_result)
    if pool_result is not None:
        return {**shard_result, "pool_complete": True, "pool_result": pool_result}
    return {**shard_result, "pool_complete": False}


def finalize_live_bon(
    context: PipelineContext,
    policy_steps: Sequence[int],
    target_pool_size: int | None = None,
) -> dict[str, Any]:
    stage_root = context.stage_root("generate-bon-live")
    configured_pool_size = int(context.raw.get("bon", {}).get("pool_size", 64))
    pool_size = target_pool_size or configured_pool_size
    if pool_size < configured_pool_size:
        raise ValueError("target_pool_size cannot be smaller than the configured pool")
    audit_prompts = _audit_prompts(context)
    rows: list[dict[str, Any]] = []
    sources = []
    for step in policy_steps:
        if pool_size == configured_pool_size:
            path = stage_root / f"pi_{step}" / f"pi_{step}.jsonl"
        else:
            path = (
                stage_root
                / f"pi_{step}"
                / "extensions"
                / f"pool-{pool_size}"
                / f"pi_{step}-pool-{pool_size}.jsonl"
            )
        if not path.is_file():
            raise StageError(f"BoN worker output is missing: {path}")
        current = read_jsonl(path)
        expected_samples = set(range(pool_size))
        for prompt in audit_prompts:
            prompt_id = str(prompt["prompt_id"])
            prompt_rows = [row for row in current if str(row["prompt_id"]) == prompt_id]
            if {int(row["sample_index"]) for row in prompt_rows} != expected_samples:
                raise StageError(f"incomplete BoN pool for pi_{step}/{prompt_id}")
        rows.extend(current)
        sources.append({"policy_step": step, "path": str(path), "sha256": sha256_file(path)})
    candidate_ids = [int(row["global_candidate_id"]) for row in rows]
    response_ids = [str(row["response_id"]) for row in rows]
    if len(candidate_ids) != len(set(candidate_ids)) or len(response_ids) != len(set(response_ids)):
        raise StageError("BoN pool contains duplicate candidate or response IDs")
    output = context.run_root / "generate-bon" / "bon_pool.jsonl"
    write_jsonl_atomic(output, rows)
    result = {
        "run_id": context.run_id,
        "focal_steps": list(policy_steps),
        "candidate_count": len(rows),
        "expected": len(policy_steps) * len(audit_prompts) * pool_size,
        "pool_size": pool_size,
        "sources": sources,
        "output_path": str(output),
        "output_sha256": sha256_file(output),
    }
    write_json_atomic(context.run_root / "generate-bon" / "result.json", result)
    return result
