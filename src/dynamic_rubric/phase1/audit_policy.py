"""Checkpoint-bound policy exports, fixed-probe pools, and policy-distance audits.

This module is deliberately separate from training.  It never restores optimizer
state and it never writes into the training run directory.  Every generated
artifact is bound to the source checkpoint bytes, the resolved training config,
the launch specification, and the fixed-train-probe manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from dynamic_rubric.artifacts import (
    artifact_record,
    read_json,
    read_jsonl,
    validate_artifact_record,
    write_json_atomic,
    write_jsonl_atomic,
)
from dynamic_rubric.hashing import sha256_file, sha256_json
from dynamic_rubric.horizon.checkpoint_kl import (
    VLLMPolicyLogprobClient,
    _decode_float32,
    score_policy_logprobs_from_files,
)
from dynamic_rubric.providers.base import GenerationRequest
from dynamic_rubric.providers.vllm_generation import VLLMPolicyGenerator
from dynamic_rubric.seeds import SeedFamily, derive_seed, response_id, vllm_seed


SCHEMA_VERSION = 1
PROBE_A_COUNT = 8
PROBE_B_COUNT = 16
DEFAULT_PROBE_COUNT = 100
DEFAULT_CHECKPOINTS = (0, 3, 6, 9, 13, 16, 24, 32, 34)
GENERATION_CONFIG = {
    "thinking": False,
    "temperature": 1.0,
    "top_p": 0.95,
    "max_output_tokens": 3584,
}
POOL_CONTRACTS = {
    "probe_A": ("pool_a", SeedFamily.HORIZON_POOL_A, PROBE_A_COUNT),
    "probe_B": ("pool_b", SeedFamily.HORIZON_POOL_B, PROBE_B_COUNT),
}


class AuditPolicyError(RuntimeError):
    """Raised when an offline policy audit would mix incompatible provenance."""


@dataclass(frozen=True, slots=True)
class RunContract:
    run_dir: Path
    run_id: str
    domain: str
    method: str
    seed: int
    model: str
    model_revision: str
    tokenizer_revision: str
    config_path: Path
    config_sha256: str
    launch_spec_path: Path
    launch_spec_sha256: str
    probe_manifest_path: Path
    probe_manifest_sha256: str
    train_path: Path
    train_sha256: str


@dataclass(frozen=True, slots=True)
class CheckpointIdentity:
    step: int
    actor_dir: Path
    source_model: Path
    source_model_sha256: str
    source_model_bytes: int


def _resolve_from_project(value: str, project_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def load_run_contract(run_dir: Path) -> RunContract:
    """Load and cross-check the immutable production-run inputs."""

    run_dir = run_dir.resolve()
    config_path = run_dir / "config.resolved.json"
    launch_path = run_dir / "launch_spec.json"
    if not config_path.is_file() or not launch_path.is_file():
        raise AuditPolicyError("run is missing config.resolved.json or launch_spec.json")
    config = read_json(config_path)
    launch = read_json(launch_path)
    project_root = Path(__file__).resolve().parents[3]
    policy = config.get("models", {}).get("policy", {})
    probe_config = config.get("data", {}).get("fixed_train_probe", {})
    train_value = str(config.get("data", {}).get("train_path", ""))
    probe_value = str(launch.get("fixed_probe_manifest", ""))
    if not isinstance(policy, Mapping) or not train_value or not probe_value:
        raise AuditPolicyError("resolved config lacks policy/train/probe identity")
    probe_path = _resolve_from_project(probe_value, project_root).resolve()
    train_path = _resolve_from_project(train_value, project_root).resolve()
    expected = {
        "domain": config.get("domain"),
        "method": config.get("method"),
        "seed": config.get("seed"),
        "run_id": run_dir.name,
    }
    actual = {
        "domain": launch.get("domain"),
        "method": launch.get("method"),
        "seed": launch.get("primary_seed"),
        "run_id": launch.get("run_id"),
    }
    if expected != actual:
        raise AuditPolicyError(f"resolved config/launch identity mismatch: {expected} != {actual}")
    if launch.get("models", {}).get("policy") != policy.get("model"):
        raise AuditPolicyError("launch policy model differs from resolved config")
    if int(probe_config.get("count", -1)) != DEFAULT_PROBE_COUNT:
        raise AuditPolicyError("Phase-1 fixed probe must contain exactly 100 prompts")
    if not probe_path.is_file() or sha256_file(probe_path) != launch.get(
        "fixed_probe_manifest_sha256"
    ):
        raise AuditPolicyError("fixed-probe manifest is absent or has changed")
    if not train_path.is_file():
        raise AuditPolicyError("training prompt source is absent")
    probe = read_json(probe_path)
    if probe.get("source_sha256") != sha256_file(train_path):
        raise AuditPolicyError("fixed-probe manifest does not bind the current train source")
    revision = str(policy.get("revision", ""))
    return RunContract(
        run_dir=run_dir,
        run_id=str(launch["run_id"]),
        domain=str(launch["domain"]),
        method=str(launch["method"]),
        seed=int(launch["primary_seed"]),
        model=str(policy["model"]),
        model_revision=revision,
        tokenizer_revision=str(policy.get("tokenizer_revision", revision)),
        config_path=config_path,
        config_sha256=sha256_file(config_path),
        launch_spec_path=launch_path,
        launch_spec_sha256=sha256_file(launch_path),
        probe_manifest_path=probe_path,
        probe_manifest_sha256=sha256_file(probe_path),
        train_path=train_path,
        train_sha256=sha256_file(train_path),
    )


def load_probe_prompts(contract: RunContract) -> list[Mapping[str, Any]]:
    manifest = read_json(contract.probe_manifest_path)
    prompt_ids = [str(value) for value in manifest.get("prompt_ids", [])]
    if len(prompt_ids) != DEFAULT_PROBE_COUNT or len(set(prompt_ids)) != len(prompt_ids):
        raise AuditPolicyError("fixed-probe manifest has an invalid prompt inventory")
    if sha256_json(prompt_ids) != manifest.get("prompt_ids_sha256"):
        raise AuditPolicyError("fixed-probe prompt ID digest mismatch")
    train_rows = read_jsonl(contract.train_path)
    by_id = {str(row.get("prompt_id", "")): row for row in train_rows}
    if len(by_id) != len(train_rows) or any(prompt_id not in by_id for prompt_id in prompt_ids):
        raise AuditPolicyError("fixed probe is not an exact subset of the training source")
    prompts = []
    for prompt_id in prompt_ids:
        row = by_id[prompt_id]
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages:
            raise AuditPolicyError(f"probe prompt lacks chat messages: {prompt_id}")
        prompts.append(
            {
                "prompt_id": prompt_id,
                "source_row_id": str(row.get("source_row_id", prompt_id)),
                "messages": [dict(message) for message in messages],
            }
        )
    return prompts


def publish_probe_prompts(contract: RunContract, output_root: Path) -> Path:
    """Publish the exact 100-prompt chat inventory consumed by generation/KL."""
    path = output_root.resolve() / "manifests" / "fixed_train_probe_prompts.jsonl"
    prompts = load_probe_prompts(contract)
    if path.exists():
        if read_jsonl(path) != prompts:
            raise AuditPolicyError("published fixed-probe prompt inventory changed")
        return path
    write_jsonl_atomic(path, prompts)
    return path


def inspect_checkpoint(contract: RunContract, step: int) -> CheckpointIdentity:
    if step < 0:
        raise ValueError("checkpoint step must be non-negative")
    actor_dir = contract.run_dir / "verl-run" / "checkpoints" / f"global_step_{step}" / "actor"
    source_model = actor_dir / "model_world_size_1_rank_0.pt"
    if not source_model.is_file() or not (actor_dir / "huggingface").is_dir():
        raise AuditPolicyError(f"checkpoint {step} lacks world-size-1 model or HF metadata")
    return CheckpointIdentity(
        step=step,
        actor_dir=actor_dir,
        source_model=source_model,
        source_model_sha256=sha256_file(source_model),
        source_model_bytes=source_model.stat().st_size,
    )


def _validate_export(export_dir: Path, checkpoint: CheckpointIdentity) -> Mapping[str, Any]:
    manifest_path = export_dir / "audit_export_manifest.json"
    config_path = export_dir / "config.json"
    shards = sorted(export_dir.glob("*.safetensors"))
    if not manifest_path.is_file() or not config_path.is_file() or not shards:
        raise AuditPolicyError(f"incomplete checkpoint export: {export_dir}")
    manifest = read_json(manifest_path)
    if (
        int(manifest.get("checkpoint_step", -1)) != checkpoint.step
        or manifest.get("source_model_sha256") != checkpoint.source_model_sha256
        or int(manifest.get("source_model_bytes", -1)) != checkpoint.source_model_bytes
    ):
        raise AuditPolicyError(f"checkpoint export provenance mismatch: {export_dir}")
    for record in manifest.get("artifacts", []):
        validate_artifact_record(record)
    return manifest


def export_checkpoint(
    contract: RunContract,
    *,
    step: int,
    export_root: Path,
    merger_python: str = sys.executable,
    verl_root: Path | None = None,
    runner: Callable[..., Any] = subprocess.run,
) -> Path:
    """Export one FSDP world-size-1 checkpoint through veRL's supported merger."""

    checkpoint = inspect_checkpoint(contract, step)
    export_root = export_root.resolve()
    export_dir = export_root / f"global_step_{step}"
    if export_dir.exists():
        _validate_export(export_dir, checkpoint)
        return export_dir
    export_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".global_step_{step}-", dir=export_root))
    try:
        command = [
            merger_python,
            "-m",
            "verl.model_merger",
            "merge",
            "--backend",
            "fsdp",
            "--use_cpu_initialization",
            "--local_dir",
            str(checkpoint.actor_dir),
            "--target_dir",
            str(temporary),
        ]
        runner(command, cwd=str(verl_root) if verl_root else None, check=True)
        config_path = temporary / "config.json"
        shards = sorted(temporary.glob("*.safetensors"))
        if not config_path.is_file() or not shards:
            raise AuditPolicyError("veRL merger produced no loadable Hugging Face export")
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "artifact_kind": "phase1_policy_checkpoint_export",
            "domain": contract.domain,
            "method": contract.method,
            "seed": contract.seed,
            "run_id": contract.run_id,
            "checkpoint_step": step,
            "checkpoint_id": f"global_step_{step}",
            "source_model_path": str(checkpoint.source_model),
            "source_model_sha256": checkpoint.source_model_sha256,
            "source_model_bytes": checkpoint.source_model_bytes,
            "base_model": contract.model,
            "model_revision": contract.model_revision,
            "tokenizer_revision": contract.tokenizer_revision,
            "config_sha256": contract.config_sha256,
            "launch_spec_sha256": contract.launch_spec_sha256,
            "artifacts": [
                {
                    "path": str(export_dir / path.name),
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                }
                for path in (config_path, *shards)
            ],
        }
        write_json_atomic(temporary / "audit_export_manifest.json", manifest)
        os.replace(temporary, export_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    _validate_export(export_dir, checkpoint)
    return export_dir


def _audit_namespace(contract: RunContract) -> str:
    return f"{contract.run_id}.fixed-train-probe-v1"


def _pool_row_identity(
    contract: RunContract,
    checkpoint: CheckpointIdentity,
    *,
    prompt_id: str,
    pool: str,
    sample_index: int,
) -> dict[str, Any]:
    pool_family, seed_family, count = POOL_CONTRACTS[pool]
    if sample_index not in range(count):
        raise ValueError("sample index is outside the requested probe pool")
    namespace = _audit_namespace(contract)
    logical_seed = derive_seed(namespace, seed_family, prompt_id, checkpoint.step, sample_index)
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "phase1_fixed_train_probe_response",
        "domain": contract.domain,
        "method": contract.method,
        "seed": contract.seed,
        "run_id": contract.run_id,
        "global_step": checkpoint.step,
        "checkpoint_id": f"global_step_{checkpoint.step}",
        "prompt_id": prompt_id,
        "response_id": response_id(
            namespace, seed_family, prompt_id, checkpoint.step, sample_index
        ),
        "pool": pool,
        "pool_family": pool_family,
        "policy_checkpoint": checkpoint.step,
        "policy_step": checkpoint.step,
        "evaluator_checkpoint": None,
        "fresh_or_stale": None,
        "sample_index": sample_index,
        "logical_seed": logical_seed,
        "vllm_seed": vllm_seed(logical_seed),
        "checkpoint_hash": checkpoint.source_model_sha256,
        "model": contract.model,
        "model_revision": contract.model_revision,
        "tokenizer_revision": contract.tokenizer_revision,
        "thinking": False,
        "generation_config": dict(GENERATION_CONFIG),
        "generation_config_hash": sha256_json(GENERATION_CONFIG),
    }


def _validate_pool_rows(
    rows: Sequence[Mapping[str, Any]],
    contract: RunContract,
    checkpoint: CheckpointIdentity,
    *,
    pool: str,
    expected_prompt_ids: set[str],
) -> None:
    _, _, expected_count = POOL_CONTRACTS[pool]
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if (
            row.get("pool") != pool
            or int(row.get("policy_step", -1)) != checkpoint.step
            or row.get("checkpoint_hash") != checkpoint.source_model_sha256
            or row.get("run_id") != contract.run_id
        ):
            raise AuditPolicyError(f"{pool} response provenance mismatch")
        groups[str(row.get("prompt_id", ""))].append(row)
    if set(groups) != expected_prompt_ids:
        raise AuditPolicyError(f"{pool} prompt inventory mismatch")
    response_ids = [str(row.get("response_id", "")) for row in rows]
    if len(response_ids) != len(set(response_ids)):
        raise AuditPolicyError(f"{pool} contains duplicate response IDs")
    for prompt_id, group in groups.items():
        if len(group) != expected_count or {int(row["sample_index"]) for row in group} != set(
            range(expected_count)
        ):
            raise AuditPolicyError(f"{pool} sample inventory mismatch for {prompt_id}")
        for row in group:
            expected = _pool_row_identity(
                contract,
                checkpoint,
                prompt_id=prompt_id,
                pool=pool,
                sample_index=int(row["sample_index"]),
            )
            if any(row.get(key) != value for key, value in expected.items()):
                raise AuditPolicyError(f"{pool} deterministic identity mismatch for {prompt_id}")
            if not str(row.get("response_text", "")).strip():
                raise AuditPolicyError(f"{pool} contains an empty response")


def generate_probe_pools(
    contract: RunContract,
    *,
    step: int,
    output_root: Path,
    base_url: str,
    concurrency: int = 32,
    timeout_seconds: float = 900.0,
    pools: Sequence[str] = ("probe_A", "probe_B"),
) -> Mapping[str, Any]:
    """Generate crash-resumable Pool A/B response shards for one checkpoint."""

    if concurrency <= 0:
        raise ValueError("concurrency must be positive")
    requested_pools = tuple(pools)
    if not requested_pools or len(set(requested_pools)) != len(requested_pools) or any(
        pool not in POOL_CONTRACTS for pool in requested_pools
    ):
        raise ValueError("pools must be a non-empty unique subset of probe_A/probe_B")
    checkpoint = inspect_checkpoint(contract, step)
    prompts = load_probe_prompts(contract)
    probe_prompts_path = publish_probe_prompts(contract, output_root)
    prompt_ids = {str(row["prompt_id"]) for row in prompts}
    output_dir = output_root.resolve() / "responses" / f"checkpoint-{step:06d}"
    provenance_path = output_dir / "provenance.json"
    existing_rows = {}
    if provenance_path.exists():
        provenance = read_json(provenance_path)
        for record in provenance.get("artifacts", []):
            validate_artifact_record(record)
        for pool in provenance.get("selected_pools", list(POOL_CONTRACTS)):
            path = output_dir / f"{pool}.jsonl"
            rows = read_jsonl(path)
            _validate_pool_rows(
                rows, contract, checkpoint, pool=pool, expected_prompt_ids=prompt_ids
            )
            existing_rows[pool] = rows
        if set(requested_pools) <= set(existing_rows):
            return {"reused": True, "provenance": str(provenance_path)}
    selected_pools = [pool for pool in POOL_CONTRACTS if pool in requested_pools or pool in existing_rows]
    pool_paths = {pool: output_dir / f"{pool}.jsonl" for pool in selected_pools}

    provider = VLLMPolicyGenerator(
        base_url,
        contract.model,
        contract.model_revision,
        contract.tokenizer_revision,
        timeout_seconds=timeout_seconds,
        expected_checkpoint_hash=checkpoint.source_model_sha256,
    )
    generator_identity = dict(provider.preflight())
    staging_root = output_dir / ".staging"
    tasks = [(prompt, pool) for prompt in prompts for pool in requested_pools if pool not in existing_rows]

    def generate_group(task: tuple[Mapping[str, Any], str]) -> tuple[str, list[dict[str, Any]]]:
        prompt, pool = task
        prompt_id = str(prompt["prompt_id"])
        _, _, count = POOL_CONTRACTS[pool]
        shard = staging_root / pool / f"{sha256_json(prompt_id)}.json"
        if shard.is_file():
            try:
                value = read_json(shard)
                rows = value.get("responses", []) if isinstance(value, Mapping) else []
                _validate_pool_rows(
                    rows, contract, checkpoint, pool=pool, expected_prompt_ids={prompt_id}
                )
                return pool, list(rows)
            except (OSError, ValueError, TypeError, AuditPolicyError):
                shard.unlink(missing_ok=True)
        identities = [
            _pool_row_identity(
                contract, checkpoint, prompt_id=prompt_id, pool=pool, sample_index=index
            )
            for index in range(count)
        ]
        requests = [
            GenerationRequest(
                prompt_id=prompt_id,
                messages=tuple(dict(message) for message in prompt["messages"]),
                family=pool,
                seed=int(identity["logical_seed"]),
                temperature=float(GENERATION_CONFIG["temperature"]),
                top_p=float(GENERATION_CONFIG["top_p"]),
                max_output_tokens=int(GENERATION_CONFIG["max_output_tokens"]),
                metadata={
                    "checkpoint_hash": checkpoint.source_model_sha256,
                    "pool": pool,
                    "sample_index": identity["sample_index"],
                },
            )
            for identity in identities
        ]
        # Outer prompt-group parallelism retains every independently derived seed.
        results = [provider.generate(request) for request in requests]
        rows = []
        for identity, result in zip(identities, results):
            if result.requested_model != contract.model or result.returned_model != contract.model:
                raise AuditPolicyError("policy model identity drifted while generating probe")
            rows.append(
                {
                    **identity,
                    "response_text": result.text,
                    "request_id": result.request_id,
                    "retry_count": result.retry_count,
                    "usage": dict(result.usage),
                    "raw_response_hash": result.raw_response_hash
                    or hashlib.sha256(result.text.encode()).hexdigest(),
                }
            )
        _validate_pool_rows(rows, contract, checkpoint, pool=pool, expected_prompt_ids={prompt_id})
        write_json_atomic(
            shard,
            {
                "schema_version": SCHEMA_VERSION,
                "artifact_kind": "phase1_probe_prompt_shard",
                "pool": pool,
                "prompt_id": prompt_id,
                "responses": rows,
            },
        )
        return pool, rows

    generated = {pool: list(existing_rows.get(pool, [])) for pool in selected_pools}
    with ThreadPoolExecutor(max_workers=min(concurrency, len(tasks))) as executor:
        for pool, rows in executor.map(generate_group, tasks):
            generated[pool].extend(rows)
    for pool, path in pool_paths.items():
        rows = sorted(
            generated[pool], key=lambda row: (str(row["prompt_id"]), int(row["sample_index"]))
        )
        _validate_pool_rows(rows, contract, checkpoint, pool=pool, expected_prompt_ids=prompt_ids)
        write_jsonl_atomic(path, rows)
    a_rows = generated.get("probe_A", [])
    b_rows = generated.get("probe_B", [])
    if {row["response_id"] for row in a_rows} & {row["response_id"] for row in b_rows}:
        raise AuditPolicyError("Pool A/B response identity intersection is non-empty")
    a_by_prompt = defaultdict(set)
    b_by_prompt = defaultdict(set)
    for row in a_rows:
        a_by_prompt[str(row["prompt_id"])].add(int(row["vllm_seed"]))
    for row in b_rows:
        b_by_prompt[str(row["prompt_id"])].add(int(row["vllm_seed"]))
    if any(a_by_prompt[prompt_id] & b_by_prompt[prompt_id] for prompt_id in prompt_ids):
        raise AuditPolicyError("Pool A/B provider seeds collide within a prompt")
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "phase1_fixed_train_probe_policy_pools",
        "domain": contract.domain,
        "method": contract.method,
        "seed": contract.seed,
        "run_id": contract.run_id,
        "global_step": step,
        "checkpoint_id": f"global_step_{step}",
        "policy_checkpoint": step,
        "checkpoint_hash": checkpoint.source_model_sha256,
        "checkpoint_bytes": checkpoint.source_model_bytes,
        "config_sha256": contract.config_sha256,
        "launch_spec_sha256": contract.launch_spec_sha256,
        "probe_manifest_sha256": contract.probe_manifest_sha256,
        "train_sha256": contract.train_sha256,
        "generator_identity": generator_identity,
        "generation_config": dict(GENERATION_CONFIG),
        "selected_pools": selected_pools,
        "pool_counts": {pool: len(generated[pool]) for pool in selected_pools},
        "pool_a_b_disjoint": True if a_rows and b_rows else None,
        "probe_responses_used_for_gradient": False,
        "artifacts": [
            artifact_record(probe_prompts_path),
            *[artifact_record(path) for path in pool_paths.values()],
        ],
    }
    write_json_atomic(provenance_path, provenance, immutable=False)
    return {"reused": False, "provenance": str(provenance_path)}


class DualEndpointPolicyLogprobClient(VLLMPolicyLogprobClient):
    """Preflight through the identity proxy and score through raw vLLM."""

    def __init__(self, identity_base_url: str, score_base_url: str, **kwargs: Any) -> None:
        super().__init__(identity_base_url, **kwargs)
        self.score_base_url = score_base_url.rstrip("/")

    def score(
        self,
        token_sequences: Sequence[Sequence[int]],
        response_starts: Sequence[int],
    ) -> list[tuple[float, ...]]:
        identity_base_url = self.base_url
        try:
            self.base_url = self.score_base_url
            return super().score(token_sequences, response_starts)
        finally:
            self.base_url = identity_base_url


def summarize_sampled_policy_distance(
    *,
    score_dir: Path,
    stale_policy_step: int,
    current_policy_step: int,
    response_policy_step: int,
    output_dir: Path,
) -> Mapping[str, Any]:
    """Summarize log pi_t(y)-log pi_tau(y) on y sampled from Pool-B pi_t.

    This is a teacher-forced, sampled on-policy log-ratio estimator.  It is not
    an exact distribution-level KL and individual finite-sample values may be
    negative.
    """

    if stale_policy_step >= current_policy_step or response_policy_step != current_policy_step:
        raise ValueError("distance requires stale < current and Pool-B responses from current")
    stale_path = score_dir / (
        f"policy-step-{stale_policy_step}_pool-step-{response_policy_step}.jsonl"
    )
    current_path = score_dir / (
        f"policy-step-{current_policy_step}_pool-step-{response_policy_step}.jsonl"
    )
    if not stale_path.is_file() or not current_path.is_file():
        raise AuditPolicyError("current/stale teacher-forced score artifacts are missing")
    stale_rows = {str(row["response_id"]): row for row in read_jsonl(stale_path)}
    current_rows = {str(row["response_id"]): row for row in read_jsonl(current_path)}
    if not stale_rows or set(stale_rows) != set(current_rows):
        raise AuditPolicyError("current/stale score response inventories differ")
    response_rows = []
    prompt_values: dict[str, list[float]] = defaultdict(list)
    token_sum = 0.0
    token_count = 0
    for response_key in sorted(current_rows):
        stale = stale_rows[response_key]
        current = current_rows[response_key]
        count = int(current["response_token_count"])
        if (
            int(stale["response_token_count"]) != count
            or stale["response_token_hash"] != current["response_token_hash"]
            or stale["prompt_id"] != current["prompt_id"]
        ):
            raise AuditPolicyError("teacher-forced token/prompt identity differs across policies")
        stale_lp = _decode_float32(stale["response_token_logprobs_f32le_b64"], count)
        current_lp = _decode_float32(current["response_token_logprobs_f32le_b64"], count)
        ratios = [new - old for new, old in zip(current_lp, stale_lp)]
        ratio_sum = float(sum(ratios))
        ratio_mean = ratio_sum / count
        prompt_id = str(current["prompt_id"])
        prompt_values[prompt_id].append(ratio_mean)
        token_sum += ratio_sum
        token_count += count
        response_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "artifact_kind": "phase1_sampled_policy_log_ratio",
                "stale_policy_checkpoint": stale_policy_step,
                "current_policy_checkpoint": current_policy_step,
                "response_policy_checkpoint": response_policy_step,
                "prompt_id": prompt_id,
                "response_id": response_key,
                "sample_index": int(current["sample_index"]),
                "response_token_count": count,
                "response_token_hash": current["response_token_hash"],
                "sampled_log_ratio_sum": ratio_sum,
                "sampled_log_ratio_mean_per_token": ratio_mean,
                "estimator": "teacher_forced_sampled_on_policy_log_ratio_current_minus_stale",
            }
        )
    prompt_means = [statistics.fmean(values) for values in prompt_values.values()]
    output_dir.mkdir(parents=True, exist_ok=True)
    response_path = output_dir / "policy_distance_response.jsonl"
    summary_path = output_dir / "policy_distance_summary.json"
    write_jsonl_atomic(response_path, response_rows)
    mean = statistics.fmean(prompt_means)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "phase1_sampled_policy_distance_summary",
        "stale_policy_checkpoint": stale_policy_step,
        "current_policy_checkpoint": current_policy_step,
        "response_policy_checkpoint": response_policy_step,
        "response_pool": "probe_B",
        "estimator": "teacher_forced_sampled_on_policy_log_ratio_current_minus_stale",
        "interpretation": "sampled estimator of KL(pi_current || pi_stale), not exact KL",
        "token_mask": "response tokens only; chat-template prompt prefix excluded",
        "thinking": False,
        "prompt_count": len(prompt_values),
        "response_count": len(response_rows),
        "response_token_count": token_count,
        "prompt_balanced_sampled_kl_mean": mean,
        "prompt_balanced_sampled_kl_standard_error": (
            statistics.stdev(prompt_means) / math.sqrt(len(prompt_means))
            if len(prompt_means) > 1
            else 0.0
        ),
        "token_weighted_sampled_kl_mean": token_sum / token_count,
        "source_scores": [artifact_record(stale_path), artifact_record(current_path)],
        "artifacts": [artifact_record(response_path)],
    }
    write_json_atomic(summary_path, summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    export = subparsers.add_parser("export")
    export.add_argument("--run-dir", type=Path, required=True)
    export.add_argument("--output-root", type=Path, required=True)
    export.add_argument("--steps", type=int, nargs="+", default=list(DEFAULT_CHECKPOINTS))
    export.add_argument("--merger-python", default=sys.executable)
    export.add_argument("--verl-root", type=Path)
    generate = subparsers.add_parser("generate")
    generate.add_argument("--run-dir", type=Path, required=True)
    generate.add_argument("--output-root", type=Path, required=True)
    generate.add_argument("--step", type=int, required=True)
    generate.add_argument("--base-url", required=True)
    generate.add_argument("--concurrency", type=int, default=32)
    generate.add_argument("--timeout-seconds", type=float, default=900.0)
    generate.add_argument("--pools", nargs="+", choices=list(POOL_CONTRACTS), default=list(POOL_CONTRACTS))
    score = subparsers.add_parser("score-kl")
    score.add_argument("--run-dir", type=Path, required=True)
    score.add_argument("--identity-base-url", required=True)
    score.add_argument("--score-base-url", required=True)
    score.add_argument("--served-model", required=True)
    score.add_argument("--probe-prompts", type=Path, required=True)
    score.add_argument("--scoring-step", type=int, required=True)
    score.add_argument("--checkpoint-hash", required=True)
    score.add_argument("--pool-b", type=Path, nargs="+", required=True)
    score.add_argument("--output-dir", type=Path, required=True)
    score.add_argument("--batch-size", type=int, default=32)
    summarize = subparsers.add_parser("summarize-kl")
    summarize.add_argument("--score-dir", type=Path, required=True)
    summarize.add_argument("--stale-step", type=int, required=True)
    summarize.add_argument("--current-step", type=int, required=True)
    summarize.add_argument("--response-step", type=int, required=True)
    summarize.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "export":
        contract = load_run_contract(args.run_dir)
        result = [
            str(
                export_checkpoint(
                    contract,
                    step=step,
                    export_root=args.output_root,
                    merger_python=args.merger_python,
                    verl_root=args.verl_root,
                )
            )
            for step in args.steps
        ]
    elif args.command == "generate":
        result = generate_probe_pools(
            load_run_contract(args.run_dir),
            step=args.step,
            output_root=args.output_root,
            base_url=args.base_url,
            concurrency=args.concurrency,
            timeout_seconds=args.timeout_seconds,
            pools=args.pools,
        )
    elif args.command == "score-kl":
        contract = load_run_contract(args.run_dir)
        client = DualEndpointPolicyLogprobClient(
            args.identity_base_url,
            args.score_base_url,
            served_model=args.served_model,
            model_revision=contract.model_revision,
            tokenizer_revision=contract.tokenizer_revision,
            checkpoint_hash=args.checkpoint_hash,
        )
        result = score_policy_logprobs_from_files(
            client,
            prompts_path=args.probe_prompts,
            pool_paths=args.pool_b,
            policy_step=args.scoring_step,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
        )
    else:
        result = summarize_sampled_policy_distance(
            score_dir=args.score_dir,
            stale_policy_step=args.stale_step,
            current_policy_step=args.current_step,
            response_policy_step=args.response_step,
            output_dir=args.output_dir,
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
