"""Live veRL orchestration for the static-R0-only pilot trajectory."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping

from dynamic_rubric.artifacts import read_jsonl, write_json_atomic, write_jsonl_atomic
from dynamic_rubric.hashing import sha256_file, sha256_json
from dynamic_rubric.providers.local_embedding import LocalBGEEmbeddingProvider
from dynamic_rubric.seeds import SeedFamily, derive_seed, response_id
from dynamic_rubric.training.policy_distance import (
    PolicyDistanceEnricher,
    summarize_policy_distance,
)
from dynamic_rubric.training.probe_export import ProbeRecord, validate_probe_inventory
from dynamic_rubric.training.verl_adapter import dependency_gate, write_launch_spec
from dynamic_rubric.training.verl_dataset import write_verl_parquets


class LiveStaticTrainingError(RuntimeError):
    pass


def _required_environment(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise LiveStaticTrainingError(f"live static training requires {name}")
    return value


def build_training_environment(
    context: Any,
    train_path: Path,
    probe_path: Path,
    rubric_path: Path,
    run_dir: Path,
) -> dict[str, str]:
    raw = context.raw.get("training", {})
    if not isinstance(raw, Mapping):
        raise LiveStaticTrainingError("config.training must be a mapping")
    grader = context.config.models["proxy_grader"]
    max_response = int(raw.get("max_response_length", 1536))
    max_model_len = int(raw.get("max_prompt_length", 4096)) + max_response
    environment = dict(os.environ)
    environment.update(
        {
            "PROJECT_ROOT": str(context.root),
            "TRAIN_FILE": str(train_path),
            "VAL_FILE": str(probe_path),
            "DYNAMIC_RUBRIC_STATIC_RUBRIC_PATH": str(rubric_path),
            "RUN_DIR": str(run_dir),
            "TRACKING_PROJECT_NAME": "dynamic_rubric_static_grpo",
            "TRACKING_EXPERIMENT_NAME": f"static_r0_grpo__{context.run_id}",
            "ROLLOUT_CACHE_DIR": str(context.run_root / "provider_cache" / "policy-rollouts"),
            "TOTAL_STEPS": str(context.config.training.max_steps),
            "TRAIN_BATCH_SIZE": str(context.config.training.train_batch_size),
            "ROLLOUT_N": str(context.config.training.rollout_n),
            "PPO_MINI_BATCH_SIZE": str(context.config.training.train_batch_size),
            "MAX_RESPONSE_LENGTH": str(max_response),
            "MAX_MODEL_LEN": str(max_model_len),
            "ACTOR_MAX_TOKEN_LEN": str(max(8192, max_model_len)),
            "TEST_FREQ": "1",
            "VAL_BEFORE_TRAIN": "False",
            "SAVE_FREQ": "1",
            "CHECKPOINT_STEPS": json.dumps(
                list(context.config.training.checkpoint_steps), separators=(",", ":")
            ),
            "RESUME_MODE": "auto",
            "POLICY_GPU": environment.get("DYNAMIC_RUBRIC_POLICY_GPU", "0"),
            "ROLLOUT_GPU_MEMORY": environment.get(
                "DYNAMIC_RUBRIC_ROLLOUT_GPU_MEMORY", "0.42"
            ),
            "DYNAMIC_RUBRIC_VLLM_URL": _required_environment("DYNAMIC_RUBRIC_VLLM_URL"),
            "DYNAMIC_RUBRIC_GRADER_MODEL": str(grader["model"]),
            "DYNAMIC_RUBRIC_GRADER_REVISION": str(grader["revision"]),
            "DYNAMIC_RUBRIC_GRADER_TOKENIZER_REVISION": str(
                grader["tokenizer_revision"]
            ),
            "DYNAMIC_RUBRIC_GRADER_TIMEOUT_SECONDS": environment.get(
                "DYNAMIC_RUBRIC_GRADER_TIMEOUT_SECONDS", "900"
            ),
        }
    )
    return environment


def _prompt_splits(public_root: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for split in ("pilot_probe", "pilot_audit"):
        for row in read_jsonl(public_root / f"{split}.jsonl"):
            prompt_id = str(row["prompt_id"])
            if prompt_id in values:
                raise LiveStaticTrainingError(f"probe prompt appears in two splits: {prompt_id}")
            values[prompt_id] = split
    return values


def _ground_probe(context: Any, raw: Mapping[str, Any], split: str, source_hash: str) -> dict[str, Any]:
    prompt_id = str(raw["prompt_id"])
    step = int(raw["policy_step"])
    family = SeedFamily(str(raw["family"]))
    sample_index = int(raw["sample_index"])
    canonical_id = response_id(context.run_id, family, prompt_id, step, sample_index)
    if raw.get("response_id") != canonical_id or int(raw.get("step", -1)) != step:
        raise LiveStaticTrainingError(f"probe identity/step mismatch: {prompt_id} step={step}")
    return {
        "run_id": context.run_id,
        "prompt_id": prompt_id,
        "split": split,
        "policy_id": f"pi_{step}",
        "policy_step": step,
        "timing": "after_optimizer_update",
        "family": family.value,
        "sample_index": sample_index,
        "seed": int(raw["logical_seed"]),
        "response_id": canonical_id,
        "response_text": str(raw["output"]),
        "checkpoint_hash": sha256_json(
            {
                "run_id": context.run_id,
                "config_hash": context.config.config_hash,
                "policy_step": step,
                "source_probe_sha256": source_hash,
            }
        ),
        "config_hash": context.config.config_hash,
        "base_policy": str(context.config.models["policy"]["model"]),
        "static_proxy_reward": float(raw["static_reward"]),
        "criterion_probabilities": [float(value) for value in raw["criterion_probabilities"]],
        "kl_from_pi0": (
            None if raw.get("kl_from_pi0") is None else float(raw["kl_from_pi0"])
        ),
        "response_embedding_distance": None,
        "source_probe_sha256": source_hash,
    }


def _probe_record(row: Mapping[str, Any]) -> ProbeRecord:
    return ProbeRecord(**{key: row[key] for key in ProbeRecord.__dataclass_fields__})


def finalize_probe_exports(
    context: Any,
    run_dir: Path,
    distance_enricher: PolicyDistanceEnricher | None = None,
) -> dict[str, Any]:
    splits = _prompt_splits(context.public_root)
    dev_prompt_count = sum(value == "pilot_probe" for value in splits.values())
    final_prompt_count = sum(value == "pilot_audit" for value in splits.values())
    development: list[dict[str, Any]] = []
    final: list[dict[str, Any]] = []
    index: list[dict[str, Any]] = []
    for step in range(1, context.config.training.max_steps + 1):
        source = run_dir / "probes" / f"{step}.jsonl"
        if not source.is_file():
            raise LiveStaticTrainingError(f"missing post-update probe export: {source}")
        source_hash = sha256_file(source)
        step_dev: list[dict[str, Any]] = []
        step_final: list[dict[str, Any]] = []
        for raw in read_jsonl(source):
            prompt_id = str(raw.get("prompt_id", ""))
            split = splits.get(prompt_id)
            if split is None:
                raise LiveStaticTrainingError(f"unknown probe prompt: {prompt_id}")
            row = _ground_probe(context, raw, split, source_hash)
            if row["policy_step"] != step:
                raise LiveStaticTrainingError(f"{source.name} contains step {row['policy_step']}")
            (step_dev if split == "pilot_probe" else step_final).append(row)
        if distance_enricher is not None:
            step_dev = distance_enricher.enrich(step_dev)
            step_final = distance_enricher.enrich(step_final)
        for rows, count in ((step_dev, dev_prompt_count), (step_final, final_prompt_count)):
            validate_probe_inventory(
                (_probe_record(row) for row in rows), count, 1, 4
            )
        dev_shard = context.stage_root() / "trajectory" / "development" / f"step-{step:06d}.jsonl"
        final_shard = context.run_root / "trajectory" / "final_sealed" / "shards" / f"step-{step:06d}.jsonl"
        write_jsonl_atomic(dev_shard, step_dev)
        write_jsonl_atomic(final_shard, step_final)
        development.extend(step_dev)
        final.extend(step_final)
        index.append(
            {
                "policy_step": step,
                "source": str(source.relative_to(context.root)),
                "source_sha256": source_hash,
                "development_records": len(step_dev),
                "sealed_final_records": len(step_final),
            }
        )
    for rows, count in ((development, dev_prompt_count), (final, final_prompt_count)):
        validate_probe_inventory(
            (_probe_record(row) for row in rows),
            count,
            context.config.training.max_steps,
            4,
        )
    dev_path = context.stage_root() / "trajectory_development.jsonl"
    final_path = context.run_root / "trajectory" / "final_sealed" / "responses.jsonl"
    write_jsonl_atomic(dev_path, development)
    write_jsonl_atomic(final_path, final)
    write_json_atomic(context.stage_root() / "probe_index.json", index)
    result = {
        "development_records": len(development),
        "sealed_final_records": len(final),
        "probe_steps": len(index),
        "development_sha256": sha256_file(dev_path),
        "sealed_final_sha256": sha256_file(final_path),
    }
    if distance_enricher is not None:
        summaries = summarize_policy_distance([*development, *final])
        summary_path = context.stage_root() / "policy_distance_summary.json"
        write_json_atomic(summary_path, summaries)
        result.update(
            {
                "policy_distance_groups": len(summaries),
                "policy_distance_summary_sha256": sha256_file(summary_path),
            }
        )
    return result


def publish_reference_export(context: Any, source: Path) -> list[dict[str, Any]]:
    prompt_splits = _prompt_splits(context.public_root)
    rows = read_jsonl(source)
    counts: dict[tuple[str, str], int] = {}
    seen_ids: set[str] = set()
    for row in rows:
        prompt_id = str(row.get("prompt_id", ""))
        family = SeedFamily(str(row.get("family", "")))
        sample_index = int(row.get("sample_index", -1))
        if family not in {
            SeedFamily.REFERENCE_DISCOVERY,
            SeedFamily.REFERENCE_VALIDATION,
        }:
            raise LiveStaticTrainingError(f"invalid pi0 reference family: {family.value}")
        if row.get("split") != prompt_splits.get(prompt_id):
            raise LiveStaticTrainingError(f"pi0 reference split mismatch: {prompt_id}")
        expected_seed = derive_seed(
            context.run_id, family, prompt_id, 0, sample_index
        )
        expected_id = response_id(
            context.run_id, family, prompt_id, 0, sample_index
        )
        if (
            int(row.get("policy_step", -1)) != 0
            or int(row.get("seed", -1)) != expected_seed
            or row.get("response_id") != expected_id
            or expected_id in seen_ids
        ):
            raise LiveStaticTrainingError(f"invalid pi0 reference identity: {prompt_id}")
        seen_ids.add(expected_id)
        key = (prompt_id, family.value)
        counts[key] = counts.get(key, 0) + 1
    expected_prompts = len(prompt_splits)
    if (
        len(counts) != expected_prompts * 2
        or any(
            count
            != (8 if family == SeedFamily.REFERENCE_DISCOVERY.value else 4)
            for (_, family), count in counts.items()
        )
    ):
        raise LiveStaticTrainingError("pi0 reference inventory is incomplete")
    destination = context.stage_root() / "reference_responses.jsonl"
    write_jsonl_atomic(destination, rows)
    return rows


def checkpoint_inventory(context: Any, run_dir: Path) -> list[dict[str, Any]]:
    root = run_dir / "checkpoints"
    actual = sorted(
        int(path.name.removeprefix("global_step_"))
        for path in root.glob("global_step_*")
        if path.is_dir() and path.name.removeprefix("global_step_").isdigit()
    )
    expected = list(context.config.training.checkpoint_steps)
    if actual != expected:
        raise LiveStaticTrainingError(
            f"focal checkpoint inventory mismatch: expected={expected}, actual={actual}"
        )
    inventory: list[dict[str, Any]] = []
    for step in expected:
        checkpoint = root / f"global_step_{step}"
        files = [
            {
                "path": str(path.relative_to(context.root)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(value for value in checkpoint.rglob("*") if value.is_file())
        ]
        if not files:
            raise LiveStaticTrainingError(f"empty focal checkpoint: {checkpoint}")
        inventory.append(
            {
                "policy_id": f"pi_{step}",
                "step": step,
                "semantics": "initial_policy" if step == 0 else "after_optimizer_update",
                "base_model": str(context.config.models["policy"]["model"]),
                "config_hash": context.config.config_hash,
                "files": files,
                "content_hash": sha256_json(files),
            }
        )
    write_json_atomic(context.stage_root() / "checkpoints.json", inventory)
    return inventory


def run_live_static_training(
    context: Any,
    rubric_path: Path,
    reference_path: Path,
    lock: Mapping[str, Any],
) -> dict[str, Any]:
    capabilities = dependency_gate(lock, context.root)
    references = publish_reference_export(context, reference_path)
    train_path, probe_path = write_verl_parquets(
        context.public_root, context.stage_root() / "verl-data", context.run_id
    )
    run_dir = context.stage_root() / "verl-run"
    training = dict(context.raw.get("training", {}))
    training.update(
        {
            "policy_gpu": int(os.environ.get("DYNAMIC_RUBRIC_POLICY_GPU", "0")),
            "grader_gpu": 1,
            "rollout_gpu_memory_utilization": float(
                os.environ.get("DYNAMIC_RUBRIC_ROLLOUT_GPU_MEMORY", "0.42")
            ),
        }
    )
    write_launch_spec(
        context.stage_root() / "launch_spec.json",
        context.config.config_hash,
        context.run_id,
        rubric_path,
        capabilities,
        training,
    )
    environment = build_training_environment(context, train_path, probe_path, rubric_path, run_dir)
    subprocess.run(
        [str(context.root / "scripts" / "run_static_grpo.sh")],
        cwd=context.root,
        env=environment,
        check=True,
    )
    embedding = context.config.models["criterion_embedding"]
    embedding_path = Path(
        _required_environment("DYNAMIC_RUBRIC_EMBEDDING_MODEL_PATH")
    )
    distance_enricher = PolicyDistanceEnricher.from_references(
        LocalBGEEmbeddingProvider(
            embedding_path,
            str(embedding["model"]),
            str(embedding["revision"]),
            device=os.environ.get("DYNAMIC_RUBRIC_EMBEDDING_DEVICE", "cpu"),
        ),
        references,
    )
    probes = finalize_probe_exports(context, run_dir, distance_enricher)
    checkpoints = checkpoint_inventory(context, run_dir)
    actual_rollouts = sorted(
        int(path.stem)
        for path in (run_dir / "rollouts").glob("*.jsonl")
        if path.stem.isdigit()
    )
    expected_rollouts = list(range(1, context.config.training.max_steps + 1))
    if actual_rollouts != expected_rollouts:
        raise LiveStaticTrainingError("training rollout inventory is incomplete")
    result = {
        "credential_free_simulation": False,
        "reward_source": "static_r0_only",
        "dynamic_artifact_inputs": 0,
        "pi0_reference_responses": len(references),
        "timing": "after_optimizer_update",
        "resume_mode": "auto",
        "gpu_topology": {
            "policy_actor_rollout": 0,
            "proxy_grader": 1,
            "rollout_gpu_memory_utilization": float(environment["ROLLOUT_GPU_MEMORY"]),
        },
        "focal_checkpoints": [row["step"] for row in checkpoints],
        **probes,
    }
    write_json_atomic(context.stage_root() / "result.json", result)
    return result
