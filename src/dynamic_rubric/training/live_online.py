"""Launch and validate paper-faithful, same-step OnlineRubrics training."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping

from dynamic_rubric.artifacts import read_json, read_jsonl, write_json_atomic
from dynamic_rubric.hashing import sha256_file, sha256_json
from dynamic_rubric.providers.vllm_generation import VLLMPolicyGenerator
from dynamic_rubric.training.checkpoint_archive import (
    CheckpointArchiveError,
    verify_public_archive,
)
from dynamic_rubric.training.verl_adapter import dependency_gate
from dynamic_rubric.training.verl_dataset import write_online_rar_verl_parquets


class LiveOnlineTrainingError(RuntimeError):
    pass


_REQUIRED_HOOK_SYMBOLS = ("prepare_rewards", "commit_step")
_PROVIDER_RECEIPTS = (
    "control_generation_receipts.jsonl",
    "extraction_receipts.jsonl",
    "dedup_receipts.jsonl",
    "grader_receipts.jsonl",
)


def _online_config(context: Any) -> Any:
    config = context.config.online_training
    if config is None:
        raise LiveOnlineTrainingError("config has no online_training section")
    return config


def online_cost_estimate(config: Any) -> dict[str, int]:
    updates = int(config.expected_updates)
    prompts = updates * int(config.prompt_batch_size)
    extractor = prompts * int(config.elicitation_pairs_per_prompt)
    dedup = prompts
    grader = prompts * int(config.rollouts_per_prompt)
    control = prompts * int(config.elicitation_pairs_per_prompt)
    return {
        "optimizer_updates": updates,
        "prompts": prompts,
        "extractor_calls": extractor,
        "dedup_calls": dedup,
        "grader_calls": grader,
        "external_calls": extractor + dedup + grader,
        "control_generations": control,
    }



def online_checkpoint_comparison_contract(config: Any) -> dict[str, Any]:
    """Canonical labels and focal checkpoints for held-out rubric staleness audits."""

    return {
        "schema_version": 1,
        "experiment_arm": config.experiment_arm,
        "baseline_arm": config.baseline_arm,
        "training_reward_source": "online_r0_union_elicited_same_step",
        "evaluation_split": "held_out_only",
        "focal_checkpoint_steps": list(config.comparison_checkpoint_steps),
        "stale_comparator": config.stale_comparator,
        "variant_aliases": {
            "r0": "r0",
            "r_t_delta": "control",
            "r_t": "current",
        },
        "variant_definitions": {
            "r0": "public static base rubric only",
            "r_t_delta": "R0 plus held-out extension from the previous focal checkpoint",
            "r_t": "R0 plus held-out extension from the current focal checkpoint",
        },
    }


def _directory_tree_hash(root: Path) -> str:
    if not root.is_dir():
        raise LiveOnlineTrainingError(f"committed checkpoint directory is missing: {root}")
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise LiveOnlineTrainingError(f"committed checkpoint directory is empty: {root}")
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def _checkpoint_tree_hash(root: Path) -> str:
    if not (root / "data.pt").is_file():
        raise LiveOnlineTrainingError(f"committed checkpoint is incomplete: {root}")
    return _directory_tree_hash(root)


def _actor_parameter_tree_hash(root: Path) -> str:
    if not root.is_dir():
        raise LiveOnlineTrainingError(f"committed actor directory is missing: {root}")
    digest = hashlib.sha256()
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and not path.name.startswith("optim_world_size_")
        and not path.name.startswith("extra_state_world_size_")
    )
    if not files:
        raise LiveOnlineTrainingError(f"committed actor parameters are missing: {root}")
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def _validate_manifest_file_hashes(
    artifact_dir: Path,
    artifacts: Mapping[str, str],
    *,
    require_provider_receipts: bool = False,
) -> None:
    for name, expected_hash in artifacts.items():
        if name in {
            "logical_policy_token",
            "resume_checkpoint_hash",
            "actor_parameter_hash",
        }:
            continue
        if Path(name).name != name or not name:
            raise LiveOnlineTrainingError(f"online artifact name is unsafe: {name!r}")
        artifact = artifact_dir / name
        label = (
            "provider receipt hash mismatch"
            if require_provider_receipts and name in _PROVIDER_RECEIPTS
            else "online artifact hash mismatch"
        )
        if not artifact.is_file() or sha256_file(artifact) != str(expected_hash):
            raise LiveOnlineTrainingError(f"{label}: {name}")


def _committed_chain(run_dir: Path) -> list[dict[str, Any]]:
    """Return the validated per-update lineage, with optional physical checkpoints."""

    online_root = run_dir / "online_steps"
    commit_paths = sorted(online_root.rglob("commit.json")) if online_root.is_dir() else []
    if not commit_paths:
        raise LiveOnlineTrainingError("resume-online requires at least one committed online step")

    chain: dict[int, dict[str, Any]] = {}
    run_id: str | None = None
    for path in commit_paths:
        relative = path.relative_to(online_root)
        if len(relative.parts) != 2 or not re.fullmatch(r"step-[0-9]{6}", relative.parts[0]):
            raise LiveOnlineTrainingError(f"online commit path is malformed: {relative}")
        directory_step = int(relative.parts[0][5:])
        try:
            from dynamic_rubric.training.online_contracts import validate_online_step_manifest

            manifest = dataclasses.asdict(validate_online_step_manifest(path))
            _validate_manifest_file_hashes(path.parent, manifest["artifacts"])
        except LiveOnlineTrainingError:
            raise
        except (KeyError, OSError, TypeError, ValueError) as error:
            raise LiveOnlineTrainingError(f"online commit is malformed: {relative}") from error
        step = int(manifest["optimizer_update_index"])
        if step in chain:
            raise LiveOnlineTrainingError(f"online commit chain is ambiguous at step {step}")
        if step != directory_step or manifest["state"] != "committed":
            raise LiveOnlineTrainingError("online commit chain contains an invalid step binding")
        if run_id is None:
            run_id = str(manifest["run_id"])
        elif str(manifest["run_id"]) != run_id:
            raise LiveOnlineTrainingError("online commit chain mixes run identities")

        artifacts = manifest["artifacts"]
        logical_token = str(artifacts.get("logical_policy_token", ""))
        if len(logical_token) != 64:
            raise LiveOnlineTrainingError("committed online step lacks a logical policy token")
        resume_hash = str(artifacts.get("resume_checkpoint_hash", ""))
        parameter_hash = str(artifacts.get("actor_parameter_hash", ""))
        if bool(resume_hash) != bool(parameter_hash):
            raise LiveOnlineTrainingError("committed online step has partial checkpoint hashes")

        record: dict[str, Any] = {
            "schema_version": 1,
            "optimizer_update_index": step,
            "manifest_hash": sha256_json(manifest),
            "logical_policy_token": logical_token,
            "checkpoint_saved": bool(resume_hash),
        }
        if resume_hash:
            checkpoint = (run_dir / "checkpoints" / f"global_step_{step}").resolve()
            actor = checkpoint / "actor"
            actor_is_archived = not actor.exists()
            if actor.exists():
                if _actor_parameter_tree_hash(actor) != parameter_hash:
                    raise LiveOnlineTrainingError(
                        f"committed actor parameter hash mismatch at step {step}"
                    )
            else:
                receipt_path = run_dir / "checkpoint_archives" / f"global_step_{step}.json"
                if not receipt_path.is_file():
                    raise LiveOnlineTrainingError(
                        f"committed actor directory is missing and has no archive receipt at step {step}"
                    )
                try:
                    receipt = read_json(receipt_path)
                    if (
                        receipt.get("checkpoint_step") != step
                        or receipt.get("run_id") != run_dir.parent.name
                        or receipt.get("actor_parameter_hash") != parameter_hash
                    ):
                        raise CheckpointArchiveError("checkpoint archive receipt binding mismatch")
                    if verify_public_archive(receipt) != parameter_hash:
                        raise CheckpointArchiveError("checkpoint archive actor hash mismatch")
                except (CheckpointArchiveError, KeyError, OSError, TypeError, ValueError) as error:
                    raise LiveOnlineTrainingError(
                        f"committed actor archive verification failed at step {step}"
                    ) from error
            if (
                not actor_is_archived
                and (checkpoint / "data.pt").is_file()
                and _checkpoint_tree_hash(checkpoint) != resume_hash
            ):
                raise LiveOnlineTrainingError(f"committed resume checkpoint hash mismatch at step {step}")
            record.update(
                {
                    "checkpoint": str(checkpoint),
                    "resume_checkpoint_hash": resume_hash,
                    "actor_parameter_hash": parameter_hash,
                }
            )
        chain[step] = record

    steps = sorted(chain)
    if steps != list(range(1, steps[-1] + 1)):
        raise LiveOnlineTrainingError(f"online commit chain is noncontiguous: {steps}")
    return [chain[step] for step in steps]


def write_online_checkpoint_comparison_plan(
    context: Any,
    run_dir: Path,
    preflight: Mapping[str, Any],
) -> Path:
    """Bind every focal comparison to an immutable checkpoint and rubric time."""

    config = _online_config(context)
    chain = _committed_chain(run_dir)
    if len(chain) != config.expected_updates:
        raise LiveOnlineTrainingError(
            "checkpoint comparison plan requires the complete committed update chain"
        )
    committed = {int(item["optimizer_update_index"]): item for item in chain}
    base_checkpoint = Path(
        str(context.config.models["policy"].get("local_snapshot", ""))
    ).resolve()
    base_hash = str(preflight["control_identity"].get("checkpoint_hash", ""))
    if not base_hash:
        raise LiveOnlineTrainingError("checkpoint comparison plan is missing the A0 hash")

    focal_records: list[dict[str, Any]] = []
    previous_step: int | None = None
    updates_per_epoch = config.expected_updates / config.epochs
    for step in config.comparison_checkpoint_steps:
        if step == 0:
            model = {
                "checkpoint": str(base_checkpoint),
                "checkpoint_hash": base_hash,
                "actor_hash": base_hash,
                "source": "pinned_base_actor",
            }
        else:
            try:
                model = dict(committed[step])
            except KeyError as error:
                raise LiveOnlineTrainingError(
                    f"focal checkpoint has no committed online step: {step}"
                ) from error
            if model.get("checkpoint_saved") is not True:
                raise LiveOnlineTrainingError(
                    f"focal checkpoint has no physical policy snapshot: {step}"
                )
            model["checkpoint_hash"] = model["actor_parameter_hash"]
            model["actor_hash"] = model["actor_parameter_hash"]
            model["source"] = "committed_online_update"

        stale = None
        if previous_step is not None:
            stale = {
                "paper_label": "R_{t-delta}",
                "canonical_key": "r_t_delta",
                "rubric_checkpoint_step": previous_step,
                "delta_updates": step - previous_step,
            }
        focal_records.append(
            {
                "policy_step": step,
                "epoch": round(step / updates_per_epoch, 6),
                "model": model,
                "rubric_comparison": {
                    "r0": {
                        "paper_label": "R0",
                        "canonical_key": "r0",
                        "rubric_checkpoint_step": 0,
                    },
                    "r_t_delta": stale,
                    "r_t": {
                        "paper_label": "R_t",
                        "canonical_key": "r_t",
                        "rubric_checkpoint_step": step,
                    },
                },
            }
        )
        previous_step = step

    value = {
        **online_checkpoint_comparison_contract(config),
        "run_id": context.run_id,
        "config_hash": context.config.config_hash,
        "base_actor": {
            "checkpoint": str(base_checkpoint),
            "checkpoint_hash": base_hash,
        },
        "focal_checkpoints": focal_records,
        "leakage_guard": (
            "R_t and R_{t-delta} are constructed only from held-out responses; "
            "training-step ephemeral rubrics are never replayed into the audit"
        ),
    }
    path = context.stage_root() / "checkpoint_comparison_plan.json"
    write_json_atomic(path, value, immutable=True)
    return path


def resolve_committed_resume(run_dir: Path) -> tuple[Path, int]:
    """Resume at veRL's latest full checkpoint and archive any logical tail."""

    chain = _committed_chain(run_dir)
    tracker = run_dir / "checkpoints" / "latest_checkpointed_iteration.txt"
    if not tracker.is_file():
        raise LiveOnlineTrainingError("resume-online requires veRL's sealed checkpoint tracker")
    try:
        resume_step = int(tracker.read_text(encoding="utf-8").strip())
    except ValueError as error:
        raise LiveOnlineTrainingError("veRL checkpoint tracker is malformed") from error
    if resume_step < 1 or resume_step > len(chain):
        raise LiveOnlineTrainingError("veRL checkpoint tracker is outside the commit chain")
    expected_latest = chain[resume_step - 1]
    if expected_latest.get("checkpoint_saved") is not True:
        raise LiveOnlineTrainingError("veRL tracker points to a logical-only online commit")
    checkpoint = Path(str(expected_latest["checkpoint"]))
    if _checkpoint_tree_hash(checkpoint) != expected_latest["resume_checkpoint_hash"]:
        raise LiveOnlineTrainingError("latest full resume checkpoint hash mismatch")

    tail_steps = list(range(resume_step + 1, len(chain) + 1))
    extra_step_dirs = sorted(
        path
        for path in (run_dir / "online_steps").glob("step-*")
        if path.is_dir() and int(path.name.removeprefix("step-")) > resume_step
    )
    extra_checkpoints = sorted(
        path
        for path in (run_dir / "checkpoints").glob("global_step_*")
        if path.is_dir() and int(path.name.removeprefix("global_step_")) > resume_step
    )
    if tail_steps or extra_step_dirs or extra_checkpoints:
        archive_base = run_dir / "replay_archive" / f"resume-from-{resume_step:06d}"
        attempt = 1
        while (archive_base / f"attempt-{attempt:03d}").exists():
            attempt += 1
        archive = archive_base / f"attempt-{attempt:03d}"
        (archive / "online_steps").mkdir(parents=True)
        (archive / "checkpoints").mkdir(parents=True)
        for path in extra_step_dirs:
            path.rename(archive / "online_steps" / path.name)
        for path in extra_checkpoints:
            path.rename(archive / "checkpoints" / path.name)
        latest_path = run_dir / "latest_commit.json"
        if latest_path.is_file():
            latest_path.rename(archive / "latest_commit.json")

    write_json_atomic(run_dir / "latest_commit.json", expected_latest, immutable=False)
    return checkpoint, resume_step


def build_online_training_environment(
    context: Any,
    train_path: Path,
    validation_path: Path,
    run_dir: Path,
    *,
    resume: bool = False,
) -> dict[str, str]:
    config = _online_config(context)
    policy = context.config.models["policy"]
    extractor = context.config.models["rubric_extractor"]
    grader = context.config.models["online_grader"]
    environment = dict(os.environ)
    model_path = str(policy.get("local_snapshot", ""))
    if not model_path:
        raise LiveOnlineTrainingError("models.policy.local_snapshot is required")
    resume_path = None
    if resume:
        resume_path, _ = resolve_committed_resume(run_dir)
    environment.update(
        {
            "PROJECT_ROOT": str(context.root),
            "CONFIG_PATH": str(context.config_path),
            "TRAIN_FILE": str(train_path),
            "VAL_FILE": str(validation_path),
            "MODEL_PATH": model_path,
            "RUN_DIR": str(run_dir),
            "ONLINE_STEP_ARTIFACT_ROOT": str(run_dir / "online_steps"),
            "ONLINE_RUN_ID": context.run_id,
            "ONLINE_CONFIG_HASH": context.config.config_hash,
            "ONLINE_EXPERIMENT_ARM": config.experiment_arm,
            "ONLINE_BASELINE_ARM": config.baseline_arm,
            "TRACKING_PROJECT_NAME": "dynamic_rubric_online_rl",
            "TRACKING_EXPERIMENT_NAME": f"{config.experiment_arm}__{context.run_id}",
            "ONLINE_CONTROL_POLICY": config.control_policy,
            "ONLINE_REPRODUCTION_CLAIM": config.runtime_claim,
            "ONLINE_CRITERIA_SCOPE": config.criteria_scope,
            "ONLINE_FAILURE_POLICY": config.failure_policy,
            "ONLINE_EXTRACTOR_MODEL": str(extractor["requested_model"]),
            "ONLINE_EXTRACTOR_RETURNED_MODEL": str(extractor["requested_model"]),
            "ONLINE_EXTRACTOR_REASONING_EFFORT": str(
                extractor.get("reasoning_effort", "medium")
            ),
            "ONLINE_GRADER_MODEL": str(grader["requested_model"]),
            "ONLINE_GRADER_RETURNED_MODEL": str(grader["requested_model"]),
            "ONLINE_ACTOR_MODEL": str(policy["model"]),
            "ONLINE_ACTOR_REVISION": str(policy["revision"]),
            "ONLINE_CONTROL_MODEL": str(policy["model"]),
            "ONLINE_CONTROL_REVISION": str(policy["revision"]),
            "ONLINE_CONTROL_TOKENIZER_REVISION": str(policy["tokenizer_revision"]),
            "ONLINE_CONTROL_CHECKPOINT_HASH": environment.get(
                "ONLINE_CONTROL_CHECKPOINT_HASH", ""
            ),
            "ONLINE_CONTROL_CONCURRENCY": environment.get(
                "ONLINE_CONTROL_CONCURRENCY", "32"
            ),
            "ONLINE_SEED": str(context.config.split_seed),
            "ONLINE_EXTRACTOR_CONCURRENCY": str(config.extractor_concurrency),
            "ONLINE_GRADER_CONCURRENCY": str(config.grader_concurrency),
            "TOTAL_EPOCHS": str(config.epochs),
            "EXPECTED_UPDATES": str(config.expected_updates),
            "TRAIN_BATCH_SIZE": str(config.prompt_batch_size),
            "ROLLOUT_N": str(config.rollouts_per_prompt),
            "ELICITATION_PAIRS": str(config.elicitation_pairs_per_prompt),
            "PPO_MINI_BATCH_SIZE": str(config.prompt_batch_size),
            "LEARNING_RATE": str(config.learning_rate),
            "WARMUP_RATIO": str(config.warmup_ratio),
            "KL_COEFFICIENT": str(config.kl_coefficient),
            "N_GPUS_PER_NODE": str(config.gpu_count),
            "CHECKPOINT_INTERVAL_STEPS": str(config.checkpoint_interval_steps),
            "CHECKPOINT_STEPS": json.dumps(config.checkpoint_steps),
            "ONLINE_STEP_HOOK_PATH": "pkg://dynamic_rubric.training.online_step",
            "ONLINE_STEP_HOOK_NAME": "prepare_rewards",
            "ONLINE_STEP_COMMIT_NAME": "commit_step",
            "ONLINE_STEP_RUNTIME_PATH": "pkg://dynamic_rubric.training.verl_online_runtime",
            "ONLINE_STEP_RUNTIME_NAME": "create_online_reward_runtime",
            "RESUME_MODE": "resume_path" if resume else "disable",
            "RESUME_FROM_PATH": str(resume_path) if resume_path is not None else "null",
        }
    )
    return environment


def validate_online_launch_environment(config: Any, environment: Mapping[str, str]) -> dict[str, Any]:
    expected = {
        "ONLINE_EXPERIMENT_ARM": config.experiment_arm,
        "ONLINE_BASELINE_ARM": config.baseline_arm,
        "TRACKING_PROJECT_NAME": "dynamic_rubric_online_rl",
        "ONLINE_CONTROL_POLICY": config.control_policy,
        "ONLINE_REPRODUCTION_CLAIM": config.runtime_claim,
        "ONLINE_CRITERIA_SCOPE": "prompt_step_ephemeral",
        "ONLINE_FAILURE_POLICY": "fail_closed",
        "ONLINE_EXTRACTOR_MODEL": config.extractor_model,
        "ONLINE_EXTRACTOR_RETURNED_MODEL": config.extractor_model,
        "ONLINE_EXTRACTOR_REASONING_EFFORT": "medium",
        "ONLINE_GRADER_MODEL": config.grader_model,
        "ONLINE_GRADER_RETURNED_MODEL": config.grader_model,
        "TOTAL_EPOCHS": str(config.epochs),
        "EXPECTED_UPDATES": str(config.expected_updates),
        "TRAIN_BATCH_SIZE": str(config.prompt_batch_size),
        "ROLLOUT_N": "16",
        "ELICITATION_PAIRS": "8",
        "LEARNING_RATE": str(config.learning_rate),
        "WARMUP_RATIO": str(config.warmup_ratio),
        "KL_COEFFICIENT": str(config.kl_coefficient),
        "N_GPUS_PER_NODE": str(config.gpu_count),
        "CHECKPOINT_INTERVAL_STEPS": str(config.checkpoint_interval_steps),
        "CHECKPOINT_STEPS": json.dumps(config.checkpoint_steps),
        "ONLINE_STEP_HOOK_PATH": "pkg://dynamic_rubric.training.online_step",
        "ONLINE_STEP_HOOK_NAME": "prepare_rewards",
        "ONLINE_STEP_COMMIT_NAME": "commit_step",
        "ONLINE_STEP_RUNTIME_PATH": "pkg://dynamic_rubric.training.verl_online_runtime",
        "ONLINE_STEP_RUNTIME_NAME": "create_online_reward_runtime",
    }
    actual = {key: str(environment.get(key, "")) for key in expected}
    if actual != expected:
        raise LiveOnlineTrainingError(f"online launcher environment drifted: {actual}")
    return {"valid": True, "expected": expected}


def preflight_online_training(context: Any, *, resume: bool = False) -> dict[str, Any]:
    config = _online_config(context)
    lock_path = context.root / "environment" / "upstream-lock.json"
    if not lock_path.is_file():
        raise LiveOnlineTrainingError("environment/upstream-lock.json is missing")
    lock = read_json(lock_path)
    capabilities = dependency_gate(lock, context.root)
    if not capabilities.online_ready:
        raise LiveOnlineTrainingError(
            "veRL online hook patch is absent, unpinned, or differs from the locked patch"
        )
    hook_module = context.root / "src" / "dynamic_rubric" / "training" / "online_step.py"
    hook_text = hook_module.read_text(encoding="utf-8") if hook_module.is_file() else ""
    missing_symbols = [name for name in _REQUIRED_HOOK_SYMBOLS if f"def {name}" not in hook_text]
    if missing_symbols:
        raise LiveOnlineTrainingError(f"online hook symbols are missing: {missing_symbols}")
    runtime_module = (
        context.root / "src" / "dynamic_rubric" / "training" / "verl_online_runtime.py"
    )
    runtime_text = runtime_module.read_text(encoding="utf-8") if runtime_module.is_file() else ""
    if "def create_online_reward_runtime" not in runtime_text:
        raise LiveOnlineTrainingError("online veRL runtime factory is missing")
    if not os.environ.get("OPENAI_API_KEY"):
        raise LiveOnlineTrainingError("online training requires OPENAI_API_KEY")
    if config.control_policy == "pi_old":
        raise LiveOnlineTrainingError(
            "pi_old is not production-ready: bounded one-batch lookahead is unimplemented"
        )
    control_url = os.environ.get("ONLINE_CONTROL_URL")
    if not control_url:
        raise LiveOnlineTrainingError("frozen pi_ref requires ONLINE_CONTROL_URL")
    policy = context.config.models["policy"]
    model_path = Path(str(policy.get("local_snapshot", "")))
    if not model_path.is_dir():
        raise LiveOnlineTrainingError(f"pinned actor snapshot is absent: {model_path}")
    expected_control_hash = os.environ.get("ONLINE_CONTROL_CHECKPOINT_HASH")
    if not expected_control_hash:
        raise LiveOnlineTrainingError("frozen pi_ref requires ONLINE_CONTROL_CHECKPOINT_HASH")
    actual_control_hash = _directory_tree_hash(model_path)
    if actual_control_hash != expected_control_hash:
        raise LiveOnlineTrainingError(
            "ONLINE_CONTROL_CHECKPOINT_HASH does not match models.policy.local_snapshot"
        )
    control_launch_spec = os.environ.get("ONLINE_CONTROL_LAUNCH_SPEC")
    control_identity = VLLMPolicyGenerator(
        control_url,
        str(policy["model"]),
        str(policy["revision"]),
        str(policy["tokenizer_revision"]),
        launch_spec_path=Path(control_launch_spec) if control_launch_spec else None,
        expected_checkpoint_hash=expected_control_hash,
    ).preflight()
    train_rows = read_jsonl(context.public_root / "train.jsonl")
    if len(train_rows) != config.expected_train_rows:
        raise LiveOnlineTrainingError(
            f"online train row count drifted: expected={config.expected_train_rows}, "
            f"actual={len(train_rows)}"
        )
    hardware = lock.get("hardware", {})
    if int(hardware.get("gpu_count", -1)) != config.gpu_count:
        raise LiveOnlineTrainingError("runtime GPU count disagrees with online config")
    if str(hardware.get("gpu_model", "")) != config.accelerator:
        raise LiveOnlineTrainingError("runtime accelerator disagrees with online config")
    if resume:
        resolve_committed_resume(context.stage_root() / "verl-run")
    result = {
        "schema_version": 1,
        "ready": True,
        "run_id": context.run_id,
        "config_hash": context.config.config_hash,
        "stage": "train-online",
        "runtime_claim": config.runtime_claim,
        "experiment_arm": config.experiment_arm,
        "baseline_arm": config.baseline_arm,
        "control_policy": config.control_policy,
        "checkpoint_comparison": online_checkpoint_comparison_contract(config),
        "paper_contract": {
            "same_step_causal": True,
            "rollouts_per_prompt": 16,
            "elicitation_pairs_per_prompt": 8,
            "criteria_scope": config.criteria_scope,
            "failure_policy": config.failure_policy,
            "checkpoint_interval_steps": config.checkpoint_interval_steps,
            "checkpoint_steps": list(config.checkpoint_steps),
        },
        "models": {
            "actor": config.actor_model,
            "actor_revision": config.actor_revision,
            "extractor": config.extractor_model,
            "grader": config.grader_model,
        },
        "control_identity": dict(control_identity),
        "hardware": {
            "gpu_count": config.gpu_count,
            "accelerator": config.accelerator,
            "effective_prompt_batch_size": config.effective_prompt_batch_size,
        },
        "cost": online_cost_estimate(config),
        "verl_capabilities": dataclasses.asdict(capabilities),
        "lock_sha256": sha256_file(lock_path),
        "train_sha256": sha256_file(context.public_root / "train.jsonl"),
    }
    write_json_atomic(context.stage_root() / "online_preflight.json", result, immutable=True)
    return result


def write_online_launch_spec(
    context: Any,
    environment: Mapping[str, str],
    preflight: Mapping[str, Any],
) -> Path:
    config = _online_config(context)
    value = {
        "schema_version": 1,
        "run_id": context.run_id,
        "config_hash": context.config.config_hash,
        "mode": config.mode,
        "runtime_claim": config.runtime_claim,
        "experiment_arm": config.experiment_arm,
        "baseline_arm": config.baseline_arm,
        "control_policy": config.control_policy,
        "same_step_causal": True,
        "reward_source": "online_r0_union_elicited_same_step",
        "checkpoint_comparison": online_checkpoint_comparison_contract(config),
        "dynamic_artifact_inputs": 0,
        "expected_updates": config.expected_updates,
        "paper_hyperparameters": {
            "epochs": config.epochs,
            "prompt_batch_size": config.prompt_batch_size,
            "rollouts_per_prompt": config.rollouts_per_prompt,
            "elicitation_pairs_per_prompt": config.elicitation_pairs_per_prompt,
            "learning_rate": config.learning_rate,
            "warmup_ratio": config.warmup_ratio,
            "kl_coefficient": config.kl_coefficient,
        },
        "hook": {
            "path": environment["ONLINE_STEP_HOOK_PATH"],
            "prepare_name": environment["ONLINE_STEP_HOOK_NAME"],
            "commit_name": environment["ONLINE_STEP_COMMIT_NAME"],
            "runtime_path": environment["ONLINE_STEP_RUNTIME_PATH"],
            "runtime_name": environment["ONLINE_STEP_RUNTIME_NAME"],
        },
        "preflight_sha256": sha256_file(context.stage_root() / "online_preflight.json"),
        "cost": dict(preflight["cost"]),
        "control_identity": dict(preflight["control_identity"]),
    }
    path = context.stage_root() / "launch_spec.json"
    write_json_atomic(path, value, immutable=True)
    return path


def validate_online_step_artifact(
    run_dir: Path,
    step: int,
    *,
    require_provider_receipts: bool = False,
) -> dict[str, Any]:
    if step <= 0:
        raise LiveOnlineTrainingError("online step must be positive")
    candidates = (
        run_dir / "online_steps" / f"step-{step:06d}" / "commit.json",
        run_dir / "online_steps" / f"step-{step:06d}" / "pre_update_seal.json",
    )
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise LiveOnlineTrainingError(f"online step artifact is missing for step {step}")
    from dynamic_rubric.training.online_contracts import validate_online_step_manifest

    manifest = validate_online_step_manifest(path)
    value = dataclasses.asdict(manifest)
    if int(value["optimizer_update_index"]) != step:
        raise LiveOnlineTrainingError("online step manifest is bound to a different step")
    artifact_dir = path.parent
    hashes = value.get("artifacts", {})
    _validate_manifest_file_hashes(
        artifact_dir,
        hashes,
        require_provider_receipts=require_provider_receipts,
    )
    if require_provider_receipts:
        for name in _PROVIDER_RECEIPTS:
            receipt = artifact_dir / name
            if not receipt.is_file() or hashes.get(name) != sha256_file(receipt):
                raise LiveOnlineTrainingError(f"provider receipt hash mismatch: {name}")
    return {"valid": True, "path": str(path), "manifest": value}


def run_live_online_training(
    context: Any,
    preflight: Mapping[str, Any],
    *,
    resume: bool = False,
) -> dict[str, Any]:
    config = _online_config(context)
    data_root = context.stage_root() / "verl-data"
    train_path, validation_path = write_online_rar_verl_parquets(
        context.public_root, data_root, context.run_id
    )
    run_dir = context.stage_root() / "verl-run"
    environment = build_online_training_environment(
        context, train_path, validation_path, run_dir, resume=resume
    )
    validate_online_launch_environment(config, environment)
    launch_spec = write_online_launch_spec(context, environment, preflight)
    subprocess.run(
        [str(context.root / "scripts" / "run_online_grpo.sh")],
        cwd=context.root,
        env=environment,
        check=True,
    )
    commits = sorted((run_dir / "online_steps").glob("step-*/commit.json"))
    if len(commits) != config.expected_updates:
        raise LiveOnlineTrainingError(
            f"committed online step count mismatch: expected={config.expected_updates}, "
            f"actual={len(commits)}"
        )
    for step in range(1, config.expected_updates + 1):
        validate_online_step_artifact(run_dir, step, require_provider_receipts=True)
    comparison_plan = write_online_checkpoint_comparison_plan(
        context, run_dir, preflight
    )
    result = {
        "credential_free_simulation": False,
        "runtime_claim": config.runtime_claim,
        "experiment_arm": config.experiment_arm,
        "baseline_arm": config.baseline_arm,
        "control_policy": config.control_policy,
        "reward_source": "online_r0_union_elicited_same_step",
        "same_step_causal": True,
        "committed_updates": len(commits),
        "expected_updates": config.expected_updates,
        "launch_spec_sha256": sha256_file(launch_spec),
        "checkpoint_comparison_plan": str(comparison_plan),
        "checkpoint_comparison_plan_sha256": sha256_file(comparison_plan),
        "resume_mode": "resume_path" if resume else "disable",
        "cost": online_cost_estimate(config),
    }
    write_json_atomic(context.stage_root() / "result.json", result, immutable=True)
    return result
