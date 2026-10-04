"""Fail-closed one-step OnlineRubrics canary for Phase-1."""
from __future__ import annotations

import os
import random
import socket
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from dynamic_rubric.artifacts import read_jsonl, write_json_atomic
from dynamic_rubric.hashing import sha256_file, sha256_json
from dynamic_rubric.providers.vllm_generation import VLLMPolicyGenerator
from dynamic_rubric.training.live_online import _directory_tree_hash, validate_online_step_artifact
from dynamic_rubric.training.verl_dataset import build_online_rar_verl_rows

from .config import GPT_OSS_MODEL, POLICY_MODEL, RUBRIC_JUDGE_MODEL, Phase1Config
from .preflight import topology_preflight


class OneStepCanaryError(RuntimeError):
    """Raised when the one-step canary contract is violated."""


CANARY_PROMPT_COUNT = 96
CANARY_ROLLOUTS = 16
CANARY_ELICITATION_PAIRS = 8
CANARY_RUN_PREFIX = "canary-online-one-step"


def validate_one_step_config(config: Phase1Config) -> None:
    if (config.domain, config.method, config.seed) != ("medicine", "online_rubrics", 11):
        raise OneStepCanaryError(
            "one-step canary requires medicine + online_rubrics + seed 11"
        )
    method = config.method_config
    actual = (
        int(config.training["global_prompt_batch"]),
        int(method["policy_responses_per_prompt"]),
        int(method["elicitation_pairs_per_prompt"]),
    )
    if actual != (CANARY_PROMPT_COUNT, CANARY_ROLLOUTS, CANARY_ELICITATION_PAIRS):
        raise OneStepCanaryError("one-step canary requires batch=96, rollouts=16, pairs=8")
    policy = config.models["policy"]
    if str(policy["model"]) != POLICY_MODEL or policy.get("thinking") is not False:
        raise OneStepCanaryError("one-step canary requires non-thinking Qwen3-4B")
    token_budgets = (
        int(config.models["extractor"]["max_output_tokens"]),
        int(config.models["deduplicator"]["max_output_tokens"]),
        int(config.models["judge"]["max_output_tokens"]),
    )
    if token_budgets != (8192, 8192, 4096):
        raise OneStepCanaryError(
            "one-step canary requires external output-token budgets 8192/8192/4096"
        )


def select_canary_train_rows(
    rows: Sequence[Mapping[str, Any]], *, seed: int = 11
) -> list[tuple[int, Mapping[str, Any]]]:
    if len(rows) != 1500:
        raise OneStepCanaryError(f"RaR-Medicine train must contain 1500 rows, got {len(rows)}")
    indices = random.Random(seed).sample(range(len(rows)), CANARY_PROMPT_COUNT)
    selected = [(index, rows[index]) for index in indices]
    if len({str(row["prompt_id"]) for _, row in selected}) != CANARY_PROMPT_COUNT:
        raise OneStepCanaryError("selected canary prompts are not unique")
    return selected


def write_canary_parquets(
    config: Phase1Config, *, repo_root: Path, run_root: Path
) -> tuple[Path, Path, Path]:
    try:
        from datasets import Dataset  # pyright: ignore[reportMissingImports]
    except ImportError as error:
        raise OneStepCanaryError("the runtime datasets package is unavailable") from error
    train_source = (repo_root / str(config.data["train_path"])).resolve()
    heldout_source = (repo_root / str(config.data["in_domain_policy_eval"]["path"])).resolve()
    train_rows = read_jsonl(train_source)
    heldout_rows = read_jsonl(heldout_source)
    if not heldout_rows:
        raise OneStepCanaryError("minimal validation source is empty")
    selected = select_canary_train_rows(train_rows, seed=config.seed)
    verl_train, verl_validation = build_online_rar_verl_rows(
        run_root.name, [row for _, row in selected], [heldout_rows[0]]
    )
    if len(verl_train) != CANARY_PROMPT_COUNT:
        raise OneStepCanaryError("veRL canary train rows drifted from 96")
    data_root = run_root / "verl-data"
    data_root.mkdir(parents=True, exist_ok=False)
    train_path = data_root / "train-online-one-step.parquet"
    validation_path = data_root / "validation-unused.parquet"
    Dataset.from_list(verl_train).to_parquet(str(train_path))
    Dataset.from_list(verl_validation[:1]).to_parquet(str(validation_path))
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "experiment": config.experiment,
        "domain": config.domain,
        "method": config.method,
        "seed": config.seed,
        "run_id": run_root.name,
        "pool": "train_batch",
        "selection": {
            "algorithm": "python_random_sample_without_replacement",
            "seed": config.seed,
            "source_row_count": len(train_rows),
            "selected_row_count": len(selected),
        },
        "source": {"path": str(train_source), "sha256": sha256_file(train_source)},
        "selected_rows": [
            {
                "selection_order": order,
                "source_index": source_index,
                "prompt_id": str(row["prompt_id"]),
                "prompt_hash": str(row["prompt_hash"]),
            }
            for order, (source_index, row) in enumerate(selected)
        ],
        "validation": {
            "enabled": False,
            "row_count": 1,
            "source_path": str(heldout_source),
            "source_prompt_id": str(heldout_rows[0]["prompt_id"]),
        },
        "train_parquet": str(train_path.resolve()),
        "validation_parquet": str(validation_path.resolve()),
    }
    manifest["selected_prompt_ids_sha256"] = sha256_json(
        [item["prompt_id"] for item in manifest["selected_rows"]]
    )
    manifest_path = run_root / "manifests" / "one_step_train_selection.json"
    write_json_atomic(manifest_path, manifest)
    return train_path.resolve(), validation_path.resolve(), manifest_path.resolve()


def _required_environment(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise OneStepCanaryError(f"{name} is required")
    return value


def build_one_step_environment(
    config: Phase1Config,
    *,
    repo_root: Path,
    run_root: Path,
    train_path: Path,
    validation_path: Path,
) -> dict[str, str]:
    validate_one_step_config(config)
    policy = config.models["policy"]
    model_path = Path(str(policy["local_snapshot"])).resolve()
    if not model_path.is_dir():
        raise OneStepCanaryError(f"pinned Qwen3-4B snapshot is absent: {model_path}")
    gpt_url = _required_environment("PHASE1_GPT_OSS_BASE_URL")
    qwen_url = _required_environment("PHASE1_QWEN32B_BASE_URL")
    control_url = _required_environment("ONLINE_CONTROL_URL")
    control_hash = _required_environment("ONLINE_CONTROL_CHECKPOINT_HASH")
    control_launch_spec = Path(_required_environment("ONLINE_CONTROL_LAUNCH_SPEC")).resolve()
    if not control_launch_spec.is_file():
        raise OneStepCanaryError(f"frozen control launch spec is absent: {control_launch_spec}")
    if _directory_tree_hash(model_path) != control_hash:
        raise OneStepCanaryError("control hash does not bind the pinned Qwen3-4B snapshot")
    control_identity = VLLMPolicyGenerator(
        control_url,
        str(policy["model"]),
        str(policy["revision"]),
        str(policy["revision"]),
        timeout_seconds=600.0,
        launch_spec_path=control_launch_spec,
        expected_checkpoint_hash=control_hash,
    ).preflight()
    if str(control_identity.get("checkpoint_hash", "")) != control_hash:
        raise OneStepCanaryError("frozen control endpoint identity drifted")
    environment = dict(os.environ)
    environment.update({
        "PROJECT_ROOT": str(repo_root.resolve()),
        "CONFIG_PATH": str(config.source_path.resolve()),
        "TRAIN_FILE": str(train_path),
        "VAL_FILE": str(validation_path),
        "MODEL_PATH": str(model_path),
        "RUN_DIR": str((run_root / "verl-run").resolve()),
        "ONLINE_STEP_ARTIFACT_ROOT": str((run_root / "verl-run/online_steps").resolve()),
        "ONLINE_RUN_ID": run_root.name,
        "ONLINE_CONFIG_HASH": config.config_hash,
        "ONLINE_EXPERIMENT_ARM": "phase1_online_rubrics_one_step_canary",
        "ONLINE_BASELINE_ARM": "static_r0_grpo",
        "ONLINE_CONTROL_POLICY": "pi_ref",
        "ONLINE_REPRODUCTION_CLAIM": "phase1_one_step_canary_not_full_training",
        "ONLINE_CRITERIA_SCOPE": "prompt_step_ephemeral",
        "ONLINE_FAILURE_POLICY": "fail_closed",
        "ONLINE_EXTRACTOR_MODEL": GPT_OSS_MODEL,
        "ONLINE_EXTRACTOR_RETURNED_MODEL": GPT_OSS_MODEL,
        "ONLINE_EXTRACTOR_REASONING_EFFORT": "medium",
        "ONLINE_EXTRACTOR_MAX_OUTPUT_TOKENS": str(
            config.models["extractor"]["max_output_tokens"]
        ),
        "ONLINE_DEDUP_MAX_OUTPUT_TOKENS": str(
            config.models["deduplicator"]["max_output_tokens"]
        ),
        "ONLINE_GRADER_MODEL": RUBRIC_JUDGE_MODEL,
        "ONLINE_GRADER_RETURNED_MODEL": RUBRIC_JUDGE_MODEL,
        "ONLINE_GRADER_MAX_OUTPUT_TOKENS": str(
            config.models["judge"]["max_output_tokens"]
        ),
        "ONLINE_ACTOR_MODEL": POLICY_MODEL,
        "ONLINE_ACTOR_REVISION": str(policy["revision"]),
        "ONLINE_CONTROL_MODEL": POLICY_MODEL,
        "ONLINE_CONTROL_REVISION": str(policy["revision"]),
        "ONLINE_CONTROL_TOKENIZER_REVISION": str(policy["revision"]),
        "ONLINE_EXTRACTOR_BASE_URL": gpt_url,
        "ONLINE_GRADER_BASE_URL": qwen_url,
        "ONLINE_EXTRACTOR_CONCURRENCY": "16",
        "ONLINE_GRADER_CONCURRENCY": "32",
        "ONLINE_CONTROL_CONCURRENCY": "16",
        "PHASE1_VLLM_TIMEOUT_SECONDS": "600",
        "PHASE1_VLLM_MAX_RETRIES": "4",
        "TOTAL_EPOCHS": "1",
        "EXPECTED_UPDATES": "1",
        "TRAIN_BATCH_SIZE": "96",
        "ROLLOUT_N": "16",
        "ELICITATION_PAIRS": "8",
        "PPO_MINI_BATCH_SIZE": "96",
        "N_GPUS_PER_NODE": "1",
        "CUDA_VISIBLE_DEVICES": "0",
        "CHECKPOINT_EVERY_STEP": "true",
        "ONLINE_STEP_HOOK_PATH": "pkg://dynamic_rubric.training.online_step",
        "ONLINE_STEP_HOOK_NAME": "prepare_rewards",
        "ONLINE_STEP_COMMIT_NAME": "commit_step",
        "ONLINE_STEP_RUNTIME_PATH": "pkg://dynamic_rubric.training.verl_online_runtime",
        "ONLINE_STEP_RUNTIME_NAME": "create_online_reward_runtime",
        "TRACKING_PROJECT_NAME": "phase1_dynamic_evaluator_updates",
        "TRACKING_EXPERIMENT_NAME": run_root.name,
    })
    return environment


def write_launch_spec(
    config: Phase1Config,
    *,
    run_root: Path,
    manifest_path: Path,
    environment: Mapping[str, str],
) -> Path:
    spec = {
        "schema_version": 1,
        "run_id": run_root.name,
        "experiment": config.experiment,
        "domain": config.domain,
        "method": config.method,
        "canary": True,
        "full_training": False,
        "trainer_total_training_steps": 1,
        "global_prompt_batch": 96,
        "rollouts_per_prompt": 16,
        "elicitation_pairs_per_prompt": 8,
        "optimizer_host": socket.gethostname().split(".", 1)[0],
        "optimizer_visible_gpus": environment["CUDA_VISIBLE_DEVICES"],
        "models": {
            "policy": POLICY_MODEL,
            "extractor_deduplicator": GPT_OSS_MODEL,
            "judge": RUBRIC_JUDGE_MODEL,
        },
        "endpoints": {
            "extractor_env": "PHASE1_GPT_OSS_BASE_URL",
            "judge_env": "PHASE1_QWEN32B_BASE_URL",
            "control_env": "ONLINE_CONTROL_URL",
        },
        "control": {
            "frozen": True,
            "launch_spec": environment["ONLINE_CONTROL_LAUNCH_SPEC"],
            "checkpoint_hash": environment["ONLINE_CONTROL_CHECKPOINT_HASH"],
        },
        "train_selection_manifest": str(manifest_path),
        "train_selection_manifest_sha256": sha256_file(manifest_path),
        "validation_enabled": False,
        "provider_limits": {
            "extractor_concurrency": int(environment["ONLINE_EXTRACTOR_CONCURRENCY"]),
            "grader_concurrency": int(environment["ONLINE_GRADER_CONCURRENCY"]),
            "control_concurrency": int(environment["ONLINE_CONTROL_CONCURRENCY"]),
            "evaluator_timeout_seconds": int(environment["PHASE1_VLLM_TIMEOUT_SECONDS"]),
            "evaluator_max_retries": int(environment["PHASE1_VLLM_MAX_RETRIES"]),
            "extractor_max_output_tokens": int(
                environment["ONLINE_EXTRACTOR_MAX_OUTPUT_TOKENS"]
            ),
            "dedup_max_output_tokens": int(
                environment["ONLINE_DEDUP_MAX_OUTPUT_TOKENS"]
            ),
            "grader_max_output_tokens": int(
                environment["ONLINE_GRADER_MAX_OUTPUT_TOKENS"]
            ),
        },
        "failure_policy": "fail_closed",
    }
    path = run_root / "launch_spec.json"
    write_json_atomic(path, spec)
    return path


def run_online_one_step(
    config: Phase1Config, *, repo_root: str | Path, run_id: str
) -> dict[str, Any]:
    validate_one_step_config(config)
    if not run_id.startswith(CANARY_RUN_PREFIX):
        raise OneStepCanaryError(f"run_id must start with {CANARY_RUN_PREFIX!r}")
    root = Path(repo_root).resolve()
    run_root = config.run_root(root, run_id).resolve()
    if run_root.exists():
        raise OneStepCanaryError(f"refusing to reuse canary output directory: {run_root}")
    topology_preflight(config, repo_root=root, require_endpoints=True)
    run_root.mkdir(parents=True)
    write_json_atomic(run_root / "config.resolved.json", dict(config.raw))
    train_path, validation_path, manifest_path = write_canary_parquets(
        config, repo_root=root, run_root=run_root
    )
    environment = build_one_step_environment(
        config, repo_root=root, run_root=run_root,
        train_path=train_path, validation_path=validation_path,
    )
    launch_spec = write_launch_spec(
        config, run_root=run_root, manifest_path=manifest_path, environment=environment
    )
    subprocess.run(
        [str(root / "scripts/phase1/run_online_one_step.sh")],
        cwd=root, env=environment, check=True,
    )
    verl_run = run_root / "verl-run"
    commit = validate_online_step_artifact(verl_run, 1, require_provider_receipts=True)
    commits = sorted((verl_run / "online_steps").glob("step-*/commit.json"))
    if len(commits) != 1:
        raise OneStepCanaryError(f"canary committed {len(commits)} steps instead of one")
    checkpoint = verl_run / "checkpoints/global_step_1"
    if not checkpoint.is_dir():
        raise OneStepCanaryError("step-1 checkpoint was not saved")
    result = {
        "schema_version": 1,
        "status": "passed",
        "run_id": run_id,
        "canary": True,
        "full_training": False,
        "committed_updates": 1,
        "checkpoint": str(checkpoint),
        "commit": str(commit["path"]),
        "launch_spec": str(launch_spec),
        "train_selection_manifest": str(manifest_path),
    }
    write_json_atomic(run_root / "one_step_complete.json", result)
    return result
