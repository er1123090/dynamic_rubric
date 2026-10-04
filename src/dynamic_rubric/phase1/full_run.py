"""Fail-closed launcher for full Phase-1 OnlineRubrics runs."""

from __future__ import annotations

import math
import os
import socket
import subprocess
from pathlib import Path
from typing import Any, Mapping

from dynamic_rubric.artifacts import read_json, read_jsonl, write_json_atomic
from dynamic_rubric.hashing import sha256_file, sha256_json
from dynamic_rubric.phase1.pi0_cache import ImmutablePi0Cache, Pi0CacheError
from dynamic_rubric.training.live_online import (
    LiveOnlineTrainingError,
    _directory_tree_hash,
    resolve_committed_resume,
)
from dynamic_rubric.training.verl_dataset import build_online_rar_verl_rows
from .config import GPT_OSS_MODEL, POLICY_MODEL, RUBRIC_JUDGE_MODEL, Phase1Config
from .preflight import topology_preflight
from .provenance import prepare_fixed_train_probe_manifest


class FullRunError(RuntimeError):
    """The full training launch contract was violated."""


FULL_RUN_PREFIX = "phase1-online-rubrics-medicine-full"
FULL_RUN_PREFIXES = {
    "medicine": FULL_RUN_PREFIX,
    "science": "phase1-online-rubrics-science-full",
}
FULL_PROMPT_COUNT = 1500
FULL_GLOBAL_STEPS = 48
CHECKPOINT_STEPS = tuple(range(1, FULL_GLOBAL_STEPS + 1))


def validate_full_run_config(config: Phase1Config) -> None:
    if config.domain not in FULL_RUN_PREFIXES or config.method != "online_rubrics":
        raise FullRunError("full run requires medicine or science + online_rubrics")
    method = config.method_config
    mode = str(config.raw.get("launch", {}).get("tuning_mode", "paper"))
    if mode not in {"paper", "custom"}:
        raise FullRunError("launch.tuning_mode must be paper or custom")
    actual = (
        int(config.data["train_prompt_count"]),
        int(config.training["epochs"]),
        int(config.training["global_prompt_batch"]),
        int(config.training["expected_global_steps"]),
        int(method["policy_responses_per_prompt"]),
        int(method["elicitation_pairs_per_prompt"]),
    )
    expected = (1500, 3, 96, 48, 16, 8)
    if mode == "paper" and config.seed != 11:
        raise FullRunError("paper mode requires seed 11")
    if mode == "paper" and actual != expected:
        raise FullRunError(f"full-run paper contract drifted: expected {expected}, got {actual}")
    if any(value < 1 for value in actual):
        raise FullRunError("full-run counts must be positive")
    if actual[-2:] != expected[-2:]:
        raise FullRunError("the Online reward backend currently requires 16 responses and 8 pairs")
    policy = config.models["policy"]
    if str(policy["model"]) != POLICY_MODEL or policy.get("thinking") is not False:
        raise FullRunError("full run requires non-thinking Qwen3-4B")
    integer_tunable = {
        "ppo_mini_batch_size": (96, 1),
        "max_prompt_length": (4096, 1),
        "max_response_length": (3584, 1),
        "rollout_tensor_parallel_size": (1, 1),
    }
    for name, (default, lower) in integer_tunable.items():
        value = config.training.get(name, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < lower:
            raise FullRunError(f"training.{name} must be an integer >= {lower}")
        if mode == "paper" and name != "rollout_tensor_parallel_size" and value != default:
            raise FullRunError(f"training.{name} requires launch.tuning_mode=custom")
    mini_batch = int(config.training.get("ppo_mini_batch_size", 96))
    prompt_batch = int(config.training["global_prompt_batch"])
    if mini_batch > prompt_batch or prompt_batch % mini_batch:
        raise FullRunError("training.ppo_mini_batch_size must divide global_prompt_batch")
    gpu_count = len(config.raw["infrastructure"]["optimizer"]["gpus"])
    rollout_tp = int(config.training.get("rollout_tensor_parallel_size", 1))
    if rollout_tp > gpu_count or gpu_count % rollout_tp:
        raise FullRunError(
            "training.rollout_tensor_parallel_size must divide the selected GPU count"
        )
    tunable = {
        "learning_rate": (5e-6, 0.0, None),
        "warmup_ratio": (0.1, 0.0, 1.0),
        "kl_coefficient": (0.01, 0.0, None),
        "rollout_temperature": (1.0, 0.0, 2.0),
        "rollout_top_p": (0.95, 0.0, 1.0),
    }
    for name, (default, lower, upper) in tunable.items():
        try:
            value = float(config.training.get(name, default))
        except (TypeError, ValueError) as error:
            raise FullRunError(f"training.{name} must be numeric") from error
        if (
            not math.isfinite(value)
            or value < lower
            or (upper is not None and value > upper)
            or (name in {"learning_rate", "rollout_top_p"} and value == 0.0)
        ):
            raise FullRunError(f"training.{name} is outside its supported range")
        if mode == "paper" and value != default:
            raise FullRunError(f"training.{name} requires launch.tuning_mode=custom")


def _unique_urls(name: str, *, expected_count: int) -> tuple[str, ...]:
    plural, singular = (
        os.environ.get(name, "").strip(),
        os.environ.get(name.removesuffix("S"), "").strip(),
    )
    if plural and singular:
        raise FullRunError(f"{name.removesuffix('S')} and {name} cannot both be set")
    urls = tuple(value.strip().rstrip("/") for value in plural.split(",") if value.strip())
    if len(urls) != expected_count or len(set(urls)) != expected_count:
        raise FullRunError(
            f"{name} must contain exactly {expected_count} unique comma-separated URLs"
        )
    if any(not value.startswith(("http://", "https://")) for value in urls):
        raise FullRunError(f"{name} entries must be HTTP(S) URLs")
    return urls  # type: ignore[return-value]


def _qwen_endpoint_count(config: Phase1Config | None = None) -> int:
    raw = os.environ.get("PHASE1_QWEN32B_EXPECTED_COUNT", "").strip()
    if not raw and config is not None and config.raw.get("launch") is not None:
        configured_urls = os.environ.get("PHASE1_QWEN32B_BASE_URLS", "")
        inferred = len([url for url in configured_urls.split(",") if url.strip()])
        if inferred:
            return inferred
        services = config.raw["infrastructure"]["services"]
        return len(services["qwen3_32b"]["instances"])
    raw = raw or "2"
    try:
        count = int(raw)
    except ValueError as error:
        raise FullRunError("PHASE1_QWEN32B_EXPECTED_COUNT must be 1 or 2") from error
    if count not in (1, 2):
        raise FullRunError("PHASE1_QWEN32B_EXPECTED_COUNT must be 1 or 2")
    return count


def _resolve_repo_path(repo_root: Path, value: object) -> Path:
    path = Path(str(value)).expanduser()
    return (path if path.is_absolute() else repo_root / path).resolve()


def write_full_run_parquets(
    config: Phase1Config, *, repo_root: Path, run_root: Path
) -> tuple[Path, Path, Path]:
    try:
        from datasets import Dataset
    except ImportError as error:
        raise FullRunError("the runtime datasets package is unavailable") from error
    train_source = _resolve_repo_path(repo_root, config.data["train_path"])
    heldout_source = _resolve_repo_path(repo_root, config.data["in_domain_policy_eval"]["path"])
    train_rows, heldout_rows = read_jsonl(train_source), read_jsonl(heldout_source)
    expected_prompts = int(config.data["train_prompt_count"])
    if len(train_rows) != expected_prompts:
        raise FullRunError(
            f"RaR-{config.domain.title()} train must contain {expected_prompts} rows, "
            f"got {len(train_rows)}"
        )
    if not heldout_rows:
        raise FullRunError("validation placeholder source is empty")
    prompt_ids = [str(row.get("prompt_id", "")) for row in train_rows]
    if any(not value for value in prompt_ids) or len(set(prompt_ids)) != expected_prompts:
        raise FullRunError("training prompt IDs must be non-empty and unique")
    if any(not str(row.get("prompt_hash", "")) for row in train_rows):
        raise FullRunError("every training row must carry prompt_hash")
    verl_train, verl_validation = build_online_rar_verl_rows(
        run_root.name, train_rows, [heldout_rows[0]]
    )
    data_root = run_root / "verl-data"
    data_root.mkdir(parents=True, exist_ok=False)
    train_path, validation_path = (
        data_root / "train-online-full.parquet",
        data_root / "validation-unused.parquet",
    )
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
        "source": {
            "path": str(train_source),
            "sha256": sha256_file(train_source),
            "row_count": len(train_rows),
        },
        "ordered_rows": [
            {
                "source_index": index,
                "prompt_id": str(row["prompt_id"]),
                "prompt_hash": str(row["prompt_hash"]),
                "source_row_sha256": sha256_json(row),
            }
            for index, row in enumerate(train_rows)
        ],
        "validation": {
            "enabled": False,
            "row_count": 1,
            "source_path": str(heldout_source),
            "source_prompt_id": str(heldout_rows[0]["prompt_id"]),
        },
        "train_parquet": str(train_path.resolve()),
        "validation_parquet": str(validation_path.resolve()),
        "ordered_prompt_ids_sha256": sha256_json(prompt_ids),
    }
    manifest_path = run_root / "manifests/full_train_selection.json"
    write_json_atomic(manifest_path, manifest)
    return train_path.resolve(), validation_path.resolve(), manifest_path.resolve()


def _validated_control_cache(
    config: Phase1Config, *, repo_root: Path
) -> tuple[Path, ImmutablePi0Cache]:
    raw = os.environ.get("ONLINE_CONTROL_CACHE", "").strip()
    if not raw:
        raise FullRunError("ONLINE_CONTROL_CACHE is required; live pi0 generation is forbidden")
    path = Path(raw).resolve()
    try:
        cache = ImmutablePi0Cache(
            path, expected_prompt_count=int(config.data["train_prompt_count"])
        )
    except (OSError, ValueError, Pi0CacheError) as error:
        raise FullRunError(f"immutable pi0 cache is invalid: {path}") from error
    policy = config.models["policy"]
    expected_hash = _directory_tree_hash(_resolve_repo_path(repo_root, policy["local_snapshot"]))
    if (
        cache.model != str(policy["model"])
        or cache.revision != str(policy["revision"])
        or cache.tokenizer_revision != str(policy["revision"])
        or cache.checkpoint_hash != expected_hash
    ):
        raise FullRunError("immutable pi0 cache does not bind the pinned initial policy")
    return path, cache


def build_full_run_environment(
    config: Phase1Config,
    *,
    repo_root: Path,
    run_root: Path,
    train_path: Path,
    validation_path: Path,
    resume_checkpoint: Path | None,
) -> dict[str, str]:
    validate_full_run_config(config)
    gpt_urls, qwen_urls = (
        _unique_urls(
            "PHASE1_GPT_OSS_BASE_URLS",
            expected_count=len(
                config.raw["infrastructure"]["services"]["gpt_oss_120b"]["instances"]
            ),
        ),
        _unique_urls(
            "PHASE1_QWEN32B_BASE_URLS",
            expected_count=_qwen_endpoint_count(config),
        ),
    )
    cache_path, cache = _validated_control_cache(config, repo_root=repo_root)
    policy = config.models["policy"]
    model_path = _resolve_repo_path(repo_root, policy["local_snapshot"])
    if not model_path.is_dir():
        raise FullRunError(f"pinned Qwen3-4B snapshot is absent: {model_path}")
    environment = dict(os.environ)
    for variable in ("RUNTIME_PYTHON", "VERL_ROOT"):
        value = environment.get(variable, "").strip()
        if value:
            environment[variable] = str(_resolve_repo_path(repo_root, value))
    try:
        extractor_concurrency = int(environment.get("ONLINE_EXTRACTOR_CONCURRENCY", "32"))
        grader_concurrency = int(environment.get("ONLINE_GRADER_CONCURRENCY", "64"))
    except ValueError as exc:
        raise FullRunError("online inference concurrency must be an integer") from exc
    if extractor_concurrency < 1 or grader_concurrency < 1:
        raise FullRunError("online inference concurrency must be positive")
    for stale in (
        "OPENAI_API_KEY",
        "PHASE1_GPT_OSS_BASE_URL",
        "PHASE1_QWEN32B_BASE_URL",
        "ONLINE_CONTROL_URL",
        "ONLINE_CONTROL_LAUNCH_SPEC",
    ):
        environment.pop(stale, None)
    environment.update(
        {
            "PROJECT_ROOT": str(repo_root.resolve()),
            "CONFIG_PATH": str(config.source_path.resolve()),
            "TRAIN_FILE": str(train_path),
            "VAL_FILE": str(validation_path),
            "MODEL_PATH": str(model_path),
            "RUN_DIR": str((run_root / "verl-run").resolve()),
            "ONLINE_STEP_ARTIFACT_ROOT": str((run_root / "verl-run/online_steps").resolve()),
            "ONLINE_RUN_ID": run_root.name,
            "ONLINE_CONFIG_HASH": config.config_hash,
            "ONLINE_EXPERIMENT_ARM": "phase1_online_rubrics_full_dynamic",
            "ONLINE_BASELINE_ARM": "static_r0_grpo",
            "ONLINE_CONTROL_POLICY": "pi_ref",
            "ONLINE_CONTROL_CACHE": str(cache_path),
            "ONLINE_CONTROL_CHECKPOINT_HASH": cache.checkpoint_hash,
            "ONLINE_REPRODUCTION_CLAIM": "phase1_full_dynamic_evaluator_training",
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
            "ONLINE_GRADER_MAX_OUTPUT_TOKENS": str(config.models["judge"]["max_output_tokens"]),
            "ONLINE_ACTOR_MODEL": POLICY_MODEL,
            "ONLINE_ACTOR_REVISION": str(policy["revision"]),
            "ONLINE_CONTROL_MODEL": POLICY_MODEL,
            "ONLINE_CONTROL_REVISION": str(policy["revision"]),
            "ONLINE_CONTROL_TOKENIZER_REVISION": str(policy["revision"]),
            "ONLINE_SEED": str(config.seed),
            "PHASE1_GPT_OSS_BASE_URLS": ",".join(gpt_urls),
            "PHASE1_QWEN32B_BASE_URLS": ",".join(qwen_urls),
            "ONLINE_EXTRACTOR_CONCURRENCY": str(extractor_concurrency),
            "ONLINE_GRADER_CONCURRENCY": str(grader_concurrency),
            "PHASE1_VLLM_TIMEOUT_SECONDS": "600",
            "PHASE1_VLLM_MAX_RETRIES": "4",
            "TOTAL_EPOCHS": str(config.training["epochs"]),
            "EXPECTED_UPDATES": str(config.training["expected_global_steps"]),
            "TRAIN_BATCH_SIZE": str(config.training["global_prompt_batch"]),
            "ROLLOUT_N": str(config.method_config["policy_responses_per_prompt"]),
            "ELICITATION_PAIRS": str(config.method_config["elicitation_pairs_per_prompt"]),
            "PPO_MINI_BATCH_SIZE": str(
                config.training.get("ppo_mini_batch_size", config.training["global_prompt_batch"])
            ),
            "MAX_PROMPT_LENGTH": str(config.training.get("max_prompt_length", 4096)),
            "MAX_RESPONSE_LENGTH": str(config.training.get("max_response_length", 3584)),
            "ROLLOUT_TENSOR_PARALLEL_SIZE": str(
                config.training.get("rollout_tensor_parallel_size", 1)
            ),
            "N_GPUS_PER_NODE": str(len(config.raw["infrastructure"]["optimizer"]["gpus"])),
            "CUDA_VISIBLE_DEVICES": ",".join(
                str(gpu) for gpu in config.raw["infrastructure"]["optimizer"]["gpus"]
            ),
            "ATTN_IMPLEMENTATION": "sdpa",
            "TRAINING_SEED": str(config.seed),
            "PHASE1_TUNING_MODE": str(config.raw.get("launch", {}).get("tuning_mode", "paper")),
            "LEARNING_RATE": str(config.training.get("learning_rate", 5e-6)),
            "WARMUP_RATIO": str(config.training.get("warmup_ratio", 0.1)),
            "KL_COEFFICIENT": str(config.training.get("kl_coefficient", 0.01)),
            "ROLLOUT_TEMPERATURE": str(config.training.get("rollout_temperature", 1.0)),
            "ROLLOUT_TOP_P": str(config.training.get("rollout_top_p", 0.95)),
            "CHECKPOINT_INTERVAL_STEPS": str(config.training["checkpoint_interval_steps"]),
            "CHECKPOINT_STEPS": "[" + ",".join(str(step) for step in config.checkpoint_steps) + "]",
            "ONLINE_STEP_HOOK_PATH": "pkg://dynamic_rubric.training.online_step",
            "ONLINE_STEP_HOOK_NAME": "prepare_rewards",
            "ONLINE_STEP_COMMIT_NAME": "commit_step",
            "ONLINE_STEP_RUNTIME_PATH": "pkg://dynamic_rubric.training.verl_online_runtime",
            "ONLINE_STEP_RUNTIME_NAME": "create_online_reward_runtime",
            "TRACKING_PROJECT_NAME": "phase1_dynamic_evaluator_updates",
            "TRACKING_EXPERIMENT_NAME": run_root.name,
            "RESUME_MODE": "resume_path" if resume_checkpoint else "disable",
            "RESUME_FROM_PATH": str(resume_checkpoint) if resume_checkpoint else "null",
        }
    )
    return environment


def latest_full_checkpoint(run_root: Path, *, expected_steps: int = 48) -> Path:
    checkpoint_root = run_root / "verl-run/checkpoints"
    tracker = checkpoint_root / "latest_checkpointed_iteration.txt"
    if not tracker.is_file():
        raise FullRunError("resume requested but the checkpoint tracker is missing")
    try:
        step = int(tracker.read_text(encoding="utf-8").strip())
    except ValueError as error:
        raise FullRunError("checkpoint tracker is malformed") from error
    checkpoint = checkpoint_root / f"global_step_{step}"
    if step >= expected_steps:
        raise FullRunError(f"run already reached the configured {expected_steps} updates")
    if not checkpoint.is_dir() or not (checkpoint / "data.pt").is_file():
        raise FullRunError("latest checkpoint is not fully resumable")
    if any(
        not tuple(checkpoint.glob(pattern))
        for pattern in (
            "actor/optim_world_size_*_rank_*.pt",
            "actor/extra_state_world_size_*_rank_*.pt",
        )
    ):
        raise FullRunError("latest checkpoint is missing actor optimizer or scheduler/RNG state")
    later = [
        path
        for path in checkpoint_root.glob("global_step_*")
        if path.is_dir()
        and path.name.removeprefix("global_step_").isdigit()
        and int(path.name.removeprefix("global_step_")) > step
    ]
    if later:
        raise FullRunError("checkpoint directories exist beyond the sealed latest checkpoint")
    return checkpoint.resolve()


def prepare_full_resume(run_root: Path, *, expected_steps: int = 48) -> Path:
    """Validate the sealed checkpoint and archive any unrecoverable logical tail."""

    verified_checkpoint = (
        latest_full_checkpoint(run_root)
        if expected_steps == 48
        else latest_full_checkpoint(run_root, expected_steps=expected_steps)
    )
    try:
        resolved_checkpoint, _ = resolve_committed_resume(run_root / "verl-run")
    except LiveOnlineTrainingError as error:
        raise FullRunError(f"committed resume state is invalid: {error}") from error
    if resolved_checkpoint.resolve() != verified_checkpoint:
        raise FullRunError("checkpoint tracker disagrees with the committed online-step chain")
    return resolved_checkpoint.resolve()


def _write_launch_spec(
    config: Phase1Config,
    *,
    run_root: Path,
    environment: Mapping[str, str],
    train_manifest: Path,
    probe_manifest: Path,
    resume_checkpoint: Path | None,
) -> Path:
    spec = {
        "schema_version": 1,
        "run_id": run_root.name,
        "experiment": config.experiment,
        "domain": config.domain,
        "method": config.method,
        "canary": False,
        "full_training": True,
        "resume": resume_checkpoint is not None,
        "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
        "trainer_total_training_steps": int(config.training["expected_global_steps"]),
        "epochs": int(config.training["epochs"]),
        "global_prompt_batch": int(config.training["global_prompt_batch"]),
        "train_drop_last": False,
        "steps_per_epoch": math.ceil(
            int(config.data["train_prompt_count"]) / int(config.training["global_prompt_batch"])
        ),
        "final_batch_prompt_count": int(config.data["train_prompt_count"])
        % int(config.training["global_prompt_batch"])
        or int(config.training["global_prompt_batch"]),
        "cumulative_prompt_exposures": int(config.data["train_prompt_count"])
        * int(config.training["epochs"]),
        "rollouts_per_prompt": int(config.method_config["policy_responses_per_prompt"]),
        "elicitation_pairs_per_prompt": int(config.method_config["elicitation_pairs_per_prompt"]),
        "checkpoint_steps": list(config.checkpoint_steps),
        "optimizer_host": socket.gethostname().split(".", 1)[0],
        "optimizer_visible_gpus": environment["CUDA_VISIBLE_DEVICES"],
        "attention_implementation": environment["ATTN_IMPLEMENTATION"],
        "primary_seed": int(environment["TRAINING_SEED"]),
        "models": {
            "policy": POLICY_MODEL,
            "extractor_deduplicator": GPT_OSS_MODEL,
            "judge": RUBRIC_JUDGE_MODEL,
        },
        "endpoints": {
            "extractor": environment["PHASE1_GPT_OSS_BASE_URLS"].split(","),
            "judge": environment["PHASE1_QWEN32B_BASE_URLS"].split(","),
        },
        "control_cache": {
            "path": environment["ONLINE_CONTROL_CACHE"],
            "manifest_sha256": sha256_file(environment["ONLINE_CONTROL_CACHE"]),
            "checkpoint_hash": environment["ONLINE_CONTROL_CHECKPOINT_HASH"],
        },
        "train_selection_manifest": str(train_manifest),
        "train_selection_manifest_sha256": sha256_file(train_manifest),
        "fixed_probe_manifest": str(probe_manifest),
        "fixed_probe_manifest_sha256": sha256_file(probe_manifest),
        "validation_enabled": False,
        "tracking_project": environment["TRACKING_PROJECT_NAME"],
        "tracking_experiment": environment["TRACKING_EXPERIMENT_NAME"],
        "failure_policy": "fail_closed",
        "execution_optimizations": {
            "online_logprob_prefetch": environment.get("ONLINE_LOGPROB_PREFETCH", "false")
            == "true",
            "prefetch_validation": "first_resumed_update_exact_match_serial_outputs",
        },
    }
    if resume_checkpoint is None:
        destination = run_root / "launch_spec.json"
    else:
        resume_root, step, attempt = (
            run_root / "resume_launches",
            resume_checkpoint.name.removeprefix("global_step_"),
            1,
        )
        while (resume_root / f"resume-from-step-{step}-{attempt}.json").exists():
            attempt += 1
        destination = resume_root / f"resume-from-step-{step}-{attempt}.json"
    write_json_atomic(destination, spec)
    return destination.resolve()


def run_online_full(
    config: Phase1Config, *, repo_root: str | Path, run_id: str, resume: bool = False
) -> dict[str, Any]:
    validate_full_run_config(config)
    run_prefix = FULL_RUN_PREFIXES[config.domain]
    if not run_id.startswith(run_prefix):
        raise FullRunError(f"run_id must start with {run_prefix!r}")
    root = Path(repo_root).resolve()
    run_root = config.run_root(root, run_id).resolve()
    topology_preflight(config, repo_root=root, require_endpoints=True)
    if resume:
        if not run_root.is_dir():
            raise FullRunError(f"resume run directory does not exist: {run_root}")
        if read_json(run_root / "config.resolved.json") != dict(config.raw):
            raise FullRunError("resolved config differs from the requested resume config")
        train_path, validation_path = (
            (run_root / "verl-data/train-online-full.parquet").resolve(),
            (run_root / "verl-data/validation-unused.parquet").resolve(),
        )
        train_manifest = (run_root / "manifests/full_train_selection.json").resolve()
        if not all(path.is_file() for path in (train_path, validation_path, train_manifest)):
            raise FullRunError("resume artifacts are incomplete")
        resume_checkpoint = prepare_full_resume(
            run_root, expected_steps=int(config.training["expected_global_steps"])
        )
        probe_manifest = Path(str(read_json(run_root / "launch_spec.json")["fixed_probe_manifest"]))
        if not probe_manifest.is_file():
            raise FullRunError("fixed probe manifest referenced by the run is missing")
        environment = build_full_run_environment(
            config,
            repo_root=root,
            run_root=run_root,
            train_path=train_path,
            validation_path=validation_path,
            resume_checkpoint=resume_checkpoint,
        )
    else:
        if run_root.exists():
            raise FullRunError(f"refusing to reuse full-run output directory: {run_root}")
        train_path = (run_root / "verl-data/train-online-full.parquet").resolve()
        validation_path = (run_root / "verl-data/validation-unused.parquet").resolve()
        resume_checkpoint = None
        # Validate endpoints, model identity, and the immutable pi0 cache before
        # creating a run directory that a retry would then be forbidden to reuse.
        environment = build_full_run_environment(
            config,
            repo_root=root,
            run_root=run_root,
            train_path=train_path,
            validation_path=validation_path,
            resume_checkpoint=None,
        )
        run_root.mkdir(parents=True)
        write_json_atomic(run_root / "config.resolved.json", dict(config.raw))
        train_path, validation_path, train_manifest = write_full_run_parquets(
            config, repo_root=root, run_root=run_root
        )
        probe_manifest, _ = prepare_fixed_train_probe_manifest(config, repo_root=root)
    launch_spec = _write_launch_spec(
        config,
        run_root=run_root,
        environment=environment,
        train_manifest=train_manifest,
        probe_manifest=probe_manifest,
        resume_checkpoint=resume_checkpoint,
    )
    subprocess.run(
        [str(root / "scripts/phase1/run_online_full.sh")], cwd=root, env=environment, check=True
    )
    result = {
        "schema_version": 1,
        "status": "completed",
        "run_id": run_id,
        "full_training": True,
        "resumed": resume,
        "launch_spec": str(launch_spec),
        "train_selection_manifest": str(train_manifest),
        "fixed_probe_manifest": str(probe_manifest),
    }
    write_json_atomic(run_root / "full_run_complete.json", result)
    return result
