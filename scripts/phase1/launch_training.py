"""One-YAML launch entry point for the RaR Medicine and Science training methods.

This only prepares and validates a launch; the existing training implementations
remain authoritative.  ``--check`` never starts a training process or service.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
from datetime import datetime, timezone
from urllib.request import Request, urlopen
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
METHODS = {"static_r0_matched", "online_rubrics", "evorubrics"}
ENV_NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")


class LaunchError(ValueError):
    """A launcher config or local prerequisite is invalid."""


@dataclass(frozen=True)
class Launch:
    method: str
    config: Path
    run_root: Path
    commands: tuple[tuple[str, ...], ...]
    environment: dict[str, str]


def _file(path: str, *, label: str) -> Path:
    result = Path(path)
    if not result.is_file():
        raise LaunchError(f"{label} is missing: {result}")
    return result


def _directory(path: str, *, label: str) -> Path:
    result = Path(path)
    if not result.is_dir():
        raise LaunchError(f"{label} is missing: {result}")
    return result


def _url(value: str, *, label: str) -> None:
    urls = [part.strip() for part in value.split(",")]
    if not urls or any(
        urlparse(part).scheme not in {"http", "https"} or not urlparse(part).netloc for part in urls
    ):
        raise LaunchError(f"{label} must contain HTTP(S) URL(s)")


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise LaunchError(f"{label} must be a mapping")
    return value


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise LaunchError(f"{label} must be a positive integer")
    return value


def _repo_path(value: object, *, label: str = "path") -> Path:
    if value is None or not str(value).strip():
        raise LaunchError(f"{label} must be configured with a non-empty path")
    path = Path(str(value))
    return path if path.is_absolute() else ROOT / path


def _run_id(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or "/" in value
        or chr(92) in value
    ):
        raise LaunchError(f"{label} must be a single non-empty path segment")
    return value


def _gpu_ids(raw: dict[str, Any]) -> list[int]:
    infra = _mapping(raw.get("infrastructure"), "infrastructure")
    optimizer = _mapping(infra.get("optimizer"), "infrastructure.optimizer")
    gpus = optimizer.get("gpus")
    if not isinstance(gpus, list) or not gpus:
        raise LaunchError("infrastructure.optimizer.gpus must be a non-empty GPU list")
    if any(isinstance(gpu, bool) or not isinstance(gpu, int) or gpu < 0 for gpu in gpus):
        raise LaunchError("optimizer GPUs must be nonnegative integers")
    if len(set(gpus)) != len(gpus):
        raise LaunchError("optimizer GPUs must be unique")
    return gpus


def _launch_section(config: Path) -> tuple[dict[str, Any], str, dict[str, Any], dict[str, str]]:
    from dynamic_rubric.phase1.config import load_yaml_config

    raw = _mapping(load_yaml_config(config), "YAML root")
    method = raw.get("method")
    if method not in METHODS:
        raise LaunchError(f"method must be one of {sorted(METHODS)}")
    launch = _mapping(raw.get("launch"), "launch")
    _run_id(launch.get("run_id"), "launch.run_id")
    if launch.get("tuning_mode") not in {"paper", "custom"}:
        raise LaunchError("launch.tuning_mode must be paper or custom")
    if method == "static_r0_matched" and launch["tuning_mode"] != "custom":
        raise LaunchError("Static recipe requires launch.tuning_mode=custom")
    values = _mapping(launch.get("environment"), "launch.environment")
    reserved = {"CUDA_VISIBLE_DEVICES", "PROJECT_ROOT", "POLICY_GPU", "N_GPUS_PER_NODE"}
    if reserved.intersection(values):
        raise LaunchError(
            "GPU and project settings are derived; edit infrastructure instead of launch.environment"
        )
    environment: dict[str, str] = {}
    for name, value in values.items():
        if not isinstance(name, str) or not ENV_NAME.fullmatch(name):
            raise LaunchError(f"invalid environment variable name: {name!r}")
        if value is None or isinstance(value, (dict, list)):
            raise LaunchError(f"{name} must have a scalar value")
        environment[name] = str(value)
    return raw, method, launch, environment


def _static_environment(
    raw: dict[str, Any], settings: dict[str, Any], overrides: dict[str, str]
) -> tuple[dict[str, str], Path]:
    data = _mapping(raw.get("data"), "data")
    models = _mapping(raw.get("models"), "models")
    policy = _mapping(models.get("policy"), "models.policy")
    judge = _mapping(models.get("judge"), "models.judge")
    training = _mapping(raw.get("training"), "training")
    output = _mapping(raw.get("output"), "output")
    tracking = _mapping(raw.get("tracking"), "tracking")
    gpus = _gpu_ids(raw)
    rollout_tp = _positive_int(
        training.get("rollout_tensor_parallel_size", 1), "training.rollout_tensor_parallel_size"
    )
    if len(gpus) % rollout_tp:
        raise LaunchError("rollout_tensor_parallel_size must divide optimizer GPU count")
    run_id = _run_id(settings["run_id"], "launch.run_id")
    if raw.get("method") != "static_r0_matched" or raw.get("domain") not in {"medicine", "science"}:
        raise LaunchError("static launcher requires Medicine or Science static_r0_matched")
    if training.get("algorithm") != "grpo":
        raise LaunchError("static training.algorithm must be grpo")
    retention = _mapping(training.get("checkpoint_retention"), "training.checkpoint_retention")
    if retention != {
        "latest_full_resume_state": True,
        "older_parameter_snapshots": True,
        "older_optimizer_state": False,
    }:
        raise LaunchError(
            "static checkpoint retention must keep older models and only latest resume state"
        )
    steps = _positive_int(training.get("expected_global_steps"), "training.expected_global_steps")
    interval = _positive_int(
        training.get("checkpoint_interval_steps"), "training.checkpoint_interval_steps"
    )
    if interval != 1:
        raise LaunchError("dense static training requires checkpoint_interval_steps=1")
    epochs = _positive_int(training.get("epochs"), "training.epochs")
    batch = _positive_int(training.get("global_prompt_batch"), "training.global_prompt_batch")
    rollouts = _positive_int(training.get("rollouts_per_prompt"), "training.rollouts_per_prompt")
    mini_batch = _positive_int(training.get("ppo_mini_batch_size"), "training.ppo_mini_batch_size")
    if batch % mini_batch:
        raise LaunchError("training.ppo_mini_batch_size must divide global_prompt_batch")
    if (mini_batch * rollouts) % len(gpus):
        raise LaunchError("mini_batch_size * rollouts_per_prompt must divide evenly across GPUs")
    prompt_length = _positive_int(
        training.get("max_prompt_length", 4096), "training.max_prompt_length"
    )
    response_length = _positive_int(
        training.get("max_response_length"), "training.max_response_length"
    )
    seed = raw.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise LaunchError("seed must be a nonnegative integer")
    for name in (
        "learning_rate",
        "warmup_ratio",
        "kl_coefficient",
        "rollout_temperature",
        "rollout_top_p",
    ):
        try:
            value = float(training[name])
        except (KeyError, TypeError, ValueError) as exc:
            raise LaunchError(f"training.{name} must be numeric") from exc
        minimum = 0.0
        maximum = (
            2.0
            if name == "rollout_temperature"
            else (1.0 if name in {"warmup_ratio", "rollout_top_p"} else None)
        )
        if (
            not math.isfinite(value)
            or value < minimum
            or (maximum is not None and value > maximum)
            or (name in {"learning_rate", "rollout_top_p"} and value == 0.0)
        ):
            raise LaunchError(f"training.{name} is outside its supported range")
    root_name = str(output.get("root", ""))
    layout = str(output.get("layout", ""))
    try:
        relative_run = Path(root_name) / layout.format(
            domain=raw["domain"], method=raw["method"], seed=seed, run_id=run_id
        )
    except (KeyError, ValueError) as exc:
        raise LaunchError("output.layout must use domain, method, seed, and run_id") from exc
    rendered_layout = Path(
        layout.format(domain=raw["domain"], method=raw["method"], seed=seed, run_id=run_id)
    )
    if (
        not root_name
        or rendered_layout.is_absolute()
        or ".." in rendered_layout.parts
        or run_id not in rendered_layout.parts
    ):
        raise LaunchError(
            "output.layout must be relative, contain run_id, and not traverse parents"
        )
    run_root = (ROOT / relative_run).resolve()
    model_path = _directory(str(_repo_path(policy.get("local_snapshot"), label="models.policy.local_snapshot")), label="policy model")
    train_path = _file(str(_repo_path(data.get("train_path"), label="data.train_path")), label="static train parquet")
    val_path = _file(
        str(_repo_path(data.get("validation_path"), label="data.validation_path")), label="static validation parquet"
    )
    rubric_path = _file(str(_repo_path(data.get("static_rubric_path"), label="data.static_rubric_path")), label="static rubrics")
    if not overrides.get("DYNAMIC_RUBRIC_VLLM_URL"):
        raise LaunchError("launch.environment needs DYNAMIC_RUBRIC_VLLM_URL")
    _url(overrides["DYNAMIC_RUBRIC_VLLM_URL"], label="static judge URL")
    reserved = {
        "MODEL_PATH",
        "TRAIN_FILE",
        "VAL_FILE",
        "DYNAMIC_RUBRIC_STATIC_RUBRIC_PATH",
        "RUN_DIR",
        "ROLLOUT_CACHE_DIR",
        "POLICY_GPU",
        "N_GPUS_PER_NODE",
        "ROLLOUT_TENSOR_PARALLEL_SIZE",
        "TOTAL_STEPS",
        "TOTAL_EPOCHS",
        "TRAIN_BATCH_SIZE",
        "ROLLOUT_N",
        "PPO_MINI_BATCH_SIZE",
        "MAX_RESPONSE_LENGTH",
        "MAX_PROMPT_LENGTH",
        "SAVE_FREQ",
        "CHECKPOINT_STEPS",
        "LEARNING_RATE",
        "WARMUP_RATIO",
        "KL_COEFFICIENT",
        "ROLLOUT_TEMPERATURE",
        "ROLLOUT_TOP_P",
        "TRAINING_SEED",
        "PYTHONHASHSEED",
        "TRACKING_PROJECT_NAME",
        "TRACKING_EXPERIMENT_NAME",
        "DYNAMIC_RUBRIC_GRADER_MODEL",
        "DYNAMIC_RUBRIC_GRADER_REVISION",
        "DYNAMIC_RUBRIC_GRADER_TOKENIZER_REVISION",
    }
    if duplicate := sorted(reserved.intersection(overrides)):
        raise LaunchError(f"move derived values out of launch.environment: {duplicate}")
    for name in ("model", "revision", "tokenizer_revision"):
        if not judge.get(name):
            raise LaunchError(f"models.judge.{name} is required")
    if not tracking.get("project"):
        raise LaunchError("tracking.project is required")
    derived = {
        "MODEL_PATH": str(model_path),
        "TRAIN_FILE": str(train_path),
        "VAL_FILE": str(val_path),
        "DYNAMIC_RUBRIC_STATIC_RUBRIC_PATH": str(rubric_path),
        "RUN_DIR": str(run_root / "verl-run"),
        "ROLLOUT_CACHE_DIR": str(run_root / "provider_cache/policy-rollouts"),
        "POLICY_GPU": ",".join(map(str, gpus)),
        "N_GPUS_PER_NODE": str(len(gpus)),
        "ROLLOUT_TENSOR_PARALLEL_SIZE": str(rollout_tp),
        "TOTAL_STEPS": str(steps),
        "TOTAL_EPOCHS": str(epochs),
        "TRAIN_BATCH_SIZE": str(batch),
        "ROLLOUT_N": str(rollouts),
        "PPO_MINI_BATCH_SIZE": str(mini_batch),
        "MAX_RESPONSE_LENGTH": str(response_length),
        "MAX_PROMPT_LENGTH": str(prompt_length),
        "SAVE_FREQ": str(interval),
        "CHECKPOINT_STEPS": "[" + ",".join(str(step) for step in range(steps + 1)) + "]",
        "LEARNING_RATE": str(training["learning_rate"]),
        "WARMUP_RATIO": str(training["warmup_ratio"]),
        "KL_COEFFICIENT": str(training["kl_coefficient"]),
        "ROLLOUT_TEMPERATURE": str(training["rollout_temperature"]),
        "ROLLOUT_TOP_P": str(training["rollout_top_p"]),
        "TRAINING_SEED": str(seed),
        "PYTHONHASHSEED": str(seed),
        "TRACKING_PROJECT_NAME": str(tracking["project"]),
        "TRACKING_EXPERIMENT_NAME": run_id,
        "DYNAMIC_RUBRIC_GRADER_MODEL": str(judge["model"]),
        "DYNAMIC_RUBRIC_GRADER_REVISION": str(judge["revision"]),
        "DYNAMIC_RUBRIC_GRADER_TOKENIZER_REVISION": str(judge["tokenizer_revision"]),
    }
    return derived, run_root


def prepare_launch(config: Path, *, resume: bool = False) -> Launch:
    config = config.resolve()
    _file(str(config), label="launch YAML")
    try:
        relative_config = config.relative_to(ROOT)
    except ValueError as exc:
        raise LaunchError(f"launch YAML must be inside {ROOT}") from exc
    raw, method, settings, overrides = _launch_section(config)
    entry_python = _file(str(_repo_path(settings.get("entry_python"), label="launch.entry_python")), label="entry Python")
    if not os.access(entry_python, os.X_OK):
        raise LaunchError(f"entry Python is not executable: {entry_python}")
    environment = dict(os.environ)
    environment.update(overrides)
    for name in ("RUNTIME_PYTHON", "VERL_ROOT", "HF_HOME"):
        if environment.get(name):
            environment[name] = str(_repo_path(environment[name], label=f"launch.environment.{name}"))
    environment["PROJECT_ROOT"] = str(ROOT)
    environment["PYTHONPATH"] = str(ROOT / "src") + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
    )

    if method == "static_r0_matched":
        derived, run_root = _static_environment(raw, settings, overrides)
        environment.update(derived)
        run_dir = Path(derived["RUN_DIR"])
        if run_root.exists() and not resume:
            raise LaunchError(f"RUN_DIR exists; use --resume or choose a new run_id: {run_dir}")
        if resume and not (run_dir / "checkpoints/latest_checkpointed_iteration.txt").is_file():
            raise LaunchError("static resume requires a checkpoint tracker")
        environment["RUNTIME_PYTHON"] = str(entry_python)
        environment.pop("DYNAMIC_RUBRIC_VLLM_URLS", None)
        environment["RESUME_MODE"] = "auto" if resume else "disable"
        environment["RESUME_FROM_PATH"] = "null"
        commands = (("bash", str(ROOT / "scripts/run_static_grpo.sh")),)

    else:
        from dynamic_rubric.phase1.config import load_phase1_config

        phase1 = load_phase1_config(config)
        if phase1.method != method or phase1.domain not in {"medicine", "science"}:
            raise LaunchError(
                "launcher must target the matching Medicine or Science Phase-1 method"
            )
        gpus = _gpu_ids(raw)
        _directory(str(_repo_path(phase1.models["policy"]["local_snapshot"], label="models.policy.local_snapshot")), label="policy model")
        _file(str(_repo_path(phase1.data["train_path"], label="data.train_path")), label="training data")
        _file(str(_repo_path(phase1.data["in_domain_policy_eval"]["path"], label="data.in_domain_policy_eval.path")), label="heldout data")
        run_id = _run_id(settings["run_id"], "launch.run_id")
        if method == "online_rubrics":
            from dynamic_rubric.phase1.full_run import FullRunError, validate_full_run_config

            try:
                validate_full_run_config(phase1)
            except FullRunError as exc:
                raise LaunchError(str(exc)) from exc
            if not run_id.startswith(f"phase1-online-rubrics-{phase1.domain}-full"):
                raise LaunchError(
                    "online launch.run_id needs the domain-specific Phase-1 full-run prefix"
                )
            run_root = phase1.run_root(ROOT, run_id)
            if run_root.exists() and not resume:
                raise LaunchError(f"online run exists; use --resume or change run_id: {run_root}")
            if resume and not run_root.is_dir():
                raise LaunchError(f"online resume directory is missing: {run_root}")
            for name in ("PHASE1_GPT_OSS_BASE_URLS", "PHASE1_QWEN32B_BASE_URLS"):
                if not overrides.get(name):
                    raise LaunchError(f"online YAML must set {name}")
                _url(overrides[name], label=name)
            cache_path = overrides.get("ONLINE_CONTROL_CACHE")
            cache_dir = overrides.get("ONLINE_CONTROL_CACHE_DIR")
            if bool(cache_path) == bool(cache_dir):
                raise LaunchError(
                    "online YAML needs exactly one of ONLINE_CONTROL_CACHE or ONLINE_CONTROL_CACHE_DIR"
                )
            if cache_dir:
                directory = _repo_path(cache_dir)
                manifests = sorted(directory.glob("manifest-*.json")) if directory.is_dir() else []
                if len(manifests) != 1:
                    raise LaunchError(
                        f"immutable pi0 cache needs exactly one manifest-*.json in {directory}; "
                        "for Science, run scripts/phase1/precompute_science_pi0.sh first"
                    )
                environment["ONLINE_CONTROL_CACHE"] = str(manifests[0])
                environment.pop("ONLINE_CONTROL_CACHE_DIR", None)
            else:
                environment["ONLINE_CONTROL_CACHE"] = str(
                    _file(str(_repo_path(cache_path)), label="immutable pi0 cache")
                )
            _file(environment.get("RUNTIME_PYTHON", ""), label="veRL runtime Python")
            environment["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
            for stale in (
                "PHASE1_GPT_OSS_BASE_URL",
                "PHASE1_QWEN32B_BASE_URL",
                "ONLINE_CONTROL_URL",
                "ONLINE_CONTROL_LAUNCH_SPEC",
            ):
                environment.pop(stale, None)
            command = "resume-online" if resume else "train-online"
            commands = (
                (
                    str(entry_python),
                    "-m",
                    "dynamic_rubric.phase1",
                    command,
                    "--config",
                    str(relative_config),
                    "--repo-root",
                    str(ROOT),
                    "--run-id",
                    run_id,
                ),
            )
        else:
            if len(gpus) != 1:
                raise LaunchError(
                    "Evo shared-backbone trainer supports one GPU; judge GPUs are independent"
                )
            if resume:
                raise LaunchError("Evo resume is managed by the existing Evo runner; omit --resume")
            smoke_id = _run_id(settings.get("smoke_run_id"), "launch.smoke_run_id")
            if not overrides.get("EVORUBRICS_JUDGE_URL"):
                raise LaunchError("Evo YAML must set EVORUBRICS_JUDGE_URL")
            _url(overrides["EVORUBRICS_JUDGE_URL"], label="Evo judge URL")
            for reserved in (
                "EVORUBRICS_DOMAIN",
                "EVORUBRICS_TRAINER_GPU",
                "EVORUBRICS_SMOKE_RUN_ID",
                "EVORUBRICS_FULL_RUN_ID",
            ):
                if reserved in overrides:
                    raise LaunchError(f"move {reserved} to launch or infrastructure")
            _file(str(ROOT / "docs/EvoRubrics-2155.zip"), label="Evo source ZIP")
            for relative in (
                "evorubric-main/config/shared_base_config.yaml",
                "evorubric-main/shared_base_trainer.py",
                "third_party/verl/verl/__init__.py",
            ):
                _file(
                    str(ROOT / "environment/upstream/EvoRubrics" / relative),
                    label="patched Evo source (run setup_evorubrics_runtime.sh --prepare-source)",
                )
            environment["EVORUBRICS_DOMAIN"] = phase1.domain
            environment["EVORUBRICS_PYTHON"] = str(entry_python)
            environment["EVORUBRICS_CONFIG"] = str(relative_config)
            environment["EVORUBRICS_TRAINER_GPU"] = str(gpus[0])
            environment["EVORUBRICS_SMOKE_RUN_ID"] = smoke_id
            environment["EVORUBRICS_FULL_RUN_ID"] = run_id
            phase1.run_root(ROOT, smoke_id)
            run_root = phase1.run_root(ROOT, run_id)
            environment["EVORUBRICS_RUN_BASE"] = str(run_root.parent)
            if run_root.exists() and (run_root / "training_complete.json").is_file():
                raise LaunchError(f"Evo run is already complete: {run_root}")
            runner = str(ROOT / f"scripts/phase1/run_{phase1.domain}_evorubrics.sh")
            commands = (("bash", runner, "smoke"), ("bash", runner, "full"))

    return Launch(method, config, run_root, commands, environment)


def check_services(launch: Launch) -> None:
    """Read-only live check; no service startup, GPU allocation, or inference."""
    raw, _, _, _ = _launch_section(launch.config)
    if launch.method == "static_r0_matched":
        endpoints = [
            (launch.environment["DYNAMIC_RUBRIC_VLLM_URL"], raw["models"]["judge"]["model"])
        ]
    elif launch.method == "online_rubrics":
        endpoints = [
            (url.strip(), raw["models"][role]["model"])
            for env, role in (
                ("PHASE1_GPT_OSS_BASE_URLS", "extractor"),
                ("PHASE1_QWEN32B_BASE_URLS", "judge"),
            )
            for url in launch.environment[env].split(",")
        ]
    else:
        endpoints = [(launch.environment["EVORUBRICS_JUDGE_URL"], raw["models"]["judge"]["model"])]
    for base, expected in endpoints:
        base = base.rstrip("/").removesuffix("/v1")
        request = Request(base + "/v1/models", headers={"Authorization": "Bearer EMPTY"})
        with urlopen(request, timeout=10) as response:
            models = json.load(response)
        if expected not in [item.get("id") for item in models.get("data", [])]:
            raise LaunchError(f"{base} does not serve {expected}")
        if launch.method == "static_r0_matched":
            with urlopen(base + "/dynamic-rubric/identity", timeout=10) as response:
                identity = json.load(response)
            judge = raw["models"]["judge"]
            for key, value in (
                ("served_model", judge["model"]),
                ("model_revision", judge["revision"]),
                ("tokenizer_revision", judge["tokenizer_revision"]),
            ):
                if identity.get(key) != value:
                    raise LaunchError(f"Static score-proxy identity mismatch: {key}")


def run_logged(command: tuple[str, ...], launch: Launch) -> None:
    # A sibling directory does not violate Online's refusal to reuse run_root.
    logs = launch.run_root.parent / "_launch_logs"
    logs.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    log_path = logs / f"{launch.run_root.name}-{stamp}.log"
    print(f"Training console log: {log_path}", flush=True)
    with log_path.open("x", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=launch.environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            bufsize=1,
        )
        try:
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            code = process.wait()
        except BaseException:
            if process.poll() is None:
                process.terminate()
                process.wait()
            raise
        if code:
            raise subprocess.CalledProcessError(code, command)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--check", action="store_true", help="validate locally without starting training"
    )
    parser.add_argument(
        "--check-services",
        action="store_true",
        help="check endpoint identities and exit without training",
    )
    parser.add_argument(
        "--resume", action="store_true", help="resume Static or Online from existing checkpoints"
    )
    args = parser.parse_args()
    try:
        launch = prepare_launch(args.config, resume=args.resume)
        if args.check_services:
            check_services(launch)
    except (LaunchError, ValueError, OSError) as exc:
        parser.exit(2, f"launch preflight failed: {exc}\n")
    if args.check or args.check_services:
        print(
            json.dumps(
                {
                    "status": "local_preflight_passed",
                    "method": launch.method,
                    "config": str(launch.config),
                    "run_root": str(launch.run_root),
                    "commands": [list(command) for command in launch.commands],
                    "services_checked": args.check_services,
                    "optimizer_gpus": _gpu_ids(_launch_section(launch.config)[0]),
                },
                indent=2,
            )
        )
        return 0
    if launch.method == "static_r0_matched":
        # Retain every parameter snapshot; only the latest sealed step keeps
        # optimizer, scheduler, RNG, and data state.
        checkpoint_root = Path(launch.environment["RUN_DIR"]) / "checkpoints"
        pruner = (
            str(launch.environment["RUNTIME_PYTHON"]),
            str(ROOT / "scripts/prune_horizon_checkpoint_state.py"),
            "--checkpoint-root",
            str(checkpoint_root),
        )
        watcher = subprocess.Popen(
            (*pruner, "--watch", "--while-pid", str(os.getpid())),
            cwd=ROOT,
            env=launch.environment,
        )
        try:
            run_logged(launch.commands[0], launch)
        finally:
            if watcher.poll() is None:
                watcher.terminate()
                watcher.wait()
            subprocess.run(pruner, cwd=ROOT, env=launch.environment, check=True)
        return 0
    for command in launch.commands:
        run_logged(command, launch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
