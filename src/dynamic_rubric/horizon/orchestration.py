"""File-level orchestration for the static-rubric discriminability horizon."""

from __future__ import annotations

from pathlib import Path
import json
from typing import Any, Iterable, Mapping

from ..artifacts import read_jsonl, write_json_atomic
from ..config import RunConfig, STATIC_EXPERIMENT_ARM
from .advantage import verify_pinned_verl_sources
from .inference import ProcessObservation, crossed_bootstrap, decide_horizon
from .observations import verify_horizon_observation_seal
from .reporting import build_horizon_report, coverage_summary, write_horizon_report


class HorizonOrchestrationError(RuntimeError):
    pass


def require_horizon(config: RunConfig):
    if config.horizon is None:
        raise HorizonOrchestrationError("config has no horizon section")
    return config.horizon


def estimate_horizon_cost(config: RunConfig) -> dict[str, Any]:
    horizon = require_horizon(config)
    prompts = horizon.development_count + horizon.final_count
    seeds = len(horizon.training_seeds)
    checkpoint_count = len(horizon.target_epochs)
    nonzero_count = checkpoint_count - 1
    fixed_and_sham = prompts * (
        horizon.fixed_control_count + horizon.sham_control_count
    )
    pool_a = prompts * seeds * nonzero_count * horizon.pool_a_count
    pool_b = prompts * seeds * checkpoint_count * horizon.pool_b_count
    extractor_regular = prompts * seeds * nonzero_count * horizon.pool_a_count
    extractor_sham = prompts * horizon.sham_control_count
    training = config.raw.get("training", {})
    epochs = int(training.get("epochs", 3))
    rollouts = int(training.get("rollout_n", config.training.rollout_n))
    training_generations = horizon.train_count * epochs * rollouts * seeds
    return {
        "schema_version": 1,
        "domain": horizon.domain,
        "prompts": {
            "development": horizon.development_count,
            "final": horizon.final_count,
            "audited_total": prompts,
        },
        "policy_generations": {
            "fixed_and_sham": fixed_and_sham,
            "pool_a": pool_a,
            "pool_b": pool_b,
            "audit_total": fixed_and_sham + pool_a + pool_b,
            "training": training_generations,
            "training_formula": {
                "train_prompts": horizon.train_count,
                "epochs": epochs,
                "rollouts_per_prompt": rollouts,
                "seeds": seeds,
            },
        },
        "extractor_calls": {
            "regular_pairs": extractor_regular,
            "sham_pairs": extractor_sham,
            "total": extractor_regular + extractor_sham,
        },
        "grader_call_formula": (
            "responses(16) * unique(R0 + current Et + matched stale/sham criteria) "
            "per seed/prompt/checkpoint; exact count is emitted after rubric construction"
        ),
    }


def validate_horizon_launch_environment(
    config: RunConfig, environment: Mapping[str, str]
) -> dict[str, Any]:
    """Fail before veRL launch when shell overrides drift from the frozen config."""

    horizon = require_horizon(config)
    if horizon.policy_training_regime != STATIC_EXPERIMENT_ARM:
        raise HorizonOrchestrationError(
            "the static horizon launcher cannot train the dynamic OnlineRubric RL arm; "
            "use train-online and audit its committed checkpoints"
        )
    expected = {
        "DOMAIN": horizon.domain,
        "TOTAL_STEPS": str(config.training.max_steps),
        "TRAIN_BATCH_SIZE": str(config.training.train_batch_size),
        "ROLLOUT_N": str(config.training.rollout_n),
        "MAX_RESPONSE_LENGTH": str(config.training.max_response_length),
        "LEARNING_RATE": f"{config.training.learning_rate:.0e}".replace("e-0", "e-"),
        "WARMUP_RATIO": str(config.training.warmup_ratio),
        "KL_COEFFICIENT": str(config.training.kl_coefficient),
        "ROLLOUT_TEMPERATURE": str(config.training.sampling_temperature),
    }
    missing = sorted(key for key in expected if key not in environment)
    if missing:
        raise HorizonOrchestrationError(f"horizon launcher environment is missing: {missing}")
    mismatched = {
        key: {"expected": value, "actual": environment[key]}
        for key, value in expected.items()
        if environment[key] != value
    }
    try:
        checkpoint_steps = tuple(int(value) for value in json.loads(environment["CHECKPOINT_STEPS"]))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise HorizonOrchestrationError("CHECKPOINT_STEPS must be a JSON integer list") from error
    if checkpoint_steps != config.training.checkpoint_steps:
        mismatched["CHECKPOINT_STEPS"] = {
            "expected": list(config.training.checkpoint_steps),
            "actual": list(checkpoint_steps),
        }
    try:
        training_seed = int(environment["TRAINING_SEED"])
    except (KeyError, ValueError) as error:
        raise HorizonOrchestrationError("TRAINING_SEED must be an integer") from error
    if training_seed not in horizon.training_seeds:
        mismatched["TRAINING_SEED"] = {
            "expected": list(horizon.training_seeds),
            "actual": training_seed,
        }
    if mismatched:
        raise HorizonOrchestrationError(f"horizon launcher settings drifted: {mismatched}")
    return {"valid": True, "domain": horizon.domain, "training_seed": training_seed, **expected}


def verify_horizon_models(config: RunConfig, *, verl_root: Path) -> dict[str, Any]:
    require_horizon(config)
    results: dict[str, Any] = {}
    for role in ("policy", "proxy_grader"):
        model = config.models.get(role)
        if not isinstance(model, Mapping):
            raise HorizonOrchestrationError(f"models.{role} is missing")
        snapshot = Path(str(model.get("local_snapshot", "")))
        required = ("config.json", "tokenizer_config.json")
        missing = [name for name in required if not (snapshot / name).is_file()]
        has_weights = (snapshot / "model.safetensors").is_file() or (
            snapshot / "model.safetensors.index.json"
        ).is_file()
        if missing or not has_weights:
            raise HorizonOrchestrationError(
                f"model snapshot incomplete for {role}: path={snapshot}, missing={missing}"
            )
        total_bytes = sum(path.stat().st_size for path in snapshot.rglob("*") if path.is_file())
        results[role] = {
            "model": model.get("model"),
            "revision": model.get("revision"),
            "snapshot": str(snapshot),
            "bytes": total_bytes,
        }
    return {
        "schema_version": 1,
        "models": results,
        "verl_commit_sources": verify_pinned_verl_sources(verl_root),
    }


def load_process_observations(rows: Iterable[Mapping[str, Any]]) -> list[ProcessObservation]:
    observations: list[ProcessObservation] = []
    for row in rows:
        if {"d_r0", "g_refresh", "g_count"} <= set(row):
            observations.append(
                ProcessObservation(
                    seed_id=str(row["seed_id"]),
                    prompt_id=str(row["prompt_id"]),
                    checkpoint=float(row["checkpoint"]),
                    d_r0=float(row["d_r0"]),
                    g_refresh=float(row["g_refresh"]),
                    g_count=float(row["g_count"]),
                )
            )
            continue
        required = {"r0_zar", "r0_baseline_zar", "current_zar", "control_zar"}
        if not required <= set(row):
            raise HorizonOrchestrationError("observation row is missing horizon process fields")
        from .inference import make_process_observation

        observations.append(
            make_process_observation(
                seed_id=str(row["seed_id"]),
                prompt_id=str(row["prompt_id"]),
                checkpoint=float(row["checkpoint"]),
                r0_zar=float(row["r0_zar"]),
                r0_baseline_zar=float(row["r0_baseline_zar"]),
                current_zar=float(row["current_zar"]),
                control_zar=float(row["control_zar"]),
            )
        )
    return observations


def analyze_horizon_observations(
    config: RunConfig,
    observation_path: Path,
    output_path: Path,
    *,
    iterations: int | None = None,
) -> dict[str, Any]:
    horizon = require_horizon(config)
    seal = verify_horizon_observation_seal(observation_path)
    if seal.get("config_hash") != config.config_hash:
        raise HorizonOrchestrationError("observation seal config hash differs from the analysis config")
    observations = load_process_observations(read_jsonl(observation_path))
    configured_iterations = int(config.raw.get("horizon", {}).get("bootstrap_replicates", 10_000))
    bootstrap = crossed_bootstrap(
        observations,
        iterations=iterations or configured_iterations,
        seed=config.bootstrap_seed,
    )
    decision = decide_horizon(
        bootstrap,
        deterioration_margin=horizon.deterioration_margin,
        refresh_margin=horizon.refresh_margin,
        equivalence_margin=horizon.equivalence_margin,
    )
    report = build_horizon_report(
        domain=horizon.domain,
        bootstrap=bootstrap,
        decision=decision,
        coverage=coverage_summary(
            valid=int(seal["coverage"]["valid"]),
            invalid=int(seal["coverage"]["invalid"]),
            missing=int(seal["coverage"]["missing"]),
            na=int(seal["coverage"]["na"]),
        ),
        revisions={
            role: str(value.get("revision"))
            for role, value in config.models.items()
            if isinstance(value, Mapping) and value.get("revision")
        },
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_horizon_report(output_path, report)
    write_json_atomic(output_path.with_name("horizon_decision.json"), report["horizon_decision"])
    return report
