"""Typed configuration loading and pipeline-boundary validation."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .hashing import sha256_json


class ConfigError(ValueError):
    pass


DEFAULT_CHECKPOINT_STEPS = (0, 1, 2, 3, 5, 10, 20, 30, 50, 75, 100)
HORIZON_TARGET_EPOCHS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0)
HORIZON_CHECKPOINT_STEPS = (0, 3, 6, 9, 13, 16, 24, 32, 40, 48)
ONLINE_HORIZON_CHECKPOINT_STEPS = (0, 3, 6, 9, 12, 15, 23, 30, 38, 45)
STATIC_EXPERIMENT_ARM = "static_r0_grpo"
ONLINE_EXPERIMENT_ARM = "dynamic_online_rubric_grpo"
HORIZON_POLICY_TRAINING_REGIMES = (STATIC_EXPERIMENT_ARM, ONLINE_EXPERIMENT_ARM)
ONLINE_CONTROL_POLICIES = ("pi_ref", "pi_old")
ONLINE_REPRODUCTION_PROFILES = (
    "paper_algorithm_faithful",
    "paper_full_reproduction",
    "paper_inspired",
)
PAPER_ACTOR_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
PAPER_EXTRACTOR_MODEL = "o3-mini"
PAPER_GRADER_MODEL = "gpt-4.1-mini"


@dataclass(frozen=True, slots=True)
class PathsConfig:
    public_data: str = "data/public"
    artifacts: str = "artifacts"
    results: str = "results"
    private_gt: str | None = None


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    max_steps: int = 100
    train_batch_size: int = 48
    rollout_n: int = 8
    checkpoint_steps: tuple[int, ...] = DEFAULT_CHECKPOINT_STEPS
    reward_source: str = "static_r0_only"
    artifact_inputs: tuple[str, ...] = ()
    epochs: int = 3
    learning_rate: float = 5e-6
    warmup_ratio: float = 0.1
    kl_coefficient: float = 0.01
    sampling_temperature: float = 1.0
    max_response_length: int = 3584

    def __post_init__(self) -> None:
        if min(
            self.max_steps,
            self.train_batch_size,
            self.rollout_n,
            self.epochs,
            self.max_response_length,
        ) <= 0:
            raise ConfigError("training counts must be positive")
        floats = (
            self.learning_rate,
            self.warmup_ratio,
            self.kl_coefficient,
            self.sampling_temperature,
        )
        if any(not math.isfinite(value) for value in floats) or self.learning_rate <= 0:
            raise ConfigError("training scalar settings must be finite and learning_rate positive")
        if not 0 <= self.warmup_ratio <= 1 or self.kl_coefficient < 0:
            raise ConfigError("warmup_ratio/kl_coefficient are outside their allowed range")
        allowed_reward_sources = {
            "static_r0_only",
            "rar_static_r0_only",
            "online_r0_union_elicited_same_step",
        }
        if self.reward_source not in allowed_reward_sources:
            raise ConfigError(
                f"training reward_source must be one of {sorted(allowed_reward_sources)}"
            )
        if any(step < 0 or step > self.max_steps for step in self.checkpoint_steps):
            raise ConfigError("checkpoint_steps must be within [0, max_steps]")
        assert_no_dynamic_training_paths(self.artifact_inputs)


@dataclass(frozen=True, slots=True)
class OnlineTrainingConfig:
    """Fail-closed contract for the causal, same-step OnlineRubrics lane."""

    mode: str = "paper_online_rubrics_v1"
    experiment_arm: str = ONLINE_EXPERIMENT_ARM
    baseline_arm: str = STATIC_EXPERIMENT_ARM
    reproduction_profile: str = "paper_algorithm_faithful"
    control_policy: str = "pi_ref"
    dataset_profile: str = "rar_medicine"
    actor_model: str = PAPER_ACTOR_MODEL
    actor_revision: str = ""
    extractor_model: str = PAPER_EXTRACTOR_MODEL
    grader_model: str = PAPER_GRADER_MODEL
    epochs: int = 3
    prompt_batch_size: int = 96
    rollouts_per_prompt: int = 16
    elicitation_pairs_per_prompt: int = 8
    learning_rate: float = 5e-6
    warmup_ratio: float = 0.1
    kl_coefficient: float = 0.01
    criteria_scope: str = "prompt_step_ephemeral"
    failure_policy: str = "fail_closed"
    checkpoint_interval_steps: int = 3
    comparison_checkpoint_steps: tuple[int, ...] = ONLINE_HORIZON_CHECKPOINT_STEPS
    stale_comparator: str = "previous_focal_checkpoint"
    drop_last: bool = True
    expected_train_rows: int = 1500
    gpu_count: int = 2
    accelerator: str = "NVIDIA H200 NVL"
    per_device_prompt_batch_size: int = 24
    gradient_accumulation_steps: int = 2
    extractor_concurrency: int = 64
    grader_concurrency: int = 128
    artifact_inputs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.mode != "paper_online_rubrics_v1":
            raise ConfigError("online_training.mode must be paper_online_rubrics_v1")
        if self.experiment_arm != ONLINE_EXPERIMENT_ARM:
            raise ConfigError(
                f"online_training.experiment_arm must be {ONLINE_EXPERIMENT_ARM}"
            )
        if self.baseline_arm != STATIC_EXPERIMENT_ARM:
            raise ConfigError(
                f"online_training.baseline_arm must be {STATIC_EXPERIMENT_ARM}"
            )
        if self.reproduction_profile not in ONLINE_REPRODUCTION_PROFILES:
            raise ConfigError("online_training.reproduction_profile is unsupported")
        if self.control_policy not in ONLINE_CONTROL_POLICIES:
            raise ConfigError("online_training.control_policy must be pi_ref or pi_old")
        if not self.dataset_profile:
            raise ConfigError("online_training.dataset_profile must be non-empty")
        counts = (
            self.epochs,
            self.prompt_batch_size,
            self.rollouts_per_prompt,
            self.elicitation_pairs_per_prompt,
            self.expected_train_rows,
            self.gpu_count,
            self.per_device_prompt_batch_size,
            self.gradient_accumulation_steps,
            self.extractor_concurrency,
            self.grader_concurrency,
        )
        if min(counts) <= 0:
            raise ConfigError("online training counts must be positive")
        scalars = (self.learning_rate, self.warmup_ratio, self.kl_coefficient)
        if any(not math.isfinite(value) for value in scalars) or self.learning_rate <= 0:
            raise ConfigError("online training scalar settings must be finite")
        if not 0 <= self.warmup_ratio <= 1 or self.kl_coefficient < 0:
            raise ConfigError("online warmup_ratio/kl_coefficient are outside their range")
        if self.criteria_scope != "prompt_step_ephemeral":
            raise ConfigError("online criteria must be prompt_step_ephemeral")
        if self.failure_policy != "fail_closed":
            raise ConfigError("online training must use fail_closed failure policy")
        if self.checkpoint_interval_steps != 3:
            raise ConfigError("full online training requires checkpoint_interval_steps=3")
        if self.stale_comparator != "previous_focal_checkpoint":
            raise ConfigError(
                "online training stale_comparator must be previous_focal_checkpoint"
            )
        if (
            not self.comparison_checkpoint_steps
            or self.comparison_checkpoint_steps[0] != 0
            or self.comparison_checkpoint_steps[-1] != self.expected_updates
            or tuple(sorted(set(self.comparison_checkpoint_steps)))
            != self.comparison_checkpoint_steps
            or any(
                step < 0 or step > self.expected_updates
                for step in self.comparison_checkpoint_steps
            )
        ):
            raise ConfigError(
                "online comparison_checkpoint_steps must be unique, increasing, start at 0, "
                "and end at expected_updates"
            )
        if not self.drop_last:
            raise ConfigError("paper online training requires drop_last=true")
        if len(self.actor_revision) != 40 or any(
            character not in "0123456789abcdef" for character in self.actor_revision.lower()
        ):
            raise ConfigError("online actor_revision must be a pinned 40-character commit")
        assert_no_online_artifact_inputs(self.artifact_inputs)
        paper_algorithm = {
            "actor_model": PAPER_ACTOR_MODEL,
            "extractor_model": PAPER_EXTRACTOR_MODEL,
            "grader_model": PAPER_GRADER_MODEL,
            "epochs": 3,
            "prompt_batch_size": 96,
            "rollouts_per_prompt": 16,
            "elicitation_pairs_per_prompt": 8,
            "learning_rate": 5e-6,
            "warmup_ratio": 0.1,
            "kl_coefficient": 0.01,
        }
        actual = {name: getattr(self, name) for name in paper_algorithm}
        if self.reproduction_profile != "paper_inspired" and actual != paper_algorithm:
            raise ConfigError(f"paper-faithful online settings drifted from the paper: {actual}")
        if self.effective_prompt_batch_size != self.prompt_batch_size:
            raise ConfigError(
                "gpu_count * per_device_prompt_batch_size * gradient_accumulation_steps "
                "must equal prompt_batch_size"
            )
        if self.reproduction_profile == "paper_full_reproduction" and not (
            self.dataset_profile == "paper_original"
            and self.gpu_count == 8
            and self.accelerator == "NVIDIA H100"
            and self.per_device_prompt_batch_size == 6
            and self.gradient_accumulation_steps == 2
        ):
            raise ConfigError(
                "paper_full_reproduction requires the paper dataset and exact 8xH100 topology"
            )

    @property
    def effective_prompt_batch_size(self) -> int:
        return self.gpu_count * self.per_device_prompt_batch_size * self.gradient_accumulation_steps

    @property
    def checkpoint_steps(self) -> tuple[int, ...]:
        resume_steps = range(
            self.checkpoint_interval_steps,
            self.expected_updates + 1,
            self.checkpoint_interval_steps,
        )
        return tuple(
            sorted(
                {
                    *resume_steps,
                    *self.comparison_checkpoint_steps,
                    self.expected_updates,
                }
                - {0}
            )
        )

    @property
    def expected_updates(self) -> int:
        return (self.expected_train_rows // self.prompt_batch_size) * self.epochs

    @property
    def runtime_claim(self) -> str:
        return self.reproduction_profile


@dataclass(frozen=True, slots=True)
class HorizonConfig:
    domain: str
    policy_training_regime: str = STATIC_EXPERIMENT_ARM
    train_count: int = 1500
    development_count: int = 150
    final_count: int = 300
    training_seeds: tuple[int, ...] = (11, 29, 47)
    target_epochs: tuple[float, ...] = HORIZON_TARGET_EPOCHS
    fixed_control_count: int = 8
    sham_control_count: int = 8
    pool_a_count: int = 8
    pool_b_count: int = 16
    max_online_criteria: int = 8
    deterioration_margin: float = 0.03
    refresh_margin: float = 0.03
    equivalence_margin: float = 0.015

    def __post_init__(self) -> None:
        if self.domain not in {"medicine", "science"}:
            raise ConfigError("horizon.domain must be medicine or science")
        if self.policy_training_regime not in HORIZON_POLICY_TRAINING_REGIMES:
            raise ConfigError(
                "horizon.policy_training_regime must identify the static or dynamic RL arm"
            )
        positive_counts = (
            self.train_count,
            self.final_count,
            self.fixed_control_count,
            self.pool_a_count,
            self.pool_b_count,
            self.max_online_criteria,
        )
        if (
            min(positive_counts) <= 0
            or self.development_count < 0
            or self.sham_control_count < 0
        ):
            raise ConfigError(
                "horizon counts must be positive, except development_count and "
                "sham_control_count may be zero"
            )
        if len(self.training_seeds) < 1 or len(set(self.training_seeds)) != len(
            self.training_seeds
        ):
            raise ConfigError("horizon training_seeds must be non-empty and unique")
        if any(seed < 0 for seed in self.training_seeds):
            raise ConfigError("horizon training_seeds must be non-negative")
        if self.target_epochs != HORIZON_TARGET_EPOCHS:
            raise ConfigError("horizon target_epochs must match the preregistered ten checkpoints")
        if self.pool_b_count != 16 or self.pool_a_count != 8:
            raise ConfigError("horizon requires Pool A=8 and Pool B=16")
        if not (
            0 <= self.equivalence_margin
            < min(self.deterioration_margin, self.refresh_margin)
        ):
            raise ConfigError("horizon equivalence margin must be below positive gain margins")


@dataclass(frozen=True, slots=True)
class RunConfig:
    experiment: str = "dynamic_rubric_staleness_audit"
    split_seed: int = 0
    bootstrap_seed: int = 0
    paths: PathsConfig = field(default_factory=PathsConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    models: Mapping[str, Any] = field(default_factory=dict)
    horizon: HorizonConfig | None = None
    online_training: OnlineTrainingConfig | None = None
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.experiment:
            raise ConfigError("experiment must be non-empty")
        if self.split_seed < 0 or self.bootstrap_seed < 0:
            raise ConfigError("seeds must be non-negative")
        if self.horizon is not None:
            dynamic_arm = self.horizon.policy_training_regime == ONLINE_EXPERIMENT_ARM
            expected = {
                "max_steps": 45 if dynamic_arm else 48,
                "train_batch_size": 96,
                "rollout_n": 16,
                "checkpoint_steps": (
                    ONLINE_HORIZON_CHECKPOINT_STEPS
                    if dynamic_arm
                    else HORIZON_CHECKPOINT_STEPS
                ),
                "reward_source": (
                    "online_r0_union_elicited_same_step"
                    if dynamic_arm
                    else "rar_static_r0_only"
                ),
                "epochs": 3,
                "learning_rate": 5e-6,
                "warmup_ratio": 0.1,
                "kl_coefficient": 0.01,
                "sampling_temperature": 1.0,
                "max_response_length": 3584,
            }
            actual = {name: getattr(self.training, name) for name in expected}
            if actual != expected:
                raise ConfigError(
                    "horizon training settings drifted from the selected policy training "
                    f"regime ({self.horizon.policy_training_regime}): {actual}"
                )
        if self.horizon is not None and self.online_training is not None:
            raise ConfigError("horizon and online_training are distinct execution lanes")
        if self.online_training is not None:
            policy = _mapping(self.models.get("policy"), "models.policy")
            extractor = _mapping(
                self.models.get("rubric_extractor"), "models.rubric_extractor"
            )
            deduplicator = _mapping(
                self.models.get("rubric_deduplicator"), "models.rubric_deduplicator"
            )
            grader = _mapping(self.models.get("online_grader"), "models.online_grader")
            expected_models = {
                "policy.model": self.online_training.actor_model,
                "policy.revision": self.online_training.actor_revision,
                "rubric_extractor.requested_model": self.online_training.extractor_model,
                "rubric_deduplicator.requested_model": self.online_training.extractor_model,
                "online_grader.requested_model": self.online_training.grader_model,
            }
            actual_models = {
                "policy.model": str(policy.get("model", "")),
                "policy.revision": str(policy.get("revision", "")),
                "rubric_extractor.requested_model": str(
                    extractor.get("requested_model", "")
                ),
                "rubric_deduplicator.requested_model": str(
                    deduplicator.get("requested_model", "")
                ),
                "online_grader.requested_model": str(grader.get("requested_model", "")),
            }
            if actual_models != expected_models:
                raise ConfigError(
                    f"online model identities disagree with online_training: {actual_models}"
                )

    @property
    def config_hash(self) -> str:
        return sha256_json(
            self.raw
            or {
                "experiment": self.experiment,
                "split_seed": self.split_seed,
                "bootstrap_seed": self.bootstrap_seed,
                "paths": self.paths,
                "training": self.training,
                "models": self.models,
                "horizon": self.horizon,
                "online_training": self.online_training,
            }
        )


def _path_text(value: object) -> str:
    return str(value).replace("\\", "/").lower()


def _contains_segment(value: object, segment: str) -> bool:
    parts = [part for part in _path_text(value).split("/") if part not in {"", "."}]
    return segment.lower() in parts


def assert_public_path(path: str | Path) -> None:
    text = _path_text(path)
    if _contains_segment(path, "private_gt") or "gold_rubric" in text or "gold-rubric" in text:
        raise ConfigError(f"private/gold path is forbidden in public stages: {path}")


def assert_no_dynamic_training_paths(value: object) -> None:
    """Reject dynamic/replay artifacts from the static-only training boundary."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            assert_no_dynamic_training_paths(key)
            assert_no_dynamic_training_paths(item)
        return
    if isinstance(value, (list, tuple, set)):
        for item in value:
            assert_no_dynamic_training_paths(item)
        return
    text = _path_text(value)
    forbidden = (
        "dynamic_fixed",
        "dynamic_prev",
        "refresh_only",
        "replay_dynamic",
        "/replay/",
        "rubric_trajectory",
        "horizon_rubric",
        "/horizon/",
        "online_criteria",
    )
    if any(token in text for token in forbidden):
        raise ConfigError(f"dynamic rubric artifact is forbidden in training: {value}")


def assert_no_online_artifact_inputs(value: object) -> None:
    """Online training may read public R0 rows, never replay/eval/generated rubrics."""

    assert_no_dynamic_training_paths(value)
    if isinstance(value, Mapping):
        for key, item in value.items():
            assert_no_online_artifact_inputs(key)
            assert_no_online_artifact_inputs(item)
        return
    if isinstance(value, (list, tuple, set)):
        for item in value:
            assert_no_online_artifact_inputs(item)
        return
    text = _path_text(value)
    forbidden = (
        "/development",
        "/final",
        "pool_a",
        "pool_b",
        "posthoc",
        "post-hoc",
        "onlinerubric_rubrics",
    )
    if any(token in text for token in forbidden):
        raise ConfigError(f"evaluation/replay artifact is forbidden in online training: {value}")


def validate_stage_paths(config: Mapping[str, Any], *, stage: str) -> None:
    public_stage = stage not in {"audit-gold", "audit_gold", "gold-audit", "gold_audit"}

    def walk(value: object) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                if public_stage:
                    assert_public_path(key)
                walk(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item)
        elif public_stage and isinstance(value, (str, Path)):
            assert_public_path(value)

    walk(config)
    if stage in {"train-static", "train_static", "training"}:
        assert_no_dynamic_training_paths(config.get("training", {}))
    if stage in {
        "train-online",
        "train_online",
        "resume-online",
        "resume_online",
        "validate-online-step",
        "validate_online_step",
    }:
        assert_no_online_artifact_inputs(config.get("online_training", {}))


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ConfigError(f"{name} must be a mapping")
    return value


def config_from_mapping(data: Mapping[str, Any], *, stage: str | None = None) -> RunConfig:
    if stage:
        validate_stage_paths(data, stage=stage)
    paths_data = _mapping(data.get("paths"), "paths")
    training_data = _mapping(data.get("training"), "training")
    paths = PathsConfig(
        public_data=str(paths_data.get("public_data", "data/public")),
        artifacts=str(paths_data.get("artifacts", "artifacts")),
        results=str(paths_data.get("results", "results")),
        private_gt=None if paths_data.get("private_gt") is None else str(paths_data["private_gt"]),
    )
    checkpoints = training_data.get("checkpoint_steps", DEFAULT_CHECKPOINT_STEPS)
    training = TrainingConfig(
        max_steps=int(training_data.get("max_steps", 100)),
        train_batch_size=int(training_data.get("train_batch_size", 48)),
        rollout_n=int(training_data.get("rollout_n", 8)),
        checkpoint_steps=tuple(int(step) for step in checkpoints),
        reward_source=str(training_data.get("reward_source", "static_r0_only")),
        artifact_inputs=tuple(str(item) for item in training_data.get("artifact_inputs", ())),
        epochs=int(training_data.get("epochs", 3)),
        learning_rate=float(training_data.get("learning_rate", 5e-6)),
        warmup_ratio=float(training_data.get("warmup_ratio", 0.1)),
        kl_coefficient=float(training_data.get("kl_coefficient", 0.01)),
        sampling_temperature=float(training_data.get("sampling_temperature", 1.0)),
        max_response_length=int(training_data.get("max_response_length", 3584)),
    )
    horizon_data = _mapping(data.get("horizon"), "horizon") if "horizon" in data else None
    horizon = None
    if horizon_data is not None:
        horizon = HorizonConfig(
            domain=str(horizon_data.get("domain", "")),
            policy_training_regime=str(
                horizon_data.get("policy_training_regime", STATIC_EXPERIMENT_ARM)
            ),
            train_count=int(horizon_data.get("train_count", 1500)),
            development_count=int(horizon_data.get("development_count", 150)),
            final_count=int(horizon_data.get("final_count", 300)),
            training_seeds=tuple(
                int(value) for value in horizon_data.get("training_seeds", (11, 29, 47))
            ),
            target_epochs=tuple(
                float(value)
                for value in horizon_data.get("target_epochs", HORIZON_TARGET_EPOCHS)
            ),
            fixed_control_count=int(horizon_data.get("fixed_control_count", 8)),
            sham_control_count=int(horizon_data.get("sham_control_count", 8)),
            pool_a_count=int(horizon_data.get("pool_a_count", 8)),
            pool_b_count=int(horizon_data.get("pool_b_count", 16)),
            max_online_criteria=int(horizon_data.get("max_online_criteria", 8)),
            deterioration_margin=float(horizon_data.get("deterioration_margin", 0.03)),
            refresh_margin=float(horizon_data.get("refresh_margin", 0.03)),
            equivalence_margin=float(horizon_data.get("equivalence_margin", 0.015)),
        )
    online_data = (
        _mapping(data.get("online_training"), "online_training")
        if "online_training" in data
        else None
    )
    online_training = None
    if online_data is not None:
        forbidden_options = {
            "checkpoint_every_step",
            "max_online_criteria",
            "admission",
            "eviction",
            "accumulate",
            "quality_filter",
            "refresh_margin",
        }
        present = sorted(forbidden_options.intersection(online_data))
        if present:
            raise ConfigError(f"paper online training forbids rubric filter/cap options: {present}")
        online_training = OnlineTrainingConfig(
            mode=str(online_data.get("mode", "paper_online_rubrics_v1")),
            experiment_arm=str(
                online_data.get("experiment_arm", ONLINE_EXPERIMENT_ARM)
            ),
            baseline_arm=str(online_data.get("baseline_arm", STATIC_EXPERIMENT_ARM)),
            reproduction_profile=str(
                online_data.get("reproduction_profile", "paper_algorithm_faithful")
            ),
            control_policy=str(online_data.get("control_policy", "pi_ref")),
            dataset_profile=str(online_data.get("dataset_profile", "rar_medicine")),
            actor_model=str(online_data.get("actor_model", PAPER_ACTOR_MODEL)),
            actor_revision=str(online_data.get("actor_revision", "")),
            extractor_model=str(online_data.get("extractor_model", PAPER_EXTRACTOR_MODEL)),
            grader_model=str(online_data.get("grader_model", PAPER_GRADER_MODEL)),
            epochs=int(online_data.get("epochs", 3)),
            prompt_batch_size=int(online_data.get("prompt_batch_size", 96)),
            rollouts_per_prompt=int(online_data.get("rollouts_per_prompt", 16)),
            elicitation_pairs_per_prompt=int(
                online_data.get("elicitation_pairs_per_prompt", 8)
            ),
            learning_rate=float(online_data.get("learning_rate", 5e-6)),
            warmup_ratio=float(online_data.get("warmup_ratio", 0.1)),
            kl_coefficient=float(online_data.get("kl_coefficient", 0.01)),
            criteria_scope=str(
                online_data.get("criteria_scope", "prompt_step_ephemeral")
            ),
            failure_policy=str(online_data.get("failure_policy", "fail_closed")),
            checkpoint_interval_steps=int(
                online_data.get("checkpoint_interval_steps", 3)
            ),
            comparison_checkpoint_steps=tuple(
                int(step)
                for step in online_data.get(
                    "comparison_checkpoint_steps", ONLINE_HORIZON_CHECKPOINT_STEPS
                )
            ),
            stale_comparator=str(
                online_data.get("stale_comparator", "previous_focal_checkpoint")
            ),
            drop_last=bool(online_data.get("drop_last", True)),
            expected_train_rows=int(online_data.get("expected_train_rows", 1500)),
            gpu_count=int(online_data.get("gpu_count", 2)),
            accelerator=str(online_data.get("accelerator", "NVIDIA H200 NVL")),
            per_device_prompt_batch_size=int(
                online_data.get("per_device_prompt_batch_size", 24)
            ),
            gradient_accumulation_steps=int(
                online_data.get("gradient_accumulation_steps", 2)
            ),
            extractor_concurrency=int(online_data.get("extractor_concurrency", 64)),
            grader_concurrency=int(online_data.get("grader_concurrency", 128)),
            artifact_inputs=tuple(
                str(item) for item in online_data.get("artifact_inputs", ())
            ),
        )
    return RunConfig(
        experiment=str(data.get("experiment", "dynamic_rubric_staleness_audit")),
        split_seed=int(data.get("split_seed", 0)),
        bootstrap_seed=int(data.get("bootstrap_seed", 0)),
        paths=paths,
        training=training,
        models=dict(_mapping(data.get("models"), "models")),
        horizon=horizon,
        online_training=online_training,
        raw=dict(data),
    )


def load_config(path: str | Path, *, stage: str | None = None) -> RunConfig:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        try:
            import yaml  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ConfigError(
                "YAML config requires PyYAML; JSON configs remain stdlib-only"
            ) from exc
        data = yaml.safe_load(text)
    if not isinstance(data, Mapping):
        raise ConfigError("config root must be a mapping")
    return config_from_mapping(data, stage=stage)


validate_config = config_from_mapping
