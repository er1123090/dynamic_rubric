"""Typed, fail-closed configuration for Phase-1 evaluator-update experiments."""

from __future__ import annotations

import math
import os
import re
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ..hashing import sha256_json


class Phase1ConfigError(ValueError):
    pass


PHASE1_EXPERIMENT = "phase1_evaluator_update_value"
PHASE1_METHODS = ("online_rubrics", "evorubrics")
PHASE1_DOMAINS = ("medicine", "science")
POLICY_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
POLICY_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
GPT_OSS_MODEL = "openai/gpt-oss-120b"
RUBRIC_JUDGE_MODEL = "Qwen/Qwen3-32B"
RUBRIC_JUDGE_REVISION = "9216db5781bf21249d130ec9da846c4624c16137"
AUDIT_CHECKPOINTS = (0, 3, 6, 9, 13, 16, 24, 32, 40, 48)
REUSE_ANCHORS = (0, 9, 16, 32, 48)
STEP_LOG_FIELDS = (
    "cumulative_prompt_exposures",
    "cumulative_completions",
    "cumulative_response_tokens",
    "adjacent_policy_kl",
    "cumulative_policy_kl_from_pi0",
    "response_length",
)
PRE_UPDATE_STATE_FIELDS = (
    "global_step",
    "evaluator_age_steps",
    "evaluator_age_checkpoints",
    "cumulative_prompts_since_evaluator",
    "cumulative_completions_since_evaluator",
    "cumulative_response_tokens_since_evaluator",
    "policy_kl_current_vs_evaluator",
    "cumulative_policy_kl_from_pi0",
    "response_length_shift",
    "stale_zar",
    "stale_pairwise_tie_rate",
    "stale_pairwise_separation_rate",
    "stale_effective_criterion_ratio",
    "stale_saturation_ratio",
    "stale_reward_std",
    "stale_top_median_margin",
)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Phase1ConfigError(f"{name} must be a mapping")
    return value


def _sequence(value: object, name: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise Phase1ConfigError(f"{name} must be a sequence")
    return value


def _require_equal(actual: object, expected: object, name: str) -> None:
    if actual != expected:
        raise Phase1ConfigError(f"{name} must be {expected!r}, got {actual!r}")


def _require_false(mapping: Mapping[str, Any], key: str, name: str) -> None:
    if mapping.get(key) is not False:
        raise Phase1ConfigError(f"{name}.{key} must be false")


def _bounded_float(
    value: object, name: str, *, minimum: float, maximum: float | None = None
) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise Phase1ConfigError(f"{name} must be numeric") from error
    if not math.isfinite(number) or number < minimum or (maximum is not None and number > maximum):
        raise Phase1ConfigError(f"{name} must be finite and in its supported range")
    return number


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = value
    return merged


_ENVIRONMENT_REFERENCE = re.compile(r"\$(?:\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))")


def _expand_config_value(value: Any, *, name: str) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _expand_config_value(item, name=f"{name}.{key}") for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _expand_config_value(item, name=f"{name}[{index}]") for index, item in enumerate(value)
        ]
    if not isinstance(value, str):
        return value
    missing = sorted(
        {
            match.group(1) or match.group(2)
            for match in _ENVIRONMENT_REFERENCE.finditer(value)
            if not os.environ.get(match.group(1) or match.group(2), "").strip()
        }
    )
    if missing:
        raise Phase1ConfigError(
            f"{name} references unset environment variables or blank values: {missing}"
        )
    expanded = os.path.expanduser(os.path.expandvars(value))
    if _ENVIRONMENT_REFERENCE.search(expanded):
        raise Phase1ConfigError(f"{name} contains an unresolved environment variable")
    return expanded


def _positive_int(value: object, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise Phase1ConfigError(f"{name} must be an integer >= {minimum}")
    return value


def _gpu_list(value: object, name: str) -> tuple[int, ...]:
    values = _sequence(value, name)
    if (
        not values
        or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in values)
        or len(set(values)) != len(values)
    ):
        raise Phase1ConfigError(f"{name} must contain unique nonnegative GPU indexes")
    return tuple(values)


def _load_yaml_with_extends(
    path: Path, *, loading: tuple[Path, ...] = ()
) -> MutableMapping[str, Any]:
    resolved_path = path.resolve()
    if resolved_path in loading:
        chain = " -> ".join(str(item) for item in (*loading, resolved_path))
        raise Phase1ConfigError(f"cyclic config extends: {chain}")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, MutableMapping):
        raise Phase1ConfigError(f"Phase-1 config root must be a mapping: {path}")
    parents = value.pop("extends", ())
    if isinstance(parents, str):
        parents = (parents,)
    parents = _sequence(parents, f"{path}.extends")
    merged: dict[str, Any] = {}
    for parent in parents:
        parent_value = _expand_config_value(str(parent), name=f"{path}.extends[]")
        parent_path = (path.parent / parent_value).resolve()
        inherited = _load_yaml_with_extends(parent_path, loading=(*loading, resolved_path))
        merged = _deep_merge(merged, inherited)
    return _deep_merge(merged, value)


def load_yaml_config(path: str | Path) -> MutableMapping[str, Any]:
    """Load inherited YAML and expand portable environment/user paths."""

    source_path = Path(path).expanduser().resolve()
    loaded = _load_yaml_with_extends(source_path)
    return dict(_expand_config_value(loaded, name=str(source_path)))


def _validate_model(
    model: Mapping[str, Any],
    *,
    name: str,
    expected_model: str,
    expected_revision: str | None = None,
    non_reasoning: bool = False,
) -> None:
    _require_equal(str(model.get("model", "")), expected_model, f"{name}.model")
    revision = str(model.get("revision", ""))
    if not revision:
        raise Phase1ConfigError(f"{name}.revision must be recorded")
    if expected_revision is not None:
        _require_equal(revision, expected_revision, f"{name}.revision")
    if non_reasoning:
        _require_false(model, "thinking", name)


@dataclass(frozen=True, slots=True)
class Phase1Config:
    source_path: Path
    raw: Mapping[str, Any]
    experiment: str
    domain: str
    method: str
    seed: int

    @property
    def config_hash(self) -> str:
        return sha256_json(self.raw)

    @property
    def data(self) -> Mapping[str, Any]:
        return _mapping(self.raw["data"], "data")

    @property
    def training(self) -> Mapping[str, Any]:
        return _mapping(self.raw["training"], "training")

    @property
    def analysis(self) -> Mapping[str, Any]:
        return _mapping(self.raw["analysis"], "analysis")

    @property
    def models(self) -> Mapping[str, Any]:
        return _mapping(self.raw["models"], "models")

    @property
    def method_config(self) -> Mapping[str, Any]:
        return _mapping(self.raw[self.method], self.method)

    @property
    def checkpoint_steps(self) -> tuple[int, ...]:
        expected = int(self.training["expected_global_steps"])
        interval = int(self.training["checkpoint_interval_steps"])
        audit = (int(step) for step in self.training["audit_checkpoints"])
        return tuple(sorted({*range(interval, expected + 1, interval), *audit, expected} - {0}))

    @property
    def output_root(self) -> Path:
        return Path(str(_mapping(self.raw["output"], "output")["root"]))

    def run_root(self, root: Path, run_id: str) -> Path:
        if not run_id or "/" in run_id or chr(92) in run_id:
            raise Phase1ConfigError("run_id must be a non-empty path segment")
        return root / self.output_root / self.domain / self.method / f"seed-{self.seed}" / run_id


def validate_phase1_mapping(data: Mapping[str, Any], *, source_path: Path) -> Phase1Config:
    _require_equal(int(data.get("schema_version", 0)), 1, "schema_version")
    experiment = str(data.get("experiment", ""))
    domain = str(data.get("domain", ""))
    method = str(data.get("method", ""))
    raw_seed = data.get("seed", -1)
    if isinstance(raw_seed, bool) or not isinstance(raw_seed, int):
        raise Phase1ConfigError("seed must be an integer")
    seed = raw_seed
    _require_equal(experiment, PHASE1_EXPERIMENT, "experiment")
    if domain not in PHASE1_DOMAINS:
        raise Phase1ConfigError(f"domain must be one of {PHASE1_DOMAINS}")
    if method not in PHASE1_METHODS:
        raise Phase1ConfigError(f"method must be one of {PHASE1_METHODS}")
    launch = data.get("launch")
    tuning_mode = (
        "paper" if launch is None else str(_mapping(launch, "launch").get("tuning_mode", "paper"))
    )
    if tuning_mode not in {"paper", "custom"}:
        raise Phase1ConfigError("launch.tuning_mode must be paper or custom")
    if tuning_mode == "paper":
        _require_equal(seed, 11, "seed")
    else:
        _positive_int(seed, "seed", minimum=0)

    data_config = _mapping(data.get("data"), "data")
    train_prompt_count = _positive_int(
        data_config.get("train_prompt_count"), "data.train_prompt_count"
    )
    if tuning_mode == "paper":
        _require_equal(train_prompt_count, 1500, "data.train_prompt_count")
    probe = _mapping(data_config.get("fixed_train_probe"), "data.fixed_train_probe")
    probe_count = _positive_int(probe.get("count"), "data.fixed_train_probe.count")
    _require_equal(probe_count, 100, "data.fixed_train_probe.count")
    if probe_count > train_prompt_count:
        raise Phase1ConfigError("data.fixed_train_probe.count cannot exceed train_prompt_count")
    _require_equal(int(probe.get("sample_seed", -1)), seed, "data.fixed_train_probe.sample_seed")
    _require_equal(str(probe.get("source", "")), "train", "data.fixed_train_probe.source")
    _require_equal(probe.get("remains_in_training"), True, "probe.remains_in_training")
    _require_false(probe, "extra_responses_used_for_gradient", "data.fixed_train_probe")
    heldout = _mapping(data_config.get("in_domain_policy_eval"), "data.in_domain_policy_eval")
    _require_equal(int(heldout.get("count", 0)), 300, "data.in_domain_policy_eval.count")
    _require_equal(heldout.get("policy_only"), True, "data.in_domain_policy_eval.policy_only")
    external = _mapping(data_config.get("external_policy_eval"), "data.external_policy_eval")
    expected_external = "HealthBench" if domain == "medicine" else "GPQA-Diamond"
    _require_equal(str(external.get("name", "")), expected_external, "external policy eval")
    _require_equal(external.get("policy_only"), True, "data.external_policy_eval.policy_only")

    training = _mapping(data.get("training"), "training")
    _require_equal(str(training.get("algorithm", "")), "grpo", "training.algorithm")
    epochs = _positive_int(training.get("epochs"), "training.epochs")
    global_prompt_batch = _positive_int(
        training.get("global_prompt_batch"), "training.global_prompt_batch"
    )
    expected_global_steps = _positive_int(
        training.get("expected_global_steps"), "training.expected_global_steps"
    )
    if tuning_mode == "paper":
        _require_equal(epochs, 3, "training.epochs")
        _require_equal(global_prompt_batch, 96, "training.global_prompt_batch")
        _require_equal(expected_global_steps, 48, "training.expected_global_steps")
    _require_equal(
        training.get("checkpoint_interval_steps"),
        1,
        "training.checkpoint_interval_steps",
    )
    retention = _mapping(training.get("checkpoint_retention"), "training.checkpoint_retention")
    _require_equal(
        retention.get("latest_full_resume_state"),
        True,
        "training.checkpoint_retention.latest_full_resume_state",
    )
    _require_equal(
        retention.get("older_parameter_snapshots"),
        True,
        "training.checkpoint_retention.older_parameter_snapshots",
    )
    _require_equal(
        retention.get("older_optimizer_state"),
        False,
        "training.checkpoint_retention.older_optimizer_state",
    )
    audit_checkpoints = tuple(training.get("audit_checkpoints", ()))
    reuse_anchors = tuple(training.get("reuse_anchors", ()))
    if tuning_mode == "paper":
        _require_equal(audit_checkpoints, AUDIT_CHECKPOINTS, "audit checkpoints")
        _require_equal(reuse_anchors, REUSE_ANCHORS, "reuse anchors")
    else:
        for name, values in (
            ("training.audit_checkpoints", audit_checkpoints),
            ("training.reuse_anchors", reuse_anchors),
        ):
            if (
                not values
                or any(isinstance(step, bool) or not isinstance(step, int) for step in values)
                or tuple(sorted(set(values))) != values
                or values[0] != 0
                or values[-1] > expected_global_steps
            ):
                raise Phase1ConfigError(
                    f"{name} must be sorted unique integer steps from 0 through expected_global_steps"
                )
    _require_equal(training.get("probe_responses_used_for_gradient"), False, "probe gradient use")
    logged = set(_sequence(training.get("per_step_logging"), "training.per_step_logging"))
    missing_logs = sorted(set(STEP_LOG_FIELDS) - logged)
    if missing_logs:
        raise Phase1ConfigError(f"training.per_step_logging is missing {missing_logs}")

    models = _mapping(data.get("models"), "models")
    _validate_model(
        _mapping(models.get("policy"), "models.policy"),
        name="models.policy",
        expected_model=POLICY_MODEL,
        expected_revision=POLICY_REVISION,
        non_reasoning=True,
    )

    analysis = _mapping(data.get("analysis"), "analysis")
    _require_equal(str(analysis.get("primary_dataset", "")), "fixed_train_probe", "primary dataset")
    _require_equal(analysis.get("same_response_pool_required"), True, "same response pool")
    _require_equal(analysis.get("same_judge_within_method"), True, "same judge")
    _require_equal(analysis.get("heldout_used_for_update_timing"), False, "heldout timing use")
    _require_equal(
        analysis.get("compare_cross_method_absolute_rewards"),
        False,
        "cross-method rewards",
    )
    _require_equal(analysis.get("compare_cross_method_absolute_zar"), False, "cross-method ZAR")
    _require_equal(analysis.get("use_response_level_ground_truth"), False, "response GT")
    _require_equal(analysis.get("use_initial_rubric_as_ground_truth"), False, "initial rubric GT")
    _require_equal(analysis.get("enable_bon"), False, "BoN")
    _require_equal(analysis.get("pool_a_b_disjoint"), True, "Pool A/B separation")
    _require_equal(
        tuple(analysis.get("reuse_anchors", ())), reuse_anchors, "analysis reuse anchors"
    )
    required_state = set(PRE_UPDATE_STATE_FIELDS)
    actual_state = set(_sequence(analysis.get("pre_update_state_fields"), "pre-update fields"))
    if missing := sorted(required_state - actual_state):
        raise Phase1ConfigError(f"analysis.pre_update_state_fields is missing {missing}")
    for key in ("epsilon_z", "epsilon_t", "practical_margin_delta_d"):
        value = float(analysis.get(key, -1.0))
        if not math.isfinite(value) or value < 0.0:
            raise Phase1ConfigError(f"analysis.{key} must be finite and non-negative")

    if method == "online_rubrics":
        online = _mapping(data.get("online_rubrics"), "online_rubrics")
        _validate_model(
            _mapping(models.get("extractor"), "models.extractor"),
            name="models.extractor",
            expected_model=GPT_OSS_MODEL,
        )
        _validate_model(
            _mapping(models.get("deduplicator"), "models.deduplicator"),
            name="models.deduplicator",
            expected_model=GPT_OSS_MODEL,
        )
        _validate_model(
            _mapping(models.get("judge"), "models.judge"),
            name="models.judge",
            expected_model=RUBRIC_JUDGE_MODEL,
            expected_revision=RUBRIC_JUDGE_REVISION,
            non_reasoning=True,
        )
        _require_equal(online.get("policy_responses_per_prompt"), 16, "online responses")
        _require_equal(online.get("elicitation_pairs_per_prompt"), 8, "online pairs")
        _require_equal(str(online.get("control_policy", "")), "pi_0_fixed", "online control")
        _require_equal(str(online.get("augmentation", "")), "step_local", "online augmentation")
        _require_equal(online.get("accumulate_previous_criteria"), False, "online accumulation")
        _require_equal(online.get("fresh_reward_used_for_training"), True, "online fresh reward")
        _require_equal(online.get("stale_shadow_only"), True, "online stale shadow")
        _require_equal(
            str(online.get("stale_lookup_key", "")),
            "prompt_id",
            "online stale lookup key",
        )
        _require_equal(online.get("require_prompt_match"), True, "online prompt match")
        _require_equal(int(online.get("pool_a_current_count", 0)), 8, "online Pool-A current")
        _require_equal(int(online.get("pool_a_pi0_count", 0)), 8, "online Pool-A pi0")
        _require_equal(int(online.get("pool_b_count", 0)), 16, "online Pool-B")
        _require_equal(
            str(online.get("control_cache", "")),
            "precomputed_immutable",
            "online control cache",
        )
    else:
        evo = _mapping(data.get("evorubrics"), "evorubrics")
        _validate_model(
            _mapping(models.get("shared_backbone"), "models.shared_backbone"),
            name="models.shared_backbone",
            expected_model=POLICY_MODEL,
            expected_revision=POLICY_REVISION,
            non_reasoning=True,
        )
        _validate_model(
            _mapping(models.get("judge"), "models.judge"),
            name="models.judge",
            expected_model=GPT_OSS_MODEL,
        )
        _require_equal(str(evo.get("policy_adapter", "")), "theta", "Evo policy adapter")
        _require_equal(str(evo.get("rubric_generator_adapter", "")), "psi", "Evo generator adapter")
        policy_responses_m = _positive_int(evo.get("policy_responses_m"), "Evo M")
        rubric_sets_n = _positive_int(evo.get("rubric_sets_n"), "Evo N")
        if tuning_mode == "paper":
            _require_equal(policy_responses_m, 4, "Evo M")
            _require_equal(rubric_sets_n, 4, "Evo N")
        _require_equal(str(evo.get("dataset_mode", "")), "open_rubrics", "Evo dataset mode")
        _require_equal(str(evo.get("training_flow", "")), "unified", "Evo training flow")
        reward_weights = _mapping(evo.get("reward_weights"), "Evo reward weights")
        expected_reward_weights = {
            "similarity": 0.25,
            "discrimination": 0.25,
            "diversity": 0.25,
            "reflect": 0.25,
        }
        _require_equal(
            set(reward_weights),
            set(expected_reward_weights),
            "Evo reward weight names",
        )
        if tuning_mode == "paper":
            _require_equal(int(evo.get("lora_rank", 0)), 32, "Evo LoRA rank")
            _require_equal(int(evo.get("lora_alpha", 0)), 64, "Evo LoRA alpha")
            for name, expected in (
                ("policy_learning_rate", 2.0e-5),
                ("rubric_generator_learning_rate", 5.0e-6),
                ("kl_loss_coefficient", 1.0e-4),
                ("generation_temperature", 0.7),
            ):
                _require_equal(float(evo.get(name, 0.0)), expected, f"Evo {name}")
            for name, expected in expected_reward_weights.items():
                _require_equal(
                    float(reward_weights.get(name, -1.0)), expected, f"Evo {name} reward weight"
                )
        else:
            for name in ("lora_rank", "lora_alpha"):
                value = evo.get(name)
                if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 256:
                    raise Phase1ConfigError(f"evorubrics.{name} must be an integer in [1, 256]")
            for name in ("policy_learning_rate", "rubric_generator_learning_rate"):
                if _bounded_float(evo.get(name), f"evorubrics.{name}", minimum=0.0) == 0.0:
                    raise Phase1ConfigError(f"evorubrics.{name} must be positive")
            _bounded_float(
                evo.get("kl_loss_coefficient"), "evorubrics.kl_loss_coefficient", minimum=0.0
            )
            _bounded_float(
                evo.get("generation_temperature"),
                "evorubrics.generation_temperature",
                minimum=0.0,
                maximum=2.0,
            )
            weights = [
                _bounded_float(
                    reward_weights[name],
                    f"evorubrics.reward_weights.{name}",
                    minimum=0.0,
                    maximum=1.0,
                )
                for name in expected_reward_weights
            ]
            if not math.isclose(sum(weights), 1.0, abs_tol=1.0e-6):
                raise Phase1ConfigError("evorubrics.reward_weights must sum to 1")
        _require_equal(
            evo.get("reflect_use_golden_rubrics"),
            True,
            "Evo golden-rubric reflection",
        )
        _require_equal(evo.get("save_policy_and_generator"), True, "Evo checkpoints")
        _require_equal(evo.get("fixed_rubric_generation_seeds"), True, "Evo fixed seeds")
        _require_equal(evo.get("fixed_anchor_is_ground_truth"), False, "Evo anchor GT")
        pool_b_count = _positive_int(evo.get("pool_b_count"), "Evo Pool-B")
        _require_equal(pool_b_count, policy_responses_m * rubric_sets_n, "Evo Pool-B")
        seeds = tuple(_sequence(evo.get("rubric_generation_seeds"), "Evo seeds"))
        if len(seeds) != rubric_sets_n or len(set(seeds)) != rubric_sets_n:
            raise Phase1ConfigError(
                "Evo rubric_generation_seeds must contain rubric_sets_n unique seeds"
            )

    infrastructure = _mapping(data.get("infrastructure"), "infrastructure")
    optimizer = _mapping(infrastructure.get("optimizer"), "infrastructure.optimizer")
    trainer_gpus = optimizer.get("gpus")
    services = _mapping(infrastructure.get("services"), "infrastructure.services")
    gpt_oss = _mapping(services.get("gpt_oss_120b"), "services.gpt_oss_120b")
    gpt_instances = tuple(
        _mapping(value, "services.gpt_oss_120b.instances[]")
        for value in _sequence(gpt_oss.get("instances"), "services.gpt_oss_120b.instances")
    )
    qwen_judge = _mapping(services.get("qwen3_32b"), "services.qwen3_32b")
    qwen_instances = tuple(
        _mapping(value, "services.qwen3_32b.instances[]")
        for value in _sequence(qwen_judge.get("instances"), "services.qwen3_32b.instances")
    )
    pi0_control = _mapping(infrastructure.get("pi0_control"), "infrastructure.pi0_control")
    # Host labels are optional documentation, never a machine identity constraint.
    if launch is None:
        _require_equal(tuple(trainer_gpus or ()), (1,), "optimizer GPUs")
        _require_equal(len(gpt_instances), 1, "gpt-oss instance count")
        _require_equal(
            tuple(tuple(v.get("gpus", ())) for v in gpt_instances),
            ((0, 1),),
            "gpt-oss instance GPUs",
        )
        _require_equal(
            tuple(int(v.get("tensor_parallel_size", 0)) for v in gpt_instances),
            (2,),
            "gpt-oss instance TP",
        )
        _require_equal(len(qwen_instances), 1, "Qwen judge instance count")
        _require_equal(
            tuple(tuple(v.get("gpus", ())) for v in qwen_instances),
            ((0, 1),),
            "Qwen judge instance GPUs",
        )
        _require_equal(
            tuple(int(v.get("tensor_parallel_size", 0)) for v in qwen_instances),
            (2,),
            "Qwen judge instance TP",
        )
        _require_equal(tuple(pi0_control.get("gpus", ())), (1,), "pi0 control GPUs")
    else:
        selected_gpus = _gpu_list(trainer_gpus, "infrastructure.optimizer.gpus")
        if method == "evorubrics" and len(selected_gpus) != 1:
            raise Phase1ConfigError(
                "EvoRubrics currently requires one optimizer GPU because distributed "
                "optimizer checkpoints are not resumable"
            )
        for service_name, instances in (("gpt-oss", gpt_instances), ("Qwen judge", qwen_instances)):
            if not 1 <= len(instances) <= 2:
                raise Phase1ConfigError(f"{service_name} must define one or two instances")
            for index, instance in enumerate(instances):
                gpus = _gpu_list(instance.get("gpus"), f"{service_name} instance {index} GPUs")
                tp = _positive_int(
                    instance.get("tensor_parallel_size"), f"{service_name} instance {index} TP"
                )
                if tp != len(gpus):
                    raise Phase1ConfigError(
                        f"{service_name} instance {index} tensor_parallel_size must equal its GPU count"
                    )
        _gpu_list(pi0_control.get("gpus"), "pi0 control GPUs")
    _require_equal(
        str(gpt_oss.get("base_url_env", "")), "PHASE1_GPT_OSS_BASE_URLS", "gpt-oss endpoint env"
    )
    _require_equal(
        str(qwen_judge.get("base_url_env", "")),
        "PHASE1_QWEN32B_BASE_URLS",
        "Qwen judge endpoint env",
    )
    _require_equal(
        str(pi0_control.get("mode", "")), "precompute_before_optimizer", "pi0 control mode"
    )
    _require_equal(pi0_control.get("immutable_cache"), True, "pi0 immutable cache")

    output = _mapping(data.get("output"), "output")
    output_root = str(output.get("root", "")).strip()
    if not output_root:
        raise Phase1ConfigError("output.root must be a non-empty path")
    if launch is None:
        _require_equal(output_root, "outputs", "output.root")
    _require_equal(
        str(output.get("layout", "")),
        "{domain}/{method}/seed-{seed}/{run_id}",
        "output.layout",
    )
    tracking = _mapping(data.get("tracking"), "tracking")
    _require_equal(
        str(tracking.get("project", "")),
        "phase1_dynamic_evaluator_updates",
        "tracking.project",
    )
    if str(tracking.get("project")) == str(tracking.get("legacy_static_project")):
        raise Phase1ConfigError("Phase-1 tracking project must differ from legacy static GRPO")

    return Phase1Config(
        source_path=source_path,
        raw=dict(data),
        experiment=experiment,
        domain=domain,
        method=method,
        seed=seed,
    )


def load_phase1_config(path: str | Path) -> Phase1Config:
    source_path = Path(path).expanduser().resolve()
    data = load_yaml_config(source_path)
    return validate_phase1_mapping(data, source_path=source_path)
