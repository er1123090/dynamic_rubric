"""Small immutable records exchanged between experiment stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .hashing import sha256_json


def _required(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


@dataclass(frozen=True, slots=True)
class Prompt:
    prompt_id: str
    messages: tuple[Mapping[str, str], ...]
    split: str

    def __post_init__(self) -> None:
        _required("prompt_id", self.prompt_id)
        _required("split", self.split)
        if not self.messages:
            raise ValueError("messages must not be empty")


@dataclass(frozen=True, slots=True)
class Response:
    response_id: str
    prompt_id: str
    text: str
    family: str
    policy_step: int
    sample_index: int
    seed: int

    def __post_init__(self) -> None:
        _required("response_id", self.response_id)
        _required("prompt_id", self.prompt_id)
        _required("family", self.family)
        if self.policy_step < 0 or self.sample_index < 0 or self.seed < 0:
            raise ValueError("policy_step, sample_index, and seed must be non-negative")


@dataclass(frozen=True, slots=True)
class PolicyCheckpoint:
    policy_id: str
    step: int
    base_model: str
    revision: str
    semantics: str = "after_optimizer_update"

    def __post_init__(self) -> None:
        _required("policy_id", self.policy_id)
        _required("base_model", self.base_model)
        _required("revision", self.revision)
        if self.step < 0:
            raise ValueError("step must be non-negative")
        if self.semantics != "after_optimizer_update" and self.step != 0:
            raise ValueError("trained checkpoints must use after_optimizer_update semantics")


@dataclass(frozen=True, slots=True)
class Criterion:
    criterion_id: str
    text: str
    source: str
    weight: float = 1.0
    created_step: int = 0
    last_validated_step: int = 0

    def __post_init__(self) -> None:
        _required("criterion_id", self.criterion_id)
        _required("text", self.text)
        _required("source", self.source)
        if self.weight <= 0:
            raise ValueError("weight must be positive")
        if min(self.created_step, self.last_validated_step) < 0:
            raise ValueError("criterion steps must be non-negative")


@dataclass(frozen=True, slots=True)
class RubricSnapshot:
    rubric_id: str
    prompt_id: str
    trajectory: str
    policy_step: int
    criteria: tuple[Criterion, ...]

    def __post_init__(self) -> None:
        _required("rubric_id", self.rubric_id)
        _required("prompt_id", self.prompt_id)
        _required("trajectory", self.trajectory)
        if self.policy_step < 0 or not self.criteria:
            raise ValueError("policy_step must be non-negative and criteria non-empty")
        ids = [criterion.criterion_id for criterion in self.criteria]
        if len(ids) != len(set(ids)):
            raise ValueError("criterion IDs must be unique")

    @property
    def content_hash(self) -> str:
        return sha256_json([(item.criterion_id, item.text, item.weight) for item in self.criteria])


@dataclass(frozen=True, slots=True)
class CandidateCriterion:
    criterion: Criterion
    satisfaction_rate: float
    current_reference_separation: float
    max_active_similarity: float
    parse_success: float
    independent_validation: bool
    utility: float | None = None

    def __post_init__(self) -> None:
        for name in ("satisfaction_rate", "max_active_similarity", "parse_success"):
            value = getattr(self, name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class AdmissionDecision:
    prompt_id: str
    policy_step: int
    admitted_criterion_id: str | None
    reason: str
    evidence: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CriterionScore:
    response_id: str
    criterion_id: str
    probability_yes: float
    parsed: bool = True

    def __post_init__(self) -> None:
        if not 0 <= self.probability_yes <= 1:
            raise ValueError("probability_yes must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class RunManifest:
    run_id: str
    stage: str
    config_hash: str
    input_hashes: Mapping[str, str]
    model_identities: Mapping[str, Any] = field(default_factory=dict)
    prompt_hashes: Mapping[str, str] = field(default_factory=dict)
    schema_hashes: Mapping[str, str] = field(default_factory=dict)
    seed_namespaces: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    manifest_version: int = 1

    def __post_init__(self) -> None:
        _required("run_id", self.run_id)
        _required("stage", self.stage)
        _required("config_hash", self.config_hash)
        if self.manifest_version != 1:
            raise ValueError("unsupported manifest version")

    @property
    def compatibility_hash(self) -> str:
        return sha256_json(
            {
                "version": self.manifest_version,
                "run_id": self.run_id,
                "stage": self.stage,
                "config_hash": self.config_hash,
                "input_hashes": self.input_hashes,
                "model_identities": self.model_identities,
                "prompt_hashes": self.prompt_hashes,
                "schema_hashes": self.schema_hashes,
                "seed_namespaces": self.seed_namespaces,
            }
        )


def criteria_tuple(items: Sequence[Criterion]) -> tuple[Criterion, ...]:
    return tuple(items)
