"""Immutable identities and inventories for paper-faithful OnlineRubrics steps."""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..artifacts import read_json
from ..hashing import sha256_json


CURRENT_ROLLOUT_COUNT = 16
CONTROL_ROLLOUT_COUNT = 8
ELICITATION_PAIR_COUNT = 8


class OnlineContractError(ValueError):
    """Raised before an incomplete or ambiguous online step can be rewarded."""


class StepState(str, Enum):
    INVENTORY_VALIDATED = "inventory_validated"
    EXTRACTIONS_SEALED = "extractions_sealed"
    DEDUP_SEALED = "dedup_sealed"
    GRADES_SEALED = "grades_sealed"
    PRE_UPDATE_SEALED = "pre_update_sealed"
    COMMITTED = "committed"


def _required(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise OnlineContractError(f"{name} must be a non-empty string")


def _fraction(value: int | float | str) -> Fraction:
    if isinstance(value, bool):
        raise OnlineContractError("criterion weights must be numeric, not boolean")
    try:
        result = Fraction(str(value))
    except (ValueError, ZeroDivisionError) as error:
        raise OnlineContractError(f"invalid criterion weight: {value!r}") from error
    if not math.isfinite(float(result)):
        raise OnlineContractError("criterion weights must be finite")
    return result


@dataclass(frozen=True, slots=True)
class PolicySnapshot:
    policy_version: int
    content_hash: str
    model: str
    revision: str

    def __post_init__(self) -> None:
        if self.policy_version < 0:
            raise OnlineContractError("policy_version must be non-negative")
        for name in ("content_hash", "model", "revision"):
            _required(name, getattr(self, name))


@dataclass(frozen=True, slots=True)
class PromptOccurrence:
    run_id: str
    optimizer_update_index: int
    batch_uid: str
    source_row_id: str
    prompt_id: str
    prompt_occurrence_id: str
    prompt: tuple[Mapping[str, str], ...]

    def __post_init__(self) -> None:
        for name in (
            "run_id",
            "batch_uid",
            "source_row_id",
            "prompt_id",
            "prompt_occurrence_id",
        ):
            _required(name, getattr(self, name))
        if self.optimizer_update_index < 1:
            raise OnlineContractError("optimizer_update_index is one-based and must be positive")
        if not self.prompt:
            raise OnlineContractError("prompt must contain at least one message")
        for message in self.prompt:
            if message.get("role") not in {"system", "user", "assistant"}:
                raise OnlineContractError("prompt message has an unsupported role")
            _required("prompt message content", str(message.get("content", "")))

    @property
    def content_hash(self) -> str:
        return sha256_json(dataclasses.asdict(self))


@dataclass(frozen=True, slots=True)
class ResponseRecord:
    prompt_occurrence_id: str
    response_id: str
    rollout_index: int
    text: str
    policy: PolicySnapshot
    family: str

    def __post_init__(self) -> None:
        for name in ("prompt_occurrence_id", "response_id", "text", "family"):
            _required(name, getattr(self, name))
        if self.rollout_index < 0:
            raise OnlineContractError("rollout_index must be non-negative")

    @property
    def text_hash(self) -> str:
        return sha256_json(self.text)


@dataclass(frozen=True, slots=True)
class WeightedCriterion:
    criterion_id: str
    text: str
    weight: int | float
    source: str

    def __post_init__(self) -> None:
        for name in ("criterion_id", "text", "source"):
            _required(name, getattr(self, name))
        _fraction(self.weight)
        if self.source == "online_pairwise" and (
            isinstance(self.weight, bool) or not isinstance(self.weight, int) or self.weight <= 0
        ):
            raise OnlineContractError("online criterion weights must be positive integers")


@dataclass(frozen=True, slots=True)
class PromptGroupInput:
    occurrence: PromptOccurrence
    offline_criteria: tuple[WeightedCriterion, ...]
    current_responses: tuple[ResponseRecord, ...]
    control_responses: tuple[ResponseRecord, ...]

    def __post_init__(self) -> None:
        if not self.offline_criteria:
            raise OnlineContractError("offline rubric must not be empty")
        validate_criterion_inventory(self.offline_criteria)
        validate_response_inventory(
            self.current_responses,
            occurrence_id=self.occurrence.prompt_occurrence_id,
            expected_count=CURRENT_ROLLOUT_COUNT,
            expected_family="current",
        )
        validate_response_inventory(
            self.control_responses,
            occurrence_id=self.occurrence.prompt_occurrence_id,
            expected_count=CONTROL_ROLLOUT_COUNT,
            expected_family="control",
        )


@dataclass(frozen=True, slots=True)
class OnlineStepInput:
    run_id: str
    optimizer_update_index: int
    batch_uid: str
    seed: int
    prompt_groups: tuple[PromptGroupInput, ...]

    def __post_init__(self) -> None:
        _required("run_id", self.run_id)
        _required("batch_uid", self.batch_uid)
        if self.optimizer_update_index < 1 or self.seed < 0:
            raise OnlineContractError("step index must be positive and seed non-negative")
        if not self.prompt_groups:
            raise OnlineContractError("an online step must contain at least one prompt group")
        occurrence_ids = [g.occurrence.prompt_occurrence_id for g in self.prompt_groups]
        if len(occurrence_ids) != len(set(occurrence_ids)):
            raise OnlineContractError("prompt occurrence IDs must be unique within a batch")
        for group in self.prompt_groups:
            occurrence = group.occurrence
            if (
                occurrence.run_id != self.run_id
                or occurrence.optimizer_update_index != self.optimizer_update_index
                or occurrence.batch_uid != self.batch_uid
            ):
                raise OnlineContractError("prompt occurrence is bound to a different online step")


@dataclass(frozen=True, slots=True)
class RubricUnion:
    prompt_occurrence_id: str
    offline_criteria: tuple[WeightedCriterion, ...]
    online_criteria: tuple[WeightedCriterion, ...]

    def __post_init__(self) -> None:
        _required("prompt_occurrence_id", self.prompt_occurrence_id)
        if not self.offline_criteria:
            raise OnlineContractError("rubric union requires offline criteria")
        validate_criterion_inventory(self.criteria)

    @property
    def criteria(self) -> tuple[WeightedCriterion, ...]:
        return self.offline_criteria + self.online_criteria

    @property
    def content_hash(self) -> str:
        return sha256_json([dataclasses.asdict(item) for item in self.criteria])


@dataclass(frozen=True, slots=True)
class RewardReceipt:
    prompt_occurrence_id: str
    response_id: str
    rollout_index: int
    grades: tuple[tuple[str, int], ...]
    numerator: int | float
    denominator: int | float
    reward: float
    rubric_hash: str

    def __post_init__(self) -> None:
        _required("prompt_occurrence_id", self.prompt_occurrence_id)
        _required("response_id", self.response_id)
        _required("rubric_hash", self.rubric_hash)
        if self.rollout_index < 0 or self.denominator <= 0:
            raise OnlineContractError("reward receipt has invalid index or denominator")
        if any(grade not in {0, 1} for _, grade in self.grades):
            raise OnlineContractError("reward receipt grades must be binary")


@dataclass(frozen=True, slots=True)
class OnlineStepManifest:
    schema_version: int
    run_id: str
    optimizer_update_index: int
    batch_uid: str
    state: StepState
    prompt_occurrence_ids: tuple[str, ...]
    current_response_count: int
    control_response_count: int
    extraction_count: int
    dedup_count: int
    grader_count: int
    reward_count: int
    artifacts: Mapping[str, str]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise OnlineContractError("unsupported online step manifest schema")
        _required("run_id", self.run_id)
        _required("batch_uid", self.batch_uid)
        if self.optimizer_update_index < 1 or not self.prompt_occurrence_ids:
            raise OnlineContractError("manifest has invalid step or empty prompt inventory")
        prompt_count = len(self.prompt_occurrence_ids)
        expected = {
            "current_response_count": prompt_count * CURRENT_ROLLOUT_COUNT,
            "control_response_count": prompt_count * CONTROL_ROLLOUT_COUNT,
            "extraction_count": prompt_count * ELICITATION_PAIR_COUNT,
            "dedup_count": prompt_count,
            "grader_count": prompt_count * CURRENT_ROLLOUT_COUNT,
            "reward_count": prompt_count * CURRENT_ROLLOUT_COUNT,
        }
        if self.state not in {StepState.PRE_UPDATE_SEALED, StepState.COMMITTED}:
            raise OnlineContractError("persisted manifest must be pre-update sealed or committed")
        for name, digest in self.artifacts.items():
            _required("artifact name", name)
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise OnlineContractError("artifact values must be lowercase SHA-256 digests")
        for field_name, expected_count in expected.items():
            if getattr(self, field_name) != expected_count:
                raise OnlineContractError(
                    f"{field_name} must equal {expected_count}, got {getattr(self, field_name)}"
                )

    @property
    def content_hash(self) -> str:
        return sha256_json(dataclasses.asdict(self))


def validate_response_inventory(
    responses: Sequence[ResponseRecord],
    *,
    occurrence_id: str,
    expected_count: int,
    expected_family: str,
) -> None:
    if len(responses) != expected_count:
        raise OnlineContractError(
            f"{expected_family} inventory must contain exactly {expected_count} responses"
        )
    indexes = [response.rollout_index for response in responses]
    if indexes != list(range(expected_count)):
        raise OnlineContractError(
            f"{expected_family} rollout indexes must be canonical 0..{expected_count - 1}"
        )
    ids = [response.response_id for response in responses]
    if len(ids) != len(set(ids)):
        raise OnlineContractError(f"{expected_family} response IDs must be unique")
    policies = {response.policy for response in responses}
    if len(policies) != 1:
        raise OnlineContractError(f"{expected_family} inventory must use one policy snapshot")
    for response in responses:
        if response.prompt_occurrence_id != occurrence_id:
            raise OnlineContractError("response is bound to a different prompt occurrence")
        if response.family != expected_family:
            raise OnlineContractError("response family does not match its inventory")


def validate_criterion_inventory(criteria: Sequence[WeightedCriterion]) -> None:
    ids = [item.criterion_id for item in criteria]
    normalized = [" ".join(item.text.casefold().split()) for item in criteria]
    if len(ids) != len(set(ids)):
        raise OnlineContractError("criterion IDs must be unique")
    if len(normalized) != len(set(normalized)):
        raise OnlineContractError("normalized criterion texts must be unique")


def manifest_from_mapping(value: Mapping[str, Any]) -> OnlineStepManifest:
    try:
        state = StepState(str(value["state"]))
        return OnlineStepManifest(
            schema_version=int(value["schema_version"]),
            run_id=str(value["run_id"]),
            optimizer_update_index=int(value["optimizer_update_index"]),
            batch_uid=str(value["batch_uid"]),
            state=state,
            prompt_occurrence_ids=tuple(str(item) for item in value["prompt_occurrence_ids"]),
            current_response_count=int(value["current_response_count"]),
            control_response_count=int(value["control_response_count"]),
            extraction_count=int(value["extraction_count"]),
            dedup_count=int(value["dedup_count"]),
            grader_count=int(value["grader_count"]),
            reward_count=int(value["reward_count"]),
            artifacts=dict(value["artifacts"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise OnlineContractError("malformed online step manifest") from error


def validate_online_step_manifest(
    value_or_path: Mapping[str, Any] | str | Path,
) -> OnlineStepManifest:
    value = read_json(value_or_path) if isinstance(value_or_path, (str, Path)) else value_or_path
    return manifest_from_mapping(value)
