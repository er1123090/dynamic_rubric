"""Immutable contracts shared by horizon extraction, grading, and reward code."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
import json
import re
from typing import Mapping, Sequence


class HorizonContractError(ValueError):
    """Raised when an artifact violates the versioned horizon contract."""


class ImportanceClass(str, Enum):
    ESSENTIAL = "essential"
    IMPORTANT = "important"
    OPTIONAL = "optional"
    PITFALL = "pitfall"


class CriterionType(str, Enum):
    QUALITY = "quality"
    PITFALL = "pitfall"


WEIGHT_UNITS: Mapping[ImportanceClass, int] = {
    ImportanceClass.ESSENTIAL: 10,
    ImportanceClass.IMPORTANT: 7,
    ImportanceClass.OPTIONAL: 3,
    ImportanceClass.PITFALL: 9,
}

_SPACE_RE = re.compile(r"\s+")


def normalized_criterion_text(text: str) -> str:
    normalized = _SPACE_RE.sub(" ", text).strip().rstrip(".").casefold()
    if not normalized:
        raise HorizonContractError("criterion text must not be empty")
    return normalized


def criterion_content_hash(text: str) -> str:
    return sha256(normalized_criterion_text(text).encode("utf-8")).hexdigest()


def make_criterion_instance_id(
    *, prompt_id: str, checkpoint_id: str, canonical_criterion_hash: str
) -> str:
    """Build checkpoint-scoped identity distinct from content identity."""

    fields = (prompt_id.strip(), checkpoint_id.strip(), canonical_criterion_hash.strip())
    if any(not field for field in fields):
        raise HorizonContractError("instance identity fields must not be empty")
    encoded = json.dumps(fields, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return f"hc_{sha256(encoded).hexdigest()}"


@dataclass(frozen=True, slots=True)
class WeightedCriterion:
    criterion_instance_id: str
    canonical_criterion_hash: str
    text: str
    importance_class: ImportanceClass
    criterion_type: CriterionType
    weight_units: int
    source_candidate_ids: tuple[str, ...] = ()
    source_checkpoint: str | None = None
    raw_paper_weight: int | None = None
    distinct_source_pair_support: int = 0

    def __post_init__(self) -> None:
        if not self.criterion_instance_id.strip():
            raise HorizonContractError("criterion_instance_id must not be empty")
        expected_hash = criterion_content_hash(self.text)
        if self.canonical_criterion_hash != expected_hash:
            raise HorizonContractError("canonical_criterion_hash does not match criterion text")
        if self.weight_units != WEIGHT_UNITS[self.importance_class]:
            raise HorizonContractError("weight_units does not match importance_class")
        if self.criterion_type is CriterionType.PITFALL:
            if self.importance_class is not ImportanceClass.PITFALL:
                raise HorizonContractError("pitfall criterion must use pitfall importance")
        elif self.importance_class is ImportanceClass.PITFALL:
            raise HorizonContractError("quality criterion cannot use pitfall importance")
        if len(set(self.source_candidate_ids)) != len(self.source_candidate_ids):
            raise HorizonContractError("source_candidate_ids must be unique")
        if self.raw_paper_weight is not None and (
            isinstance(self.raw_paper_weight, bool)
            or not isinstance(self.raw_paper_weight, int)
            or self.raw_paper_weight < 1
        ):
            raise HorizonContractError("raw_paper_weight must be a positive integer")
        if (
            isinstance(self.distinct_source_pair_support, bool)
            or not isinstance(self.distinct_source_pair_support, int)
            or self.distinct_source_pair_support < 0
        ):
            raise HorizonContractError(
                "distinct_source_pair_support must be a non-negative integer"
            )


@dataclass(frozen=True, slots=True)
class WeightedRubric:
    prompt_id: str
    criteria: tuple[WeightedCriterion, ...]
    rubric_id: str = ""

    def __post_init__(self) -> None:
        if not self.prompt_id.strip():
            raise HorizonContractError("prompt_id must not be empty")
        if not self.criteria:
            raise HorizonContractError("weighted rubric must contain criteria")
        instance_ids = tuple(item.criterion_instance_id for item in self.criteria)
        if len(set(instance_ids)) != len(instance_ids):
            raise HorizonContractError("criterion instance IDs must be unique within a rubric")

    @property
    def content_hash(self) -> str:
        payload = [(item.canonical_criterion_hash, item.weight_units) for item in self.criteria]
        encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        return sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class RationalScore:
    """Exact weighted hard-binary score; ``value`` is display-only."""

    numerator: int
    denominator: int

    def __post_init__(self) -> None:
        if self.denominator <= 0:
            raise HorizonContractError("score denominator must be positive")
        if not 0 <= self.numerator <= self.denominator:
            raise HorizonContractError("score numerator must be in [0, denominator]")

    @property
    def value(self) -> float:
        return self.numerator / self.denominator

    @classmethod
    def from_grades(
        cls,
        criteria: Sequence[WeightedCriterion],
        grades: Mapping[str, int],
    ) -> "RationalScore":
        if not criteria:
            raise HorizonContractError("cannot score an empty rubric")
        expected = {criterion.criterion_instance_id for criterion in criteria}
        missing = expected.difference(grades)
        if missing:
            raise HorizonContractError(f"missing criterion grades: {sorted(missing)}")
        numerator = 0
        denominator = 0
        for criterion in criteria:
            grade = grades[criterion.criterion_instance_id]
            if isinstance(grade, bool) or grade not in (0, 1):
                raise HorizonContractError("hard criterion grades must be integer 0 or 1")
            numerator += criterion.weight_units * grade
            denominator += criterion.weight_units
        return cls(numerator=numerator, denominator=denominator)
