"""Criterion-wise hard grading, identity-safe caching, and score assembly."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
import math
from typing import Callable, Mapping, Sequence

from dynamic_rubric.horizon.contracts import RationalScore, WeightedRubric


YES_TARGET = " YES"
NO_TARGET = " NO"
TARGET_ENCODING_VERSION = "yes_no_prompt_logprob_v2"


class GradingContractError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class HardGrade:
    parse_status: str
    public_label: str | None
    grade: int | None
    probability_present: float
    yes_logprob: float
    no_logprob: float
    retry_count: int = 0

    @property
    def valid(self) -> bool:
        return self.parse_status in {"ok", "ambiguous_target_tie"} and self.grade in {0, 1}


def _full_target_logprob(values: Sequence[float]) -> float:
    if not values:
        raise GradingContractError("target token log-probability sequence must not be empty")
    converted: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise GradingContractError("target token log probabilities must be numeric")
        converted.append(float(value))
    if any(not math.isfinite(value) for value in converted):
        raise GradingContractError("target token log probabilities must be finite")
    return math.fsum(converted)


def hard_grade_from_target_logprobs(
    target_token_logprobs: Mapping[str, Sequence[float]], *, retry_count: int = 0
) -> HardGrade:
    """Compare exact full-sequence targets; sampled-text fallback is intentionally absent."""

    if set(target_token_logprobs) != {YES_TARGET, NO_TARGET}:
        raise GradingContractError('expected exact targets " YES" and " NO"')
    yes = _full_target_logprob(target_token_logprobs[YES_TARGET])
    no = _full_target_logprob(target_token_logprobs[NO_TARGET])
    maximum = max(yes, no)
    yes_exp = math.exp(yes - maximum)
    no_exp = math.exp(no - maximum)
    probability = yes_exp / (yes_exp + no_exp)
    if yes == no:
        return HardGrade(
            parse_status="ambiguous_target_tie",
            public_label="NOT_PRESENT",
            grade=0,
            probability_present=probability,
            yes_logprob=yes,
            no_logprob=no,
            retry_count=retry_count,
        )
    present = yes > no
    return HardGrade(
        parse_status="ok",
        public_label="PRESENT" if present else "NOT_PRESENT",
        grade=1 if present else 0,
        probability_present=probability,
        yes_logprob=yes,
        no_logprob=no,
        retry_count=retry_count,
    )


def grade_cache_key(
    *,
    grader_model_revision: str,
    tokenizer_revision: str,
    grader_prompt_hash: str,
    prompt_hash: str,
    response_hash: str,
    canonical_criterion_hash: str,
    target_encoding_version: str = TARGET_ENCODING_VERSION,
) -> str:
    fields = (
        grader_model_revision,
        tokenizer_revision,
        grader_prompt_hash,
        prompt_hash,
        response_hash,
        canonical_criterion_hash,
        target_encoding_version,
    )
    if any(not field.strip() for field in fields):
        raise GradingContractError("grade cache identity fields must not be empty")
    encoded = json.dumps(fields, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return sha256(encoded).hexdigest()


@dataclass(slots=True)
class HardGradeCache:
    values: dict[str, HardGrade] = field(default_factory=dict)
    hits: int = 0
    misses: int = 0

    def get_or_grade(self, *, key: str, grader: Callable[[], HardGrade]) -> HardGrade:
        if key in self.values:
            self.hits += 1
            return self.values[key]
        result = grader()
        if not isinstance(result, HardGrade):
            raise GradingContractError("grader must return HardGrade")
        self.values[key] = result
        self.misses += 1
        return result

    def artifact(self, key: str) -> dict[str, object]:
        if key not in self.values:
            raise KeyError(key)
        return asdict(self.values[key])


def assemble_rubric_score(
    rubric: WeightedRubric, grades_by_content_hash: Mapping[str, HardGrade]
) -> RationalScore:
    """Assemble a rational score, failing closed on missing or invalid grades."""

    instance_grades: dict[str, int] = {}
    for criterion in rubric.criteria:
        grade = grades_by_content_hash.get(criterion.canonical_criterion_hash)
        if grade is None:
            raise GradingContractError(
                f"missing grade for criterion {criterion.canonical_criterion_hash}"
            )
        if not grade.valid or grade.grade is None:
            raise GradingContractError(
                f"invalid grade for criterion {criterion.canonical_criterion_hash}: "
                f"{grade.parse_status}"
            )
        instance_grades[criterion.criterion_instance_id] = grade.grade
    return RationalScore.from_grades(rubric.criteria, instance_grades)


def assemble_variant_scores(
    rubrics: Mapping[str, WeightedRubric],
    grades_by_content_hash: Mapping[str, HardGrade],
) -> dict[str, RationalScore]:
    return {
        rubric_name: assemble_rubric_score(rubric, grades_by_content_hash)
        for rubric_name, rubric in rubrics.items()
    }
