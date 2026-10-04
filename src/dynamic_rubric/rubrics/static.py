"""Static rubric construction and criterion validation.

This module deliberately contains no provider or artifact dependencies.  The
objects are frozen and tuple-backed so callers can safely serialize snapshots
without later replay steps mutating an earlier rubric.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
import re
from typing import Callable, Iterable, Sequence


TASK_SPECIFIC_CRITERIA_COUNT = 6
UNIVERSAL_CRITERIA_COUNT = 2
STATIC_CRITERIA_COUNT = TASK_SPECIFIC_CRITERIA_COUNT + UNIVERSAL_CRITERIA_COUNT
EQUAL_CRITERION_WEIGHT = 1.0 / STATIC_CRITERIA_COUNT
SEMANTIC_DUPLICATE_THRESHOLD = 0.85

UNIVERSAL_CRITERIA: tuple[str, str] = (
    "Direct relevance without irrelevant or excessive content",
    "Context-appropriate certainty, tone, and caution",
)

STATIC_RUBRIC_INSTRUCTIONS = """Create exactly six task-specific criteria for evaluating a
candidate response to the supplied health question. Use only the question and candidate
responses; never assume a physician answer, reference answer, hidden rubric, or external
document. Each criterion must describe one positive behavior that can be judged from the
response alone with a YES or NO decision. Keep each criterion atomic: do not use conjunctions
such as 'and' or 'or', do not mention candidate numbers, and do not duplicate another
criterion. The six task-specific criteria must complement the two universal criteria
'Direct relevance without irrelevant or excessive content' and
'Context-appropriate certainty, tone, and caution'. Return only the required JSON schema."""


class CriterionValidationError(ValueError):
    """Raised when a generated criterion violates the experiment contract."""


@dataclass(frozen=True, slots=True)
class Criterion:
    criterion_id: str
    text: str
    source: str
    weight: float = EQUAL_CRITERION_WEIGHT
    created_step: int = 0

    def __post_init__(self) -> None:
        if not self.criterion_id.strip():
            raise ValueError("criterion_id must not be empty")
        if not self.text.strip():
            raise ValueError("criterion text must not be empty")
        if not math.isfinite(self.weight) or self.weight <= 0:
            raise ValueError("criterion weight must be finite and positive")
        if self.created_step < 0:
            raise ValueError("created_step must be non-negative")


@dataclass(frozen=True, slots=True)
class StaticRubric:
    prompt_id: str
    criteria: tuple[Criterion, ...]
    version: int = 0

    def __post_init__(self) -> None:
        if not self.prompt_id.strip():
            raise ValueError("prompt_id must not be empty")
        if len(self.criteria) != STATIC_CRITERIA_COUNT:
            raise ValueError("a static rubric must contain exactly 8 criteria")
        if any(item.source not in {"task_specific", "universal"} for item in self.criteria):
            raise ValueError("static criteria must have static provenance")
        expected = EQUAL_CRITERION_WEIGHT
        if any(
            not math.isclose(item.weight, expected, rel_tol=0, abs_tol=1e-12)
            for item in self.criteria
        ):
            raise ValueError("all static criteria must have equal weight")

    @property
    def content_hash(self) -> str:
        payload = [
            (item.criterion_id, item.text, item.source, item.weight) for item in self.criteria
        ]
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
        return sha256(encoded).hexdigest()


SimilarityFunction = Callable[[str, str], float]


_SPACE_RE = re.compile(r"\s+")
_CANDIDATE_NUMBER_RE = re.compile(
    r"(?:\b(?:candidate|response|answer|output|sample)\s*(?:#|no\.?\s*)?\d+\b|"
    r"\b(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\s+"
    r"(?:candidate|response|answer|output|sample)\b)",
    re.IGNORECASE,
)
_EXTERNAL_REFERENCE_RE = re.compile(
    r"\b(?:physician|doctor[- ]written|gold(?:en)?\s+(?:answer|rubric|criterion)|"
    r"reference\s+answer|other\s+(?:answer|response)|source\s+(?:text|document))\b",
    re.IGNORECASE,
)
_NEGATIVE_FORM_RE = re.compile(
    r"^(?:no\b|do(?:es)?\s+not|never|avoid|omit|fail(?:s)?\s+to|must\s+not|should\s+not)\b",
    re.IGNORECASE,
)
_MULTI_REQUIREMENT_RE = re.compile(
    r"(?:;|\n|\band\b|\bor\b|\bas\s+well\s+as\b|\badditionally\b)",
    re.IGNORECASE,
)


def canonical_criterion_text(text: str) -> str:
    """Return the canonical form used only for exact duplicate detection."""

    return _SPACE_RE.sub(" ", text).strip().rstrip(".").casefold()


def validate_criterion_text(text: str) -> str:
    """Validate structural properties that can be checked without an LLM.

    Atomicity and response-only judgeability are intentionally conservative:
    ambiguous multi-clause or externally-referenced criteria are rejected.
    """

    normalized = _SPACE_RE.sub(" ", text).strip()
    if not normalized:
        raise CriterionValidationError("criterion must not be empty")
    if len(normalized.split()) < 2:
        raise CriterionValidationError("criterion must state a yes/no judgeable behavior")
    if len(normalized) > 500:
        raise CriterionValidationError("criterion must be at most 500 characters")
    if normalized.endswith("?"):
        raise CriterionValidationError(
            "criterion must describe a judgeable behavior, not ask a question"
        )
    if _CANDIDATE_NUMBER_RE.search(normalized):
        raise CriterionValidationError("criterion must not identify a numbered candidate")
    if _EXTERNAL_REFERENCE_RE.search(normalized):
        raise CriterionValidationError("criterion must be judgeable from the response alone")
    if _NEGATIVE_FORM_RE.search(normalized):
        raise CriterionValidationError("criterion must use positive behavior form")
    if _MULTI_REQUIREMENT_RE.search(normalized):
        raise CriterionValidationError(
            f"criterion must be single and atomic: {normalized!r}"
        )
    return normalized


def validate_criteria(
    texts: Iterable[str],
    *,
    expected_count: int | None = None,
    similarity: SimilarityFunction | None = None,
    semantic_threshold: float = SEMANTIC_DUPLICATE_THRESHOLD,
) -> tuple[str, ...]:
    """Validate a criterion set, including exact and optional semantic duplicates."""

    values = tuple(validate_criterion_text(text) for text in texts)
    if expected_count is not None and len(values) != expected_count:
        raise CriterionValidationError(
            f"expected exactly {expected_count} criteria, got {len(values)}"
        )
    canonical = tuple(canonical_criterion_text(text) for text in values)
    if len(set(canonical)) != len(canonical):
        raise CriterionValidationError("criteria must not contain exact duplicates")
    if not 0 <= semantic_threshold <= 1:
        raise ValueError("semantic_threshold must be in [0, 1]")
    if similarity is not None:
        for index, left in enumerate(values):
            for right in values[index + 1 :]:
                score = similarity(left, right)
                if not math.isfinite(score) or not -1 <= score <= 1:
                    raise CriterionValidationError("similarity adapter returned an invalid score")
                if score >= semantic_threshold:
                    raise CriterionValidationError(
                        "criteria must not contain semantic duplicates: "
                        f"{left!r} <> {right!r} (similarity={score:.6f})"
                    )
    return values


def build_static_rubric(
    prompt_id: str,
    task_specific_texts: Sequence[str],
    *,
    similarity: SimilarityFunction | None = None,
) -> StaticRubric:
    """Build the immutable six task-specific plus two universal ``R_0`` rubric."""

    task_texts = validate_criteria(
        task_specific_texts,
        expected_count=TASK_SPECIFIC_CRITERIA_COUNT,
        similarity=similarity,
    )
    # Validate duplicates across generated and universal criteria as well.  The
    # universal wording itself is trusted by the experiment contract.
    all_texts = task_texts + UNIVERSAL_CRITERIA
    canonical = tuple(canonical_criterion_text(text) for text in all_texts)
    if len(set(canonical)) != STATIC_CRITERIA_COUNT:
        raise CriterionValidationError("task-specific criterion duplicates a universal criterion")
    if similarity is not None:
        for index, left in enumerate(all_texts):
            for right in all_texts[index + 1 :]:
                score = similarity(left, right)
                if not math.isfinite(score) or not -1 <= score <= 1:
                    raise CriterionValidationError("similarity adapter returned an invalid score")
                if score >= SEMANTIC_DUPLICATE_THRESHOLD:
                    raise CriterionValidationError(
                        "criterion semantically duplicates an active criterion: "
                        f"{left!r} <> {right!r} (similarity={score:.6f})"
                    )

    criteria = tuple(
        Criterion(f"task-{index:02d}", text, "task_specific")
        for index, text in enumerate(task_texts, start=1)
    ) + tuple(
        Criterion(f"universal-{index:02d}", text, "universal")
        for index, text in enumerate(UNIVERSAL_CRITERIA, start=1)
    )
    return StaticRubric(prompt_id=prompt_id, criteria=criteria)


# Backwards-friendly explicit names for callers that use verbs.
create_static_rubric = build_static_rubric
validate_static_criteria = validate_criteria
