"""Frozen proxy-grader score normalization and criterion-level caching."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import TypeVar


class MalformedLogProbError(ValueError):
    """Raised when a canonical YES/NO score cannot be safely interpreted."""


def normalized_yes_probability(yes_logprob: float, no_logprob: float) -> float:
    """Return P(YES) after normalizing exactly two finite log-likelihoods.

    This intentionally rejects missing, boolean, NaN and infinite values rather
    than silently falling back to a parsed Boolean judgment.
    """

    values = (yes_logprob, no_logprob)
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
        raise MalformedLogProbError("YES and NO log-likelihoods must be numeric")
    yes, no = map(float, values)
    if not (math.isfinite(yes) and math.isfinite(no)):
        raise MalformedLogProbError("YES and NO log-likelihoods must be finite")
    maximum = max(yes, no)
    yes_exp = math.exp(yes - maximum)
    no_exp = math.exp(no - maximum)
    return yes_exp / (yes_exp + no_exp)


def normalize_yes_no_logprobs(logprobs: Mapping[str, float]) -> dict[str, float]:
    """Normalize a mapping containing exactly the canonical ``YES``/``NO`` labels."""

    if set(logprobs) != {"YES", "NO"}:
        raise MalformedLogProbError("expected exactly the labels YES and NO")
    yes = normalized_yes_probability(logprobs["YES"], logprobs["NO"])
    return {"YES": yes, "NO": 1.0 - yes}


def rubric_mean_score(criterion_scores: Iterable[float]) -> float:
    """Assemble an equal-weight rubric reward from cached criterion scores."""

    scores = tuple(float(score) for score in criterion_scores)
    if not scores:
        raise ValueError("a rubric must contain at least one active criterion")
    if any(not math.isfinite(score) or not 0.0 <= score <= 1.0 for score in scores):
        raise ValueError("criterion scores must be finite values in [0, 1]")
    return math.fsum(scores) / len(scores)


def criterion_cache_key(
    prompt_id: str,
    response_text: str,
    criterion_id: str,
    grader_identity: str = "",
) -> str:
    payload = [
        prompt_id,
        hashlib.sha256(response_text.encode("utf-8")).hexdigest(),
        criterion_id,
        grader_identity,
    ]
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


T = TypeVar("T")


@dataclass
class CriterionScoreCache:
    """In-memory cache ensuring each response/criterion pair is graded once."""

    values: dict[str, float] = field(default_factory=dict)
    hits: int = 0
    misses: int = 0

    def score(
        self,
        *,
        prompt_id: str,
        response_text: str,
        criterion_id: str,
        grader: Callable[[], float],
        grader_identity: str = "",
    ) -> float:
        key = criterion_cache_key(prompt_id, response_text, criterion_id, grader_identity)
        if key in self.values:
            self.hits += 1
            return self.values[key]
        value = float(grader())
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("grader score must be a finite value in [0, 1]")
        self.values[key] = value
        self.misses += 1
        return value

    def rubric_score(self, criterion_ids: Iterable[str], scores: Mapping[str, float]) -> float:
        ids = tuple(criterion_ids)
        missing = [criterion_id for criterion_id in ids if criterion_id not in scores]
        if missing:
            raise KeyError(f"missing criterion scores: {missing}")
        return rubric_mean_score(scores[criterion_id] for criterion_id in ids)
