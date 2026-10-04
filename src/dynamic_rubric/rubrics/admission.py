"""GT-free admission gate and utility calculation."""

from __future__ import annotations

from dataclasses import dataclass
import math


MIN_SATISFACTION_RATE = 0.10
MAX_SATISFACTION_RATE = 0.90
MIN_CURRENT_REFERENCE_SEPARATION = 0.15
MAX_ACTIVE_SIMILARITY_EXCLUSIVE = 0.85
MIN_PARSE_SUCCESS = 0.95


@dataclass(frozen=True, slots=True)
class AdmissionEvidence:
    satisfaction_rate: float
    current_mean: float
    reference_mean: float
    max_active_similarity: float
    parse_success: float
    independent_validation: bool
    score_variance: float = 0.0
    recent_validation_failure_rate: float = 0.0

    @property
    def current_reference_separation(self) -> float:
        return abs(self.current_mean - self.reference_mean)


@dataclass(frozen=True, slots=True)
class AdmissionDecision:
    admitted: bool
    utility: float
    failed_gates: tuple[str, ...]


def _finite(values: tuple[float, ...]) -> bool:
    return all(math.isfinite(value) for value in values)


def candidate_utility(evidence: AdmissionEvidence) -> float:
    """Compute the fixed, equally-weighted GT-free eviction utility."""

    if not _finite(
        (
            evidence.score_variance,
            evidence.current_reference_separation,
            evidence.max_active_similarity,
            evidence.recent_validation_failure_rate,
        )
    ):
        raise ValueError("utility evidence must be finite")
    components = (
        min(1.0, max(0.0, 4.0 * evidence.score_variance)),
        min(1.0, max(0.0, evidence.current_reference_separation)),
        min(1.0, max(0.0, 1.0 - evidence.max_active_similarity)),
        1.0 - min(1.0, max(0.0, evidence.recent_validation_failure_rate)),
    )
    return sum(components) / len(components)


def evaluate_admission(evidence: AdmissionEvidence) -> AdmissionDecision:
    """Apply exact inclusive/exclusive gate semantics from the PRD."""

    numeric = (
        evidence.satisfaction_rate,
        evidence.current_mean,
        evidence.reference_mean,
        evidence.max_active_similarity,
        evidence.parse_success,
        evidence.score_variance,
        evidence.recent_validation_failure_rate,
    )
    failed: list[str] = []
    if not _finite(numeric):
        failed.append("non_finite_evidence")
    else:
        if not MIN_SATISFACTION_RATE <= evidence.satisfaction_rate <= MAX_SATISFACTION_RATE:
            failed.append("satisfaction_rate")
        if evidence.current_reference_separation < MIN_CURRENT_REFERENCE_SEPARATION:
            failed.append("current_reference_separation")
        if evidence.max_active_similarity >= MAX_ACTIVE_SIMILARITY_EXCLUSIVE:
            failed.append("max_active_similarity")
        if evidence.parse_success < MIN_PARSE_SUCCESS:
            failed.append("parse_success")
    if evidence.independent_validation is not True:
        failed.append("independent_validation")
    utility = candidate_utility(evidence) if _finite(numeric) else 0.0
    return AdmissionDecision(not failed, utility, tuple(failed))


admit_candidate = evaluate_admission
