"""CI-gated preliminary interpretation of a rubric-staleness audit."""

from __future__ import annotations

from dataclasses import dataclass

from ..evaluation.bootstrap import BootstrapResult


@dataclass(frozen=True)
class InterpretationDecision:
    label: str
    evidence: tuple[str, ...]


def _meaningful_positive(result: BootstrapResult) -> bool:
    return result.classification == "meaningful_difference" and result.ci_low > 0.0


def classify_interpretation(
    *,
    max_policy_kl: float,
    admissions: int,
    primary: BootstrapResult,
    previous: BootstrapResult,
    refresh: BootstrapResult,
) -> InterpretationDecision:
    """Choose one interpretation using only treatment strength and CI states."""

    primary_gain = _meaningful_positive(primary)
    previous_gain = _meaningful_positive(previous)
    refresh_gain = _meaningful_positive(refresh)
    if max_policy_kl < 0.02:
        return InterpretationDecision("weak_policy_drift", (f"max_kl={max_policy_kl:.6g}<0.02",))
    if refresh_gain:
        return InterpretationDecision(
            "refresh_noise",
            (f"refresh_ci=[{refresh.ci_low:.6g},{refresh.ci_high:.6g}]",),
        )
    if previous_gain and not primary_gain:
        return InterpretationDecision(
            "frequent_update_need",
            (
                f"previous_ci=[{previous.ci_low:.6g},{previous.ci_high:.6g}]",
                "fixed_reference_gain_not_established",
            ),
        )
    if primary_gain and previous_gain:
        return InterpretationDecision(
            "general_rubric_improvement",
            (
                f"fixed_ci=[{primary.ci_low:.6g},{primary.ci_high:.6g}]",
                f"previous_ci=[{previous.ci_low:.6g},{previous.ci_high:.6g}]",
            ),
        )
    if primary_gain:
        return InterpretationDecision(
            "policy_adaptive_gain",
            (f"fixed_ci=[{primary.ci_low:.6g},{primary.ci_high:.6g}]",),
        )
    if primary.classification == "local_equivalence" and admissions > 0:
        return InterpretationDecision(
            "textual_only_churn",
            (f"admissions={admissions}", "fixed_effect_ci_is_equivalent"),
        )
    if admissions == 0:
        return InterpretationDecision(
            "updater_miss", ("no_dynamic_criteria_admitted", "policy_drift_is_nontrivial")
        )
    return InterpretationDecision(
        "local_redundancy",
        (f"fixed_bootstrap={primary.classification}", f"admissions={admissions}"),
    )
