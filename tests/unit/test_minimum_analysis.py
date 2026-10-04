from __future__ import annotations

from dynamic_rubric.minimum_analysis import _onset


def _checkpoint(step: int, point: float, low: float, classification: str) -> dict[str, object]:
    return {
        "policy_id": f"pi_{step}",
        "policy_step": step,
        "paired_bootstrap": {
            "point_estimate": point,
            "ci_low": low,
            "ci_high": point + 0.02,
            "classification": classification,
        },
        "policy_distance": {"mean_kl_from_pi0": step / 1000},
    }


def test_onset_is_first_ci_positive_observed_checkpoint() -> None:
    result = _onset(
        [
            _checkpoint(3, 0.0, -0.01, "local_equivalence"),
            _checkpoint(10, 0.02, -0.005, "inconclusive"),
            _checkpoint(30, 0.04, 0.01, "meaningful_difference"),
            _checkpoint(50, 0.05, 0.02, "meaningful_difference"),
        ]
    )
    assert result["statistically_positive_first_policy"] == "pi_30"
    assert result["meaningful_positive_first_step"] == 30
    assert result["estimated_kl_at_statistical_onset"] == 0.03
    assert result["interpretation"] == "staleness_onset_pi_30"


def test_onset_remains_unobserved_when_intervals_cross_zero() -> None:
    result = _onset([_checkpoint(50, 0.04, -0.01, "inconclusive")])
    assert result["statistically_positive_first_step"] is None
    assert result["meaningful_positive_first_step"] is None
    assert result["interpretation"] == "not_observed_through_pi_50"
