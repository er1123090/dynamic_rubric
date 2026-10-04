from __future__ import annotations

from scripts.benchmark_judge_replicas import _comparison


def test_replica_comparison_reports_raw_and_normalized_deltas() -> None:
    reference = [[{"YES": -0.5, "NO": -1.5}]]
    same_offset = [[{"YES": -0.25, "NO": -1.25}]]

    result = _comparison(reference, same_offset)

    assert result["max_abs_target_logprob_delta"] == 0.25
    assert result["mean_abs_target_logprob_delta"] == 0.25
    assert result["max_abs_probability_yes_delta"] == 0.0
    assert result["mean_abs_probability_yes_delta"] == 0.0
