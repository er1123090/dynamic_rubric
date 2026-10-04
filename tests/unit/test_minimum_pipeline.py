from __future__ import annotations

from dynamic_rubric import minimum_analysis


def test_analysis_bootstrap_contract_is_fixed_for_the_minimum_experiment() -> None:
    assert minimum_analysis.BOOTSTRAP_ITERATIONS == 10_000
    assert minimum_analysis.EXPECTED_AUDIT_PROMPTS == 96
    assert minimum_analysis.EXPECTED_POOL_SIZE == 1024
