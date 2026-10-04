import pytest
import random

from dynamic_rubric.horizon.inference import (
    ConfidenceInterval,
    CrossedBootstrapResult,
    ProcessObservation,
    crossed_bootstrap,
    decide_horizon,
    make_process_observation,
)
from dynamic_rubric.horizon.reporting import build_horizon_report, canonical_report_json, coverage_summary


def _observations() -> list[ProcessObservation]:
    rows = []
    for seed in ("s1", "s2"):
        for prompt_index in range(4):
            for checkpoint in (0.2, 0.4, 0.6):
                base = 0.1 * prompt_index + (0.05 if seed == "s2" else 0.0)
                rows.append(
                    ProcessObservation(seed, f"p{prompt_index}", checkpoint, base, base / 2, base / 3)
                )
    return rows


def test_make_process_observation_builds_paired_processes() -> None:
    row = make_process_observation(
        seed_id="s1",
        prompt_id="p1",
        checkpoint=0.2,
        r0_zar=1,
        r0_baseline_zar=0,
        current_zar=0,
        control_zar=1,
    )
    assert (row.d_r0, row.g_refresh, row.g_count) == (1.0, 1.0, 1.0)


def test_crossed_bootstrap_is_deterministic_and_uses_max_error_bands() -> None:
    first = crossed_bootstrap(_observations(), iterations=200, seed=7)
    second = crossed_bootstrap(_observations(), iterations=200, seed=7)
    assert first == second
    assert first.seed_count == 2
    assert first.prompt_count == 4
    assert first.small_seed_cluster_warning
    for process in ("d_r0", "g_refresh", "g_count"):
        q = first.simultaneous_quantiles[process]
        for checkpoint, interval in first.bands[process].items():
            assert interval.high - interval.point == pytest.approx(q)
            assert interval.point - interval.low == pytest.approx(q)


def test_crossed_bootstrap_rejects_incomplete_or_duplicate_grid() -> None:
    rows = _observations()
    with pytest.raises(ValueError, match="complete"):
        crossed_bootstrap(rows[:-1], iterations=2)
    with pytest.raises(ValueError, match="duplicate"):
        crossed_bootstrap(rows + [rows[0]], iterations=2)


def test_crossed_random_effect_fixture_covers_known_trajectory_mean() -> None:
    rng = random.Random(19)
    checkpoints = (0.2, 0.4, 0.6)
    truth = {0.2: 0.0, 0.4: 0.08, 0.6: 0.16}
    seed_effects = {f"s{index}": rng.gauss(0.0, 0.02) for index in range(6)}
    prompt_effects = {f"p{index}": rng.gauss(0.0, 0.04) for index in range(80)}
    rows = [
        ProcessObservation(
            seed,
            prompt,
            checkpoint,
            truth[checkpoint] + seed_effect + prompt_effect,
            truth[checkpoint] / 2 + seed_effect + prompt_effect,
            truth[checkpoint] / 3 + seed_effect + prompt_effect,
        )
        for seed, seed_effect in seed_effects.items()
        for prompt, prompt_effect in prompt_effects.items()
        for checkpoint in checkpoints
    ]
    result = crossed_bootstrap(rows, iterations=500, seed=23)
    assert all(
        result.bands["d_r0"][checkpoint].low
        <= truth[checkpoint]
        <= result.bands["d_r0"][checkpoint].high
        for checkpoint in checkpoints
    )


def _result(
    d_lows: list[float], refresh_lows: list[float], count_lows: list[float],
    *, equivalent_at: int | None = 0,
) -> CrossedBootstrapResult:
    checkpoints = (0.2, 0.4, 0.6, 0.8)
    lows = {"d_r0": d_lows, "g_refresh": refresh_lows, "g_count": count_lows}
    bands = {
        name: {
            checkpoint: ConfidenceInterval(point=value + 0.01, low=value, high=value + 0.02)
            for checkpoint, value in zip(checkpoints, values)
        }
        for name, values in lows.items()
    }
    pointwise = {
        name: {
            checkpoint: ConfidenceInterval(point=0.0, low=-0.01, high=0.01)
            if name == "g_refresh" and index == equivalent_at
            else ConfidenceInterval(point=0.1, low=0.08, high=0.12)
            for index, checkpoint in enumerate(checkpoints)
        }
        for name in lows
    }
    return CrossedBootstrapResult(
        checkpoints=checkpoints,
        bands=bands,
        pointwise=pointwise,
        simultaneous_quantiles={name: 0.01 for name in lows},
        seed_count=3,
        prompt_count=300,
        iterations=100,
        bootstrap_seed=1,
        small_seed_cluster_warning=True,
    )


def test_horizon_requires_early_equivalence_and_next_checkpoint_persistence() -> None:
    result = _result(
        [0.0, 0.04, 0.05, 0.05],
        [0.0, 0.04, 0.05, 0.05],
        [0.0, 0.01, 0.02, 0.02],
    )
    decision = decide_horizon(result)
    assert decision.status == "observed"
    assert decision.t_star == 0.4
    assert decision.t_refresh == 0.4
    assert decision.early_equivalence_checkpoint == 0.2


def test_first_checkpoint_onset_is_left_censored_without_early_equivalence() -> None:
    result = _result(
        [0.04, 0.05, 0.05, 0.05],
        [0.04, 0.05, 0.05, 0.05],
        [0.01, 0.02, 0.02, 0.02],
        equivalent_at=None,
    )
    decision = decide_horizon(result)
    assert decision.status == "left_censored_before_first_checkpoint"
    assert decision.t_star is None
    assert decision.candidate_checkpoint == 0.2


def test_final_only_onset_is_right_censored() -> None:
    result = _result(
        [0.0, 0.0, 0.0, 0.04],
        [0.0, 0.0, 0.0, 0.04],
        [0.0, 0.0, 0.0, 0.01],
    )
    decision = decide_horizon(result)
    assert decision.status == "candidate_onset_right_censored_at_final"
    assert decision.t_star is None
    assert decision.candidate_checkpoint == 0.8


def test_gate_failure_prevents_horizon_onset() -> None:
    result = _result(
        [0.0, 0.04, 0.05, 0.05],
        [0.0, 0.04, 0.05, 0.05],
        [0.0, 0.01, 0.02, 0.02],
    )
    decision = decide_horizon(result, gate_passed={0.4: False, 0.6: False})
    assert decision.t_star is None


def test_report_is_machine_readable_deterministic_and_claim_limited() -> None:
    bootstrap = crossed_bootstrap(_observations(), iterations=10, seed=3)
    decision = decide_horizon(bootstrap)
    report = build_horizon_report(
        domain="medicine",
        bootstrap=bootstrap,
        decision=decision,
        coverage=coverage_summary(valid=10, invalid=1, missing=1),
        revisions={"grader": "abc"},
    )
    assert report["claim_scope"] == "empirical_discriminability_only"
    assert report["limitations"]
    assert canonical_report_json(report) == canonical_report_json(report)
