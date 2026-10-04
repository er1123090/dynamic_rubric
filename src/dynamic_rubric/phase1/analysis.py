"""Adjacent-update and triangular reuse-horizon analysis over saved score groups."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .config import REUSE_ANCHORS
from .metrics import (
    MetricContractError,
    ScoreGroup,
    aggregate_comparisons,
    compare_fresh_stale,
    empirical_reuse_horizon,
)


def adjacent_update_value(
    stale_by_prompt: Mapping[str, ScoreGroup],
    fresh_by_prompt: Mapping[str, ScoreGroup],
    *,
    epsilon_z: float,
    epsilon_t: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if set(stale_by_prompt) != set(fresh_by_prompt):
        raise MetricContractError(
            "adjacent fresh/stale evaluators must cover identical prompt IDs"
        )
    comparisons = [
        compare_fresh_stale(
            stale_by_prompt[prompt_id],
            fresh_by_prompt[prompt_id],
            epsilon_z=epsilon_z,
            epsilon_t=epsilon_t,
        )
        for prompt_id in sorted(stale_by_prompt)
    ]
    return comparisons, aggregate_comparisons(
        comparisons, comparison_kind="adjacent_update_value"
    )


def reuse_horizon_matrix(
    groups: Mapping[tuple[int, int, str], ScoreGroup],
    *,
    prompt_ids: Sequence[str],
    anchors: Sequence[int] = REUSE_ANCHORS,
    epsilon_z: float,
    epsilon_t: float,
    practical_margin_delta_d: float,
) -> dict[str, Any]:
    """Build the fixed-probe triangular matrix from saved score groups.

    Keys are (evaluator_step, policy_step, prompt_id). The current evaluator
    at policy step t is always keyed by (t, t, prompt_id). No responses are
    generated or mutated here.
    """

    ordered_anchors = tuple(int(value) for value in anchors)
    if ordered_anchors != tuple(sorted(set(ordered_anchors))):
        raise MetricContractError("reuse anchors must be unique and increasing")
    prompts = tuple(str(prompt_id) for prompt_id in prompt_ids)
    if not prompts or len(prompts) != len(set(prompts)):
        raise MetricContractError("fixed-probe prompt IDs must be non-empty and unique")

    rows: list[dict[str, Any]] = []
    prompt_rows: list[dict[str, Any]] = []
    for policy_step in ordered_anchors:
        for evaluator_step in ordered_anchors:
            if evaluator_step > policy_step:
                continue
            comparisons: list[dict[str, Any]] = []
            for prompt_id in prompts:
                stale_key = (evaluator_step, policy_step, prompt_id)
                fresh_key = (policy_step, policy_step, prompt_id)
                if stale_key not in groups or fresh_key not in groups:
                    raise MetricContractError(
                        f"missing triangular score group for {stale_key} or {fresh_key}"
                    )
                comparison = compare_fresh_stale(
                    groups[stale_key],
                    groups[fresh_key],
                    epsilon_z=epsilon_z,
                    epsilon_t=epsilon_t,
                )
                comparisons.append(comparison)
                prompt_rows.append(
                    {
                        "evaluator_step": evaluator_step,
                        "policy_step": policy_step,
                        "l_zar": comparison["v_adj_zar"],
                        **comparison,
                    }
                )
            summary = aggregate_comparisons(
                comparisons, comparison_kind="reuse_horizon"
            )
            rows.append(
                {
                    "evaluator_step": evaluator_step,
                    "policy_step": policy_step,
                    "evaluator_age_steps": policy_step - evaluator_step,
                    "l_zar": summary["l_zar"],
                    "delta_tie_rate": summary["delta_tie_rate"],
                    "delta_separation_rate": summary["delta_separation_rate"],
                    "delta_effective_criterion_ratio": summary[
                        "delta_effective_criterion_ratio"
                    ],
                    "delta_top_median_margin": summary["delta_top_median_margin"],
                    "kendall_tau_b": summary["kendall_tau_b"],
                    "prompt_count": summary["prompt_count"],
                    "same_pool_b": True,
                }
            )

    horizons = {
        str(tau): empirical_reuse_horizon(
            rows,
            tau=tau,
            practical_margin_delta_d=practical_margin_delta_d,
        )
        for tau in ordered_anchors
    }
    return {
        "schema_version": 1,
        "primary_dataset": "fixed_train_probe",
        "anchors": list(ordered_anchors),
        "practical_margin_delta_d": float(practical_margin_delta_d),
        "matrix": rows,
        "prompt_comparisons": prompt_rows,
        "empirical_reuse_horizon": horizons,
        "heldout_used": False,
        "actual_training_batches_used": False,
        "ranking_similarity_is_correctness": False,
    }
