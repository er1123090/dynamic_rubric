"""Deterministic scoring and statistical evaluation utilities."""

from .bon import BON_SIZES, fixed_candidate_permutations, select_best_of_n, shared_pool_hash
from .bootstrap import BootstrapResult, paired_prompt_bootstrap
from .gold_score import gold_cache_key, weighted_gold_score
from .metrics import (
    gt_auc,
    kendall_tau_b,
    reward_resolution,
    stale_rubric_regret,
    top1_agreement,
)
from .proxy_score import (
    CriterionScoreCache,
    MalformedLogProbError,
    normalized_yes_probability,
    rubric_mean_score,
)

__all__ = [
    "BON_SIZES",
    "BootstrapResult",
    "CriterionScoreCache",
    "MalformedLogProbError",
    "fixed_candidate_permutations",
    "gold_cache_key",
    "gt_auc",
    "kendall_tau_b",
    "normalized_yes_probability",
    "paired_prompt_bootstrap",
    "reward_resolution",
    "rubric_mean_score",
    "select_best_of_n",
    "shared_pool_hash",
    "stale_rubric_regret",
    "top1_agreement",
    "weighted_gold_score",
]
