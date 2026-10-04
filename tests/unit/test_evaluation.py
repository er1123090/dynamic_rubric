import math

import pytest

from dynamic_rubric.evaluation.bon import (
    BON_SIZES,
    fixed_candidate_permutations,
    select_best_of_n,
    shared_pool_hash,
)
from dynamic_rubric.evaluation.bootstrap import classify_difference, paired_prompt_bootstrap
from dynamic_rubric.evaluation.gold_score import gold_cache_key, weighted_gold_score
from dynamic_rubric.evaluation.metrics import (
    gt_auc,
    kendall_tau_b,
    reward_resolution,
    stale_rubric_regret,
    top1_agreement,
)
from dynamic_rubric.evaluation.proxy_score import (
    CriterionScoreCache,
    MalformedLogProbError,
    normalize_yes_no_logprobs,
    normalized_yes_probability,
    rubric_mean_score,
)


def test_yes_no_logprob_normalization_and_malformed_rejection():
    assert normalized_yes_probability(0.0, 0.0) == pytest.approx(0.5)
    assert normalized_yes_probability(math.log(3), 0.0) == pytest.approx(0.75)
    assert normalize_yes_no_logprobs({"YES": 0.0, "NO": 0.0}) == {"YES": 0.5, "NO": 0.5}
    for malformed in [(float("nan"), 0), (float("inf"), 0), (True, 0), ("0", 0)]:
        with pytest.raises(MalformedLogProbError):
            normalized_yes_probability(*malformed)
    with pytest.raises(MalformedLogProbError):
        normalize_yes_no_logprobs({"yes": 0.0, "NO": 0.0})


def test_proxy_cache_grades_once_and_rubric_is_equal_weight_mean():
    calls = []
    cache = CriterionScoreCache()

    def grader():
        return calls.append(1) or 0.8

    kwargs = dict(prompt_id="p", response_text="response", criterion_id="c", grader=grader)
    assert cache.score(**kwargs) == cache.score(**kwargs) == 0.8
    assert len(calls) == 1
    assert (cache.hits, cache.misses) == (1, 1)
    assert rubric_mean_score([0.2, 0.4, 0.6]) == pytest.approx(0.4)
    assert cache.rubric_score(["a", "b"], {"a": 0.25, "b": 0.75}) == 0.5


def test_bon_permutations_grid_tie_break_and_shared_pool_hash():
    assert BON_SIZES == (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
    ids = list(range(1024))
    first = fixed_candidate_permutations(ids, seed=9)
    assert first == fixed_candidate_permutations(ids, seed=9)
    assert len(first) == 5 and all(sorted(order) == ids for order in first)
    permutation = (7, 3, 9, 1)
    assert select_best_of_n({7: 0.1, 3: 0.9, 9: 0.9, 1: 0.0}, permutation, 4) == 3
    assert select_best_of_n({item: float(item) for item in ids}, ids, 1024) == 1023
    pool = [{"candidate_id": 2, "response_text": "b"}, {"candidate_id": 1, "response_text": "a"}]
    assert shared_pool_hash(pool) == shared_pool_hash(list(reversed(pool)))
    changed = [{"candidate_id": 1, "response_text": "changed"}, pool[0]]
    assert shared_pool_hash(pool) != shared_pool_hash(changed)


def test_gold_weighted_score_and_cache_key_invalidates_every_setting():
    assert weighted_gold_score({"a": 1.0, "b": 0.25}, {"a": 3, "b": 1}) == pytest.approx(0.8125)
    assert weighted_gold_score(
        {"good": 1.0, "harm": 1.0}, {"good": 4, "harm": -2}
    ) == pytest.approx(0.5)
    assert weighted_gold_score({"good": 1.0, "harm": 0.0}, {"good": 4, "harm": -2}) == 1.0
    assert weighted_gold_score({"good": 0.0, "harm": 1.0}, {"good": 4, "harm": -2}) == 0.0
    kwargs = dict(
        prompt_id="p",
        response_text_hash="r",
        gold_rubric_hash="g",
        requested_model="gpt-5",
        returned_model="gpt-5-x",
        grader_prompt_hash="gp",
        schema_hash="s",
        reasoning_effort="high",
    )
    original = gold_cache_key(**kwargs)
    for key in kwargs:
        changed = dict(kwargs)
        changed[key] += "-changed"
        assert gold_cache_key(**changed) != original


def test_gt_auc_regret_agreement_and_kendall_ties():
    linear = {n: index / (len(BON_SIZES) - 1) for index, n in enumerate(BON_SIZES)}
    assert gt_auc(linear) == pytest.approx(0.5)
    assert stale_rubric_regret(0.7, 0.6) == pytest.approx(0.1)
    assert top1_agreement([1, 2, 3], [1, 9, 3]) == pytest.approx(2 / 3)
    assert kendall_tau_b([1, 2, 2], [1, 2, 3]) == pytest.approx(2 / math.sqrt(6))
    assert kendall_tau_b([1, 1], [2, 2]) == 0.0


def test_reward_resolution_reports_pair_ties_and_repeat_rank_stability():
    result = reward_resolution([0.0, 0.0, 1.0], repeat_scores=[[0.1, 0.1, 0.9]])
    assert result["tie_rate"] == pytest.approx(1 / 3)
    assert result["variance"] == pytest.approx(2 / 9)
    assert result["top_median_margin"] == 1.0
    assert result["judge_repeat_stability"] == 1.0


def test_paired_bootstrap_clusters_prompts_is_seeded_and_classifies_by_ci():
    rows = [
        ("p1", 0.51, 0.50),
        ("p1", 0.53, 0.50),
        ("p2", 0.49, 0.50),
        ("p2", 0.51, 0.50),
    ]
    first = paired_prompt_bootstrap(rows, iterations=500, seed=7)
    assert first == paired_prompt_bootstrap(rows, iterations=500, seed=7)
    assert first.n_prompt_clusters == 2
    assert first.point_estimate == pytest.approx(0.01)
    assert classify_difference(0.0, -0.014, 0.015) == "local_equivalence"
    assert classify_difference(0.03, 0.001, 0.06) == "meaningful_difference"
    assert classify_difference(0.02, 0.001, 0.03) == "inconclusive"
