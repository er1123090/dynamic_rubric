from __future__ import annotations

from pathlib import Path

import pytest

from dynamic_rubric.horizon.contracts import (
    CriterionType,
    ImportanceClass,
    WEIGHT_UNITS,
    WeightedCriterion,
    WeightedRubric,
    criterion_content_hash,
)
from dynamic_rubric.providers.vllm import FullCriterionScore
from dynamic_rubric.training.rar_reward import (
    RaRRewardConfig,
    RaRRewardContractError,
    score_static_r0,
    weighted_rational_score,
)
from dynamic_rubric.training import verl_reward
from dynamic_rubric.training.verl_reward import _grader_base_url, _score_rar_response


def criterion(identity: str, text: str, importance: ImportanceClass) -> WeightedCriterion:
    return WeightedCriterion(
        criterion_instance_id=identity,
        canonical_criterion_hash=criterion_content_hash(text),
        text=text,
        importance_class=importance,
        criterion_type=CriterionType.QUALITY,
        weight_units=WEIGHT_UNITS[importance],
        source_checkpoint="step0",
    )


def test_weighted_hard_binary_score_preserves_exact_rational_representation() -> None:
    criteria = (
        criterion("essential", "States the main diagnosis", ImportanceClass.ESSENTIAL),
        criterion("important", "Explains the next action", ImportanceClass.IMPORTANT),
        criterion("optional", "Adds useful context", ImportanceClass.OPTIONAL),
    )
    score = weighted_rational_score(criteria, {"essential": 1, "important": 0, "optional": 1})
    assert (score.numerator, score.denominator) == (13, 20)
    assert score.value == 0.65


def test_reward_rejects_soft_boolean_and_missing_grades() -> None:
    criteria = (criterion("c", "States the main diagnosis", ImportanceClass.ESSENTIAL),)
    with pytest.raises(RaRRewardContractError, match="integer 0 or 1"):
        weighted_rational_score(criteria, {"c": True})
    with pytest.raises(RaRRewardContractError, match="missing criterion grades"):
        weighted_rational_score(criteria, {})


def test_training_boundary_accepts_only_static_r0_artifacts() -> None:
    RaRRewardConfig(Path("artifacts/rar/static_r0.json")).validate()
    with pytest.raises(RaRRewardContractError, match="rar_static_r0_only"):
        RaRRewardConfig(Path("artifacts/rar/static_r0.json"), reward_source="dynamic").validate()
    with pytest.raises(RaRRewardContractError, match="cannot read dynamic"):
        RaRRewardConfig(Path("artifacts/horizon/r0.json")).validate()


def test_static_reward_rejects_online_checkpoint_provenance() -> None:
    base = criterion("base", "States the main diagnosis", ImportanceClass.ESSENTIAL)
    online = WeightedCriterion(
        criterion_instance_id="online",
        canonical_criterion_hash=criterion_content_hash("Identifies a subtle failure"),
        text="Identifies a subtle failure",
        importance_class=ImportanceClass.IMPORTANT,
        criterion_type=CriterionType.QUALITY,
        weight_units=7,
        source_checkpoint="step3",
    )
    rubric = WeightedRubric("p", (base, online))
    with pytest.raises(RaRRewardContractError, match="non-R0"):
        score_static_r0(rubric, {"base": 1, "online": 1})


def test_verl_rar_hook_emits_reusable_probability_and_exact_rational_trace() -> None:
    class Grader:
        def score_many_full(self, items, *, prompt_text_by_id):
            assert prompt_text_by_id == {"p": "[]"}
            return tuple(
                FullCriterionScore(
                    prompt_id=item[0],
                    response_id=item[1],
                    criterion_id=item[3],
                    yes_logprob=-0.1 if probability > 0.5 else -1.0,
                    no_logprob=-1.0 if probability > 0.5 else -0.1,
                    probability_present=probability,
                    parse_status="ok",
                )
                for item, probability in zip(items, (0.9, 0.1))
            )

    rubric = {
        "r0": {
            "criteria": [
                {
                    "criterion_id": "essential",
                    "criterion": "Is correct",
                    "weight_units": 10,
                },
                {
                    "criterion_id": "optional",
                    "criterion": "Is concise",
                    "weight_units": 3,
                },
            ]
        }
    }
    reward, trace, numerator, denominator = _score_rar_response(
        Grader(), "p", "r", "answer", rubric
    )
    assert (numerator, denominator, reward) == (10, 13, 10 / 13)
    assert [item["probability_yes"] for item in trace] == [0.9, 0.1]
    assert [item["probability_present"] for item in trace] == [0.9, 0.1]


def test_static_reward_workers_distribute_across_configured_judges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "DYNAMIC_RUBRIC_VLLM_URLS",
        "http://judge-a/v1,http://judge-b/v1,http://judge-c/v1",
    )
    monkeypatch.setattr(verl_reward.os, "getpid", lambda: 4)
    assert _grader_base_url() == "http://judge-b/v1"
