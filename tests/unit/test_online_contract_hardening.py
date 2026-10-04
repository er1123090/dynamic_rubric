from __future__ import annotations

import json
from pathlib import Path

import pytest

from dynamic_rubric.providers.base import GenerationResult
from dynamic_rubric.training.online_contracts import OnlineContractError, WeightedCriterion
from dynamic_rubric.training.online_step import OnlineStepError, _validate_model_identity


FIXTURES = Path(__file__).parents[1] / "fixtures" / "paper_online_rubrics"


def test_returned_model_must_match_explicit_pinned_identity() -> None:
    result = GenerationResult("{}", "o3-mini", "o3-mini-2026-01-01", "request", 1, 0)
    with pytest.raises(OnlineStepError, match="returned-model identity mismatch"):
        _validate_model_identity(
            (result,),
            expected_requested_model="o3-mini",
            expected_returned_model="o3-mini",
            family="extractor",
        )
    _validate_model_identity(
        (result,),
        expected_requested_model="o3-mini",
        expected_returned_model="o3-mini-2026-01-01",
        family="extractor",
    )


def test_offline_weights_must_be_numeric_and_finite() -> None:
    malformed = json.loads((FIXTURES / "malformed_weights.json").read_text(encoding="utf-8"))
    for item in malformed.values():
        with pytest.raises(OnlineContractError, match="criterion weight"):
            WeightedCriterion(
                item["criterion_id"], item["criterion"], item["weight"], item["source"]
            )
