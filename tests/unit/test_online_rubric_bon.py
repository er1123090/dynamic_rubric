from __future__ import annotations

import json
from pathlib import Path

import pytest

from dynamic_rubric.minimum_staleness import MinimumExperimentError
from dynamic_rubric.online_rubric_bon import load_online_rubrics, weighted_proxy_score


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_weighted_proxy_score_uses_online_rubric_integer_weights() -> None:
    criteria = [
        {"criterion_key": "major", "weight": 3},
        {"criterion_key": "minor", "weight": 1},
    ]
    assert weighted_proxy_score({"major": 1.0, "minor": 0.0}, criteria) == pytest.approx(0.75)


@pytest.mark.parametrize("weight", [0, -1, 1.5, True])
def test_weighted_proxy_score_rejects_non_positive_integer_weights(weight: object) -> None:
    with pytest.raises(MinimumExperimentError, match="invalid OnlineRubric weight"):
        weighted_proxy_score({"criterion": 1.0}, [{"criterion_key": "criterion", "weight": weight}])


def test_load_online_rubrics_keeps_modes_separate_and_hashes_criterion_text(
    tmp_path: Path,
) -> None:
    base = {
        "prompt_id": "prompt-1",
        "policy_step": 3,
        "rubric_id": "prompt-1:onlinerubric_R_3",
        "criteria": [{"criterion_id": "criterion-1", "text": "Is correct", "weight": 2}],
    }
    for stage in ("onlinerubric_dedup_fixed_batch", "onlinerubric_dedup_prev_batch"):
        _write_jsonl(tmp_path / stage / "onlinerubric_rubrics.jsonl", [base])

    rubrics = load_online_rubrics(tmp_path)

    assert set(rubrics) == {
        ("pi_ref", "prompt-1", 3),
        ("pi_old", "prompt-1", 3),
    }
    fixed = rubrics["pi_ref", "prompt-1", 3]
    previous = rubrics["pi_old", "prompt-1", 3]
    assert fixed["criteria"][0]["criterion_key"] == previous["criteria"][0]["criterion_key"]
    assert fixed["experiment_mode"] == "pi_ref"
    assert previous["experiment_mode"] == "pi_old"
