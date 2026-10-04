from __future__ import annotations

from dynamic_rubric.training.verl_dataset import (
    build_online_rar_verl_rows,
    build_rar_verl_rows,
)


PROMPT = {
    "prompt_id": "p1",
    "domain": "medicine",
    "messages": [{"role": "user", "content": "Question"}],
    "r0": {"criteria": [{"criterion_id": "c1", "criterion": "Good", "weight_units": 10}, {"criterion_id": "c2", "criterion": "Bad", "weight_units": -3}]},
}


def test_online_rows_add_stable_occurrence_identity_without_changing_static_rows() -> None:
    static_train, _ = build_rar_verl_rows("run", [PROMPT], [])
    online_a, _ = build_online_rar_verl_rows("run", [PROMPT], [])
    online_b, _ = build_online_rar_verl_rows("run", [PROMPT], [])

    assert "prompt_occurrence_id" not in static_train[0]
    assert online_a == online_b
    row = online_a[0]
    assert row["data_source"] == "rar_online_rubrics_v1"
    assert row["source_row_id"] == "p1"
    assert row["prompt_occurrence_id"].startswith("train:0:")
    assert row["extra_info"]["prompt_occurrence_id"] == row["prompt_occurrence_id"]
    assert [item["weight"] for item in row["offline_criteria"]] == [10, -3]
    assert row["extra_info"]["offline_criteria"] == row["offline_criteria"]
