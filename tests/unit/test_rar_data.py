from __future__ import annotations

import json

import pytest

from dynamic_rubric.data.rar import RaRDataError, normalize_rar_row, prepare_rar_source


def _row(index: int) -> dict:
    return {
        "id": f"p{index}",
        "prompt": f"question {index}",
        "strata": {"topic": "a" if index % 2 else "b"},
        "rubric": [
            {"criterion": "Gives the correct conclusion", "importance": "essential"},
            {
                "criterion": "Avoids an unsafe recommendation",
                "type": "pitfall",
                "positive_form": True,
            },
        ],
    }


def test_normalize_rar_row_maps_exact_weights() -> None:
    value = normalize_rar_row(_row(1), domain="medicine")
    assert value["prompt_id"] == "rar_medicine_p1"
    assert [item["weight_units"] for item in value["r0"]["criteria"]] == [10, 9]
    assert value["r0"]["criteria"][1]["criterion_type"] == "pitfall"


def test_normalize_official_rar_shape_maps_labels_and_negative_pitfalls() -> None:
    value = normalize_rar_row(
        {
            "question": "official question",
            "reference_answer": "official answer",
            "question_source": "source-dataset",
            "rubric": [
                {
                    "description": "Important Criteria: Explains the central mechanism.",
                    "title": "Mechanism",
                    "weight": 5,
                },
                {
                    "description": "Pitfall Criteria: Recommends an unsafe treatment.",
                    "title": "Unsafe",
                    "weight": -1,
                },
            ],
        },
        domain="medicine",
    )
    assert [item["weight_units"] for item in value["r0"]["criteria"]] == [7, 9]
    assert value["r0"]["criteria"][0]["criterion"] == "Explains the central mechanism."
    assert value["r0"]["criteria"][1]["criterion"].startswith("Avoids this pitfall:")
    assert value["strata"] == {"question_source": "source-dataset"}
    assert value["reference_answer"] == "official answer"


def test_pitfall_must_be_positive() -> None:
    row = _row(1)
    row["rubric"][1]["positive_form"] = False
    with pytest.raises(RaRDataError, match="positive avoidance"):
        normalize_rar_row(row, domain="medicine")

    row = _row(2)
    row["rubric"][1].pop("positive_form")
    row["rubric"][1]["criterion"] = "Recommends an unsafe treatment"
    with pytest.raises(RaRDataError, match="positive avoidance"):
        normalize_rar_row(row, domain="medicine")


def test_prepare_rar_source_is_deterministic_and_disjoint(tmp_path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text("".join(json.dumps(_row(index)) + "\n" for index in range(8)))
    first = prepare_rar_source(
        source,
        tmp_path / "one",
        domain="science",
        seed=4,
        train_count=4,
        development_count=2,
        final_count=2,
    )
    second = prepare_rar_source(
        source,
        tmp_path / "two",
        domain="science",
        seed=4,
        train_count=4,
        development_count=2,
        final_count=2,
    )
    assert first["splits"] == second["splits"]
    assert first["stratification"] == second["stratification"]
    assert first["stratification"]["keys"] == ["topic"]
    prompt_sets = [set(item["prompt_ids"]) for item in first["splits"].values()]
    assert not any(left & right for index, left in enumerate(prompt_sets) for right in prompt_sets[index + 1 :])


def test_duplicate_prompt_hashes_are_rejected_even_with_different_source_ids() -> None:
    first = _row(1)
    second = _row(2)
    second["prompt"] = first["prompt"]
    with pytest.raises(RaRDataError, match="prompt hashes"):
        from dynamic_rubric.data.rar import normalize_rar_rows

        normalize_rar_rows((first, second), domain="science")
