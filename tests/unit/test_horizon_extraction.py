from __future__ import annotations

from dynamic_rubric.horizon.extraction import (
    prepare_dedup_request,
    prepare_extraction_requests,
)


def rows(family: str) -> list[dict[str, str]]:
    return [
        {"prompt_id": "medicine:p", "response_text": f"{family} response {index}"}
        for index in range(8)
    ]


def test_exactly_eight_blinded_paper_requests_are_stable_and_source_blind() -> None:
    kwargs = {
        "prompt_id": "medicine:p",
        "checkpoint_id": "step3",
        "prompt": [{"role": "user", "content": "What is the answer?"}],
        "existing_r0": [{"criterion": "Is correct", "weight": 1.0}],
        "current_rows": rows("current"),
        "control_rows": rows("control"),
        "pairing_seed": 17,
    }
    requests = prepare_extraction_requests(**kwargs)
    assert requests == prepare_extraction_requests(**kwargs)
    assert len(requests) == 8
    assert len({item["source_pair_id"] for item in requests}) == 8
    serialized = repr(requests)
    assert "current_label" not in serialized and "control_label" not in serialized
    assert all(len(item["messages"]) == 2 for item in requests)


def test_dedup_request_binds_allowed_candidate_identity_set() -> None:
    request = prepare_dedup_request(
        prompt_id="science:p",
        checkpoint_id="step6",
        prompt=[{"role": "user", "content": "Explain."}],
        existing_r0=[{"criterion": "Is correct", "weight": 1.0}],
        candidate_criteria=[
            {"candidate_id": "b", "criterion": "B", "weight": 1},
            {"candidate_id": "a", "criterion": "A", "weight": 2},
        ],
    )
    assert request["allowed_source_candidate_ids"] == ["a", "b"]
