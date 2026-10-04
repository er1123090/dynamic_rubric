from __future__ import annotations

from pathlib import Path

import pytest

from dynamic_rubric.artifacts import write_json_atomic
from dynamic_rubric.minimum_gold import MinimumGoldError, _egress_approval_sha256


def _approval() -> dict[str, object]:
    return {
        "approved": True,
        "destination": "OpenAI GPT-5 Batch API",
        "endpoint": "/v1/responses",
        "purpose": "hidden-gold-evaluation",
        "requested_model": "gpt-5",
        "payload_categories": [
            "selected_medical_response_texts",
            "private_physician_rubrics",
            "criterion_weights",
        ],
    }


def test_egress_approval_binds_exact_destination_payload_and_purpose(tmp_path: Path) -> None:
    path = tmp_path / "train-static" / "gold-egress-approval.json"
    write_json_atomic(path, _approval())
    assert len(_egress_approval_sha256(tmp_path)) == 64


def test_egress_approval_rejects_scope_drift(tmp_path: Path) -> None:
    value = _approval()
    value["payload_categories"] = ["selected_medical_response_texts"]
    write_json_atomic(tmp_path / "train-static" / "gold-egress-approval.json", value)
    with pytest.raises(MinimumGoldError, match="scope"):
        _egress_approval_sha256(tmp_path)
