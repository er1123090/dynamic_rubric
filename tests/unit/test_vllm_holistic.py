from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import pytest

from dynamic_rubric.artifacts import write_json_atomic
from dynamic_rubric.providers.vllm_holistic import (
    NO_TARGET,
    TARGETS,
    YES_TARGET,
    HybridVLLMFullRubricGrader,
)


def _grader(tmp_path: Path) -> HybridVLLMFullRubricGrader:
    return HybridVLLMFullRubricGrader(
        base_urls=["http://trainer", "http://inference_a-0", "http://inference_a-1"],
        served_model="grader",
        model_revision="revision",
        tokenizer_revision="tokenizer",
        criterion_cache_dir=tmp_path / "criterion-cache",
        holistic_cache_dir=tmp_path / "holistic-cache",
        max_workers=2,
    )


def _items() -> list[tuple[str, str, str, str, str]]:
    return [
        ("p1", "r1", "response", "c1", "criterion one"),
        ("p1", "r1", "response", "c2", "criterion two"),
    ]


def _write_criterion_cache(
    grader: HybridVLLMFullRubricGrader,
    item: tuple[str, str, str, str, str],
    prompt_text: str,
    *,
    yes: float,
    no: float,
) -> None:
    rendered = grader._render_criterion_prompt(prompt_text, item[2], item[4])
    key = grader._criterion_cache_key(rendered)
    path = grader.criterion_cache_dir / key[:2] / f"{key}.json"
    write_json_atomic(
        path,
        {
            "schema_version": 1,
            "key": key,
            "identity": grader.criterion_cache_identity,
            "rendered_prompt_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
            "targets": TARGETS,
            "target_logprobs": {YES_TARGET: yes, NO_TARGET: no},
        },
    )


def test_reuses_legacy_cache_only_when_the_entire_response_rubric_is_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    grader = _grader(tmp_path)
    items = _items()
    prompt_text = '[{"content": "question", "role": "user"}]'
    _write_criterion_cache(grader, items[0], prompt_text, yes=-0.1, no=-2.0)
    _write_criterion_cache(grader, items[1], prompt_text, yes=-2.0, no=-0.1)

    def fail_request(**_: Any) -> tuple[dict[str, str], int]:
        raise AssertionError("complete response cache should avoid a holistic request")

    monkeypatch.setattr(grader, "_request_holistic", fail_request)
    scores = grader.score_many_full(items, prompt_text_by_id={"p1": prompt_text})

    assert [score.criterion_id for score in scores] == ["c1", "c2"]
    assert [score.yes_logprob for score in scores] == [-0.1, -2.0]
    stats = grader.artifact_metadata["stats"]
    assert stats["reused_complete_responses"] == 1
    assert stats["reused_criterion_items"] == 2
    assert stats["holistic_responses"] == 0


def test_partial_legacy_cache_is_ignored_and_full_rubric_is_graded_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    grader = _grader(tmp_path)
    items = _items()
    prompt_text = '[{"content": "question", "role": "user"}]'
    _write_criterion_cache(grader, items[0], prompt_text, yes=-0.1, no=-2.0)
    calls: list[tuple[list[str], list[str]]] = []

    def request_holistic(*, request_payload, criteria, **_):
        required = request_payload["response_format"]["json_schema"]["schema"]["required"]
        calls.append((required, [criterion.criterion_id for criterion in criteria]))
        return {"c1": "NOT_PRESENT", "c2": "PRESENT"}, 0

    monkeypatch.setattr(grader, "_request_holistic", request_holistic)
    scores = grader.score_many_full(items, prompt_text_by_id={"p1": prompt_text})

    assert calls == [(["1", "2"], ["c1", "c2"])]
    assert [score.yes_logprob for score in scores] == [-1.0, 0.0]
    stats = grader.artifact_metadata["stats"]
    assert stats["ignored_partial_criterion_items"] == 1
    assert stats["holistic_responses"] == 1
    assert stats["reused_complete_responses"] == 0

    grader.score_many_full(items, prompt_text_by_id={"p1": prompt_text})
    assert len(calls) == 1
    assert grader.artifact_metadata["stats"]["holistic_cache_hits"] == 1


def test_transport_uses_complete_exact_key_regex_and_preserves_logical_request(
    tmp_path: Path,
) -> None:
    grader = _grader(tmp_path)
    logical = grader._holistic_request(
        prompt=[{"role": "user", "content": "question"}],
        response_text="response",
        items=_items(),
    )

    transport = grader._transport_request(logical)

    assert "response_format" in logical
    assert "structured_outputs" not in logical
    assert "response_format" not in transport
    pattern = transport["structured_outputs"]["regex"]
    assert re.fullmatch(pattern, '{"1":"PRESENT","2":"NOT_PRESENT"}')
    assert not re.fullmatch(pattern, '{"1":"PRESENT"}')
    assert not re.fullmatch(pattern, '{"1": "PRESENT","2":"NOT_PRESENT"}')
    assert (
        grader.artifact_metadata["vllm_structured_output_transport"]
        == "exact_key_regex_no_whitespace_v1"
    )


def test_concurrent_cache_winner_is_canonical_when_labels_differ(
    tmp_path: Path,
) -> None:
    grader = _grader(tmp_path)
    items = _items()
    logical = grader._holistic_request(
        prompt=[{"role": "user", "content": "question"}],
        response_text="response",
        items=items,
    )
    key = grader._holistic_cache_key(logical)
    labels = {"c1": "PRESENT", "c2": "NOT_PRESENT"}

    grader._publish_holistic_cache(
        key=key,
        request_payload=logical,
        labels=labels,
        retry_count=0,
    )
    published = grader._publish_holistic_cache(
        key=key,
        request_payload=logical,
        labels={"c1": "NOT_PRESENT", "c2": "PRESENT"},
        retry_count=1,
    )

    assert grader._read_holistic_cache(
        key=key,
        request_payload=logical,
        criterion_ids=["c1", "c2"],
    ) == (labels, 0)
    assert published == (labels, 0)
