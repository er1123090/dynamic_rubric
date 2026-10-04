from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from dynamic_rubric.artifacts import read_json, write_jsonl_atomic
from dynamic_rubric.horizon.checkpoint_kl import (
    VLLMPolicyLogprobClient,
    _encode_float32,
    analyze_adjacent_checkpoint_kl,
)


class _Response:
    def __init__(self, payload: object) -> None:
        self.payload = payload

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode()


def test_vllm_policy_logprob_client_extracts_only_response_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[Any] = []

    def urlopen(request: Any, *, timeout: float) -> _Response:
        captured.append((json.loads(request.data), timeout))
        return _Response(
            {
                "model": "policy",
                "choices": [
                    {
                        "index": 0,
                        "prompt_token_ids": [10, 11, 12, 13],
                        "prompt_logprobs": [
                            None,
                            {"11": {"logprob": -0.1}},
                            {"12": {"logprob": -0.2}},
                            {"13": {"logprob": -0.3}},
                        ],
                    }
                ],
            }
        )

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    client = VLLMPolicyLogprobClient(
        "http://policy",
        served_model="policy",
        model_revision="revision",
        tokenizer_revision="tokenizer",
        checkpoint_hash="hash",
    )

    assert client.score([[10, 11, 12, 13]], [2]) == [pytest.approx((-0.2, -0.3))]
    assert captured[0][0]["prompt_logprobs"] == 0
    assert captured[0][0]["max_tokens"] == 1


def _score_row(
    *,
    policy_step: int,
    pool_step: int,
    prompt_id: str,
    sample_index: int,
    logprobs: list[float],
) -> dict[str, object]:
    response_id = f"{prompt_id}-{sample_index}"
    return {
        "schema_version": 1,
        "policy_step": policy_step,
        "pool_policy_step": pool_step,
        "scoring_checkpoint_hash": f"checkpoint-{policy_step}",
        "source_checkpoint_hash": f"checkpoint-{pool_step}",
        "prompt_id": prompt_id,
        "response_id": response_id,
        "sample_index": sample_index,
        "response_token_count": len(logprobs),
        "response_token_hash": f"token-hash-{response_id}",
        "response_token_logprobs_f32le_b64": _encode_float32(logprobs),
    }


def test_adjacent_checkpoint_kl_is_prompt_balanced_and_sealed(tmp_path: Path) -> None:
    prompts = tmp_path / "prompts.jsonl"
    write_jsonl_atomic(
        prompts,
        [
            {"prompt_id": "p1", "messages": [{"role": "user", "content": "a"}]},
            {"prompt_id": "p2", "messages": [{"role": "user", "content": "b"}]},
        ],
    )
    score_dir = tmp_path / "scores"
    score_dir.mkdir()
    old_rows = []
    new_rows = []
    for prompt_id in ("p1", "p2"):
        for sample_index in range(2):
            old = [-1.0, -2.0, -3.0]
            new = [value - 0.2 for value in old]
            old_rows.append(
                _score_row(
                    policy_step=0,
                    pool_step=0,
                    prompt_id=prompt_id,
                    sample_index=sample_index,
                    logprobs=old,
                )
            )
            new_rows.append(
                _score_row(
                    policy_step=3,
                    pool_step=0,
                    prompt_id=prompt_id,
                    sample_index=sample_index,
                    logprobs=new,
                )
            )
    write_jsonl_atomic(score_dir / "policy-step-0_pool-step-0.jsonl", old_rows)
    write_jsonl_atomic(score_dir / "policy-step-3_pool-step-0.jsonl", new_rows)
    output_dir = tmp_path / "result"

    result = analyze_adjacent_checkpoint_kl(
        score_dir=score_dir,
        checkpoint_steps=[0, 3],
        prompts_path=prompts,
        output_dir=output_dir,
        expected_responses_per_prompt=2,
    )
    summary = read_json(output_dir / "adjacent_kl_summary.json")
    pair = summary["pairs"][0]

    assert result["pairs"] == 1
    assert pair["k1_prompt_balanced_mean"] == pytest.approx(0.2)
    assert pair["k1_prompt_balanced_per_step_proxy"] == pytest.approx(0.2 / 3)
    assert pair["k3_clipped_prompt_balanced_mean"] == pytest.approx(
        math.expm1(-0.2) + 0.2
    )
    assert pair["k3_clipped_token_fraction"] == 0.0
    assert (output_dir / "adjacent_kl_seal.json").is_file()

    reused = analyze_adjacent_checkpoint_kl(
        score_dir=score_dir,
        checkpoint_steps=[0, 3],
        prompts_path=prompts,
        output_dir=output_dir,
        expected_responses_per_prompt=2,
    )
    assert reused["reused"] is True
