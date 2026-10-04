from __future__ import annotations

import json
from pathlib import Path

import pytest

from dynamic_rubric.artifacts import read_json
from dynamic_rubric.minimum_gold import GOLD_MAX_OUTPUT_TOKENS, MinimumGoldError
from dynamic_rubric.minimum_gold_streaming import prepare_gold_selection_shard
from dynamic_rubric.minimum_staleness import _shard_name, select_bon_shard


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_select_bon_shard_publishes_full_group_n_curve(tmp_path: Path) -> None:
    candidates = [
        {
            "global_candidate_id": index,
            "response_id": f"r-{index}",
            "response_text": f"response {index}",
        }
        for index in range(1024)
    ]
    policy_id = "pi_3"
    prompt_id = "prompt-1"
    shard = f"{policy_id}-{_shard_name(policy_id, prompt_id)}.jsonl"
    score_rows = []
    for rubric_id, mode, rubric_step in (
        ("static-rubric", "static", 0),
        ("dynamic-rubric", "dynamic_fixed_budgeted", 3),
    ):
        for candidate in candidates:
            score_rows.append(
                {
                    "policy_id": policy_id,
                    "policy_step": 3,
                    "prompt_id": prompt_id,
                    "rubric_id": rubric_id,
                    "mode": mode,
                    "rubric_step": rubric_step,
                    "global_candidate_id": candidate["global_candidate_id"],
                    "score": float(candidate["global_candidate_id"]),
                }
            )
    _write_jsonl(tmp_path / "generate-bon" / "bon_pool.jsonl", [{}])
    _write_jsonl(tmp_path / "score-proxy-minimum" / "manifest.json", [{}])
    _write_jsonl(
        tmp_path / "score-proxy-minimum" / "shards" / shard,
        score_rows,
    )

    output = select_bon_shard(
        tmp_path, policy_id, prompt_id, candidates  # type: ignore[arg-type]
    )

    rows = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert len(rows) == 2 * 11 * 5
    assert {row["mode"] for row in rows} == {"static", "dynamic_fixed_budgeted"}
    assert read_json(tmp_path / "select-bon-minimum" / "progress.json")[
        "completed_prompt_policy_shards"
    ] == 1


def test_prepare_streaming_gold_deduplicates_one_selection_group(
    tmp_path: Path,
) -> None:
    selection = tmp_path / "select-bon-minimum" / "shards" / "pi_3-group.jsonl"
    selected = {
        "policy_id": "pi_3",
        "prompt_id": "prompt-1",
        "response_id": "response-1",
        "response_text": "safe answer",
    }
    _write_jsonl(selection, [selected, {**selected, "n": 2}])
    private_gt = tmp_path / "private.jsonl"
    _write_jsonl(
        private_gt,
        [
            {
                "prompt_id": "prompt-1",
                "gold_rubric": [{"criterion": "Be safe", "points": 1}],
            }
        ],
    )
    schema = tmp_path / "schema.json"
    schema.write_text(
        json.dumps(
            {
                "type": "object",
                "properties": {
                    "criterion_scores": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "criterion_id": {"type": "string"},
                                "score": {"type": "number"},
                            },
                            "required": ["criterion_id", "score"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["criterion_scores"],
                "additionalProperties": False,
            }
        ),
        encoding="utf-8",
    )

    first = prepare_gold_selection_shard(tmp_path, selection, private_gt, schema)
    second = prepare_gold_selection_shard(tmp_path, selection, private_gt, schema)

    assert first == second
    assert first["requests"] == 1
    input_path = Path(str(first["input_file"]["path"]))
    lines = input_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["body"]["max_output_tokens"] == GOLD_MAX_OUTPUT_TOKENS


def test_prepare_streaming_gold_rejects_selection_outside_stage(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside.jsonl"
    _write_jsonl(outside, [{"prompt_id": "p", "response_id": "r", "response_text": "x"}])
    private_gt = tmp_path / "private.jsonl"
    schema = tmp_path / "schema.json"
    _write_jsonl(private_gt, [])
    schema.write_text("{}", encoding="utf-8")

    with pytest.raises(MinimumGoldError, match="outside"):
        prepare_gold_selection_shard(tmp_path, outside, private_gt, schema)
