from __future__ import annotations

import json
from pathlib import Path

from dynamic_rubric.artifacts import read_json
from dynamic_rubric.human_static_gold_reuse import prepare_reused_gold_selection_shard
from dynamic_rubric.human_static_online_bon import load_human_rubrics
from dynamic_rubric.minimum_interim import _human_static_repeat_compatible_scores
from dynamic_rubric.judge_prompts import PAPER_JUDGE_PROMPT_VERSION
from dynamic_rubric.minimum_gold import REASONING_EFFORT, REQUESTED_MODEL
from dynamic_rubric.hashing import sha256_file


def _jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_load_human_rubrics_preserves_signed_points(tmp_path: Path) -> None:
    private_gt = tmp_path / "private.jsonl"
    _jsonl(
        private_gt,
        [
            {
                "prompt_id": "p1",
                "gold_rubric": [
                    {"criterion": "required", "points": 3},
                    {"criterion": "unsafe", "points": -2},
                ],
            }
        ],
    )

    rubric = load_human_rubrics(private_gt, ("p1",))["p1"]

    assert [row["points"] for row in rubric] == [3.0, -2.0]
    assert len({row["criterion_key"] for row in rubric}) == 2


def test_human_static_repeat_compatibility_is_scoped_to_human_manifest(
    tmp_path: Path,
) -> None:
    stage = tmp_path / "score-proxy-minimum"
    stage.mkdir(parents=True)
    (stage / "manifest.json").write_text('{"static_r0":"human_gt"}')
    rows = [{"score": 0.75, "mode": "static"}]

    compatible = _human_static_repeat_compatible_scores(tmp_path, rows)

    assert compatible[0]["judge_repeat_score"] == 0.75
    assert "judge_repeat_score" not in rows[0]


def test_repeat_compatibility_keeps_non_human_runs_strict(tmp_path: Path) -> None:
    stage = tmp_path / "score-proxy-minimum"
    stage.mkdir(parents=True)
    (stage / "manifest.json").write_text('{"static_r0":"generated"}')
    rows = [{"score": 0.75, "mode": "static"}]

    assert _human_static_repeat_compatible_scores(tmp_path, rows) == rows


def test_prepare_reused_gold_finishes_cache_only_group(
    tmp_path: Path, monkeypatch
) -> None:
    run_root = tmp_path / "runs" / "new"
    selection = run_root / "select-bon-minimum" / "shards" / "pi_3-group.jsonl"
    _jsonl(
        selection,
        [
            {
                "prompt_id": "p1",
                "response_id": "r1",
                "response_text": "answer",
            }
        ],
    )
    private_gt = tmp_path / "private.jsonl"
    _jsonl(
        private_gt,
        [{"prompt_id": "p1", "gold_rubric": [{"criterion": "good", "points": 1}]}],
    )
    schema = tmp_path / "schema.json"
    schema.write_text("{}")
    reuse = tmp_path / "runs" / "reuse"
    group = reuse / "audit-gold-streaming-private" / "groups" / "old"
    _jsonl(
        group / "gold_scores.jsonl",
        [
            {
                "prompt_id": "p1",
                "response_id": "r1",
                "gold_score": 1.0,
                "criterion_scores": {"gold-000": 1.0},
                "requested_model": REQUESTED_MODEL,
                "returned_model": "gpt-5-2025-08-07",
                "reasoning_effort": REASONING_EFFORT,
            }
        ],
    )
    (group / "manifest.json").write_text(
        json.dumps(
            {
                "prompt_version": PAPER_JUDGE_PROMPT_VERSION,
                "private_gt_sha256": sha256_file(private_gt),
                "schema_sha256": sha256_file(schema),
                "requested_model": REQUESTED_MODEL,
                "reasoning_effort": REASONING_EFFORT,
            }
        )
    )
    monkeypatch.setattr(
        "dynamic_rubric.human_static_gold_reuse._audit_conversations",
        lambda _run_root: {"p1": [{"role": "user", "content": "question"}]},
    )

    manifest = prepare_reused_gold_selection_shard(
        run_root,
        selection,
        private_gt,
        schema,
        reuse_roots=(reuse,),
    )

    assert manifest["requests"] == 0
    assert manifest["cached_responses"] == 1
    result = read_json(
        run_root / "audit-gold-streaming-private" / "groups" / "pi_3-group" / "result.json"
    )
    assert result["grader_calls"] == 0
    assert result["cached_responses"] == 1
