from __future__ import annotations

import json
from pathlib import Path

from dynamic_rubric import paper_judge_experiment
from dynamic_rubric.artifacts import read_json
from dynamic_rubric.judge_prompts import PAPER_JUDGE_PROMPT_VERSION
from dynamic_rubric.minimum_gold_streaming import prepare_gold_selection_shard


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_paper_gold_preparation_audits_old_new_selection_union(tmp_path: Path) -> None:
    run_root = tmp_path / "artifacts" / "runs" / "paper"
    primary = run_root / "select-bon-minimum" / "shards" / "pi_3-group.jsonl"
    old = tmp_path / "old-selection.jsonl"
    _write_jsonl(
        primary,
        [
            {
                "prompt_id": "prompt-1",
                "response_id": "new-response",
                "response_text": "new answer",
            }
        ],
    )
    _write_jsonl(
        old,
        [
            {
                "prompt_id": "prompt-1",
                "response_id": "old-response",
                "response_text": "old answer",
            },
            {
                "prompt_id": "prompt-1",
                "response_id": "new-response",
                "response_text": "new answer",
            },
        ],
    )
    _write_jsonl(
        tmp_path / "data" / "public" / "pilot_audit.jsonl",
        [
            {
                "prompt_id": "prompt-1",
                "messages": [{"role": "user", "content": "medical question"}],
            }
        ],
    )
    private_gt = tmp_path / "private.jsonl"
    _write_jsonl(
        private_gt,
        [
            {
                "prompt_id": "prompt-1",
                "gold_rubric": [{"criterion": "is safe", "points": 1}],
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
                                "score": {"type": "integer", "enum": [0, 1]},
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

    manifest = prepare_gold_selection_shard(
        run_root,
        primary,
        private_gt,
        schema,
        additional_selection_paths=(old,),
        prompt_version=PAPER_JUDGE_PROMPT_VERSION,
    )

    assert manifest["requests"] == 2
    assert manifest["prompt_version"] == PAPER_JUDGE_PROMPT_VERSION
    assert len(manifest["selection_shards"]) == 2
    identities = [
        json.loads(line)
        for line in Path(manifest["request_map"]["path"]).read_text(encoding="utf-8").splitlines()
    ]
    assert {row["response_id"] for row in identities} == {
        "old-response",
        "new-response",
    }
    saved = read_json(
        run_root / "audit-gold-streaming-private" / "groups" / primary.stem / "manifest.json"
    )
    assert saved == manifest


def test_paper_gold_subset_excludes_old_selection(tmp_path: Path, monkeypatch) -> None:
    run_root = tmp_path / "paper"
    source_root = tmp_path / "source"
    new_selection = run_root / "select-bon-minimum" / "shards" / "new.jsonl"
    old_selection = source_root / "select-bon-minimum" / "shards" / "old.jsonl"
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        paper_judge_experiment,
        "ordered_prompt_subset",
        lambda _run_root, _prompt_count: ("prompt-1",),
    )
    monkeypatch.setattr(
        paper_judge_experiment,
        "_group_paths",
        lambda _run_root, _source_root, _prompt_ids: iter(
            [("pi_3", "prompt-1", new_selection, old_selection)]
        ),
    )

    def fake_prepare(*args, **kwargs):
        captured["selection_path"] = args[1]
        captured["kwargs"] = kwargs
        return {"requests": 1}

    monkeypatch.setattr(
        paper_judge_experiment,
        "prepare_gold_selection_shard",
        fake_prepare,
    )

    result = paper_judge_experiment.prepare_or_submit_gold_subset(
        run_root,
        source_root,
        tmp_path / "private.jsonl",
        tmp_path / "schema.json",
        prompt_count=1,
        submit=False,
    )

    assert result["requests"] == 1
    assert captured["selection_path"] == new_selection
    assert isinstance(captured["kwargs"], dict)
    assert "additional_selection_paths" not in captured["kwargs"]
