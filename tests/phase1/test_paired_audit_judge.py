from __future__ import annotations

import json

from dynamic_rubric.phase1.audit_run import AuditTask
from dynamic_rubric.phase1.audit_scoring import AuditScoreConfig
from scripts.phase1 import paired_audit_judge as paired


def _task() -> AuditTask:
    responses = [
        {
            "response_id": f"r{i}",
            "prompt_id": "p",
            "prompt_messages": [{"role": "user", "content": "question"}],
            "text": "answer",
            "rollout_index": i,
            "global_step": 3,
            "policy_step": 2,
        }
        for i in range(2)
    ]
    canonical = [
        {"response_id": f"r{i}", "grades": [["fresh", i]], "reward": float(i)} for i in range(2)
    ]
    return AuditTask(
        3,
        1,
        "p",
        "train:0:p",
        responses,
        canonical,
        [{"criterion_id": "stale", "text": "old", "weight": 1}],
        "fresh-hash",
        "stale-hash",
        {},
    )


def test_loads_same_occurrence_fresh_rubric_and_verifies_hash(tmp_path) -> None:
    task = _task()
    path = tmp_path / "verl-run" / "online_steps" / "step-000003" / "rubric_unions.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "prompt_occurrence_id": task.occurrence_id,
                "offline_criteria": [{"criterion_id": "base", "text": "base", "weight": 1}],
                "online_criteria": [{"criterion_id": "fresh", "text": "new", "weight": 1}],
                "content_hash": task.fresh_rubric_hash,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = paired.load_fresh_rubrics(tmp_path, [task])
    assert [row["criterion_id"] for row in result[(3, task.occurrence_id)]] == [
        "base",
        "fresh",
    ]


def test_paired_group_scores_same_responses_and_resumes_immutably(tmp_path, monkeypatch) -> None:
    task = _task()
    fresh_rubric = [{"criterion_id": "fresh", "text": "new", "weight": 1}]
    calls: list[str] = []

    def score_pool(responses, rubric_by_prompt, **kwargs):
        criterion = rubric_by_prompt["p"][0]["criterion_id"]
        calls.append(criterion)
        rewards = [0.0, 1.0] if criterion == "fresh" else [0.0, 0.0]
        return [
            {
                "response_id": row["response_id"],
                "grades": [[criterion, int(rewards[index] > 0)]],
                "numerator": rewards[index],
                "denominator": 1.0,
                "reward": rewards[index],
            }
            for index, row in enumerate(responses)
        ]

    monkeypatch.setattr(paired, "score_pool", score_pool)
    config = AuditScoreConfig("medicine", concurrency=8)
    first = paired.run_group(
        task,
        fresh_rubric,
        tmp_path,
        config,
        object(),
        0.01,
        0.01,
        "endpoint-hash",
    )
    second = paired.run_group(
        task,
        fresh_rubric,
        tmp_path,
        config,
        object(),
        0.01,
        0.01,
        "endpoint-hash",
    )

    assert first == second
    assert calls == ["fresh", "stale"]
    assert [row["response_id"] for row in first["fresh"]] == ["r0", "r1"]
    assert [row["response_id"] for row in first["stale"]] == ["r0", "r1"]
    assert first["comparison"]["v_adj_zar"] == 1.0
    assert first["canonical_fresh_reference"] == task.fresh
    assert "reference-only" in first["canonical_fresh_reference_role"]
    assert "same-H200 paired sensitivity" in first["interpretation"]
