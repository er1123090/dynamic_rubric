import dataclasses

import pytest

from dynamic_rubric.phase1.audit_run import AuditTask, equivalent_metrics, run_task
from dynamic_rubric.phase1.audit_scoring import AuditScoreConfig
from dynamic_rubric.providers.base import GenerationResult


class Judge:
    def __init__(self):
        self.calls = []

    def generate(self, request):
        self.calls.append(request)
        return GenerationResult(
            '{"1":"PRESENT"}', "Qwen/Qwen3-32B", "Qwen/Qwen3-32B", "test", None, 0
        )


def test_derived_metric_rounding_is_tolerated_but_structure_is_exact():
    assert equivalent_metrics({"std": 0.09819474285785312}, {"std": 0.09819474285785311})
    assert not equivalent_metrics({"std": 0.1}, {"std": 0.10000001})
    assert not equivalent_metrics({"n": 1}, {"n": True})
    assert not equivalent_metrics({"n": 1}, {"n": 1, "extra": 0})


def task():
    responses = [
        dict(
            prompt_id="A",
            response_id=f"r{i}",
            rollout_index=i,
            text="answer",
            prompt_messages=[dict(role="user", content="question")],
            global_step=3,
            policy_step=2,
        )
        for i in range(2)
    ]
    fresh = [
        dict(response_id=f"r{i}", rollout_index=i, reward=float(i), grades=[["c", i]])
        for i in range(2)
    ]
    return AuditTask(
        3,
        1,
        "A",
        "train:0:A",
        responses,
        fresh,
        [dict(criterion_id="old", text="old criterion", weight=1, source="online_pairwise")],
        "fresh-hash",
        "stale-hash",
        {},
    )


def test_training_pair_is_not_probe_and_resumes(tmp_path):
    judge = Judge()
    cfg = AuditScoreConfig("medicine", concurrency=1)
    value = run_task(task(), tmp_path, cfg, judge, 0.01, 0.01)
    assert [r.seed for r in judge.calls] == [11, 12]
    assert len(judge.calls) == 2  # never regrade fresh
    assert value["comparison"]["v_adj_zar"] == 1
    assert value["comparison"]["same_pool_b"] is False
    assert value["stale_creation_update"] == 1
    assert value["policy_step"] == 2
    assert all(r["fresh_or_stale"] == "stale" for r in value["stale"])
    assert run_task(task(), tmp_path, cfg, judge, 0.01, 0.01) == value
    assert len(judge.calls) == 2


def test_resume_rejects_changed_rubric(tmp_path):
    cfg = AuditScoreConfig("medicine", concurrency=1)
    run_task(task(), tmp_path, cfg, Judge(), 0.01, 0.01)
    with pytest.raises(ValueError, match="identity changed"):
        run_task(
            dataclasses.replace(task(), stale_rubric_hash="changed"),
            tmp_path,
            cfg,
            Judge(),
            0.01,
            0.01,
        )


def test_cli_mutable_status_and_resume(tmp_path, monkeypatch):
    import json
    import sys
    from dynamic_rubric.artifacts import write_json_atomic
    from dynamic_rubric.phase1 import audit_run

    run = tmp_path / "run"
    output = tmp_path / "audit"
    write_json_atomic(
        run / "config.resolved.json",
        dict(
            domain="medicine",
            seed=11,
            models={"judge": {"revision": "pin"}},
            analysis={"epsilon_z": 0.01, "epsilon_t": 0.01},
        ),
    )
    monkeypatch.setattr(
        audit_run, "build_tasks", lambda *_: ([task()], {"fresh_all": [], "eligible_groups": 1})
    )
    monkeypatch.setattr(
        audit_run,
        "endpoint_identity",
        lambda url, *_: {
            "version": "test",
            "model": {"root": "pin" if url.endswith("28002") else "/ram/pin"},
        },
    )
    judge = Judge()
    monkeypatch.setattr(audit_run, "VLLMChatAdapter", lambda *_, **__: judge)
    monkeypatch.setattr(
        sys, "argv", ["audit", "--run", str(run), "--output", str(output), "--limit-groups", "1"]
    )
    audit_run.main()
    audit_run.main()
    assert json.loads((output / "status.json").read_text())["state"] == "smoke_complete"
    assert len(judge.calls) == 2
