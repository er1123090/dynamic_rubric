from __future__ import annotations

import argparse
import json
from pathlib import Path

from dynamic_rubric.hashing import sha256_file
from dynamic_rubric.phase1.audit_run import AuditTask
from scripts.phase1 import certify_audit_judge as canary


def test_selection_is_deterministic_and_spans_ordered_inventory() -> None:
    candidates = [
        {"step": step, "rubric_size": 20 - step, "prompt_id": f"p{step}"} for step in range(1, 17)
    ]
    first = canary.select_groups(list(reversed(candidates)), 8)
    second = canary.select_groups(candidates, 8)

    assert first == second
    assert len(first) == 8
    assert first[0]["step"] == 1
    assert first[-1]["step"] == 16


def _fixture(tmp_path: Path, monkeypatch, *, candidate_reward: float = 1.0):
    run, audit, output = tmp_path / "run", tmp_path / "audit", tmp_path / "canary"
    run.mkdir()
    audit.mkdir()
    (run / "config.resolved.json").write_text(
        json.dumps(
            {
                "domain": "medicine",
                "seed": 11,
                "models": {"judge": {"model": "Qwen/Qwen3-32B", "revision": "revision-pin"}},
            }
        ),
        encoding="utf-8",
    )
    (audit / "judge_identity.json").write_text(
        json.dumps(
            {
                "model": "Qwen/Qwen3-32B",
                "revision": "revision-pin",
                "model_root": "/inference_b/revision-pin",
                "vllm_version": {"version": "0.19.1"},
            }
        ),
        encoding="utf-8",
    )
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
    rubric = [{"criterion_id": "c", "text": "correct", "weight": 1}]
    task = AuditTask(3, 1, "p", "train:0:p", responses, [], rubric, "fresh", "stale", {})
    reference = [
        {
            "response_id": f"r{i}",
            "grades": [["c", 1]],
            "numerator": 1.0,
            "denominator": 1.0,
            "reward": 1.0,
        }
        for i in range(2)
    ]
    group = audit / "groups" / "step-000003" / "p.json"
    group.parent.mkdir(parents=True)
    group.write_text(json.dumps({"stale": reference}), encoding="utf-8")
    original_group_hash = sha256_file(group)
    monkeypatch.setattr(
        canary,
        "build_tasks",
        lambda *_args: ([task], {"source_hashes": [{"sha256": "source"}]}),
    )

    def identity(url: str, model: str, revision: str) -> dict:
        assert (output / "manifest.json").is_file()  # selection froze before endpoint access
        assert model == "Qwen/Qwen3-32B"
        assert revision == "revision-pin"
        return {
            "url": url,
            "model": {"id": model, "root": f"/trainer/{revision}"},
            "version": {"version": "0.19.1"},
        }

    monkeypatch.setattr(canary, "endpoint_identity", identity)
    monkeypatch.setattr(canary, "VLLMChatAdapter", lambda *_args, **_kwargs: object())

    def scoring(responses_arg, rubric_arg, **kwargs):
        assert [row["response_id"] for row in responses_arg] == ["r0", "r1"]
        assert rubric_arg == {"p": rubric}
        assert kwargs["config"].concurrency == 8
        assert kwargs["config"].seed == 11
        rows = [dict(row) for row in reference]
        rows[-1]["reward"] = candidate_reward
        return rows

    monkeypatch.setattr(canary, "score_pool", scoring)
    args = argparse.Namespace(
        run=run,
        audit=audit,
        output=output,
        candidate_url="http://127.0.0.1:28007",
        through=34,
        groups=1,
        group_workers=8,
    )
    return args, group, original_group_hash


def test_certification_freezes_manifest_and_preserves_canonical_receipts(
    tmp_path: Path, monkeypatch
) -> None:
    args, group, original_group_hash = _fixture(tmp_path, monkeypatch)
    result = canary.certify(args)

    assert result == {
        "passed": True,
        "selected_groups": 1,
        "compared_responses": 2,
        "exact_mismatches": 0,
        "tolerance_only_rewards": 0,
        "candidate_url": args.candidate_url,
        "reference": "existing immutable Inference B stale receipts",
    }
    assert sha256_file(group) == original_group_hash
    manifest = json.loads((args.output / "manifest.json").read_text())
    assert manifest["selection_policy"].startswith("even indices")
    assert manifest["selected"][0]["group_sha256"] == original_group_hash
    assert (args.output / "mismatches.jsonl").read_text() == ""


def test_exact_reward_gate_reports_tolerance_only_difference(tmp_path: Path, monkeypatch) -> None:
    args, _group, _hash = _fixture(tmp_path, monkeypatch, candidate_reward=1.0 + 2e-15)
    result = canary.certify(args)
    mismatch = json.loads((args.output / "mismatches.jsonl").read_text())

    assert result["passed"] is False
    assert result["exact_mismatches"] == 1
    assert result["tolerance_only_rewards"] == 1
    assert mismatch["exact"]["grades"] is True
    assert mismatch["exact"]["reward"] is False
    assert mismatch["tolerance_only"] is True
    frozen_manifest_hash = sha256_file(args.output / "manifest.json")
    manifest = json.loads((args.output / "manifest.json").read_text())
    frozen = manifest["selected"][0]
    newly_completed = {
        **frozen,
        "step": 34,
        "prompt_id": "later-group",
        "group_sha256": "later-hash",
    }
    monkeypatch.setattr(canary, "_completed_inventory", lambda *_args: [frozen, newly_completed])
    assert canary.certify(args)["passed"] is False
    assert sha256_file(args.output / "manifest.json") == frozen_manifest_hash
