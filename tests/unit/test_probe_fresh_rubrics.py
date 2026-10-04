from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from dynamic_rubric.artifacts import read_json, read_jsonl, write_json_atomic, write_jsonl_atomic
from dynamic_rubric.hashing import sha256_file, sha256_json
from dynamic_rubric.phase1 import probe_fresh_rubrics
from dynamic_rubric.phase1.probe_fresh_rubrics import (
    ProbeFreshRubricError,
    build_probe_fresh_rubrics,
)
from dynamic_rubric.providers.base import GenerationResult
from dynamic_rubric.training.online_contracts import PolicySnapshot, ResponseRecord


class _Provider:
    def __init__(self) -> None:
        self.calls = 0
        self.lock = threading.Lock()

    def generate(self, request):
        with self.lock:
            self.calls += 1
        if request.family == "online_rubric_extraction":
            value = {"analysis": "no addition", "new_criteria": []}
        else:
            value = {"analysis": "nothing to deduplicate", "final_criteria": []}
        return GenerationResult(
            text=json.dumps(value),
            requested_model="openai/gpt-oss-120b",
            returned_model="openai/gpt-oss-120b",
            request_id=f"request-{self.calls}",
            created_at=1,
            retry_count=0,
            usage={"prompt_tokens": 1, "completion_tokens": 1},
            raw_response_hash=sha256_json(value),
        )


class _ControlCache:
    model = "Qwen/Qwen3-4B-Instruct-2507"
    revision = "base-revision"
    checkpoint_hash = "b" * 64

    def __init__(self, _path: Path, *, expected_prompt_count: int) -> None:
        assert expected_prompt_count == 1500

    def bind(self, occurrence, *, step: int):
        snapshot = PolicySnapshot(0, self.checkpoint_hash, self.model, self.revision)
        records = tuple(
            ResponseRecord(
                occurrence.prompt_occurrence_id,
                f"control-{occurrence.prompt_id}-{index}",
                index,
                f"control response {index}",
                snapshot,
                "control",
            )
            for index in range(8)
        )
        receipts = tuple(
            {"response_id": item.response_id, "prompt_id": occurrence.prompt_id}
            for item in records
        )
        return records, receipts


def _inputs(tmp_path: Path, *, pool_name: str = "probe_A") -> dict[str, Path]:
    prompt_ids = [f"prompt-{index:03d}" for index in range(100)]
    train = tmp_path / "train.jsonl"
    write_jsonl_atomic(
        train,
        [
            {
                "prompt_id": prompt_id,
                "messages": [{"role": "user", "content": f"question {prompt_id}"}],
                "r0": {
                    "criteria": [
                        {
                            "criterion_id": f"{prompt_id}:r0:0",
                            "criterion": "Answers the question correctly.",
                            "weight_units": 10,
                        }
                    ]
                },
            }
            for prompt_id in prompt_ids
        ],
    )
    probe = tmp_path / "probe.json"
    write_json_atomic(
        probe,
        {
            "prompt_ids": prompt_ids,
            "prompt_ids_sha256": sha256_json(prompt_ids),
            "source_sha256": sha256_file(train),
        },
    )
    pool_a = tmp_path / "probe_A.jsonl"
    write_jsonl_atomic(
        pool_a,
        [
            {
                "prompt_id": prompt_id,
                "pool": pool_name,
                "policy_checkpoint": 3,
                "checkpoint_hash": "a" * 64,
                "sample_index": sample,
                "response_id": f"current-{prompt_id}-{sample}",
                "response_text": f"current response {sample}",
                "model": "Qwen/Qwen3-4B-Instruct-2507",
                "model_revision": "policy-revision",
            }
            for prompt_id in prompt_ids
            for sample in range(8)
        ],
    )
    pi0 = tmp_path / "pi0.json"
    pi0.write_text("{}\n", encoding="utf-8")
    return {"train": train, "probe": probe, "pool_a": pool_a, "pi0": pi0}


@pytest.mark.parametrize('workers', [4, 8])
def test_fixed_train_fresh_rubrics_are_resumable_and_record_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, workers: int
) -> None:
    paths = _inputs(tmp_path)
    monkeypatch.setattr(probe_fresh_rubrics, "ImmutablePi0Cache", _ControlCache)
    provider = _Provider()
    kwargs = {
        "run_id": "phase1-run",
        "train_path": paths["train"],
        "probe_manifest_path": paths["probe"],
        "pool_a_path": paths["pool_a"],
        "pi0_manifest_path": paths["pi0"],
        "checkpoint_step": 3,
        "checkpoint_hash": "a" * 64,
        "seed": 11,
        "extractor_model": "openai/gpt-oss-120b",
        "extractor_returned_model": "openai/gpt-oss-120b",
        "output_root": tmp_path / "output",
        "prompt_workers": workers,
        "extractor_concurrency": 8,
    }

    result = build_probe_fresh_rubrics(provider, **kwargs)

    assert result["prompt_count"] == 100
    assert result["extraction_request_count"] == 800
    assert result["dedup_request_count"] == 100
    assert provider.calls == 900
    rows = read_jsonl(tmp_path / "output/checkpoint-000003/fresh_rubrics.jsonl")
    assert len(rows) == 100
    assert all(len(row["pool_a_provenance"]) == 8 for row in rows)
    assert all(len(row["pi0_control_provenance"]) == 8 for row in rows)
    assert all(row["fresh_rubric"]["online_criteria"] == [] for row in rows)
    assert all(
        (row["domain"], row["method"], row["global_step"], row["checkpoint_id"],
         row["pool"], row["policy_checkpoint"], row["evaluator_checkpoint"],
         row["fresh_or_stale"])
        == ("medicine", "online_rubrics", 3, "global_step_3", "probe_A", 3, 3, "fresh")
        for row in rows
    )
    assert all(item["fresh_or_stale"] == "fresh" for item in rows[0]["pool_a_provenance"])
    assert read_json(tmp_path / "output/checkpoint-000003/status.json")["state"] == "complete"

    assert build_probe_fresh_rubrics(provider, **kwargs) == result
    assert provider.calls == 900

    # Simulate a restart before final status publication, with prompt caches intact.
    status_path = tmp_path / 'output/checkpoint-000003/status.json'
    write_json_atomic(status_path, {'state': 'running'}, immutable=False)
    resumed = build_probe_fresh_rubrics(provider, **kwargs)
    assert resumed['reused_prompts'] == 100
    assert resumed['newly_completed_prompts'] == 0
    assert provider.calls == 900


def test_fixed_train_fresh_rubrics_reject_non_pool_a_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _inputs(tmp_path, pool_name="probe_B")
    monkeypatch.setattr(probe_fresh_rubrics, "ImmutablePi0Cache", _ControlCache)

    with pytest.raises(ProbeFreshRubricError, match="Pool A provenance mismatch"):
        build_probe_fresh_rubrics(
            _Provider(),
            run_id="phase1-run",
            train_path=paths["train"],
            probe_manifest_path=paths["probe"],
            pool_a_path=paths["pool_a"],
            pi0_manifest_path=paths["pi0"],
            checkpoint_step=3,
            checkpoint_hash="a" * 64,
            seed=11,
            extractor_model="openai/gpt-oss-120b",
            extractor_returned_model="openai/gpt-oss-120b",
            output_root=tmp_path / "output",
        )


def test_fixed_train_fresh_rubrics_reject_duplicate_ids_within_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _inputs(tmp_path)
    rows = read_jsonl(paths["pool_a"])
    rows[1]["response_id"] = rows[0]["response_id"]
    paths["pool_a"].unlink()
    write_jsonl_atomic(paths["pool_a"], rows)
    monkeypatch.setattr(probe_fresh_rubrics, "ImmutablePi0Cache", _ControlCache)

    with pytest.raises(ProbeFreshRubricError, match="globally unique"):
        build_probe_fresh_rubrics(
            _Provider(),
            run_id="phase1-run",
            train_path=paths["train"],
            probe_manifest_path=paths["probe"],
            pool_a_path=paths["pool_a"],
            pi0_manifest_path=paths["pi0"],
            checkpoint_step=3,
            checkpoint_hash="a" * 64,
            seed=11,
            extractor_model="openai/gpt-oss-120b",
            extractor_returned_model="openai/gpt-oss-120b",
            output_root=tmp_path / "output",
        )


def test_fixed_train_fresh_rubrics_reject_train_source_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _inputs(tmp_path)
    paths["train"].write_text(paths["train"].read_text() + "\n", encoding="utf-8")
    monkeypatch.setattr(probe_fresh_rubrics, "ImmutablePi0Cache", _ControlCache)

    with pytest.raises(ProbeFreshRubricError, match="does not bind the train source"):
        build_probe_fresh_rubrics(
            _Provider(),
            run_id="phase1-run",
            train_path=paths["train"],
            probe_manifest_path=paths["probe"],
            pool_a_path=paths["pool_a"],
            pi0_manifest_path=paths["pi0"],
            checkpoint_step=3,
            checkpoint_hash="a" * 64,
            seed=11,
            extractor_model="openai/gpt-oss-120b",
            extractor_returned_model="openai/gpt-oss-120b",
            output_root=tmp_path / "output",
        )
