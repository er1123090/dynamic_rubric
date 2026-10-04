import math
from types import SimpleNamespace

import pytest

from dynamic_rubric.artifacts import (
    artifact_record,
    read_json,
    write_json_atomic,
    write_jsonl_atomic,
)
from dynamic_rubric.horizon.checkpoint_kl import _encode_float32
from scripts.phase1.run_historical_kl import (
    matrix_cells,
    summarize,
    validate_scores,
    delete_download,
    serving_options,
    wait_for_memory,
    validate_context_budget,
    CachedClient,
)


def test_full_long_context_fits_existing_kv_budget(tmp_path):
    write_json_atomic(tmp_path / "config.json", {
        "max_position_embeddings": 262144, "num_hidden_layers": 36,
        "num_key_value_heads": 8, "head_dim": 128,
    })
    validate_context_budget(tmp_path, 10240, 1610612736)
    with pytest.raises(ValueError, match="KV bytes"):
        validate_context_budget(tmp_path, 16384, 1610612736)
    with pytest.raises(ValueError, match="position capacity"):
        validate_context_budget(tmp_path, 262145, None)


def test_long_input_preserved_and_completion_token_reserved(tmp_path, monkeypatch):
    import scripts.phase1.run_historical_kl as runner

    calls = []

    def score(self, tokens, starts):
        calls.append((tokens, starts))
        return [[-1.0] * (len(t) - s) for t, s in zip(tokens, starts)]

    monkeypatch.setattr(runner.VLLMPolicyLogprobClient, "score", score)
    client = CachedClient(
        "http://unused", served_model="test", model_revision="rev",
        tokenizer_revision="tok", checkpoint_hash="hash", model_path=tmp_path,
        cache=tmp_path / "cache", progress=tmp_path / "progress.json",
        max_model_len=10240,
    )
    tokens = [1] * 10132
    assert len(client.score([tokens], [10])[0]) == 10122
    assert calls == [([tokens], [10])]
    client.score([tokens], [10])
    assert len(calls) == 1  # Runtime context setting does not invalidate token cache.
    client.score([[1] * 10239], [10])
    with pytest.raises(ValueError, match="requires context 10241"):
        client.score([[1] * 10240], [10])
    assert len(calls) == 2


def test_shared_gpu_requires_small_explicit_budget():
    args = SimpleNamespace(
        gpu_memory_utilization=0.70, max_num_seqs=64, max_num_batched_tokens=16384, share_gpu=False
    )
    assert serving_options(args)[1] == "0.7"
    args.share_gpu = True
    with pytest.raises(ValueError, match="capped"):
        serving_options(args)
    args.gpu_memory_utilization = 0.08
    args.max_num_seqs = 4
    args.max_num_batched_tokens = 1024
    with pytest.raises(ValueError, match="explicit KV"):
        serving_options(args)
    args.kv_cache_memory_bytes = 1610612736
    assert serving_options(args) == [
        "--gpu-memory-utilization",
        "0.08",
        "--max-num-seqs",
        "4",
        "--max-num-batched-tokens",
        "1024",
        "--kv-cache-memory-bytes",
        "1610612736",
    ]
    args.max_num_seqs = 0
    with pytest.raises(ValueError, match="positive"):
        serving_options(args)


def test_shared_memory_guard_waits_without_signalling_training(tmp_path, monkeypatch):
    import scripts.phase1.run_historical_kl as runner

    args = SimpleNamespace(gpu_memory_utilization=0.08, memory_wait_seconds=60)
    readings = iter(["1000, 143771", "20000, 143771"])
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *a, **kw: next(readings))
    sleeps = []
    monkeypatch.setattr(runner.time, "sleep", sleeps.append)
    wait_for_memory(args, tmp_path, 36)
    assert sleeps == [10]
    assert read_json(tmp_path / "memory-wait.json")["required_mib"] == 13550


def config(cells):
    return {
        "primary_dataset": "fixed_train_probe_100",
        "prompt_count": 100,
        "responses_per_prompt": 16,
        "cells": [{"evaluator_step": a, "policy_step": t} for a, t in cells],
    }


def test_matrix_preserves_missing48():
    active, missing = matrix_cells(config([(0, 0), (0, 3), (3, 3), (0, 48), (48, 48)]), {0, 3})
    assert active == [(0, 0), (0, 3), (3, 3)] and missing == [(0, 48), (48, 48)]


@pytest.mark.parametrize("cells", [[(0, 0), (0, 0)], [(3, 0)], [(0, 3)]])
def test_invalid_matrix(cells):
    with pytest.raises(ValueError):
        matrix_cells(config(cells), {0, 3})


def rows(step, lp):
    return [
        {
            "policy_step": step,
            "pool_policy_step": 3,
            "scoring_checkpoint_hash": f"hash{step}",
            "source_checkpoint_hash": "hash3",
            "prompt_id": f"p{i}",
            "response_id": f"p{i}-r{j}",
            "sample_index": j,
            "response_token_count": 2,
            "response_token_hash": "tokens",
            "response_token_logprobs_f32le_b64": _encode_float32([lp, lp]),
        }
        for i in range(100)
        for j in range(16)
    ]


def test_current_pool_direction_and_diagonal(tmp_path):
    write_jsonl_atomic(tmp_path / "scores/policy-step-0_pool-step-3.jsonl", rows(0, -2))
    write_jsonl_atomic(tmp_path / "scores/policy-step-3_pool-step-3.jsonl", rows(3, -1))
    summarize(tmp_path, 0, 3)
    s = read_json(tmp_path / "pairs/stale-000000_current-000003/summary.json")
    assert s["k1_prompt_mean"] == 1
    assert s["k3_prompt_mean"] == pytest.approx(math.exp(-1))
    assert s["prompt_count"] == 100 and s["response_count"] == 1600
    assert "not exact" in s["estimator"]
    summarize(tmp_path, 3, 3)
    d = read_json(tmp_path / "pairs/stale-000003_current-000003/summary.json")
    assert d["k1_prompt_mean"] == d["k3_prompt_mean"] == 0
    summarize(tmp_path, 0, 3)  # sealed restart


def test_scores_reject_duplicates(tmp_path):
    scored = rows(3, -1)
    pool = [
        {
            "response_id": r["response_id"],
            "prompt_id": r["prompt_id"],
            "sample_index": r["sample_index"],
            "policy_step": 3,
            "checkpoint_hash": "hash3",
        }
        for r in scored
    ]
    path = tmp_path / "scores.jsonl"
    write_jsonl_atomic(path, scored[:-1] + [scored[0]])
    with pytest.raises(ValueError, match="inventory"):
        validate_scores(path, pool, 3, "hash3")


def test_deletion_rejects_non_owned_path(tmp_path):
    target = tmp_path / "protected"
    target.mkdir()
    with pytest.raises(ValueError, match="Unsafe"):
        delete_download(target, tmp_path / "downloads", {"checkpoint_step": 3}, tmp_path, [])
    assert target.exists()


def test_delete_only_after_score_seal_and_preserve_training(tmp_path):
    root = tmp_path / "kl"
    download_root = root / "temporary_models"
    target = download_root / "global_step_3"
    target.mkdir(parents=True)
    training = tmp_path / "training/global_step_46"
    training.mkdir(parents=True)
    receipt = {"checkpoint_step": 3, "repo_id": "org/model", "revision": "pinned"}
    write_json_atomic(
        target / "kl-download-owner.json",
        {"checkpoint_step": 3, "repo_id": "org/model", "revision": "pinned", "kl_root": str(root)},
    )
    with pytest.raises(FileNotFoundError):
        delete_download(target, download_root, receipt, root, [(3, 3)])
    assert target.exists()
    score = root / "scores.jsonl"
    write_jsonl_atomic(score, [{"verified": True}])
    write_json_atomic(root / "seals/model-3_pool-3.json", {"scores": artifact_record(score)})
    delete_download(target, download_root, receipt, root, [(3, 3)])
    assert not target.exists()
    assert training.exists() and score.exists()
    assert read_json(root / "downloads/step-3-deleted.json")["revision"] == "pinned"
