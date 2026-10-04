from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path

import pytest

from dynamic_rubric.artifacts import artifact_record, read_json, write_json_atomic
from scripts.phase1.prefetch_historical_kl import download_model, model_scored


def fixture(tmp_path):
    run, root = tmp_path / "run", tmp_path / "kl"
    run.mkdir()
    root.mkdir()
    pool = root / "pool.json"
    pool.write_text("[]")
    write_json_atomic(
        root / "plan.json",
        {
            "run_id": run.name,
            "cells": [[3, 3]],
            "pool_files": {"3": artifact_record(pool)},
        },
    )
    content = b"fake-weights"
    receipt = {
        "run_id": run.name,
        "state": "verified",
        "checkpoint_step": 3,
        "repo_id": "HYU-NLP-EVAL/qwen3-4b-rar-medicine-onlinerubrics-seed11-step-003",
        "revision": "a" * 40,
        "audit_export_manifest": {"source_model_sha256": "b" * 64},
        "public_export_files": [
            {
                "remote_path": "model.safetensors",
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        ],
    }
    return run, root, receipt, content


def test_pinned_download_and_concurrent_reuse(tmp_path, monkeypatch):
    import scripts.phase1.prefetch_historical_kl as module

    run, root, receipt, content = fixture(tmp_path)
    calls = []

    def fake(argv, **kwargs):
        calls.append(argv)
        assert argv[argv.index("--revision") + 1] == receipt["revision"]
        assert "original_checkpoint" not in " ".join(argv)
        (Path(argv[argv.index("--local-dir") + 1]) / "model.safetensors").write_bytes(content)

    monkeypatch.setattr(module.subprocess, "run", fake)
    with ThreadPoolExecutor(max_workers=2) as executor:
        jobs = [
            executor.submit(download_model, run, root, receipt, hf_cli="hf", env={})
            for _ in range(2)
        ]
        paths = [f.result() for f in jobs]
    assert paths[0] == paths[1] and len(calls) == 1
    assert read_json(root / "downloads/step-3-verified.json")["revision"] == receipt["revision"]


@pytest.mark.parametrize("bad", ["revision", "path", "run", "hash"])
def test_reject_unpinned_unsafe_or_corrupt_download(tmp_path, monkeypatch, bad):
    import scripts.phase1.prefetch_historical_kl as module

    run, root, receipt, content = fixture(tmp_path)
    if bad == "revision":
        receipt["revision"] = "main"
    elif bad == "path":
        receipt["public_export_files"][0]["remote_path"] = "../model.safetensors"
    elif bad == "run":
        receipt["run_id"] = "another-run"

    def fake(argv, **kwargs):
        assert bad == "hash", "Invalid receipt reached network download"
        (Path(argv[argv.index("--local-dir") + 1]) / "model.safetensors").write_bytes(
            b"x" * len(content)
        )

    monkeypatch.setattr(module.subprocess, "run", fake)
    with pytest.raises(ValueError):
        download_model(run, root, receipt, hf_cli="hf", env={})
    assert not (root / "downloads/step-3-verified.json").exists()


def test_do_not_recreate_consumed_model(tmp_path, monkeypatch):
    import scripts.phase1.prefetch_historical_kl as module

    run, root, receipt, _ = fixture(tmp_path)
    score = root / "scores.json"
    score.write_text("[]")
    write_json_atomic(
        root / "seals/model-3_pool-3.json",
        {
            "scores": artifact_record(score),
            "checkpoint_hash": "b" * 64,
            "pool": read_json(root / "plan.json")["pool_files"]["3"],
        },
    )
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **kw: pytest.fail("Redownload"))
    assert model_scored(root, receipt)
    assert download_model(run, root, receipt, hf_cli="hf", env={}, skip_if_scored=True) is None
    assert not (root / "temporary_models/global_step_3").exists()
