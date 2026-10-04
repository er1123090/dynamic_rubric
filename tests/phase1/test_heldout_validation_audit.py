from __future__ import annotations

import json
import inspect
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from dynamic_rubric.hashing import sha256_file
from dynamic_rubric.providers.base import GenerationResult
from scripts.phase1 import run_heldout_validation_audit as audit


def test_heatmap_titles_use_shared_positive_current_semantics() -> None:
    source = inspect.getsource(audit._plot_outputs)
    assert "Positive = current rubric distinguishes responses better" in source
    assert "current rubric - reused rubric" not in source


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _row(prefix: str, index: int) -> dict:
    return {
        "prompt_id": f"{prefix}-{index:03d}",
        "messages": [{"role": "user", "content": f"q{index}"}],
        "r0": {
            "criteria": [{"criterion_id": f"c{index}", "criterion": "correct", "weight_units": 1}]
        },
    }


def _config(tmp_path: Path) -> Path:
    sources = {
        "train": [_row("train", i) for i in range(12)],
        "development": [_row("dev", i) for i in range(100)],
        "heldout_test": [_row("test", i) for i in range(4)],
    }
    dataset_specs = {}
    for name, rows in sources.items():
        path = tmp_path / f"{name}.jsonl"
        _write_jsonl(path, rows)
        dataset_specs[name] = {
            "path": str(path),
            "expected_count": len(rows),
            "sha256": sha256_file(path),
        }
    config = {
        "schema_version": 1,
        "data_role": audit.DATA_ROLE,
        "output_root": str(tmp_path / "heldout-output"),
        "seed": 11,
        "training_enabled": False,
        "optimizer_required": False,
        "validation": {
            "source_split": "development",
            "source_count": 100,
            "count": 100,
            "sample_seed": 11,
        },
        "datasets": dataset_specs,
        "checkpoint_steps": {
            "static": list(audit.STATIC_STEPS),
            "online": list(audit.ONLINE_STEPS),
        },
        "methods": {
            "static": {
                "base_repo": "static-base",
                "base_revision": "sha",
                "checkpoint_repo_template": "static-{step:03d}",
                "checkpoint_revision": "main",
            },
            "online": {
                "base_repo": "online-base",
                "base_revision": "sha",
                "checkpoint_repo_template": "online-{step:03d}",
                "checkpoint_revision": "main",
            },
        },
        "policy_generation": {
            "base_url": "http://policy",
            "temperature": 1.0,
            "top_p": 0.95,
            "max_output_tokens": 10,
            "workers": 2,
            "max_in_flight": 2,
            "max_retries": 0,
        },
        "rubric_generation": {
            "base_url": "http://extractor",
            "model": "gpt-oss",
            "revision": "extractor-sha",
            "workers": 2,
            "max_in_flight": 2,
            "max_retries": 0,
        },
        "grading": {
            "base_urls": [
                "http://127.0.0.1:28132/v1",
                "http://127.0.0.1:28133/v1",
            ],
            "model": "qwen32b",
            "revision": "judge-sha",
            "workers": 2,
            "max_in_flight": 2,
            "max_retries": 0,
        },
        "policy_lifecycle": {
            "gpu": 1,
            "vllm": "/bin/false",
            "port": 28131,
            "temporary_download_root": str(tmp_path / "policy-downloads"),
            "temporary_root_sentinel": ".heldout_validation_root.json",
            "temporary_root_identity": "test-heldout-policy-cache-v1",
            "gpu_memory_utilization": 0.9,
            "max_model_len": 32768,
            "startup_timeout_seconds": 1,
        },
        "metrics": {"pairwise_tie_epsilon": 0.01},
    }
    download_root = Path(config["policy_lifecycle"]["temporary_download_root"])
    download_root.mkdir()
    (download_root / config["policy_lifecycle"]["temporary_root_sentinel"]).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "data_role": audit.DATA_ROLE,
                "root_identity": config["policy_lifecycle"]["temporary_root_identity"],
            }
        ),
        encoding="utf-8",
    )
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


class FakeAdapter:
    calls = 0

    def __init__(self, base_url, model, cache_dir, **kwargs):
        self.model = model
        self.base_urls = (base_url,) if isinstance(base_url, str) else tuple(base_url)

    def request_provenance(self, request):
        index = sum(request.prompt_id.encode()) % len(self.base_urls)
        return {
            "selected_base_url": self.base_urls[index],
            "provider_cache_path": "fake-cache",
        }

    def generate(self, request):
        type(self).calls += 1
        if request.family == "online_rubric_extraction":
            text = json.dumps(
                {
                    "analysis": "candidate",
                    "new_criteria": [
                        {"quote": "answer", "criterion": "new useful distinction", "weight": 2}
                    ],
                }
            )
        elif request.family == "online_rubric_dedup":
            text = json.dumps(
                {
                    "analysis": "dedup",
                    "final_criteria": [{"criterion": "new useful distinction", "weight": 2}],
                }
            )
        elif request.family == "phase1_audit_grading":
            text = json.dumps({key: "PRESENT" for key in request.json_schema["required"]})
        else:
            text = f"answer-{request.metadata['pool']}-{request.metadata['sample_index']}"
        return GenerationResult(
            text=text,
            requested_model=self.model,
            returned_model=self.model,
            request_id="r",
            created_at=1,
            retry_count=0,
            usage={"completion_tokens": 1},
            raw_response_hash="h",
        )


def test_config_requires_exact_schedule_and_analysis_only(tmp_path: Path) -> None:
    path = _config(tmp_path)
    raw = yaml.safe_load(path.read_text())
    raw["checkpoint_steps"]["static"] = [0, 3, 48]
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(audit.HeldoutAuditError, match="schedule drift"):
        audit.load_config(path)
    raw["checkpoint_steps"]["static"] = list(audit.STATIC_STEPS)
    raw["training_enabled"] = True
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(audit.HeldoutAuditError, match="must not enable training"):
        audit.load_config(path)


def test_prepare_freezes_same_validation_ids_for_both_methods(tmp_path: Path, monkeypatch) -> None:
    path = _config(tmp_path)
    monkeypatch.setattr(
        audit, "_sample_validation", lambda rows, count, seed: [dict(row) for row in rows[:3]]
    )
    config = audit.load_config(path)
    root = audit.prepare(config)
    prompts = audit.read_jsonl(root / "manifests/validation_prompts.jsonl")
    plan = audit.read_jsonl(root / "manifests/checkpoint_work_plan.jsonl")
    assert {row["data_role"] for row in prompts} == {"heldout_validation"}
    assert {row["source_split"] for row in prompts} == {"development"}
    assert len(plan) == 32
    assert {row["optimizer_required"] for row in plan} == {False}
    assert {row["inference_parameters_only"] for row in plan} == {True}
    assert {row["policy_step"] for row in plan if row["method"] == "static"} == set(
        audit.STATIC_STEPS
    )
    first = (root / "manifest.json").read_bytes()
    audit.prepare(config)
    assert (root / "manifest.json").read_bytes() == first


def test_generate_pools_are_disjoint_and_resume_without_calls(tmp_path: Path, monkeypatch) -> None:
    path = _config(tmp_path)
    monkeypatch.setattr(
        audit, "_sample_validation", lambda rows, count, seed: [dict(row) for row in rows[:2]]
    )
    monkeypatch.setattr(audit, "VLLMChatAdapter", FakeAdapter)
    monkeypatch.setattr(audit, "_verify_served_model", lambda url, model: {"served_model": model})
    FakeAdapter.calls = 0
    config = audit.load_config(path)
    target = audit.generate(config, "static", 3, "served-static-3")
    first_calls = FakeAdapter.calls
    a = audit.read_jsonl(target / "probe_A.jsonl")
    b = audit.read_jsonl(target / "probe_B.jsonl")
    assert len(a) == 16 and len(b) == 32
    assert {row["response_id"] for row in a}.isdisjoint({row["response_id"] for row in b})
    assert {row["data_role"] for row in a + b} == {"heldout_validation"}
    assert {row["used_for_gradient"] for row in a + b} == {False}
    audit.generate(config, "static", 3, "served-static-3")
    assert FakeAdapter.calls == first_calls


def test_gain_signs_make_positive_mean_current_better() -> None:
    stale = {"mad": 0.10, "zar": 0.20, "ptr": 0.40}
    current = {"mad": 0.15, "zar": 0.05, "ptr": 0.25}
    assert audit.gain_values(stale, current) == pytest.approx(
        {"mad_gain": 0.05, "zar_gain": 0.15, "ptr_gain": 0.15}
    )


def test_preflight_is_offline_and_separate_from_train_probe(tmp_path: Path, monkeypatch) -> None:
    path = _config(tmp_path)
    monkeypatch.setattr(
        audit, "_sample_validation", lambda rows, count, seed: [dict(row) for row in rows[:3]]
    )
    config = audit.load_config(path)
    report = audit.preflight(config)
    assert report["offline_manifest_valid"] is False
    assert report["remote_endpoints_called"] is False
    assert report["gpu_required"] is False
    assert report["data_role"] == "heldout_validation"
    assert "train_probe" not in str(audit.output_root(config))
    assert report["matrix_cells_per_method"] == {"static": 55, "online": 253}


def test_mocked_rubric_and_score_cell_are_resumable(tmp_path: Path, monkeypatch) -> None:
    path = _config(tmp_path)
    monkeypatch.setattr(
        audit, "_sample_validation", lambda rows, count, seed: [dict(row) for row in rows[:2]]
    )
    monkeypatch.setattr(audit, "VLLMChatAdapter", FakeAdapter)
    monkeypatch.setattr(audit, "_verify_served_model", lambda url, model: {"served_model": model})
    FakeAdapter.calls = 0
    config = audit.load_config(path)
    audit.generate(config, "online", 0, "online-base")
    audit.generate(config, "online", 3, "online-step3")
    audit.generate_control(config, "online", "online-base")
    audit.build_rubrics(config, "online", 0, "gpt-oss")
    rubric_path = audit.build_rubrics(config, "online", 3, "gpt-oss")
    rubric_rows = audit.read_jsonl(rubric_path)
    assert all(
        row["construction"] == "step_local_r0_union_semantically_deduplicated_new_criteria"
        for row in rubric_rows
    )
    assert all(len(row["criteria_after_deduplication"]) == 1 for row in rubric_rows)
    score_path = audit.score_cell(config, "online", 3, 0, "qwen32b")
    first_calls = FakeAdapter.calls
    scores = audit.read_jsonl(score_path)
    assert len(scores) == 32
    assert {row["fresh_or_stale"] for row in scores} == {"stale"}
    assert {row["pool"] for row in scores} == {"probe_B"}
    audit.score_cell(config, "online", 3, 0, "qwen32b")
    assert FakeAdapter.calls == first_calls


def test_build_rubrics_runtime_endpoint_preserves_canonical_config(
    tmp_path: Path, monkeypatch
) -> None:
    path = _config(tmp_path)
    monkeypatch.setattr(
        audit, "_sample_validation", lambda rows, count, seed: [dict(row) for row in rows[:2]]
    )
    monkeypatch.setattr(audit, "VLLMChatAdapter", FakeAdapter)
    observed_urls = []

    def fake_verify(url, model):
        observed_urls.append(url)
        return {"base_url": url, "served_model": model}

    monkeypatch.setattr(audit, "_verify_served_model", fake_verify)
    config = audit.load_config(path)
    canonical_url = config["rubric_generation"]["base_url"]
    audit.generate(config, "online", 0, "online-base")
    audit.generate(config, "online", 3, "online-step3")
    audit.generate_control(config, "online", "online-base")
    rubric_path = audit.build_rubrics(
        config,
        "online",
        3,
        "gpt-oss",
        runtime_base_url="http://inference_a-extractor/v1",
    )

    endpoint = json.loads(
        (rubric_path.parent / "extractor_endpoint_receipt.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (audit.output_root(config) / "manifest.json").read_text(encoding="utf-8")
    )
    assert observed_urls[-1] == "http://inference_a-extractor/v1"
    assert endpoint["configured_base_url"] == canonical_url
    assert endpoint["runtime_base_url"] == "http://inference_a-extractor/v1"
    assert manifest["resolved_config"]["rubric_generation"]["base_url"] == canonical_url
    assert config["rubric_generation"]["base_url"] == canonical_url


def test_integration_smoke_components_follow_model_switch_order(
    tmp_path: Path, monkeypatch
) -> None:
    path = _config(tmp_path)
    monkeypatch.setattr(audit, "VLLMChatAdapter", FakeAdapter)
    monkeypatch.setattr(audit, "_verify_served_model", lambda url, model: {"served_model": model})
    config = audit.load_config(path)
    stages = (
        ("policy-base", "online-base"),
        ("policy-previous", "online-step3"),
        ("policy-current", "online-step6"),
        ("rubric", "gpt-oss"),
        ("score", "qwen32b"),
    )
    reports = [
        audit.integration_smoke(config, "online", 6, component, model)
        for component, model in stages
    ]
    assert [report["component"] for report in reports] == [item[0] for item in stages]
    assert reports[-1]["state"] == "passed"
    assert reports[-1]["evaluator_steps_compared"] == [0, 3, 6]
    assert reports[-1]["identical_response_ids_and_pool_hash"] is True
    assert reports[-1]["signed_gains_positive_means_current_better"] == {
        "R0_to_Rt": {"mad_gain": 0.0, "zar_gain": 0.0, "ptr_gain": 0.0},
        "Rprev_to_Rt": {"mad_gain": 0.0, "zar_gain": 0.0, "ptr_gain": 0.0},
    }
    assert len(reports[-1]["endpoint_model_identity"]) == 6
    assert {item["base_url"] for item in reports[-1]["endpoint_model_identity"]} == set(
        audit.APPROVED_JUDGE_BASE_URLS
    )
    assert all(
        receipt["served_model"] == receipt["configured_model"] == "qwen32b"
        and receipt["revision"] == "judge-sha"
        for receipt in reports[-1]["endpoint_model_identity"]
    )
    assert reports[-1]["training_updates"] == 0
    assert reports[-1]["optimizer_loaded"] is False


def test_policy_lifecycle_rejects_resume_identity_drift(tmp_path: Path) -> None:
    config = audit.load_config(_config(tmp_path))
    root = audit.prepare(config)
    state_path = root / "lifecycle" / "online" / "step-003.json"
    audit.write_json_atomic(
        state_path,
        {"state": "downloading", "identity_sha256": "wrong"},
        immutable=False,
    )
    with pytest.raises(audit.HeldoutAuditError, match="lifecycle resume identity drift"):
        audit.policy_lifecycle(config, "online", 3)


def test_lifecycle_rejects_broad_temporary_root(tmp_path: Path) -> None:
    config = audit.load_config(_config(tmp_path))
    config["policy_lifecycle"]["temporary_download_root"] = "/"
    with pytest.raises(audit.HeldoutAuditError, match="unsafe lifecycle"):
        audit._lifecycle_download_root(config)


def test_lifecycle_requires_preexisting_matching_root_sentinel(tmp_path: Path) -> None:
    config = audit.load_config(_config(tmp_path))
    root = Path(config["policy_lifecycle"]["temporary_download_root"])
    (root / config["policy_lifecycle"]["temporary_root_sentinel"]).unlink()
    with pytest.raises(audit.HeldoutAuditError, match="root sentinel missing"):
        audit._lifecycle_download_root(config)


def test_lifecycle_unique_directory_resume_and_unrelated_preexisting_rejection(
    tmp_path: Path,
) -> None:
    config = audit.load_config(_config(tmp_path))
    identity = "identity-sha"
    first = audit._acquire_checkpoint_dir(config, "online", 3, identity)
    second = audit._acquire_checkpoint_dir(config, "online", 3, identity)
    assert first != second
    assert audit._acquire_checkpoint_dir(config, "online", 3, identity, str(first)) == first

    unrelated = Path(config["policy_lifecycle"]["temporary_download_root"]) / "unrelated"
    unrelated.mkdir()
    with pytest.raises(audit.HeldoutAuditError, match="ownership sentinel missing"):
        audit._acquire_checkpoint_dir(config, "online", 3, identity, str(unrelated))


def test_extractor_and_judge_reject_wrong_served_model_before_artifacts(tmp_path: Path) -> None:
    config = audit.load_config(_config(tmp_path))
    with pytest.raises(audit.HeldoutAuditError, match="extractor served-model"):
        audit.build_rubrics(config, "online", 0, "wrong-extractor")
    with pytest.raises(audit.HeldoutAuditError, match="judge served-model"):
        audit.score_cell(config, "online", 3, 0, "wrong-judge")


def test_smoke_rejects_tampered_prerequisite_receipt(tmp_path: Path, monkeypatch) -> None:
    config = audit.load_config(_config(tmp_path))
    monkeypatch.setattr(audit, "VLLMChatAdapter", FakeAdapter)
    monkeypatch.setattr(audit, "_verify_served_model", lambda url, model: {"served_model": model})
    for component, model in (
        ("policy-base", "online-base"),
        ("policy-previous", "online-step3"),
        ("policy-current", "online-step6"),
        ("rubric", "gpt-oss"),
    ):
        audit.integration_smoke(config, "online", 6, component, model)
    receipt = (
        audit.output_root(config)
        / "integration_smoke"
        / "_smoke_receipts"
        / "online"
        / "step-006"
        / "rubric.json"
    )
    receipt.write_text(receipt.read_text() + " ")
    with pytest.raises(audit.HeldoutAuditError, match="artifact (size|hash) mismatch"):
        audit.integration_smoke(config, "online", 6, "score", "qwen32b")


def test_score_all_uses_complete_configured_lower_triangles(tmp_path: Path, monkeypatch) -> None:
    config = audit.load_config(_config(tmp_path))
    calls = []
    monkeypatch.setattr(audit, "_verify_served_model", lambda url, model: {"served_model": model})

    def fake_score(config, method, policy_step, evaluator_step, served_model):
        calls.append((method, policy_step, evaluator_step, served_model))
        return tmp_path / f"{method}-{policy_step}-{evaluator_step}.jsonl"

    monkeypatch.setattr(audit, "score_cell", fake_score)
    report = audit.score_all_cells(config, "qwen32b")
    assert report["matrix_cells_per_method"] == {"static": 55, "online": 253}
    assert report["completed_cells"] == 308
    assert len(calls) == 308
    assert all(evaluator <= policy for _, policy, evaluator, _ in calls)
    assert {model for *_, model in calls} == {"qwen32b"}


def test_runtime_judge_overlay_preserves_canonical_config(tmp_path: Path) -> None:
    config = audit.load_config(_config(tmp_path))
    canonical_urls = list(config["grading"]["base_urls"])
    runtime_urls = [
        "http://127.0.0.1:28132/v1",
        "http://127.0.0.1:28134/v1",
        "http://127.0.0.1:28135/v1",
        "http://127.0.0.1:28136/v1",
        "http://127.0.0.1:28137/v1",
        "http://127.0.0.1:28138/v1",
    ]
    spec = audit._runtime_judge_spec(config, runtime_urls, 160)
    assert spec["base_urls"] == runtime_urls
    assert spec["workers"] == 160
    assert config["grading"]["base_urls"] == canonical_urls


def test_runtime_judge_overlay_rejects_unapproved_endpoint(tmp_path: Path) -> None:
    config = audit.load_config(_config(tmp_path))
    with pytest.raises(audit.HeldoutAuditError, match="unapproved runtime judge endpoint"):
        audit._runtime_judge_spec(config, ["http://remote-host:8000/v1"], 32)


def test_run_all_is_smoke_gated_and_phase_ordered(tmp_path: Path, monkeypatch) -> None:
    config = audit.load_config(_config(tmp_path))
    config["execution"] = {
        "smoke_gates": [
            {"method": "static", "step": 6},
            {"method": "online", "step": 6},
        ]
    }
    smoke_root = audit.output_root(config) / "integration_smoke"
    for method in audit.METHODS:
        receipt_root = smoke_root / "_smoke_receipts" / method / "step-006"
        for component in ("policy-base", "policy-previous", "policy-current", "rubric"):
            receipt = receipt_root / f"{component}.json"
            component_report = {
                "method": method,
                "policy_step": 6,
                "component": component,
                "endpoint_model_identity": (
                    [
                        {
                            "served_model": "gpt-oss",
                            "configured_model": "gpt-oss",
                            "revision": "extractor-sha",
                        }
                        for _ in range(2)
                    ]
                    if component == "rubric"
                    else []
                ),
            }
            audit.write_json_atomic(receipt, component_report)
            audit._seal(
                receipt.with_suffix(".seal.json"), {"gate": f"{method}-{component}"}, [receipt]
            )
        prerequisite_hashes = {
            component: sha256_file((receipt_root / f"{component}.json").with_suffix(".seal.json"))
            for component in ("policy-base", "policy-previous", "policy-current", "rubric")
        }
        receipt = receipt_root / "score.json"
        report = {
            "state": "passed",
            "method": method,
            "policy_step": 6,
            "component": "score",
            "full_chain_verified": True,
            "evaluator_steps_compared": [0, 3, 6],
            "identical_response_ids_and_pool_hash": True,
            "signed_gains_positive_means_current_better": {
                "R0_to_Rt": {"mad_gain": 0.1, "zar_gain": 0.2, "ptr_gain": 0.3},
                "Rprev_to_Rt": {"mad_gain": 0.0, "zar_gain": 0.1, "ptr_gain": 0.2},
            },
            "endpoint_model_identity": [
                {
                    "evaluator_step": evaluator_step,
                    "base_url": base_url,
                    "served_model": "qwen32b",
                    "configured_model": "qwen32b",
                    "revision": "judge-sha",
                }
                for evaluator_step in (0, 3, 6)
                for base_url in audit.APPROVED_JUDGE_BASE_URLS
            ],
            "prerequisite_seal_sha256": prerequisite_hashes,
        }
        audit.write_json_atomic(receipt, report)
        audit._seal(receipt.with_suffix(".seal.json"), {"gate": method}, [receipt])

    calls = []

    def fake_lifecycle(config, method, step):
        calls.append(("policy", method, step))
        return {
            "state": "complete",
            "identity_sha256": f"{method}-{step}",
            "response_seal_sha256": f"seal-{method}-{step}",
        }

    rubric = tmp_path / "rubric" / "rubric_unions.jsonl"
    rubric.parent.mkdir()
    rubric.write_text("{}\n")
    (rubric.parent / "sealed_manifest.json").write_text("{}")

    def fake_build(config, method, step, model):
        calls.append(("rubric", method, step, model))
        return rubric

    phases = []
    monkeypatch.setattr(audit, "policy_lifecycle", fake_lifecycle)
    monkeypatch.setattr(audit, "validate_pools", lambda config: {"complete": True})
    monkeypatch.setattr(audit, "build_rubrics", fake_build)
    monkeypatch.setattr(
        audit,
        "score_all_cells",
        lambda config, model: {"completed_cells": 308, "judge_model": model},
    )
    monkeypatch.setattr(audit, "analyze", lambda config: {"matrix_cells": 308})
    monkeypatch.setattr(
        audit,
        "_phase_receipt",
        lambda config, phase, details: phases.append(phase) or {"phase": phase},
    )
    result = audit.run_all(config, "gpt-oss", "qwen32b")
    assert result["state"] == "complete"
    assert phases == ["smoke-gate", "policies", "rubrics", "scores", "analyze"]
    assert len([call for call in calls if call[0] == "policy"]) == 32
    assert len([call for call in calls if call[0] == "rubric"]) == 32


def test_all_cli_stages_are_exposed_and_offline_dry_runnable(tmp_path: Path) -> None:
    config = _config(tmp_path)
    script = Path(audit.__file__).resolve()
    stages = (
        "preflight",
        "prepare",
        "generate-control",
        "generate",
        "validate-pools",
        "build-rubrics",
        "score-cell",
        "score-all",
        "integration-smoke",
        "policy-lifecycle",
        "run-all",
        "analyze",
    )
    for stage in stages:
        completed = subprocess.run(
            [sys.executable, str(script), "--config", str(config), "--stage", stage, "--dry-run"],
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(completed.stdout)
        assert payload == {
            "data_role": "heldout_validation",
            "dry_run": True,
            "gpu_called": False,
            "network_called": False,
            "stage": stage,
        }


def test_smoke_cli_uses_component_specific_model_without_served_model(tmp_path: Path) -> None:
    config = _config(tmp_path)
    script = Path(audit.__file__).resolve()
    for component, flag, model in (
        ("rubric", "--extractor-model", "gpt-oss"),
        ("score", "--judge-model", "qwen32b"),
    ):
        completed = subprocess.run(
            [
                sys.executable,
                str(script),
                "--config",
                str(config),
                "--stage",
                "integration-smoke",
                "--smoke-component",
                component,
                flag,
                model,
                "--dry-run",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        assert json.loads(completed.stdout)["resolved_smoke_model"] == model


def test_judge_replica_adapter_shards_and_fails_over(tmp_path: Path, monkeypatch) -> None:
    calls = []

    class ReplicaAdapter:
        def __init__(self, base_url, model, cache_dir, **kwargs):
            self.base_urls = (base_url,) if isinstance(base_url, str) else tuple(base_url)
            self.model = model

        def request_provenance(self, request):
            selected = self.base_urls[0 if request.prompt_id == "fail" else -1]
            return {"selected_base_url": selected, "provider_cache_path": "fake-cache"}

        def generate(self, request):
            url = self.base_urls[0]
            calls.append((request.prompt_id, url))
            if request.prompt_id == "fail" and url == audit.APPROVED_JUDGE_BASE_URLS[0]:
                raise audit.VLLMChatError("primary unavailable")
            return GenerationResult(
                text="ok",
                requested_model=self.model,
                returned_model=self.model,
                request_id="r",
                created_at=1,
                retry_count=0,
                usage={},
                raw_response_hash="h",
            )

    monkeypatch.setattr(audit, "VLLMChatAdapter", ReplicaAdapter)
    adapter = audit._JudgeReplicaAdapter(
        {
            "base_url": list(audit.APPROVED_JUDGE_BASE_URLS),
            "workers": 2,
            "max_retries": 0,
        },
        tmp_path,
        "qwen32b",
    )
    healthy = audit.GenerationRequest(
        prompt_id="healthy",
        messages=({"role": "user", "content": "x"},),
        family="phase1_audit_grading",
        seed=11,
        max_output_tokens=8,
    )
    assert adapter.generate(healthy).returned_model == "qwen32b"
    healthy_provenance = adapter.request_provenance(healthy)
    assert healthy_provenance["selected_base_url"] == audit.APPROVED_JUDGE_BASE_URLS[1]
    assert healthy_provenance["failover_used"] is False

    request = audit.GenerationRequest(
        prompt_id="fail",
        messages=({"role": "user", "content": "x"},),
        family="phase1_audit_grading",
        seed=11,
        max_output_tokens=8,
    )
    assert adapter.generate(request).returned_model == "qwen32b"
    provenance = adapter.request_provenance(request)
    assert calls == [
        ("healthy", audit.APPROVED_JUDGE_BASE_URLS[1]),
        ("fail", audit.APPROVED_JUDGE_BASE_URLS[0]),
        ("fail", audit.APPROVED_JUDGE_BASE_URLS[1]),
    ]
    assert provenance["primary_base_url"] == audit.APPROVED_JUDGE_BASE_URLS[0]
    assert provenance["selected_base_url"] == audit.APPROVED_JUDGE_BASE_URLS[1]
    assert provenance["failover_used"] is True


def test_judge_endpoints_are_exactly_the_approved_replicas(tmp_path: Path) -> None:
    path = _config(tmp_path)
    raw = yaml.safe_load(path.read_text())
    raw["grading"]["base_urls"] = ["http://inference_b:28021/v1"]
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(audit.HeldoutAuditError, match="approved Inference B tunnel and Trainer local"):
        audit.load_config(path)


def test_analyze_requires_sealed_cells_and_writes_requested_figures(
    tmp_path: Path, monkeypatch
) -> None:
    config = audit.load_config(_config(tmp_path))
    monkeypatch.setattr(audit, "EXPECTED_STEPS_BY_METHOD", {"static": (0,), "online": (0,)})
    root = audit.prepare(config)
    prompts = audit.read_jsonl(root / "manifests/validation_prompts.jsonl")
    assert len(prompts) == 100
    for method in audit.METHODS:
        target = root / "grades" / "cells" / method / "policy-000" / "evaluator-000.jsonl"
        rows = []
        for prompt in prompts:
            for index in range(audit.POOL_B_COUNT):
                rows.append(
                    {
                        "data_role": audit.DATA_ROLE,
                        "pool": "probe_B",
                        "method": method,
                        "policy_step": 0,
                        "evaluator_step": 0,
                        "prompt_id": prompt["prompt_id"],
                        "response_id": f"{method}-{prompt['prompt_id']}-{index}",
                        "reward": float(index % 2),
                    }
                )
        audit.write_jsonl_atomic(target, rows)
        audit._seal(target.with_suffix(".seal.json"), {"test": method}, [target])
    summary = audit.analyze(config)
    assert summary["matrix_cells"] == 2
    assert summary["matrix_cells_per_method"] == {"static": 1, "online": 1}
    metric_root = root / "metrics"
    assert (metric_root / "prompt_level_metrics.csv").is_file()
    assert (metric_root / "reuse_matrix_cells.csv").is_file()
    assert (metric_root / "static_mad_gain_lower_triangle.csv").is_file()
    assert (metric_root / "static_ptr_gain_lower_triangle.csv").is_file()
    assert (metric_root / "static_zar_gain_lower_triangle.csv").is_file()
    assert (metric_root / "online_mad_gain_lower_triangle.csv").is_file()
    assert (metric_root / "online_ptr_gain_lower_triangle.csv").is_file()
    assert (metric_root / "online_zar_gain_lower_triangle.csv").is_file()
    assert (metric_root / "static_positive_current_gain_matrices.png").is_file()
    assert (metric_root / "online_positive_current_gain_matrices.png").is_file()
    assert (metric_root / "static_online_r0_previous_to_current_2x3.png").is_file()
