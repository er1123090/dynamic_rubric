from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from dynamic_rubric.training.live_online import LiveOnlineTrainingError
from dynamic_rubric.phase1.config import load_phase1_config
from dynamic_rubric.phase1.full_run import (
    CHECKPOINT_STEPS,
    FULL_RUN_PREFIX,
    FULL_RUN_PREFIXES,
    FullRunError,
    build_full_run_environment,
    latest_full_checkpoint,
    prepare_full_resume,
    run_online_full,
    validate_full_run_config,
    write_full_run_parquets,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/phase1/medicine_online_rubrics.yaml"
SCIENCE_CONFIG = ROOT / "configs/phase1/science_online_rubrics.yaml"


@pytest.fixture(autouse=True)
def local_model_fixture(tmp_path, monkeypatch):
    """Config/environment tests use a temporary model directory, never real weights."""
    original_load = load_phase1_config
    model = tmp_path / "policy-model"
    model.mkdir()

    def load_with_local_model(path):
        config = original_load(path)
        models = {
            **config.models,
            "policy": {**config.models["policy"], "local_snapshot": str(model)},
        }
        return replace(config, raw={**config.raw, "models": models})

    monkeypatch.setattr(sys.modules[__name__], "load_phase1_config", load_with_local_model)


def medicine():
    return load_phase1_config(CONFIG)


def science():
    return load_phase1_config(SCIENCE_CONFIG)


def test_science_full_contract_uses_dense_checkpoints() -> None:
    config = science()
    validate_full_run_config(config)
    assert config.training["checkpoint_interval_steps"] == 1
    assert FULL_RUN_PREFIXES["science"] == "phase1-online-rubrics-science-full"


def fake_cache(monkeypatch, tmp_path: Path):
    policy = medicine().models["policy"]
    cache_path = tmp_path / "manifest-cache.json"
    cache_path.write_text("{}")
    cache = SimpleNamespace(
        model=policy["model"],
        revision=policy["revision"],
        tokenizer_revision=policy["revision"],
        checkpoint_hash="a" * 64,
    )
    monkeypatch.setenv("ONLINE_CONTROL_CACHE", str(cache_path))
    monkeypatch.setenv("PHASE1_GPT_OSS_BASE_URLS", "http://gpt-1")
    monkeypatch.setenv("PHASE1_QWEN32B_BASE_URLS", "http://qwen-1,http://qwen-2")
    monkeypatch.delenv("PHASE1_QWEN32B_EXPECTED_COUNT", raising=False)
    monkeypatch.delenv("PHASE1_GPT_OSS_BASE_URL", raising=False)
    monkeypatch.delenv("PHASE1_QWEN32B_BASE_URL", raising=False)
    monkeypatch.delenv("ONLINE_EXTRACTOR_CONCURRENCY", raising=False)
    monkeypatch.delenv("ONLINE_GRADER_CONCURRENCY", raising=False)
    monkeypatch.setattr("dynamic_rubric.phase1.full_run.ImmutablePi0Cache", lambda *a, **k: cache)
    monkeypatch.setattr(
        "dynamic_rubric.phase1.full_run._directory_tree_hash", lambda *a, **k: "a" * 64
    )
    return cache_path


def test_full_contract_and_environment(monkeypatch, tmp_path: Path) -> None:
    config = medicine()
    validate_full_run_config(config)
    fake_cache(monkeypatch, tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "must-be-removed")
    monkeypatch.setenv("ONLINE_CONTROL_URL", "must-be-removed")
    env = build_full_run_environment(
        config,
        repo_root=ROOT,
        run_root=tmp_path / FULL_RUN_PREFIX,
        train_path=tmp_path / "train.parquet",
        validation_path=tmp_path / "val.parquet",
        resume_checkpoint=None,
    )
    assert env["CUDA_VISIBLE_DEVICES"] == "1"
    assert env["N_GPUS_PER_NODE"] == "1"
    assert env["ATTN_IMPLEMENTATION"] == "sdpa"
    assert env["TRAINING_SEED"] == "11"
    assert env["EXPECTED_UPDATES"] == "48"
    assert env["TOTAL_EPOCHS"] == "3"
    assert env["TRAIN_BATCH_SIZE"] == "96"
    assert env["ROLLOUT_N"] == "16"
    assert env["ELICITATION_PAIRS"] == "8"
    assert env["ONLINE_EXTRACTOR_CONCURRENCY"] == "32"
    assert env["ONLINE_GRADER_CONCURRENCY"] == "64"
    assert env["CHECKPOINT_INTERVAL_STEPS"] == "1"
    assert CHECKPOINT_STEPS == tuple(range(1, 49))
    assert env["CHECKPOINT_STEPS"] == "[" + ",".join(map(str, CHECKPOINT_STEPS)) + "]"
    assert env["TRACKING_PROJECT_NAME"] == "phase1_dynamic_evaluator_updates"
    assert env["ONLINE_EXPERIMENT_ARM"] == "phase1_online_rubrics_full_dynamic"
    assert "OPENAI_API_KEY" not in env
    assert "ONLINE_CONTROL_URL" not in env


def test_custom_online_yaml_values_reach_verl_environment(monkeypatch, tmp_path: Path) -> None:
    base = medicine()
    training = {
        **base.training,
        "learning_rate": 1e-5,
        "warmup_ratio": 0.2,
        "kl_coefficient": 0.03,
        "rollout_temperature": 0.8,
        "rollout_top_p": 0.9,
    }
    raw = {**base.raw, "training": training, "launch": {"tuning_mode": "custom"}}
    config = replace(base, raw=raw)
    fake_cache(monkeypatch, tmp_path)
    env = build_full_run_environment(
        config,
        repo_root=ROOT,
        run_root=tmp_path / FULL_RUN_PREFIX,
        train_path=tmp_path / "train.parquet",
        validation_path=tmp_path / "val.parquet",
        resume_checkpoint=None,
    )
    assert env["PHASE1_TUNING_MODE"] == "custom"
    assert float(env["LEARNING_RATE"]) == 1e-5
    assert float(env["WARMUP_RATIO"]) == 0.2
    assert float(env["KL_COEFFICIENT"]) == 0.03
    assert float(env["ROLLOUT_TEMPERATURE"]) == 0.8
    assert float(env["ROLLOUT_TOP_P"]) == 0.9


def test_online_launch_gpu_selection_reaches_training(monkeypatch, tmp_path: Path) -> None:
    base = medicine()
    infrastructure = {
        **base.raw["infrastructure"],
        "optimizer": {**base.raw["infrastructure"]["optimizer"], "gpus": [0]},
    }
    config = replace(
        base,
        raw={**base.raw, "infrastructure": infrastructure, "launch": {"tuning_mode": "paper"}},
    )
    fake_cache(monkeypatch, tmp_path)
    env = build_full_run_environment(
        config,
        repo_root=ROOT,
        run_root=tmp_path / FULL_RUN_PREFIX,
        train_path=tmp_path / "train.parquet",
        validation_path=tmp_path / "val.parquet",
        resume_checkpoint=None,
    )
    assert env["CUDA_VISIBLE_DEVICES"] == "0"


def test_paper_mode_rejects_changed_online_tuning() -> None:
    base = medicine()
    config = replace(base, raw={**base.raw, "training": {**base.training, "learning_rate": 1e-5}})
    with pytest.raises(FullRunError, match="tuning_mode=custom"):
        validate_full_run_config(config)


def test_full_environment_honors_explicit_inference_concurrency(
    monkeypatch, tmp_path: Path
) -> None:
    config = medicine()
    fake_cache(monkeypatch, tmp_path)
    monkeypatch.setenv("ONLINE_EXTRACTOR_CONCURRENCY", "48")
    monkeypatch.setenv("ONLINE_GRADER_CONCURRENCY", "64")

    env = build_full_run_environment(
        config,
        repo_root=ROOT,
        run_root=tmp_path / FULL_RUN_PREFIX,
        train_path=tmp_path / "train.parquet",
        validation_path=tmp_path / "val.parquet",
        resume_checkpoint=None,
    )

    assert env["ONLINE_EXTRACTOR_CONCURRENCY"] == "48"
    assert env["ONLINE_GRADER_CONCURRENCY"] == "64"


def test_requires_exact_endpoint_counts(monkeypatch, tmp_path: Path) -> None:
    config = medicine()
    fake_cache(monkeypatch, tmp_path)
    monkeypatch.setenv("PHASE1_GPT_OSS_BASE_URLS", "http://gpt-1,http://extra")
    with pytest.raises(FullRunError, match="exactly 1 unique"):
        build_full_run_environment(
            config,
            repo_root=ROOT,
            run_root=tmp_path / "run",
            train_path=tmp_path / "train",
            validation_path=tmp_path / "val",
            resume_checkpoint=None,
        )


def test_allows_explicit_single_qwen_endpoint(monkeypatch, tmp_path: Path) -> None:
    config = medicine()
    fake_cache(monkeypatch, tmp_path)
    monkeypatch.setenv("PHASE1_QWEN32B_BASE_URLS", "http://qwen-inference_b")
    monkeypatch.setenv("PHASE1_QWEN32B_EXPECTED_COUNT", "1")
    environment = build_full_run_environment(
        config,
        repo_root=ROOT,
        run_root=tmp_path / "run",
        train_path=tmp_path / "train",
        validation_path=tmp_path / "val",
        resume_checkpoint=None,
    )
    assert environment["PHASE1_QWEN32B_BASE_URLS"] == "http://qwen-inference_b"
    assert environment["PHASE1_QWEN32B_EXPECTED_COUNT"] == "1"


def test_full_manifest_contains_all_prompt_ids_and_hashes(monkeypatch, tmp_path: Path) -> None:
    class FakeDataset:
        def __init__(self, rows):
            self.rows = rows

        @classmethod
        def from_list(cls, rows):
            return cls(rows)

        def to_parquet(self, path):
            Path(path).write_text(json.dumps(self.rows))

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(Dataset=FakeDataset))
    run_root = tmp_path / FULL_RUN_PREFIX
    run_root.mkdir()
    train, validation, manifest_path = write_full_run_parquets(
        medicine(), repo_root=ROOT, run_root=run_root
    )
    manifest = json.loads(manifest_path.read_text())
    assert train.is_file() and validation.is_file()
    assert manifest["source"]["row_count"] == 1500
    assert len(manifest["ordered_rows"]) == 1500
    assert len({row["prompt_id"] for row in manifest["ordered_rows"]}) == 1500
    assert all(
        row["prompt_hash"] and len(row["source_row_sha256"]) == 64
        for row in manifest["ordered_rows"]
    )
    assert manifest["validation"]["enabled"] is False


def test_resume_selects_only_latest_full_checkpoint(tmp_path: Path) -> None:
    root = tmp_path / "run/verl-run/checkpoints"
    old = root / "global_step_3"
    latest = root / "global_step_6"
    for checkpoint in (old, latest):
        (checkpoint / "actor").mkdir(parents=True)
    (root / "latest_checkpointed_iteration.txt").write_text("6")
    (latest / "data.pt").write_bytes(b"data")
    (latest / "actor/optim_world_size_1_rank_0.pt").write_bytes(b"optimizer")
    (latest / "actor/extra_state_world_size_1_rank_0.pt").write_bytes(b"state")
    assert latest_full_checkpoint(tmp_path / "run") == latest.resolve()
    (latest / "actor/optim_world_size_1_rank_0.pt").unlink()
    with pytest.raises(FullRunError, match="optimizer"):
        latest_full_checkpoint(tmp_path / "run")


def test_prepare_full_resume_reconciles_logical_tail(monkeypatch, tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    checkpoint = (run_root / "verl-run/checkpoints/global_step_3").resolve()
    observed = {}
    monkeypatch.setattr(
        "dynamic_rubric.phase1.full_run.latest_full_checkpoint", lambda _: checkpoint
    )

    def fake_resolve(run_dir):
        observed["run_dir"] = run_dir
        return checkpoint, 3

    monkeypatch.setattr("dynamic_rubric.phase1.full_run.resolve_committed_resume", fake_resolve)
    assert prepare_full_resume(run_root) == checkpoint
    assert observed["run_dir"] == run_root / "verl-run"


def test_prepare_full_resume_rejects_chain_error(monkeypatch, tmp_path: Path) -> None:
    checkpoint = (tmp_path / "run/verl-run/checkpoints/global_step_3").resolve()
    monkeypatch.setattr(
        "dynamic_rubric.phase1.full_run.latest_full_checkpoint", lambda _: checkpoint
    )

    def fail(_):
        raise LiveOnlineTrainingError("bad chain")

    monkeypatch.setattr("dynamic_rubric.phase1.full_run.resolve_committed_resume", fail)
    with pytest.raises(FullRunError, match="committed resume state is invalid: bad chain"):
        prepare_full_resume(tmp_path / "run")


def test_new_run_writes_specs_before_subprocess(monkeypatch, tmp_path: Path) -> None:
    config = medicine()
    raw = dict(config.raw)
    raw["output"] = {**config.raw["output"], "root": str(tmp_path / "outputs")}
    config = replace(config, raw=raw)
    cache_path = fake_cache(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "dynamic_rubric.phase1.full_run.topology_preflight", lambda *a, **k: {"passed": True}
    )

    def fake_parquets(config, *, repo_root, run_root):
        data = run_root / "verl-data"
        manifests = run_root / "manifests"
        data.mkdir()
        manifests.mkdir()
        train, val, manifest = (
            data / "train-online-full.parquet",
            data / "validation-unused.parquet",
            manifests / "full_train_selection.json",
        )
        train.write_bytes(b"train")
        val.write_bytes(b"val")
        manifest.write_text("{}")
        return train.resolve(), val.resolve(), manifest.resolve()

    monkeypatch.setattr("dynamic_rubric.phase1.full_run.write_full_run_parquets", fake_parquets)
    probe = tmp_path / "probe.json"
    probe.write_text("{}")
    monkeypatch.setattr(
        "dynamic_rubric.phase1.full_run.prepare_fixed_train_probe_manifest",
        lambda *a, **k: (probe, {}),
    )
    observed = {}

    def fake_run(argv, *, cwd, env, check):
        run_root = Path(env["RUN_DIR"]).parent
        observed["launch_exists"] = (run_root / "launch_spec.json").is_file()
        observed["env"] = env

    monkeypatch.setattr("dynamic_rubric.phase1.full_run.subprocess.run", fake_run)
    result = run_online_full(config, repo_root=ROOT, run_id=FULL_RUN_PREFIX + "-pytest")
    assert observed["launch_exists"] is True
    assert observed["env"]["ONLINE_CONTROL_CACHE"] == str(cache_path.resolve())
    assert result["status"] == "completed"
    run_root = config.run_root(ROOT, FULL_RUN_PREFIX + "-pytest")
    launch_spec = json.loads((run_root / "launch_spec.json").read_text())
    assert launch_spec["fixed_probe_manifest"] == str(probe)
    assert launch_spec["train_drop_last"] is False
    assert launch_spec["attention_implementation"] == "sdpa"
    assert launch_spec["primary_seed"] == 11
    assert launch_spec["steps_per_epoch"] == 16
    assert launch_spec["final_batch_prompt_count"] == 60
    assert launch_spec["cumulative_prompt_exposures"] == 4500
    assert (run_root / "config.resolved.json").is_file()


def test_full_shell_has_exact_step_limit_and_no_api_key() -> None:
    shell = (ROOT / "scripts/phase1/run_online_full.sh").read_text()
    assert '[ "${EXPECTED_UPDATES}" = "48" ]' in shell
    assert '[[ "${CUDA_VISIBLE_DEVICES:-}" =~ ^[0-9]+$ ]]' in shell
    assert 'override_config.attn_implementation="${ATTN_IMPLEMENTATION}"' in shell
    assert 'data.seed="${TRAINING_SEED}"' in shell
    assert 'actor.fsdp_config.seed="${TRAINING_SEED}"' in shell
    assert 'actor.data_loader_seed="${TRAINING_SEED}"' in shell
    assert 'rollout.seed="${TRAINING_SEED}"' in shell
    assert 'ref.fsdp_config.seed="${TRAINING_SEED}"' in shell
    assert "ONLINE_CONTROL_CACHE" in shell
    assert "OPENAI_API_KEY" not in shell
    assert 'trainer.total_training_steps="${EXPECTED_UPDATES}"' in shell
    assert "+data.train_drop_last=False" in shell


@pytest.mark.parametrize("cache_bytes", [None, "34359738368", "-1"])
def test_full_shell_optional_kv_cap_does_not_change_training(tmp_path, cache_bytes):
    import subprocess

    control = tmp_path / "control.json"
    control.write_text("{}")
    env = {
        "PROJECT_ROOT": str(ROOT),
        "VERL_ROOT": str(tmp_path),
        "RUNTIME_PYTHON": "/bin/echo",
        "MODEL_PATH": "model",
        "TRAIN_FILE": "train",
        "VAL_FILE": "val",
        "RUN_DIR": str(tmp_path),
        "ONLINE_STEP_ARTIFACT_ROOT": str(tmp_path),
        "ONLINE_CONTROL_POLICY": "pi_ref",
        "ONLINE_STEP_HOOK_PATH": "hook.py",
        "ONLINE_STEP_RUNTIME_PATH": "runtime.py",
        "CHECKPOINT_STEPS": "[1,2,3,48]",
        "CUDA_VISIBLE_DEVICES": "1",
        "PHASE1_GPT_OSS_BASE_URLS": "http://extractor",
        "PHASE1_QWEN32B_BASE_URLS": "http://judge",
        "ONLINE_CONTROL_CACHE": str(control),
        "ONLINE_CONTROL_CHECKPOINT_HASH": "bound",
        "ROLLOUT_GPU_MEMORY": "0.43",
    }
    if cache_bytes is not None:
        env["ROLLOUT_KV_CACHE_MEMORY_BYTES"] = cache_bytes
    result = subprocess.run(
        ["/bin/bash", str(ROOT / "scripts/phase1/run_online_full.sh")],
        env=env,
        text=True,
        capture_output=True,
    )
    if cache_bytes == "-1":
        assert result.returncode == 2 and "positive integer" in result.stderr
        return
    assert result.returncode == 0, result.stderr
    assert "data.train_batch_size=96" in result.stdout
    assert "actor_rollout_ref.rollout.n=16" in result.stdout
    assert "actor_rollout_ref.actor.optim.lr=5e-6" in result.stdout
    assert "actor_rollout_ref.rollout.gpu_memory_utilization=0.43" in result.stdout
    key = "++actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_memory_bytes="
    assert (key in result.stdout) == (cache_bytes is not None)
    if cache_bytes:
        assert key + cache_bytes in result.stdout
