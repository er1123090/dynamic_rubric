from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from dynamic_rubric.phase1.config import (
    Phase1ConfigError,
    load_phase1_config,
    load_yaml_config,
    validate_phase1_mapping,
)
from dynamic_rubric.phase1.evorubrics_run import build_training_config, runtime_environment
from dynamic_rubric.phase1.full_run import build_full_run_environment, validate_full_run_config
from dynamic_rubric.phase1.preflight import topology_preflight


ROOT = Path(__file__).resolve().parents[2]


def _custom(path: str) -> dict:
    base = load_phase1_config(ROOT / path)
    raw = copy.deepcopy(base.raw)
    raw["launch"]["tuning_mode"] = "custom"
    raw["seed"] = 7
    raw["data"]["fixed_train_probe"]["sample_seed"] = 7
    raw["training"].update(
        {
            "epochs": 2,
            "global_prompt_batch": 32,
            "expected_global_steps": 64,
            "ppo_mini_batch_size": 32,
        }
    )
    return raw


def test_yaml_loader_expands_environment_and_home(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PORTABLE_MODEL_ROOT", str(tmp_path / "models"))
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "model: ${PORTABLE_MODEL_ROOT}/policy\ncache: ~/hf-cache\n",
        encoding="utf-8",
    )

    loaded = load_yaml_config(config_path)

    assert loaded["model"] == str(tmp_path / "models/policy")
    assert loaded["cache"] == str(Path.home() / "hf-cache")


def test_yaml_loader_rejects_unresolved_environment(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model: ${PORTABLE_VARIABLE_THAT_IS_NOT_SET}/policy\n")

    with pytest.raises(Phase1ConfigError, match="unset environment variables"):
        load_yaml_config(config_path)


def test_custom_online_accepts_portable_hosts_and_multiple_training_gpus() -> None:
    raw = _custom("configs/launch/science_online_rubric.yaml")
    raw["infrastructure"]["code_host"] = "colleague-workstation"
    raw["infrastructure"]["optimizer"] = {"host": "local", "gpus": [2, 4]}
    raw["infrastructure"]["services"]["gpt_oss_120b"].update(
        {
            "host": "judge-a",
            "instances": [
                {"host": "judge-a", "gpus": [0], "tensor_parallel_size": 1},
                {"host": "judge-b", "gpus": [3], "tensor_parallel_size": 1},
            ],
        }
    )
    raw["infrastructure"]["services"]["qwen3_32b"].update(
        {
            "hosts": ["judge-c"],
            "instances": [{"host": "judge-c", "gpus": [6], "tensor_parallel_size": 1}],
        }
    )
    raw["infrastructure"]["pi0_control"].update({"host": "local", "gpus": [7]})

    config = validate_phase1_mapping(raw, source_path=ROOT / "portable-online.yaml")

    validate_full_run_config(config)
    assert config.raw["infrastructure"]["optimizer"]["gpus"] == [2, 4]


def test_custom_online_values_reach_verl_environment(monkeypatch, tmp_path: Path) -> None:
    raw = _custom("configs/launch/science_online_rubric.yaml")
    raw["models"]["policy"]["local_snapshot"] = "models/policy"
    raw["infrastructure"]["optimizer"]["gpus"] = [2, 4]
    raw["training"].update(
        {
            "ppo_mini_batch_size": 16,
            "max_prompt_length": 2048,
            "max_response_length": 1024,
            "rollout_tensor_parallel_size": 2,
        }
    )
    config = validate_phase1_mapping(raw, source_path=ROOT / "portable-online.yaml")
    model = tmp_path / "models/policy"
    model.mkdir(parents=True)
    cache_path = tmp_path / "pi0.json"
    cache_path.write_text("{}")
    cache = SimpleNamespace(
        model=config.models["policy"]["model"],
        revision=config.models["policy"]["revision"],
        tokenizer_revision=config.models["policy"]["revision"],
        checkpoint_hash="a" * 64,
    )
    monkeypatch.setenv("ONLINE_CONTROL_CACHE", str(cache_path))
    monkeypatch.setenv("PHASE1_GPT_OSS_BASE_URLS", "http://gpt")
    monkeypatch.setenv("PHASE1_QWEN32B_BASE_URLS", "http://qwen")
    monkeypatch.setenv("RUNTIME_PYTHON", ".venvs/online/bin/python")
    monkeypatch.setenv("VERL_ROOT", "environment/upstream/verl")
    monkeypatch.setattr("dynamic_rubric.phase1.full_run.ImmutablePi0Cache", lambda *a, **k: cache)
    monkeypatch.setattr(
        "dynamic_rubric.phase1.full_run._directory_tree_hash", lambda *a, **k: "a" * 64
    )

    environment = build_full_run_environment(
        config,
        repo_root=tmp_path,
        run_root=tmp_path / "run",
        train_path=tmp_path / "train.parquet",
        validation_path=tmp_path / "validation.parquet",
        resume_checkpoint=None,
    )

    assert environment["CUDA_VISIBLE_DEVICES"] == "2,4"
    assert environment["N_GPUS_PER_NODE"] == "2"
    assert environment["TOTAL_EPOCHS"] == "2"
    assert environment["EXPECTED_UPDATES"] == "64"
    assert environment["TRAIN_BATCH_SIZE"] == "32"
    assert environment["PPO_MINI_BATCH_SIZE"] == "16"
    assert environment["MAX_PROMPT_LENGTH"] == "2048"
    assert environment["MAX_RESPONSE_LENGTH"] == "1024"
    assert environment["ROLLOUT_TENSOR_PARALLEL_SIZE"] == "2"
    assert environment["CHECKPOINT_STEPS"].endswith(",64]")
    assert environment["RUNTIME_PYTHON"] == str(tmp_path / ".venvs/online/bin/python")
    assert environment["VERL_ROOT"] == str(tmp_path / "environment/upstream/verl")


def test_custom_evo_counts_propagate_but_multigpu_is_rejected(tmp_path: Path) -> None:
    raw = _custom("configs/launch/science_evorubric.yaml")
    raw["models"]["policy"]["local_snapshot"] = "models/policy"
    raw["models"]["shared_backbone"]["local_snapshot"] = "models/policy"
    raw["evorubrics"].update(
        {
            "policy_responses_m": 2,
            "rubric_sets_n": 3,
            "pool_b_count": 6,
            "rubric_generation_seeds": [7, 8, 9],
        }
    )
    config = validate_phase1_mapping(raw, source_path=ROOT / "portable-evo.yaml")
    upstream = build_training_config(
        config,
        repo_root=ROOT,
        run_root=tmp_path / "run",
        train_path=tmp_path / "train.json",
    )

    assert upstream["data"]["num_answers"] == 2
    assert upstream["data"]["num_rubrics"] == 3
    assert upstream["actor_rollout_ref"]["rollout"]["n"] == 2
    assert upstream["trainer"]["total_epochs"] == 2
    assert upstream["trainer"]["total_training_steps"] == 64
    assert upstream["trainer"]["save_freq"] == 1
    assert upstream["rq2"]["prune_old_optimizers"] is True

    raw["infrastructure"]["optimizer"]["gpus"] = [0, 1]
    with pytest.raises(Phase1ConfigError, match="distributed optimizer checkpoints"):
        validate_phase1_mapping(raw, source_path=ROOT / "portable-evo.yaml")


def test_evo_runtime_preserves_user_hf_cache(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))

    environment = runtime_environment(tmp_path, tmp_path / "run", "http://judge", "3")

    assert environment["HF_HOME"] == str(tmp_path / "hf")
    assert environment["CUDA_VISIBLE_DEVICES"] == "3"


def test_preflight_does_not_require_configured_hostname(tmp_path: Path) -> None:
    raw = _custom("configs/launch/science_online_rubric.yaml")
    raw["infrastructure"]["code_host"] = "a-different-machine"
    raw["models"]["policy"]["local_snapshot"] = "models/policy"
    raw["data"]["train_path"] = "data/train.jsonl"
    raw["data"]["in_domain_policy_eval"]["path"] = "data/heldout.jsonl"
    config = validate_phase1_mapping(raw, source_path=ROOT / "portable-preflight.yaml")
    (tmp_path / "models/policy").mkdir(parents=True)
    (tmp_path / "data").mkdir()
    (tmp_path / "data/train.jsonl").write_text("{}\n")
    (tmp_path / "data/heldout.jsonl").write_text("{}\n")

    result = topology_preflight(config, repo_root=tmp_path, require_endpoints=False)

    assert result["status"] == "passed"
    assert result["checks"]["code_host"]["required"] is False
    assert result["checks"]["code_host"]["matched"] is False
