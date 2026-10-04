"""The three user-facing YAML/SH training entry points."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

from dynamic_rubric.phase1.config import load_phase1_config
from scripts.phase1 import launch_training


ROOT = Path(__file__).resolve().parents[2]


def _portable_prerequisites(tmp_path: Path, document: dict) -> None:
    document["launch"]["entry_python"] = sys.executable
    if document["method"] == "online_rubrics":
        document["launch"]["environment"]["RUNTIME_PYTHON"] = sys.executable
    model = tmp_path / "model"
    model.mkdir(exist_ok=True)
    document["models"]["policy"]["local_snapshot"] = str(model)
    for key in ("train.jsonl", "heldout.jsonl"):
        (tmp_path / key).touch()
    document["data"]["train_path"] = str(tmp_path / "train.jsonl")
    document["data"]["in_domain_policy_eval"]["path"] = str(tmp_path / "heldout.jsonl")
    for relative in (
        "environment/upstream/verl/verl/trainer/main_ppo.py",
        "environment/upstream/EvoRubrics/evorubric-main/config/shared_base_config.yaml",
        "environment/upstream/EvoRubrics/evorubric-main/shared_base_trainer.py",
        "environment/upstream/EvoRubrics/third_party/verl/verl/__init__.py",
        "environment/evorubrics-runtime-lock.txt",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()


def _static_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(launch_training, "ROOT", tmp_path)
    model = tmp_path / "model"
    model.mkdir()
    for name in ("train.parquet", "val.parquet", "rubrics.jsonl"):
        (tmp_path / name).touch()
    config = tmp_path / "configs/static.yaml"
    config.parent.mkdir()
    config.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "experiment": "matched_static_r0_grpo",
                "domain": "medicine",
                "method": "static_r0_matched",
                "seed": 11,
                "data": {
                    "train_path": str(tmp_path / "train.parquet"),
                    "validation_path": str(tmp_path / "val.parquet"),
                    "static_rubric_path": str(tmp_path / "rubrics.jsonl"),
                },
                "models": {
                    "policy": {"local_snapshot": str(model)},
                    "judge": {
                        "model": "Qwen/Qwen3-32B",
                        "revision": "pinned",
                        "tokenizer_revision": "pinned",
                    },
                },
                "training": {
                    "algorithm": "grpo",
                    "epochs": 3,
                    "global_prompt_batch": 96,
                    "expected_global_steps": 2,
                    "checkpoint_interval_steps": 1,
                    "checkpoint_retention": {
                        "latest_full_resume_state": True,
                        "older_parameter_snapshots": True,
                        "older_optimizer_state": False,
                    },
                    "rollouts_per_prompt": 16,
                    "ppo_mini_batch_size": 96,
                    "learning_rate": 5e-6,
                    "warmup_ratio": 0.1,
                    "kl_coefficient": 0.01,
                    "max_response_length": 256,
                    "rollout_temperature": 1.0,
                    "rollout_top_p": 0.95,
                },
                "infrastructure": {"optimizer": {"gpus": [1]}},
                "output": {"root": "outputs", "layout": "{run_id}"},
                "tracking": {"project": "test_static"},
                "launch": {
                    "entry_python": sys.executable,
                    "run_id": "run",
                    "tuning_mode": "custom",
                    "environment": {"DYNAMIC_RUBRIC_VLLM_URL": "http://127.0.0.1:28137"},
                },
            }
        ),
        encoding="utf-8",
    )
    return config


def test_static_recipe_prepares_one_training_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _static_config(tmp_path, monkeypatch)
    launch = launch_training.prepare_launch(config)
    assert launch.method == "static_r0_matched"
    assert launch.commands == (("bash", str(tmp_path / "scripts/run_static_grpo.sh")),)
    assert launch.environment["RESUME_MODE"] == "disable"
    assert launch.environment["TOTAL_STEPS"] == "2"
    assert launch.environment["SAVE_FREQ"] == "1"
    assert launch.environment["CHECKPOINT_STEPS"] == "[0,1,2]"
    assert launch.environment["TRAIN_BATCH_SIZE"] == "96"
    assert launch.environment["ROLLOUT_N"] == "16"
    assert float(launch.environment["LEARNING_RATE"]) == 5e-6
    assert launch.environment["POLICY_GPU"] == "1"


def test_static_yaml_edits_change_real_training_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _static_config(tmp_path, monkeypatch)
    document = yaml.safe_load(config.read_text(encoding="utf-8"))
    document["training"]["expected_global_steps"] = 4
    document["training"]["global_prompt_batch"] = 64
    document["training"]["ppo_mini_batch_size"] = 64
    document["training"]["learning_rate"] = 1e-5
    document["infrastructure"]["optimizer"]["gpus"] = [0]
    config.write_text(yaml.safe_dump(document), encoding="utf-8")
    environment = launch_training.prepare_launch(config).environment
    assert environment["TOTAL_STEPS"] == "4"
    assert environment["CHECKPOINT_STEPS"] == "[0,1,2,3,4]"
    assert environment["TRAIN_BATCH_SIZE"] == "64"
    assert float(environment["LEARNING_RATE"]) == 1e-5
    assert environment["POLICY_GPU"] == "0"


def test_static_recipe_rejects_non_dense_saving(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _static_config(tmp_path, monkeypatch)
    document = yaml.safe_load(config.read_text(encoding="utf-8"))
    document["training"]["checkpoint_interval_steps"] = 2
    config.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(launch_training.LaunchError, match="checkpoint_interval_steps=1"):
        launch_training.prepare_launch(config)


def test_static_recipe_protects_existing_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _static_config(tmp_path, monkeypatch)
    (tmp_path / "outputs/run/verl-run").mkdir(parents=True)
    with pytest.raises(launch_training.LaunchError, match="RUN_DIR exists"):
        launch_training.prepare_launch(config)
    with pytest.raises(launch_training.LaunchError, match="checkpoint tracker"):
        launch_training.prepare_launch(config, resume=True)


@pytest.mark.parametrize(
    ("filename", "method"),
    [
        ("medicine_online_rubric.yaml", "online_rubrics"),
        ("medicine_evorubric.yaml", "evorubrics"),
        ("science_online_rubric.yaml", "online_rubrics"),
        ("science_evorubric.yaml", "evorubrics"),
    ],
)
def test_phase1_launch_yaml_has_same_editable_sections(filename: str, method: str) -> None:
    path = ROOT / "configs/launch" / filename
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert "extends" not in document
    assert load_phase1_config(path).method == method
    for section in ("data", "models", "training", "infrastructure", "output", "tracking", "launch"):
        assert section in document
    parsed, parsed_method, settings, environment = launch_training._launch_section(path)
    assert parsed == document
    assert parsed_method == method
    assert settings["entry_python"]
    assert settings["run_id"]
    assert environment


@pytest.mark.parametrize("domain", ["medicine", "science"])
def test_static_yaml_has_same_editable_sections(domain: str) -> None:
    path = ROOT / "configs/launch" / f"{domain}_static_rubric.yaml"
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    for section in ("data", "models", "training", "infrastructure", "output", "tracking", "launch"):
        assert section in document


def test_all_launch_yamls_keep_common_section_order() -> None:
    common = [
        "schema_version",
        "experiment",
        "domain",
        "method",
        "seed",
        "data",
        "models",
        "training",
        "infrastructure",
        "output",
        "tracking",
        "launch",
    ]
    for name in (
        "medicine_static_rubric.yaml",
        "medicine_online_rubric.yaml",
        "medicine_evorubric.yaml",
        "science_static_rubric.yaml",
        "science_online_rubric.yaml",
        "science_evorubric.yaml",
    ):
        document = yaml.safe_load((ROOT / "configs/launch" / name).read_text())
        assert list(document)[: len(common)] == common


def test_static_rejects_fractional_step_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _static_config(tmp_path, monkeypatch)
    document = yaml.safe_load(config.read_text(encoding="utf-8"))
    document["training"]["expected_global_steps"] = 2.5
    config.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(launch_training.LaunchError, match="positive integer"):
        launch_training.prepare_launch(config)


@pytest.mark.parametrize("domain", ["medicine", "science"])
@pytest.mark.parametrize("method", ["static", "online", "evo"])
def test_six_training_routes_have_yaml_and_shell(domain: str, method: str) -> None:
    root = ROOT
    config_name = (
        f"{domain}_evorubric.yaml" if method == "evo" else f"{domain}_{method}_rubric.yaml"
    )
    config = root / "configs/launch" / config_name
    script = root / "scripts/phase1" / f"train_{domain}_{method}_rubric.sh"
    assert config.is_file()
    assert script.is_file()
    assert yaml.safe_load(config.read_text())["domain"] == domain


def test_science_static_recipe_uses_science_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _static_config(tmp_path, monkeypatch)
    document = yaml.safe_load(config.read_text())
    document["domain"] = "science"
    document["output"]["layout"] = "{domain}/{method}/seed-{seed}/{run_id}"
    config.write_text(yaml.safe_dump(document))
    prepared = launch_training.prepare_launch(config)
    assert prepared.run_root == tmp_path / "outputs/science/static_r0_matched/seed-11/run"
    assert prepared.environment["RUN_DIR"].endswith(
        "outputs/science/static_r0_matched/seed-11/run/verl-run"
    )


def test_science_online_resolves_one_pi0_cache_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(launch_training, "ROOT", tmp_path)
    document = yaml.safe_load((ROOT / "configs/launch/science_online_rubric.yaml").read_text())
    _portable_prerequisites(tmp_path, document)
    cache_dir = tmp_path / "pi0-cache"
    cache_dir.mkdir()
    manifest = cache_dir / "manifest-sealed.json"
    manifest.touch()
    environment = document["launch"]["environment"]
    environment.pop("ONLINE_CONTROL_CACHE", None)
    environment["ONLINE_CONTROL_CACHE_DIR"] = str(cache_dir)
    config = tmp_path / "configs/launch/science_online_rubric.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(yaml.safe_dump(document))
    prepared = launch_training.prepare_launch(config)
    assert prepared.environment["ONLINE_CONTROL_CACHE"] == str(manifest)
    assert prepared.environment["CUDA_VISIBLE_DEVICES"] == "1"
    assert any(arg.endswith("science_online_rubric.yaml") for arg in prepared.commands[0])


def test_science_evo_routes_smoke_and_full_to_science_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(launch_training, "ROOT", tmp_path)
    document = yaml.safe_load((ROOT / "configs/launch/science_evorubric.yaml").read_text())
    _portable_prerequisites(tmp_path, document)
    config = tmp_path / "configs/launch/science_evorubric.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(yaml.safe_dump(document))
    source_zip = tmp_path / "docs/EvoRubrics-2155.zip"
    source_zip.parent.mkdir()
    source_zip.touch()
    prepared = launch_training.prepare_launch(config)
    assert prepared.environment["EVORUBRICS_DOMAIN"] == "science"
    assert prepared.run_root == (
        tmp_path / "outputs/science/evorubrics/seed-11" / document["launch"]["run_id"]
    )
    assert prepared.commands == (
        ("bash", str(tmp_path / "scripts/phase1/run_science_evorubrics.sh"), "smoke"),
        ("bash", str(tmp_path / "scripts/phase1/run_science_evorubrics.sh"), "full"),
    )


def test_science_online_requires_exactly_one_pi0_cache_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(launch_training, "ROOT", tmp_path)
    document = yaml.safe_load((ROOT / "configs/launch/science_online_rubric.yaml").read_text())
    _portable_prerequisites(tmp_path, document)
    cache_dir = tmp_path / "pi0-cache"
    cache_dir.mkdir()
    document["launch"]["environment"]["ONLINE_CONTROL_CACHE_DIR"] = str(cache_dir)
    config = tmp_path / "configs/launch/science_online_rubric.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(yaml.safe_dump(document))
    with pytest.raises(launch_training.LaunchError, match="exactly one manifest"):
        launch_training.prepare_launch(config)
    (cache_dir / "manifest-one.json").touch()
    (cache_dir / "manifest-two.json").touch()
    with pytest.raises(launch_training.LaunchError, match="exactly one manifest"):
        launch_training.prepare_launch(config)
