from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from scripts.phase1 import launch_training, serve_training

REPO = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("domain", ["medicine", "science"])
@pytest.mark.parametrize("recipe", ["static_rubric", "online_rubric", "evorubric"])
def test_six_recipes_prepare_on_an_unrelated_machine(tmp_path, monkeypatch, domain, recipe):
    """No /path/to/workspace runtime/model files, GPU, endpoints or training are used."""
    filename = f"{domain}_{recipe}.yaml"
    text = (REPO / "configs/launch" / filename).read_text()
    assert "/path/to/workspace" not in text and "/path/to/user-home" not in text
    raw = yaml.safe_load(text)
    for model in raw["models"].values():
        if "local_snapshot" in model:
            (tmp_path / model["local_snapshot"]).mkdir(parents=True, exist_ok=True)
    data_paths = [raw["data"]["train_path"]]
    if recipe == "static_rubric":
        data_paths += [raw["data"]["validation_path"], raw["data"]["static_rubric_path"]]
    else:
        data_paths += [raw["data"]["in_domain_policy_eval"]["path"]]
    for relative in data_paths:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    for relative in [
        raw["launch"]["entry_python"],
        ".venvs/judge/bin/python",
        ".venvs/judge/bin/vllm",
    ]:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.symlink_to(sys.executable)
    if recipe == "online_rubric":
        directory = tmp_path / raw["launch"]["environment"]["ONLINE_CONTROL_CACHE_DIR"]
        directory.mkdir(parents=True)
        (directory / "manifest-sealed.json").touch()
    if recipe == "evorubric":
        for relative in (
            "docs/EvoRubrics-2155.zip",
            "environment/upstream/EvoRubrics/evorubric-main/config/shared_base_config.yaml",
            "environment/upstream/EvoRubrics/evorubric-main/shared_base_trainer.py",
            "environment/upstream/EvoRubrics/third_party/verl/verl/__init__.py",
        ):
            path = tmp_path / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
    # Host labels must not be tied to the author's machine, even in paper mode.
    raw["infrastructure"]["code_host"] = "colleague-workstation"
    raw["infrastructure"]["optimizer"]["host"] = "colleague-workstation"
    raw["infrastructure"]["optimizer"]["gpus"] = [0]
    config = tmp_path / "configs/launch" / filename
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(yaml.safe_dump(raw))
    monkeypatch.setattr(launch_training, "ROOT", tmp_path)
    prepared = launch_training.prepare_launch(config)
    assert prepared.run_root.is_relative_to(tmp_path)
    assert not prepared.run_root.exists()
    assert prepared.environment["PROJECT_ROOT"] == str(tmp_path)
    for role in ["judge", "extractor"] if recipe == "online_rubric" else ["judge"]:
        plan = serve_training.prepare_service(config, role, root=tmp_path)
        assert plan["gpus"] == [0, 1]
        assert len(plan["commands"]) == (2 if recipe == "static_rubric" else 1)


def test_static_service_check_uses_real_identity_fields(tmp_path, monkeypatch):
    raw = yaml.safe_load((REPO / "configs/launch/science_static_rubric.yaml").read_text())
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(raw))
    judge = raw["models"]["judge"]
    payloads = [
        {"data": [{"id": judge["model"]}]},
        {
            "served_model": judge["model"],
            "model_revision": judge["revision"],
            "tokenizer_revision": judge["tokenizer_revision"],
        },
    ]
    monkeypatch.setattr(
        launch_training, "urlopen", lambda *a, **k: io.BytesIO(json.dumps(payloads.pop(0)).encode())
    )
    launch = launch_training.Launch(
        "static_r0_matched",
        config,
        tmp_path / "run",
        (),
        {"DYNAMIC_RUBRIC_VLLM_URL": "http://judge:28137"},
    )
    launch_training.check_services(launch)
    assert not payloads


def test_gitignore_excludes_data_and_outputs_but_keeps_code(tmp_path):
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text((REPO / ".gitignore").read_text())
    candidates = [
        "outputs/science/checkpoint/model.safetensors",
        "models/qwen/config.json",
        "data/source/download.jsonl",
        ".env",
        "environment/upstream/verl/file.py",
        "data/rar/science/public/train.jsonl",
        "data/rar/medicine/eval300/final.jsonl",
        "data/rar/science/verl/train.parquet",
        "docs/EvoRubrics-2155.zip",
        "docs/2606.23038v1.pdf",
        "docs/figures/static-rl-reward-steps-1-50.svg",
        "docs/phase1_evorubrics_validation.json",
        "configs/phase1/ssh_config",
        "outputsl",
        "configs/launch/science_online_rubric.yaml",
        "environment/source-snapshots/EvoRubrics-2155-rq2-patch-manifest.json",
    ]
    result = subprocess.run(
        ["git", "-c", "core.excludesFile=/dev/null", "check-ignore", "--no-index", "--stdin"],
        cwd=tmp_path,
        input="\n".join(candidates),
        text=True,
        capture_output=True,
    )
    assert set(result.stdout.splitlines()) == set(candidates[:-2])
