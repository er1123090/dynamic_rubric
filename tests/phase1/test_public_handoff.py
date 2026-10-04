from __future__ import annotations

import copy
import shutil
from pathlib import Path

import pytest

from dynamic_rubric.phase1.config import (
    Phase1ConfigError,
    load_phase1_config,
    load_yaml_config,
    validate_phase1_mapping,
)
from dynamic_rubric.phase1.preflight import PreflightError, topology_preflight

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("host", ["", "my-training-server"])
def test_legacy_config_does_not_require_original_machine_names(host):
    config = load_phase1_config(ROOT / "configs/phase1/science_online_rubrics.yaml")
    raw = copy.deepcopy(config.raw)
    infrastructure = raw["infrastructure"]
    infrastructure["code_host"] = host
    infrastructure["optimizer"]["host"] = host
    infrastructure["pi0_control"]["host"] = host
    for service in infrastructure["services"].values():
        service["host"] = host
        service["hosts"] = [host]
        for instance in service["instances"]:
            instance["host"] = host
    assert validate_phase1_mapping(raw, source_path=config.source_path).method == "online_rubrics"


@pytest.mark.parametrize("value", [None, "", "  "])
def test_unfilled_environment_path_is_rejected(tmp_path, monkeypatch, value):
    path = tmp_path / "config.yaml"
    path.write_text('model: "${COLLEAGUE_MODEL_DIRECTORY}"\n')
    if value is None:
        monkeypatch.delenv("COLLEAGUE_MODEL_DIRECTORY", raising=False)
    else:
        monkeypatch.setenv("COLLEAGUE_MODEL_DIRECTORY", value)
    with pytest.raises(Phase1ConfigError, match="COLLEAGUE_MODEL_DIRECTORY"):
        load_yaml_config(path)


@pytest.mark.parametrize("value", [None, "", "  "])
def test_blank_model_path_cannot_pass_as_repository_directory(tmp_path, value):
    config = load_phase1_config(ROOT / "configs/launch/science_online_rubric.yaml")
    raw = copy.deepcopy(config.raw)
    raw["models"]["policy"]["local_snapshot"] = value
    config = validate_phase1_mapping(raw, source_path=config.source_path)
    with pytest.raises(PreflightError, match="models.policy.local_snapshot"):
        topology_preflight(config, repo_root=tmp_path, require_endpoints=False)


def test_six_recipes_load_without_docker(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda command: None)
    for path in sorted((ROOT / "configs/launch").glob("*.yaml")):
        raw = load_yaml_config(path)
        assert "container" not in raw["infrastructure"].get("optimizer", {})
        assert "docker" not in raw.get("launch", {}).get("environment", {})
        assert not any(personal_root in path.read_text() for personal_root in ("/home/", "/data/"))
