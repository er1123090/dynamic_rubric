from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
import yaml

from scripts.phase1 import launch_training, serve_training
from test_launch_training import _static_config


def test_static_multi_gpu_and_rollout_tp_reach_trainer(tmp_path, monkeypatch):
    path = _static_config(tmp_path, monkeypatch)
    raw = yaml.safe_load(path.read_text())
    raw["infrastructure"]["optimizer"]["gpus"] = [2, 3]
    raw["training"]["rollout_tensor_parallel_size"] = 2
    raw["launch"]["entry_python"] = "${HANDOFF_PYTHON}"
    monkeypatch.setenv("HANDOFF_PYTHON", sys.executable)
    path.write_text(yaml.safe_dump(raw))
    env = launch_training.prepare_launch(path).environment
    assert env["POLICY_GPU"] == "2,3"
    assert env["N_GPUS_PER_NODE"] == "2"
    assert env["ROLLOUT_TENSOR_PARALLEL_SIZE"] == "2"
    raw["training"]["rollout_tensor_parallel_size"] = 3
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="divide"):
        launch_training.prepare_launch(path)


def test_static_external_output_root_keeps_run_isolation(tmp_path, monkeypatch):
    path = _static_config(tmp_path, monkeypatch)
    raw = yaml.safe_load(path.read_text())
    raw["output"]["root"] = str(tmp_path / "separate-disk")
    path.write_text(yaml.safe_dump(raw))
    assert launch_training.prepare_launch(path).run_root == tmp_path / "separate-disk/run"
    raw["output"]["layout"] = "../{run_id}"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="output.layout"):
        launch_training.prepare_launch(path)


def test_console_logging_does_not_precreate_online_run(tmp_path, monkeypatch):
    monkeypatch.setattr(launch_training, "ROOT", tmp_path)
    root = tmp_path / "runs/run01"
    launch = launch_training.Launch(
        "online_rubrics", tmp_path / "config.yaml", root, (), dict(os.environ)
    )
    launch_training.run_logged((sys.executable, "-c", "print('durable training log')"), launch)
    assert not root.exists()
    logs = list((root.parent / "_launch_logs").glob("run01-*.log"))
    assert len(logs) == 1
    assert "durable training log" in logs[0].read_text()


def _service_config(tmp_path: Path, method: str) -> Path:
    model = tmp_path / "model"
    model.mkdir()
    instance = {
        "gpus": [1, 3],
        "tensor_parallel_size": 2,
        "vllm_bin": sys.executable,
        "python": sys.executable,
        "port": 28014,
        "proxy_port": 28137,
    }
    raw = {
        "method": method,
        "models": {
            "judge": {
                "model": "Qwen/Qwen3-32B",
                "local_snapshot": str(model),
                "revision": "pinned",
                "tokenizer_revision": "pinned",
            }
        },
        "infrastructure": {"services": {"qwen3_32b": {"instances": [instance]}}},
    }
    path = tmp_path / "service.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def test_static_service_includes_mandatory_score_proxy(tmp_path):
    path = _service_config(tmp_path, "static_r0_matched")
    plan = serve_training.prepare_service(path, "judge", root=tmp_path)
    assert plan["gpus"] == [1, 3]
    assert len(plan["commands"]) == 2
    assert "dynamic_rubric.services.vllm_score_proxy" in plan["commands"][1]
    assert plan["client_url"].endswith(":28137")


def test_online_judge_is_raw_chat_and_rejects_tp_mismatch(tmp_path):
    path = _service_config(tmp_path, "online_rubrics")
    assert len(serve_training.prepare_service(path, "judge", root=tmp_path)["commands"]) == 1
    raw = yaml.safe_load(path.read_text())
    raw["infrastructure"]["services"]["qwen3_32b"]["instances"][0]["tensor_parallel_size"] = 1
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="GPU count"):
        serve_training.prepare_service(path, "judge", root=tmp_path)


def test_service_check_does_not_spawn(tmp_path, monkeypatch, capsys):
    path = _service_config(tmp_path, "online_rubrics")
    monkeypatch.setattr(
        sys, "argv", ["serve_training.py", "--config", str(path), "--service", "judge", "--check"]
    )
    monkeypatch.setattr(
        serve_training.subprocess, "Popen", lambda *a, **k: pytest.fail("spawned a server")
    )
    assert serve_training.main() == 0
    assert json.loads(capsys.readouterr().out)["gpus"] == [1, 3]
