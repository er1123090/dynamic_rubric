from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/phase1/finish48_start_judges.py"


def load_script():
    spec = importlib.util.spec_from_file_location("finish48_start_judges", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_remote_stop_is_pid_starttime_and_command_bound() -> None:
    module = load_script()
    code = module.remote_stop_code(
        [{"pid": 123, "starttime": "456", "required_parts": ["model", "8001"]}]
    )
    assert "remote PID reused" in code
    assert "remote command mismatch" in code
    assert "os.getpgid(p) != p" in code
    assert "os.killpg" in code
    assert "123" in code and "456" in code


def test_judge_commands_use_dedicated_gpu_capacity() -> None:
    module = load_script()
    trainer = module.judge_command(port=28012, tensor_parallel_size=1)
    inference_a = module.judge_command(port=8004, tensor_parallel_size=2)
    assert trainer[trainer.index("--tensor-parallel-size") + 1] == "1"
    assert inference_a[inference_a.index("--tensor-parallel-size") + 1] == "2"
    for command in (trainer, inference_a):
        assert command[command.index("--gpu-memory-utilization") + 1] == "0.90"
        assert command[command.index("--max-num-seqs") + 1] == "128"
        assert command[command.index("--max-num-batched-tokens") + 1] == "16384"
    assert inference_a[inference_a.index("--host") + 1] == "0.0.0.0"


def test_scorer_command_keeps_plan_and_output_lanes_separate(tmp_path: Path) -> None:
    module = load_script()
    command = module.scorer_command(
        tmp_path / "run",
        tmp_path / "artifacts",
        tmp_path / "scores/trainer",
        tmp_path / "trainer-plan.json",
        "http://127.0.0.1:28012",
        concurrency=64,
    )
    assert command[command.index("--output-root") + 1].endswith("scores/trainer")
    assert command[command.index("--cell-plan") + 1].endswith("trainer-plan.json")
    assert command[command.index("--judge-urls") + 1] == "http://127.0.0.1:28012"
    assert command[command.index("--concurrency") + 1] == "64"
    assert "--bounded-grading-whitespace" in command
    assert command[command.index("--max-client-restarts") + 1] == "8"
    assert command[command.index("--restart-delay") + 1] == "30"


def test_remote_command_defaults_to_direct_python_without_container(monkeypatch):
    module = load_script()
    monkeypatch.setenv("PHASE1_REMOTE_PROJECT_ROOT", "/srv/your-project")
    monkeypatch.setenv("PHASE1_REMOTE_PYTHON", "/srv/your-venv/bin/python")
    monkeypatch.delenv("PHASE1_REMOTE_CONTAINER", raising=False)
    monkeypatch.delenv("PHASE1_REMOTE_SSH_PORT", raising=False)
    command = module.ssh_command("user@inference-host", "print('ready')")
    assert command[command.index("-p") + 1] == "22"
    assert "docker" not in command[-1]
    assert "cd /srv/your-project" in command[-1]
    assert "/srv/your-venv/bin/python" in command[-1]


def test_remote_container_is_explicitly_opt_in(monkeypatch):
    module = load_script()
    monkeypatch.setenv("PHASE1_REMOTE_PROJECT_ROOT", "/srv/your-project")
    monkeypatch.setenv("PHASE1_REMOTE_PYTHON", "/srv/your-venv/bin/python")
    monkeypatch.setenv("PHASE1_REMOTE_CONTAINER", "your-container")
    command = module.ssh_command("user@inference-host", "print('ready')")
    assert "docker exec -w /srv/your-project your-container" in command[-1]


def test_unfilled_remote_settings_fail_before_execution(monkeypatch):
    import pytest

    module = load_script()
    monkeypatch.delenv("PHASE1_REMOTE_PROJECT_ROOT", raising=False)
    with pytest.raises(module.Final48TransitionError, match="PHASE1_REMOTE_PROJECT_ROOT"):
        module.ssh_command("user@inference-host", "print('ready')")
