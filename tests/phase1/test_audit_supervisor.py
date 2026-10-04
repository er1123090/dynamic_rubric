from __future__ import annotations

import fcntl
from itertools import count
import json
from pathlib import Path

import pytest

from dynamic_rubric.phase1.audit_supervisor import (
    AuditSupervisorError,
    SupervisorConfig,
    child_command,
    run_supervisor,
)


def _source_run(tmp_path: Path) -> Path:
    run = tmp_path / "source-run"
    run.mkdir()
    (run / "config.resolved.json").write_text("{}", encoding="utf-8")
    (run / "launch_spec.json").write_text("{}", encoding="utf-8")
    return run


def _config(tmp_path: Path, **overrides: object) -> SupervisorConfig:
    values = {
        "run_dir": _source_run(tmp_path),
        "output_dir": tmp_path / "audit-output",
        "judge_urls": ("http://inference_b-a:8000", "http://inference_b-b:8000"),
        "expected_groups": 2,
        "max_restarts": 1,
        "restart_backoff_seconds": 2.0,
        "max_restart_backoff_seconds": 2.0,
        "poll_seconds": 1.0,
        "snapshot_seconds": 1.0,
        "snapshot_bootstrap_iterations": 7,
        "python": "/venv/bin/python",
    }
    values.update(overrides)
    return SupervisorConfig(**values)  # type: ignore[arg-type]


class _DoneProcess:
    def __init__(self, return_code: int, pid: int) -> None:
        self.returncode = return_code
        self.pid = pid

    def poll(self) -> int:
        return self.returncode


def test_restarts_bounded_and_writes_unique_partial_snapshots(tmp_path: Path) -> None:
    config = _config(tmp_path)
    commands: list[list[str]] = []
    snapshots: list[dict[str, object]] = []
    sleeps: list[float] = []

    def popen(command: list[str], **_: object) -> _DoneProcess:
        commands.append(command)
        attempt = len(commands)
        group = config.output_dir / "groups" / "step-000001" / f"p{attempt}.json"
        group.parent.mkdir(parents=True, exist_ok=True)
        group.write_text("{}", encoding="utf-8")
        state = "complete" if attempt == 2 else "failed"
        (config.output_dir / "status.json").write_text(
            json.dumps({"state": state}), encoding="utf-8"
        )
        return _DoneProcess(0 if attempt == 2 else 1, 100 + attempt)

    def snapshot_runner(**kwargs: object) -> dict[str, object]:
        snapshots.append(kwargs)
        Path(kwargs["output_dir"]).mkdir(parents=True)
        return {"ok": True}

    result = run_supervisor(
        config,
        popen_factory=popen,
        sleep=sleeps.append,
        monotonic=iter(range(20)).__next__,
        snapshot_runner=snapshot_runner,
    )

    assert result["state"] == "complete"
    assert [row["attempt"] for row in result["attempts"]] == [1, 2]
    assert sleeps == [2.0]
    assert len(snapshots) == 2
    assert snapshots[0]["training_groups"] == config.output_dir / "groups"
    assert snapshots[0]["expected_groups"] == 2
    assert snapshots[0]["bootstrap_iterations"] == 7
    assert len(list((config.output_dir / "analysis_snapshots").iterdir())) == 2
    assert len(list((config.output_dir / "attempts").glob("attempt-*.json"))) == 2
    assert (
        commands[0]
        == commands[1]
        == child_command(config, config.run_dir.resolve(), config.output_dir.resolve())
    )
    assert "dynamic_rubric.phase1.audit_run" in commands[0]
    assert all("train" not in argument for argument in commands[0])


def test_refuses_source_output_and_concurrent_audit_worker(tmp_path: Path) -> None:
    run = _source_run(tmp_path)
    with pytest.raises(AuditSupervisorError, match="outside"):
        run_supervisor(
            SupervisorConfig(run, run / "audit", ("http://inference_b:8000",)),
            popen_factory=lambda *_args, **_kwargs: pytest.fail("must not launch"),
        )

    config = SupervisorConfig(run, tmp_path / "audit", ("http://inference_b:8000",))
    config.output_dir.mkdir()
    with (config.output_dir / "audit.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(AuditSupervisorError, match="worker already owns"):
            run_supervisor(
                config,
                popen_factory=lambda *_args, **_kwargs: pytest.fail("must not launch"),
            )


def test_new_invocation_continues_attempt_numbering(tmp_path: Path) -> None:
    config = _config(tmp_path, expected_groups=1, max_restarts=0)

    with pytest.raises(AuditSupervisorError, match="exhausted 1 attempts"):
        run_supervisor(
            config,
            popen_factory=lambda *_args, **_kwargs: _DoneProcess(1, 101),
            monotonic=iter(range(10)).__next__,
            snapshot_runner=lambda **_kwargs: {},
        )

    group = config.output_dir / "groups" / "step-000001" / "p1.json"
    group.parent.mkdir(parents=True, exist_ok=True)
    group.write_text("{}", encoding="utf-8")
    (config.output_dir / "status.json").write_text(
        json.dumps({"state": "complete"}), encoding="utf-8"
    )
    result = run_supervisor(
        config,
        popen_factory=lambda *_args, **_kwargs: _DoneProcess(0, 102),
        monotonic=iter(range(10)).__next__,
        snapshot_runner=lambda **_kwargs: {},
    )

    assert result["attempts"][0]["attempt"] == 2
    assert (config.output_dir / "attempts" / "attempt-001.json").is_file()
    assert (config.output_dir / "attempts" / "attempt-002.json").is_file()


def test_stalled_worker_is_terminated_and_recorded(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        expected_groups=1,
        max_restarts=0,
        stall_seconds=0.5,
    )

    class StalledProcess:
        pid = 303
        returncode: int | None = None
        terminated = False

        def poll(self) -> int | None:
            return self.returncode

        def terminate(self) -> None:
            self.terminated = True
            self.returncode = -15

        def wait(self, timeout: int) -> int:
            assert timeout == 30
            assert self.returncode is not None
            return self.returncode

    process = StalledProcess()
    clock = count().__next__
    with pytest.raises(AuditSupervisorError, match="exhausted 1 attempts"):
        run_supervisor(
            config,
            popen_factory=lambda *_args, **_kwargs: process,
            monotonic=clock,
            snapshot_runner=lambda **_kwargs: {},
        )

    assert process.terminated is True
    record = json.loads(
        (config.output_dir / "attempts" / "attempt-001.json").read_text(encoding="utf-8")
    )
    assert record["stalled"] is True
    assert record["return_code"] == -15
