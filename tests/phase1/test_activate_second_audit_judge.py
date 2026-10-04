from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.phase1 import activate_second_audit_judge as controller


def _args(tmp_path: Path) -> argparse.Namespace:
    run = tmp_path / "run"
    run.mkdir()
    (run / "config.resolved.json").write_text(
        json.dumps(
            {"models": {"judge": {"model": "Qwen/Qwen3-32B", "revision": "pinned-revision"}}}
        ),
        encoding="utf-8",
    )
    return argparse.Namespace(
        run=run,
        output=tmp_path / "audit",
        smoke_output=tmp_path / "smoke",
        primary_url="http://127.0.0.1:28002",
        secondary_url="http://127.0.0.1:28004",
        current_supervisor_pid=4067167,
        timeout=30.0,
    )


def test_smoke_failure_never_signals_primary_supervisor(tmp_path: Path, monkeypatch) -> None:
    args = _args(tmp_path)
    monkeypatch.setattr(
        controller,
        "wait_for_judge",
        lambda *_args, **_kwargs: {"url": args.secondary_url},
    )
    monkeypatch.setattr(
        controller.subprocess,
        "run",
        lambda command, check, timeout: SimpleNamespace(returncode=1),
    )
    signaled: list[tuple[int, int]] = []
    monkeypatch.setattr(controller.os, "killpg", lambda *values: signaled.append(values))

    with pytest.raises(controller.HandoffError, match="primary supervisor left untouched"):
        controller.handoff(args)

    assert signaled == []


def test_vllm_version_mismatch_never_signals_primary_supervisor(
    tmp_path: Path, monkeypatch
) -> None:
    args = _args(tmp_path)
    monkeypatch.setattr(
        controller,
        "wait_for_judge",
        lambda *_args, **_kwargs: {"url": args.secondary_url, "version": "vllm-secondary"},
    )

    def smoke_run(command: list[str], check: bool, timeout: float) -> SimpleNamespace:
        assert check is False
        assert timeout == args.timeout
        args.smoke_output.mkdir()
        (args.smoke_output / "status.json").write_text(
            json.dumps({"state": "smoke_complete"}), encoding="utf-8"
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(controller.subprocess, "run", smoke_run)
    monkeypatch.setattr(
        controller,
        "endpoint_identity",
        lambda *_args, **_kwargs: {"url": args.primary_url, "version": "vllm-primary"},
    )
    mismatch_command = (
        "python",
        "-m",
        "dynamic_rubric.phase1.audit_supervisor",
        "--output",
        str(args.output.resolve()),
        "--judge-urls",
        args.primary_url,
        "--through",
        "34",
    )
    mismatch_session = controller.VerifiedSession(mismatch_command, 777, {4067167: 777})
    monkeypatch.setattr(
        controller,
        "verified_supervisor_session",
        lambda *_args, **_kwargs: mismatch_session,
    )
    signaled: list[tuple[int, int]] = []
    monkeypatch.setattr(controller.os, "killpg", lambda *values: signaled.append(values))

    with pytest.raises(controller.HandoffError, match="do not share pinned"):
        controller.handoff(args)

    assert signaled == []


def test_successful_smoke_preserves_existing_judges_and_adds_third_once(
    tmp_path: Path, monkeypatch
) -> None:
    args = _args(tmp_path)
    retained_secondary = "http://inference_b-retained:28004"
    args.secondary_url = "http://trainer-new:28004"
    old_command = [
        "/venv/python",
        "-m",
        "dynamic_rubric.phase1.audit_supervisor",
        "--run",
        str(args.run.resolve()),
        "--output",
        str(args.output.resolve()),
        "--judge-urls",
        args.primary_url,
        retained_secondary,
        "--through",
        "34",
        "--group-workers",
        "8",
        "--requests-per-group",
        "8",
    ]
    monkeypatch.setattr(
        controller,
        "wait_for_judge",
        lambda *_args, **_kwargs: {"url": args.secondary_url, "version": "vllm"},
    )
    checked_urls: list[str] = []

    def endpoint(url: str, *_args, **_kwargs) -> dict:
        checked_urls.append(url)
        return {"url": url, "version": "vllm"}

    monkeypatch.setattr(controller, "endpoint_identity", endpoint)

    def smoke_run(command: list[str], check: bool, timeout: float) -> SimpleNamespace:
        assert timeout == args.timeout
        assert check is False
        assert command[command.index("--judge-urls") + 1] == args.secondary_url
        assert command[command.index("--limit-groups") + 1] == "2"
        assert command[command.index("--group-workers") + 1] == "8"
        assert command[command.index("--requests-per-group") + 1] == "8"
        args.smoke_output.mkdir()
        (args.smoke_output / "status.json").write_text(
            json.dumps({"state": "smoke_complete"}), encoding="utf-8"
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(controller.subprocess, "run", smoke_run)
    old_session = controller.VerifiedSession(tuple(old_command), 777, {4067167: 777, 4067170: 778})
    monkeypatch.setattr(
        controller,
        "verified_supervisor_session",
        lambda pid, output: old_session,
    )
    signaled: list[tuple[int, int]] = []
    monkeypatch.setattr(controller.os, "killpg", lambda *values: signaled.append(values))
    monkeypatch.setattr(controller, "wait_for_exit", lambda session, output: None)
    launched: list[tuple[list[str], dict]] = []

    def popen(command: list[str], **kwargs) -> SimpleNamespace:
        launched.append((command, kwargs))
        return SimpleNamespace(pid=5000)

    monkeypatch.setattr(controller.subprocess, "Popen", popen)
    result = controller.handoff(args)

    assert signaled == [(args.current_supervisor_pid, controller.signal.SIGTERM)]
    assert launched[0][0] == controller.replace_judge_urls(
        old_command, args.primary_url, args.secondary_url
    )
    assert controller.judge_urls(launched[0][0]) == [
        args.primary_url,
        retained_secondary,
        args.secondary_url,
    ]
    assert checked_urls == [args.primary_url, retained_secondary]
    assert controller.judge_urls(
        controller.replace_judge_urls(launched[0][0], args.primary_url, args.secondary_url)
    ) == [args.primary_url, retained_secondary, args.secondary_url]

    assert launched[0][1]["start_new_session"] is True
    assert result["state"] == "activated"
    assert result["new_pid"] == 5000
    assert json.loads((args.output / "controller_status.json").read_text())["state"] == (
        "activated"
    )
    assert len(list((args.output / "controller").glob("intent-*.json"))) == 1


def test_pid_validation_resolves_relative_output_from_process_cwd(
    tmp_path: Path, monkeypatch
) -> None:
    cwd = tmp_path / "cwd"
    output = cwd / "audit"
    process = tmp_path / "proc" / "123"
    cwd.mkdir()
    process.mkdir(parents=True)
    (process / "cwd").symlink_to(cwd, target_is_directory=True)
    (process / "cmdline").write_bytes(
        b"python\0-m\0dynamic_rubric.phase1.audit_supervisor\0"
        b"--output\0audit\0--judge-urls\0http://primary\0"
    )
    fields = ["S", "1", "123"] + ["0"] * 16 + ["999"]
    (process / "stat").write_text(f"123 (python worker) {' '.join(fields)}")
    monkeypatch.setattr(controller.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(controller.os, "getsid", lambda pid: pid)

    session = controller.verified_supervisor_session(123, output, tmp_path / "proc")
    assert session.leader_start_time == 999
    assert session.members == {123: 999}

    monkeypatch.setattr(controller.os, "getsid", lambda pid: pid + 1)
    with pytest.raises(controller.HandoffError, match="isolated process session"):
        controller.verified_supervisor_session(123, output, tmp_path / "proc")


def test_wait_for_entire_group_treats_zombies_as_exited_and_waits_for_locks(
    tmp_path: Path, monkeypatch
) -> None:
    session = controller.VerifiedSession(("python",), 10, {1: 10, 2: 20})
    child_checks = iter([("S", 1, 20), ("Z", 1, 20), ("Z", 1, 20)])

    def identity(pid: int):
        if pid == 1:
            return "Z", 1, 10
        return next(child_checks)

    lock_checks = iter([False, True])
    monkeypatch.setattr(controller, "_proc_identity", identity)
    monkeypatch.setattr(controller, "_locks_released", lambda output: next(lock_checks))
    monkeypatch.setattr(controller.time, "monotonic", iter(range(10)).__next__)
    monkeypatch.setattr(controller.time, "sleep", lambda _: None)

    controller.wait_for_exit(session, tmp_path)
