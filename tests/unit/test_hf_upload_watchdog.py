from __future__ import annotations

import importlib.util
from pathlib import Path
import signal
import sys

import pytest


MODULE_PATH = Path(__file__).parents[2] / "scripts/phase1/hf_upload_watchdog.py"
SPEC = importlib.util.spec_from_file_location("hf_upload_watchdog", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
watchdog = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = watchdog
SPEC.loader.exec_module(watchdog)


class FakeProcess:
    next_pid = 1000

    def __init__(self, outcomes):
        self.pid = FakeProcess.next_pid
        FakeProcess.next_pid += 1
        self._outcomes = iter(outcomes)
        self.returncode = None
        self.wait_calls = 0

    def poll(self):
        if self.returncode is not None:
            return self.returncode
        value = next(self._outcomes, None)
        if value is not None:
            self.returncode = value
        return value

    def wait(self, timeout=None):
        self.wait_calls += 1
        if self.returncode is None:
            self.returncode = -signal.SIGTERM
        return self.returncode


def install_clock(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(watchdog.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(watchdog.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    return clock


def test_retries_failed_child_and_preserves_launch_contract(monkeypatch):
    install_clock(monkeypatch)
    processes = [FakeProcess([7]), FakeProcess([0])]
    launches = []

    def popen(command, **kwargs):
        launches.append((command, kwargs))
        return processes[len(launches) - 1]

    monkeypatch.setattr(watchdog.subprocess, "Popen", popen)
    monkeypatch.setattr(watchdog, "_read_activity", lambda pid: watchdog._Activity(0, 0, 0, 0, 0))

    result = watchdog.run_upload(["hf", "upload-large-folder", "repo", "folder"], env={"SAFE": "1"})

    assert result.returncode == 0
    assert len(launches) == 2
    assert all(call[1] == {"env": {"SAFE": "1"}, "start_new_session": True} for call in launches)
    assert all(process.wait_calls == 1 for process in processes)


def test_stall_kills_only_child_group_reaps_then_retries(monkeypatch):
    install_clock(monkeypatch)
    processes = [FakeProcess([None] * 10), FakeProcess([0])]
    launches = []
    signals = []

    def popen(command, **kwargs):
        launches.append(kwargs)
        return processes[len(launches) - 1]

    monkeypatch.setattr(watchdog.subprocess, "Popen", popen)
    monkeypatch.setattr(watchdog, "_read_activity", lambda pid: watchdog._Activity(0, 0, 0, 0, 0))
    monkeypatch.setattr(watchdog.os, "killpg", lambda pid, sig: signals.append((pid, sig)))

    result = watchdog.run_upload(["hf", "upload"], idle_timeout=30, poll_seconds=10, max_retries=1)

    assert result.returncode == 0
    assert signals == [(processes[0].pid, signal.SIGTERM)]
    assert processes[0].wait_calls == 1
    assert processes[1].wait_calls == 1


def test_cpu_or_non_stdout_io_resets_idle_timer(monkeypatch):
    install_clock(monkeypatch)
    process = FakeProcess([None, None, None, 0])
    samples = iter(
        [
            watchdog._Activity(0, 0, 0, 0, 0),
            watchdog._Activity(0, 0, 0, 0, 0),
            watchdog._Activity(watchdog._CPU_TICKS_PER_SECOND, 0, 0, 0, 0),
            watchdog._Activity(watchdog._CPU_TICKS_PER_SECOND, 0, 0, 0, 0),
        ]
    )
    monkeypatch.setattr(watchdog.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(watchdog, "_read_activity", lambda pid: next(samples))
    monkeypatch.setattr(watchdog.os, "killpg", lambda *args: pytest.fail("active process was killed"))

    assert watchdog.run_upload(["hf", "upload"], idle_timeout=25, poll_seconds=10).returncode == 0


def test_small_stdout_heartbeat_does_not_reset_idle_timer(monkeypatch):
    install_clock(monkeypatch)
    process = FakeProcess([None] * 10)
    samples = iter(
        [watchdog._Activity(0, 0, offset, 0, 0) for offset in (0, 100, 200, 300)]
    )
    signals = []
    monkeypatch.setattr(watchdog.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(watchdog, "_read_activity", lambda pid: next(samples))
    monkeypatch.setattr(watchdog.os, "killpg", lambda pid, sig: signals.append((pid, sig)))

    with pytest.raises(watchdog.UploadWatchdogError, match="inactive"):
        watchdog.run_upload(["hf", "upload"], idle_timeout=30, poll_seconds=10, max_retries=0)

    assert signals == [(process.pid, signal.SIGTERM)]


def test_retry_limit_is_bounded(monkeypatch):
    install_clock(monkeypatch)
    processes = [FakeProcess([2]), FakeProcess([2]), FakeProcess([2])]
    monkeypatch.setattr(watchdog.subprocess, "Popen", lambda *args, **kwargs: processes.pop(0))
    monkeypatch.setattr(watchdog, "_read_activity", lambda pid: watchdog._Activity(0, 0, 0, 0, 0))

    with pytest.raises(watchdog.UploadWatchdogError, match="exhausted 2 retries"):
        watchdog.run_upload(["hf", "upload"], max_retries=2)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"idle_timeout": 0}, "idle_timeout"),
        ({"poll_seconds": float("inf")}, "poll_seconds"),
        ({"max_retries": -1}, "max_retries"),
    ],
)
def test_rejects_invalid_limits(kwargs, message):
    with pytest.raises(ValueError, match=message):
        watchdog.run_upload(["hf"], **kwargs)
