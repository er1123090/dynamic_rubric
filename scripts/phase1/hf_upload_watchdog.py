#!/usr/bin/env python3
"""Run one resumable Hugging Face upload with bounded stall retries."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Mapping, Sequence


@dataclass(frozen=True)
class _Activity:
    cpu_ticks: int
    read_chars: int
    written_chars: int
    read_bytes: int
    written_bytes: int


class UploadWatchdogError(RuntimeError):
    """Raised when an upload exhausts its bounded retry allowance."""


_MEANINGFUL_IO_BYTES = 64 * 1024
_CPU_TICKS_PER_SECOND = int(os.sysconf("SC_CLK_TCK"))


def _event(state: str, **fields: object) -> None:
    print(json.dumps({"state": state, **fields}, sort_keys=True), flush=True)


def _read_activity(pid: int) -> _Activity | None:
    """Read cumulative CPU and process-I/O counters for one child."""

    try:
        stat = (Path("/proc") / str(pid) / "stat").read_text()
        # The command name is parenthesized and may contain spaces.
        fields = stat.rsplit(")", 1)[1].split()
        cpu_ticks = int(fields[11]) + int(fields[12])
        io_values = {}
        for line in (Path("/proc") / str(pid) / "io").read_text().splitlines():
            name, value = line.split(":", 1)
            io_values[name] = int(value)
    except (FileNotFoundError, PermissionError, OSError, ValueError, IndexError):
        return None
    return _Activity(
        cpu_ticks=cpu_ticks,
        read_chars=io_values.get("rchar", 0),
        written_chars=io_values.get("wchar", 0),
        read_bytes=io_values.get("read_bytes", 0),
        written_bytes=io_values.get("write_bytes", 0),
    )


def _meaningful_progress(previous: _Activity | None, current: _Activity | None) -> bool:
    if current is None:
        return False
    if previous is None:
        return True
    cpu_delta = max(0, current.cpu_ticks - previous.cpu_ticks)
    io_delta = sum(
        max(0, new - old)
        for old, new in (
            (previous.read_chars, current.read_chars),
            (previous.written_chars, current.written_chars),
            (previous.read_bytes, current.read_bytes),
            (previous.written_bytes, current.written_bytes),
        )
    )
    return cpu_delta >= _CPU_TICKS_PER_SECOND or io_delta >= _MEANINGFUL_IO_BYTES


def _terminate_and_reap(process: subprocess.Popen, timeout: float = 30.0) -> None:
    """Terminate only this child's process group and always reap the leader."""

    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=timeout)
            return
        except subprocess.TimeoutExpired:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
    process.wait()


def run_upload(
    command: Sequence[str],
    env: Mapping[str, str] | None = None,
    idle_timeout: float = 300,
    poll_seconds: float = 10,
    max_retries: int = 3,
) -> subprocess.CompletedProcess:
    """Run an upload command, retrying failures or genuine inactivity.

    ``max_retries`` counts retries after the initial attempt. Each attempt is a
    new process group, while Hugging Face's on-disk upload cache is untouched.
    Activity requires at least one CPU-second or 64 KiB of cumulative I/O, so
    small periodic stdout reports do not disguise a stalled transfer.
    """

    if not command or any(not isinstance(part, str) or not part for part in command):
        raise ValueError("command must be a non-empty sequence of strings")
    if not math.isfinite(idle_timeout) or idle_timeout <= 0:
        raise ValueError("idle_timeout must be positive and finite")
    if not math.isfinite(poll_seconds) or poll_seconds <= 0:
        raise ValueError("poll_seconds must be positive and finite")
    if not isinstance(max_retries, int) or isinstance(max_retries, bool) or max_retries < 0:
        raise ValueError("max_retries must be a non-negative integer")

    last_reason = "upload failed"
    last_returncode = 1
    for attempt in range(max_retries + 1):
        process = subprocess.Popen(
            list(command),
            env=None if env is None else dict(env),
            start_new_session=True,
        )
        _event("attempt_started", attempt=attempt + 1, pid=process.pid)
        last_activity = _read_activity(process.pid)
        active_at = time.monotonic()
        stalled = False
        try:
            while True:
                returncode = process.poll()
                if returncode is not None:
                    process.wait()
                    last_returncode = returncode
                    if returncode == 0:
                        _event("completed", attempt=attempt + 1, pid=process.pid)
                        return subprocess.CompletedProcess(list(command), returncode)
                    last_reason = f"upload exited with status {returncode}"
                    break

                time.sleep(poll_seconds)
                now = time.monotonic()
                activity = _read_activity(process.pid)
                if _meaningful_progress(last_activity, activity):
                    last_activity = activity
                    active_at = now
                if now - active_at >= idle_timeout:
                    stalled = True
                    last_reason = f"upload was inactive for {idle_timeout:g} seconds"
                    _event("attempt_stalled", attempt=attempt + 1, pid=process.pid)
                    _terminate_and_reap(process)
                    last_returncode = process.returncode if process.returncode is not None else 1
                    break
        except BaseException:
            _terminate_and_reap(process)
            raise

        if not stalled and process.poll() is None:
            _terminate_and_reap(process)
        if attempt == max_retries:
            raise UploadWatchdogError(
                f"{last_reason}; exhausted {max_retries} retries (last status {last_returncode})"
            )
        _event("attempt_retry", attempt=attempt + 2, reason=last_reason)

    raise AssertionError("unreachable")
