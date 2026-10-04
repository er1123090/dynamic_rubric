#!/usr/bin/env python3
"""Safely hand an active audit supervisor from one to two pinned judge replicas."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Callable, Sequence

from dynamic_rubric.artifacts import read_json, write_json_atomic
from dynamic_rubric.phase1.audit_run import endpoint_identity


class HandoffError(RuntimeError):
    pass


def wait_for_judge(
    url: str,
    model: str,
    revision: str,
    timeout: float,
    *,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> dict:
    deadline = clock() + timeout
    while True:
        try:
            return endpoint_identity(url, model, revision)
        except Exception as error:
            if clock() >= deadline:
                raise HandoffError(f"secondary judge was not ready within {timeout}s") from error
            sleep(min(10.0, max(0.0, deadline - clock())))


@dataclass(frozen=True)
class VerifiedSession:
    command: tuple[str, ...]
    leader_start_time: int
    members: dict[int, int]


def _proc_identity(pid: int, proc_root: Path = Path("/proc")) -> tuple[str, int, int]:
    raw = (proc_root / str(pid) / "stat").read_text(encoding="utf-8")
    fields = raw[raw.rfind(")") + 2 :].split()
    return fields[0], int(fields[2]), int(fields[19])


def _group_members(pgid: int, proc_root: Path = Path("/proc")) -> dict[int, int]:
    members: dict[int, int] = {}
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            _state, member_pgid, start_time = _proc_identity(int(entry.name), proc_root)
        except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError, IndexError):
            continue
        if member_pgid == pgid:
            members[int(entry.name)] = start_time
    return members


def verified_supervisor_session(
    pid: int, output: Path, proc_root: Path = Path("/proc")
) -> VerifiedSession:
    process_root = proc_root / str(pid)
    command = tuple(
        part.decode() for part in (process_root / "cmdline").read_bytes().split(b"\0") if part
    )
    if "dynamic_rubric.phase1.audit_supervisor" not in command:
        raise HandoffError("PID is not the expected audit supervisor module")
    try:
        output_value = Path(command[command.index("--output") + 1])
    except (ValueError, IndexError) as error:
        raise HandoffError("supervisor command has no parseable --output") from error
    cwd = (process_root / "cwd").resolve(strict=True)
    actual_output = (
        (cwd / output_value).resolve() if not output_value.is_absolute() else output_value.resolve()
    )
    if actual_output != output.resolve():
        raise HandoffError("supervisor PID does not own the canonical audit output")
    if os.getpgid(pid) != pid or os.getsid(pid) != pid:
        raise HandoffError("supervisor does not own an isolated process session")
    _state, pgid, start_time = _proc_identity(pid, proc_root)
    if pgid != pid:
        raise HandoffError("supervisor stat does not match its process group")
    members = _group_members(pid, proc_root)
    if members.get(pid) != start_time:
        raise HandoffError("supervisor leader is absent from its process group")
    return VerifiedSession(command, start_time, members)


def _common_judge_identity(identity: dict, model: str, revision: str) -> dict:
    return {
        "model": model,
        "revision": revision,
        "vllm_version": identity["version"],
    }


def judge_urls(command: Sequence[str]) -> list[str]:
    result = list(command)
    try:
        start = result.index("--judge-urls") + 1
    except ValueError as error:
        raise HandoffError("supervisor command has no --judge-urls") from error
    end = next((i for i in range(start, len(result)) if result[i].startswith("--")), len(result))
    if start == end:
        raise HandoffError("supervisor command has an empty --judge-urls")
    return result[start:end]


def replace_judge_urls(command: Sequence[str], primary: str, secondary: str) -> list[str]:
    result = list(command)
    existing = judge_urls(result)
    if primary not in existing:
        raise HandoffError("configured primary URL is absent from the active supervisor")
    combined = list(dict.fromkeys([*existing, secondary]))
    start = result.index("--judge-urls") + 1
    return result[:start] + combined + result[start + len(existing) :]


def _locks_released(output: Path) -> bool:
    locks = []
    try:
        for name in ("supervisor.lock", "audit.lock"):
            lock = (output / name).open("a")
            locks.append(lock)
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except BlockingIOError:
        return False
    finally:
        for lock in locks:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            finally:
                lock.close()


def wait_for_exit(session: VerifiedSession, output: Path, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while True:
        alive = []
        for pid, expected_start in session.members.items():
            try:
                state, _pgid, current_start = _proc_identity(pid)
            except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError, IndexError):
                continue
            if state != "Z" and current_start == expected_start:
                alive.append(pid)
        if not alive and _locks_released(output):
            return
        if time.monotonic() >= deadline:
            raise HandoffError(f"old audit process group/locks remained after 30s: {alive}")
        time.sleep(0.25)


def handoff(args: argparse.Namespace) -> dict:
    if not 0 < args.timeout <= 3600:
        raise HandoffError("--timeout must be in (0, 3600]")
    run, output, smoke = args.run.resolve(), args.output.resolve(), args.smoke_output.resolve()
    if len({run, output, smoke}) != 3 or output in smoke.parents or smoke in output.parents:
        raise HandoffError("run, audit output, and smoke output must be separate")
    cfg = read_json(run / "config.resolved.json")
    judge = cfg["models"]["judge"]
    identity = wait_for_judge(args.secondary_url, judge["model"], judge["revision"], args.timeout)
    smoke_command = [
        sys.executable,
        "-m",
        "dynamic_rubric.phase1.audit_run",
        "--run",
        str(run),
        "--output",
        str(smoke),
        "--through",
        "34",
        "--judge-urls",
        args.secondary_url,
        "--group-workers",
        "8",
        "--requests-per-group",
        "8",
        "--limit-groups",
        "2",
    ]
    try:
        smoke_result = subprocess.run(smoke_command, check=False, timeout=args.timeout)
    except subprocess.TimeoutExpired as error:
        raise HandoffError(
            "secondary-only audit smoke timed out; primary supervisor left untouched"
        ) from error
    smoke_status = read_json(smoke / "status.json") if (smoke / "status.json").is_file() else {}
    if smoke_result.returncode or smoke_status.get("state") != "smoke_complete":
        raise HandoffError("secondary-only audit smoke failed; primary supervisor left untouched")
    old_session = verified_supervisor_session(args.current_supervisor_pid, output)
    new_command = replace_judge_urls(old_session.command, args.primary_url, args.secondary_url)
    retained_urls = judge_urls(old_session.command)
    endpoint_identities = {args.secondary_url: identity}
    try:
        for url in retained_urls:
            endpoint_identities[url] = endpoint_identity(url, judge["model"], judge["revision"])
    except Exception as error:
        raise HandoffError(
            "an active judge identity could not be verified; supervisor left untouched"
        ) from error
    expected_identity = _common_judge_identity(identity, judge["model"], judge["revision"])
    if any(
        _common_judge_identity(item, judge["model"], judge["revision"]) != expected_identity
        for item in endpoint_identities.values()
    ):
        raise HandoffError(
            "judge replicas do not share pinned model/revision/vLLM version; "
            "supervisor left untouched"
        )

    intent = {
        "state": "handoff_intent",
        "old_pid": args.current_supervisor_pid,
        "old_command": list(old_session.command),
        "old_start_time": old_session.leader_start_time,
        "old_group_members": old_session.members,
        "new_command": new_command,
        "endpoint_identities": endpoint_identities,
        "smoke_output": str(smoke),
    }
    write_json_atomic(output / "controller" / f"intent-{time.time_ns()}.json", intent)
    write_json_atomic(output / "controller_status.json", intent, immutable=False)
    rechecked = verified_supervisor_session(args.current_supervisor_pid, output)
    if (
        rechecked.command != old_session.command
        or rechecked.leader_start_time != old_session.leader_start_time
    ):
        raise HandoffError("supervisor identity changed before signal; refusing handoff")
    os.killpg(args.current_supervisor_pid, signal.SIGTERM)
    wait_for_exit(rechecked, output)
    log_path = output / "logs" / f"audit-supervisor-two-judges-{time.time_ns()}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as log:
        process = subprocess.Popen(
            new_command,
            cwd=str(Path(__file__).resolve().parents[2]),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    result = {**intent, "state": "activated", "new_pid": process.pid, "log": str(log_path)}
    write_json_atomic(output / "controller_status.json", result, immutable=False)
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--run", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--smoke-output", type=Path, required=True)
    value.add_argument("--primary-url", required=True)
    value.add_argument("--secondary-url", required=True)
    value.add_argument("--current-supervisor-pid", type=int, required=True)
    value.add_argument("--timeout", type=float, default=3600.0)
    return value


def main(argv: Sequence[str] | None = None) -> None:
    print(json.dumps(handoff(parser().parse_args(argv)), sort_keys=True))


if __name__ == "__main__":
    main()
