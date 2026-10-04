"""Durable, audit-only supervisor for Phase-1 training-batch stale scoring.

The child command is fixed to :mod:`dynamic_rubric.phase1.audit_run`.  This
module never invokes veRL, restores an optimizer, or writes below the source
training run.  Completed group files are reused by ``audit_run`` after a
bounded child restart.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Sequence

from dynamic_rubric.artifacts import read_json, write_json_atomic
from dynamic_rubric.phase1.audit_analysis import run_training_group_analysis


class AuditSupervisorError(RuntimeError):
    """Raised when safe supervision cannot continue."""


@dataclass(frozen=True, slots=True)
class SupervisorConfig:
    run_dir: Path
    output_dir: Path
    judge_urls: tuple[str, ...]
    through_step: int = 34
    group_workers: int = 8
    requests_per_group: int = 8
    expected_groups: int = 1692
    max_restarts: int = 12
    restart_backoff_seconds: float = 15.0
    max_restart_backoff_seconds: float = 300.0
    poll_seconds: float = 10.0
    snapshot_seconds: float = 300.0
    stall_seconds: float = 1800.0
    snapshot_bootstrap_iterations: int = 200
    epsilon_z: float = 0.01
    epsilon_t: float = 0.01
    python: str = sys.executable

    def __post_init__(self) -> None:
        if not self.judge_urls or any(not value.startswith("http") for value in self.judge_urls):
            raise ValueError("at least one HTTP judge URL is required")
        positive = (
            self.through_step,
            self.group_workers,
            self.requests_per_group,
            self.expected_groups,
            self.poll_seconds,
            self.stall_seconds,
            self.snapshot_seconds,
            self.snapshot_bootstrap_iterations,
        )
        if any(float(value) <= 0 for value in positive):
            raise ValueError("steps, workers, counts, polling, and snapshots must be positive")
        if self.max_restarts < 0 or self.restart_backoff_seconds < 0:
            raise ValueError("restart counts/backoff must be non-negative")
        if self.max_restart_backoff_seconds < self.restart_backoff_seconds:
            raise ValueError("maximum restart backoff cannot be smaller than initial backoff")
        if self.epsilon_z < 0 or self.epsilon_t < 0:
            raise ValueError("metric epsilons must be non-negative")


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _validate_paths(config: SupervisorConfig) -> tuple[Path, Path]:
    run_dir = config.run_dir.resolve()
    output_dir = config.output_dir.resolve()
    if (
        not (run_dir / "config.resolved.json").is_file()
        or not (run_dir / "launch_spec.json").is_file()
    ):
        raise AuditSupervisorError("source run lacks its resolved config or launch spec")
    if output_dir == run_dir or _is_relative_to(output_dir, run_dir):
        raise AuditSupervisorError("audit output must be outside the immutable source run")
    return run_dir, output_dir


def child_command(config: SupervisorConfig, run_dir: Path, output_dir: Path) -> list[str]:
    return [
        config.python,
        "-m",
        "dynamic_rubric.phase1.audit_run",
        "--run",
        str(run_dir),
        "--output",
        str(output_dir),
        "--through",
        str(config.through_step),
        "--judge-urls",
        *config.judge_urls,
        "--group-workers",
        str(config.group_workers),
        "--requests-per-group",
        str(config.requests_per_group),
    ]


def _group_count(output_dir: Path) -> int:
    return sum(1 for _ in (output_dir / "groups").glob("step-*/*.json"))


def _child_state(output_dir: Path) -> str | None:
    path = output_dir / "status.json"
    if not path.is_file():
        return None
    value = read_json(path)
    return str(value.get("state")) if isinstance(value, Mapping) else None


def _snapshot_name(group_count: int, attempt: int) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return f"groups-{group_count:04d}-attempt-{attempt:03d}-{stamp}"


def _write_supervisor_status(output_dir: Path, value: Mapping[str, Any]) -> None:
    write_json_atomic(output_dir / "supervisor_status.json", dict(value), immutable=False)


def _next_attempt_number(output_dir: Path) -> int:
    numbers: list[int] = []
    for path in (output_dir / "attempts").glob("attempt-*.json"):
        try:
            numbers.append(int(path.stem.removeprefix("attempt-")))
        except ValueError:
            continue
    return max(numbers, default=0) + 1


def run_supervisor(
    config: SupervisorConfig,
    *,
    popen_factory: Callable[..., Any] = subprocess.Popen,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
    snapshot_runner: Callable[..., Mapping[str, Any]] = run_training_group_analysis,
) -> dict[str, Any]:
    """Run and restart the audit worker, producing immutable-named snapshots."""

    run_dir, output_dir = _validate_paths(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    supervisor_lock_path = output_dir / "supervisor.lock"
    with supervisor_lock_path.open("a") as supervisor_lock:
        try:
            fcntl.flock(supervisor_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise AuditSupervisorError("another audit supervisor owns this output") from error

        # An independently launched audit_run is authoritative while it owns this lock.
        audit_lock_path = output_dir / "audit.lock"
        with audit_lock_path.open("a") as audit_lock:
            try:
                fcntl.flock(audit_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise AuditSupervisorError("an audit worker already owns this output") from error
            finally:
                try:
                    fcntl.flock(audit_lock.fileno(), fcntl.LOCK_UN)
                except OSError:
                    pass

        last_snapshot_count = -1
        last_snapshot_time = monotonic() - config.snapshot_seconds
        attempts: list[dict[str, Any]] = []

        def snapshot(attempt: int, *, force: bool = False) -> str | None:
            nonlocal last_snapshot_count, last_snapshot_time
            count = _group_count(output_dir)
            now = monotonic()
            if not count or count == last_snapshot_count:
                return None
            if not force and now - last_snapshot_time < config.snapshot_seconds:
                return None
            destination = output_dir / "analysis_snapshots" / _snapshot_name(count, attempt)
            snapshot_runner(
                training_groups=output_dir / "groups",
                output_dir=destination,
                expected_groups=config.expected_groups,
                epsilon_z=config.epsilon_z,
                epsilon_t=config.epsilon_t,
                bootstrap_iterations=config.snapshot_bootstrap_iterations,
            )
            last_snapshot_count = count
            last_snapshot_time = now
            return str(destination)

        first_attempt = _next_attempt_number(output_dir)
        for local_attempt in range(1, config.max_restarts + 2):
            attempt = first_attempt + local_attempt - 1
            log_path = output_dir / "logs" / f"audit-worker-attempt-{attempt:03d}.log"
            progress_count = _group_count(output_dir)
            progress_at = monotonic()
            stalled = False
            log_path.parent.mkdir(parents=True, exist_ok=True)
            command = child_command(config, run_dir, output_dir)
            started = monotonic()
            with log_path.open("ab") as log:
                process = popen_factory(
                    command,
                    cwd=str(Path(__file__).resolve().parents[3]),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                try:
                    while process.poll() is None:
                        completed_groups = _group_count(output_dir)
                        now = monotonic()
                        if completed_groups > progress_count:
                            progress_count = completed_groups
                            progress_at = now
                        elif now - progress_at >= config.stall_seconds:
                            stalled = True
                            process.terminate()
                            try:
                                process.wait(timeout=30)
                            except subprocess.TimeoutExpired:
                                process.kill()
                                process.wait(timeout=30)
                            break
                        snapshot(attempt)
                        _write_supervisor_status(
                            output_dir,
                            {
                                "state": "running",
                                "attempt": attempt,
                                "max_attempts_this_invocation": config.max_restarts + 1,
                                "worker_pid": getattr(process, "pid", None),
                                "completed_groups": completed_groups,
                                "expected_groups": config.expected_groups,
                                "training_started": False,
                                "source_run_mutated": False,
                            },
                        )
                        sleep(config.poll_seconds)
                    return_code = int(process.returncode if process.returncode is not None else -1)
                except BaseException:
                    if process.poll() is None:
                        process.terminate()
                        process.wait(timeout=30)
                    raise
            snapshot_path = snapshot(attempt, force=True)
            attempt_record = {
                "attempt": attempt,
                "return_code": return_code,
                "elapsed_seconds": monotonic() - started,
                "completed_groups": _group_count(output_dir),
                "child_state": _child_state(output_dir),
                "log": str(log_path),
                "stalled": stalled,
                "snapshot": snapshot_path,
            }
            attempts.append(attempt_record)
            write_json_atomic(
                output_dir / "attempts" / f"attempt-{attempt:03d}.json", attempt_record
            )
            complete = (
                return_code == 0
                and attempt_record["child_state"] == "complete"
                and attempt_record["completed_groups"] == config.expected_groups
            )
            if complete:
                result = {
                    "state": "complete",
                    "attempts": attempts,
                    "completed_groups": config.expected_groups,
                    "expected_groups": config.expected_groups,
                    "training_started": False,
                    "source_run_mutated": False,
                }
                _write_supervisor_status(output_dir, result)
                return result
            if local_attempt > config.max_restarts:
                break
            backoff = min(
                config.restart_backoff_seconds * (2 ** (local_attempt - 1)),
                config.max_restart_backoff_seconds,
            )
            _write_supervisor_status(
                output_dir,
                {
                    "state": "restarting",
                    "next_attempt": attempt + 1,
                    "backoff_seconds": backoff,
                    "last_attempt": attempt_record,
                    "training_started": False,
                },
            )
            sleep(backoff)

        result = {
            "state": "failed",
            "attempts": attempts,
            "completed_groups": _group_count(output_dir),
            "expected_groups": config.expected_groups,
            "training_started": False,
            "source_run_mutated": False,
        }
        _write_supervisor_status(output_dir, result)
        raise AuditSupervisorError(
            f"audit worker exhausted {config.max_restarts + 1} attempts; "
            f"completed {result['completed_groups']}/{config.expected_groups} groups"
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--judge-urls", nargs="+", required=True)
    parser.add_argument("--through", type=int, default=34)
    parser.add_argument("--group-workers", type=int, default=8)
    parser.add_argument("--requests-per-group", type=int, default=8)
    parser.add_argument("--expected-groups", type=int, default=1692)
    parser.add_argument("--max-restarts", type=int, default=12)
    parser.add_argument("--restart-backoff-seconds", type=float, default=15.0)
    parser.add_argument("--max-restart-backoff-seconds", type=float, default=300.0)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--snapshot-seconds", type=float, default=300.0)
    parser.add_argument("--stall-seconds", type=float, default=1800.0)
    parser.add_argument("--snapshot-bootstrap-iterations", type=int, default=200)
    parser.add_argument("--epsilon-z", type=float, default=0.01)
    parser.add_argument("--epsilon-t", type=float, default=0.01)
    parser.add_argument("--python", default=sys.executable)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    result = run_supervisor(
        SupervisorConfig(
            run_dir=args.run,
            output_dir=args.output,
            judge_urls=tuple(args.judge_urls),
            through_step=args.through,
            group_workers=args.group_workers,
            requests_per_group=args.requests_per_group,
            expected_groups=args.expected_groups,
            max_restarts=args.max_restarts,
            restart_backoff_seconds=args.restart_backoff_seconds,
            max_restart_backoff_seconds=args.max_restart_backoff_seconds,
            poll_seconds=args.poll_seconds,
            snapshot_seconds=args.snapshot_seconds,
            stall_seconds=args.stall_seconds,
            snapshot_bootstrap_iterations=args.snapshot_bootstrap_iterations,
            epsilon_z=args.epsilon_z,
            epsilon_t=args.epsilon_t,
            python=args.python,
        )
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
