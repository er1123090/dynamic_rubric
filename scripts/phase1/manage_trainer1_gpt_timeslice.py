#!/usr/bin/env python3
"""Start Trainer GPU 1 GPT only during OnlineRubrics extraction/deduplication."""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


def command(*args: str, check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, check=check, text=True, capture_output=True)


def session_exists(name: str) -> bool:
    return command("tmux", "has-session", "-t", name).returncode == 0


def stop_session(name: str) -> None:
    if session_exists(name):
        command("tmux", "kill-session", "-t", name)


def trainer_exists(pattern: str) -> bool:
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            cmdline = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if "verl.trainer.main_ppo" in cmdline and pattern in cmdline:
            return True
    return False


def proxy_status(url: str) -> dict[str, object] | None:
    try:
        with urllib.request.urlopen(url, timeout=1.0) as response:
            return json.load(response)
    except (OSError, urllib.error.URLError, json.JSONDecodeError):
        return None


def gpu_free_mib(index: int) -> int:
    result = command(
        "nvidia-smi",
        f"--id={index}",
        "--query-gpu=memory.free",
        "--format=csv,noheader,nounits",
    )
    if result.returncode != 0:
        return 0
    return int(result.stdout.strip().splitlines()[0])


def port_has_established_connection(port: int) -> bool:
    target = f"{port:04X}"
    for table in (Path("/proc/net/tcp"), Path("/proc/net/tcp6")):
        try:
            lines = table.read_text().splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            fields = line.split()
            if len(fields) < 4 or fields[3] != "01":
                continue
            local_port = fields[1].rsplit(":", 1)[-1]
            remote_port = fields[2].rsplit(":", 1)[-1]
            if target in {local_port, remote_port}:
                return True
    return False


def start_backend(session: str, serve_script: Path, log_path: Path) -> None:
    shell_command = f"exec {serve_script} >> {log_path} 2>&1"
    command("tmux", "new-session", "-d", "-s", session, shell_command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer-pattern", required=True)
    parser.add_argument("--proxy-status", default="http://127.0.0.1:28011/_proxy_status")
    parser.add_argument("--qwen-port", type=int, default=28014)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--min-free-mib", type=int, default=100000)
    parser.add_argument("--backend-session", default="online64-trainer1-gpt")
    parser.add_argument("--serve-script", type=Path, required=True)
    parser.add_argument("--backend-log", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    args = parser.parse_args()

    stopping = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    last_start_attempt = 0.0
    initial_status = proxy_status(args.proxy_status)
    served_until = int(initial_status["chat_requests"]) if initial_status else 0
    print(
        f"[{time.strftime('%FT%T%z')}] proxy request baseline={served_until}",
        flush=True,
    )
    observed_trainer = False

    try:
        while not stopping:
            trainer_up = trainer_exists(args.trainer_pattern)
            observed_trainer = observed_trainer or trainer_up
            if observed_trainer and not trainer_up:
                print(
                    f"[{time.strftime('%FT%T%z')}] trainer exited; supervisor stopping",
                    flush=True,
                )
                break

            status = proxy_status(args.proxy_status)
            backend_up = session_exists(args.backend_session)
            qwen_active = port_has_established_connection(args.qwen_port)

            if backend_up and qwen_active:
                print(
                    f"[{time.strftime('%FT%T%z')}] Qwen judge active; "
                    "stopping Trainer GPU 1 GPT",
                    flush=True,
                )
                stop_session(args.backend_session)
                if status:
                    served_until = int(status["chat_requests"])
            elif trainer_up and not backend_up and not qwen_active and status:
                chat_requests = int(status["chat_requests"])
                active = int(status["active_chat_requests"])
                ready = bool(status["primary_ready"])
                now = time.monotonic()
                if (
                    active > 0
                    and chat_requests > served_until
                    and not ready
                    and now - last_start_attempt >= 30.0
                ):
                    free = gpu_free_mib(args.gpu)
                    if free >= args.min_free_mib:
                        print(
                            f"[{time.strftime('%FT%T%z')}] GPT work detected; "
                            f"starting backend on GPU {args.gpu} (free={free} MiB)",
                            flush=True,
                        )
                        start_backend(
                            args.backend_session,
                            args.serve_script.resolve(),
                            args.backend_log.resolve(),
                        )
                        last_start_attempt = now
            time.sleep(args.poll_seconds)
    finally:
        stop_session(args.backend_session)


if __name__ == "__main__":
    main()
