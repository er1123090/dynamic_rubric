from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import threading


SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts" / "phase1" / "run_final_eval_grade_queue.sh"
)


def test_dry_run_exposes_order_without_side_effects(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            str(SCRIPT),
            "--config",
            str(tmp_path / "missing-is-allowed-in-dry-run.yaml"),
            "--run-dir",
            str(tmp_path / "missing-is-allowed-in-dry-run"),
            "--expected-grades-per-model",
            "60091",
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "dry_run=true" in completed.stdout
    positions = [
        completed.stdout.index(model) for model in ("static_final", "online_base", "online_final")
    ]
    assert positions == sorted(positions)
    assert "--stage summarize" in completed.stdout


def test_queue_checks_endpoint_counts_and_runs_in_order(tmp_path: Path) -> None:
    request_count = 0

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            nonlocal request_count
            request_count += 1
            body = b'{"data":[{"id":"Qwen/Qwen3-32B"}]}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    run_dir = tmp_path / "full-run"
    (run_dir / "prepared").mkdir(parents=True)
    (run_dir / "prepared" / "prompts.jsonl").write_text(
        json.dumps(
            {
                "dataset": "medicine",
                "prompt_id": "p1",
                "criteria": [
                    {
                        "criterion_id": "c1",
                        "criterion": "correct",
                        "points": 1,
                        "tags": [],
                    }
                ],
            }
        )
        + "\n"
    )
    config = tmp_path / "config.yaml"
    config.write_text("{}")
    calls = tmp_path / "calls.txt"
    runner = tmp_path / "fake_runner.py"
    runner.write_text(
        r"""
import argparse
import json
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--config")
parser.add_argument("--stage")
parser.add_argument("--model")
parser.add_argument("--output-dir")
args = parser.parse_args()
root = Path(os.environ["TEST_RUN_DIR"])
log = Path(os.environ["TEST_CALL_LOG"])
with log.open("a") as stream:
    stream.write(f"{args.stage}:{args.model or ''}\n")
if args.stage == "grade":
    response = {
        "schema_version": 1, "dataset": "medicine", "prompt_id": "p1",
        "model_name": args.model, "response_id": f"{args.model}-response", "text": "answer",
    }
    response_path = root / "responses" / args.model / "medicine" / "response.json"
    response_path.parent.mkdir(parents=True, exist_ok=True)
    response_path.write_text(json.dumps(response))
    grade = {
        "schema_version": 1, "dataset": "medicine", "model_name": args.model,
        "prompt_id": "p1", "response_id": response["response_id"], "criterion_id": "c1",
        "criterion": "correct", "points": 1, "tags": [], "criteria_met": True,
        "explanation": "ok", "requested_model": "Qwen/Qwen3-32B",
        "returned_model": "Qwen/Qwen3-32B",
    }
    grade_path = root / "grades" / args.model / "medicine" / "grade.json"
    grade_path.parent.mkdir(parents=True, exist_ok=True)
    grade_path.write_text(json.dumps(grade))
""".strip()
    )
    environment = {
        **os.environ,
        "TEST_RUN_DIR": str(run_dir),
        "TEST_CALL_LOG": str(calls),
    }
    subprocess.run(
        [
            sys.executable,
            str(runner),
            "--config",
            str(config),
            "--stage",
            "grade",
            "--model",
            "static_base",
        ],
        check=True,
        env=environment,
    )
    calls.write_text("")
    try:
        completed = subprocess.run(
            [
                str(SCRIPT),
                "--config",
                str(config),
                "--run-dir",
                str(run_dir),
                "--expected-grades-per-model",
                "1",
                "--tmux-session",
                "definitely-not-a-real-test-session",
                "--judge-url",
                f"http://127.0.0.1:{server.server_port}/v1",
                "--python",
                sys.executable,
                "--runner",
                str(runner),
                "--poll-seconds",
                "1",
            ],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        static_grade = next((run_dir / "grades" / "static_base").rglob("*.json"))
        corrupt = json.loads(static_grade.read_text())
        corrupt["response_id"] = "wrong-response"
        static_grade.write_text(json.dumps(corrupt))
        rejected = subprocess.run(
            [
                str(SCRIPT),
                "--config",
                str(config),
                "--run-dir",
                str(run_dir),
                "--expected-grades-per-model",
                "1",
                "--tmux-session",
                "definitely-not-a-real-test-session",
                "--judge-url",
                f"http://127.0.0.1:{server.server_port}/v1",
                "--python",
                sys.executable,
                "--runner",
                str(runner),
                "--poll-seconds",
                "1",
            ],
            capture_output=True,
            text=True,
            env=environment,
        )
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
    assert calls.read_text().splitlines() == [
        "grade:static_final",
        "grade:online_base",
        "grade:online_final",
        "summarize:",
    ]
    assert request_count == 10
    assert "queue_complete=true" in completed.stdout
    assert rejected.returncode != 0
    assert "grade response identity mismatch" in rejected.stderr
