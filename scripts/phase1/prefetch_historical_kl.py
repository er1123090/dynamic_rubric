"""CPU/network-only, pinned HF prefetch for the existing immutable KL plan."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time

from dynamic_rubric.artifacts import read_json, validate_artifact_record, write_json_atomic
from dynamic_rubric.hashing import sha256_file


@contextmanager
def model_lock(root, step):
    directory = root / "model_locks"
    directory.mkdir(exist_ok=True)
    if directory.is_symlink():
        raise ValueError("Unsafe model lock directory")
    path = directory / f"step-{step}.lock"
    if path.is_symlink():
        raise ValueError("Unsafe model lock file")
    with path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def model_scored(root, receipt):
    plan = read_json(root / "plan.json")
    step = receipt["checkpoint_step"]
    cells = [(a, t) for a, t in plan["cells"] if a == step]
    if not cells:
        raise ValueError("Model is not required by the KL plan")
    paths = [root / f"seals/model-{a}_pool-{t}.json" for a, t in cells]
    if not all(path.is_file() for path in paths):
        return False
    for (_, t), path in zip(cells, paths):
        seal = read_json(path)
        if (
            seal["checkpoint_hash"] != receipt["audit_export_manifest"]["source_model_sha256"]
            or seal["pool"] != plan["pool_files"][str(t)]
        ):
            raise ValueError("Completed score seal identity mismatch")
        validate_artifact_record(seal["scores"])
    return True


def validate_receipt(run, receipt):
    if (
        receipt.get("state") != "verified"
        or receipt.get("run_id") != run.name
        or not re.fullmatch(r"[0-9a-f]{40}", receipt.get("revision", ""))
    ):
        raise ValueError("Unverified or unpinned archive receipt")
    step = receipt["checkpoint_step"]
    if (
        receipt["repo_id"]
        != f"HYU-NLP-EVAL/qwen3-4b-rar-medicine-onlinerubrics-seed11-step-{step:03d}"
    ):
        raise ValueError("Unexpected policy archive repository")
    files = receipt["public_export_files"]
    names = [f["remote_path"] for f in files]
    metadata = {
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "model.safetensors.index.json",
    }
    if not files or len(names) != len(set(names)):
        raise ValueError("Missing or duplicate inference inventory")
    for f in files:
        name = f["remote_path"]
        if (
            Path(name).name != name
            or (
                name not in metadata
                and not re.fullmatch(r"model(?:-\d+-of-\d+)?\.safetensors", name)
            )
            or f["bytes"] <= 0
            or not re.fullmatch(r"[0-9a-f]{64}", f["sha256"])
        ):
            raise ValueError("Unsafe inference file inventory")


def download_model(run, root, receipt, *, hf_cli, env, skip_if_scored=False):
    validate_receipt(run, receipt)
    step = receipt["checkpoint_step"]
    with model_lock(root, step):
        scored = model_scored(root, receipt)
        if scored and skip_if_scored:
            return None
        parent = root / "temporary_models"
        parent.mkdir(exist_ok=True)
        target = parent / f"global_step_{step}"
        if parent.is_symlink() or target.is_symlink():
            raise ValueError("Unsafe temporary model directory")
        target.mkdir(exist_ok=True)
        if any(p.is_symlink() for p in target.rglob("*")):
            raise ValueError("Unsafe temporary model symlink")
        write_json_atomic(
            target / "kl-download-owner.json",
            {
                "repo_id": receipt["repo_id"],
                "revision": receipt["revision"],
                "checkpoint_step": step,
                "kl_root": str(root),
            },
        )
        files = receipt["public_export_files"]
        if not all(
            (target / f["remote_path"]).is_file()
            and (target / f["remote_path"]).stat().st_size == f["bytes"]
            for f in files
        ):
            argv = [
                hf_cli,
                "download",
                receipt["repo_id"],
                *[f["remote_path"] for f in files],
                "--revision",
                receipt["revision"],
                "--local-dir",
                str(target),
                "--max-workers",
                "4",
                "--quiet",
            ]
            with (root / f"download-step-{step}.log").open("ab") as log:
                for attempt in range(3):
                    try:
                        subprocess.run(
                            argv,
                            env=env,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            check=True,
                            timeout=3600,
                        )
                        break
                    except (subprocess.SubprocessError, OSError):
                        if attempt == 2:
                            raise
                        time.sleep(10)
        for f in files:
            path = target / f["remote_path"]
            if (
                path.is_symlink()
                or path.stat().st_size != f["bytes"]
                or sha256_file(path) != f["sha256"]
            ):
                raise ValueError("Downloaded policy checksum mismatch")
        write_json_atomic(
            root / "downloads" / f"step-{step}-verified.json",
            {
                "repo_id": receipt["repo_id"],
                "revision": receipt["revision"],
                "checkpoint_hash": receipt["audit_export_manifest"]["source_model_sha256"],
                "files": files,
            },
        )
        return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--exclude-steps", nargs="*", type=int, default=[])
    parser.add_argument("--hf-cli", default="hf")
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error("Use one to four concurrent model downloads")
    run, root = args.run_dir.resolve(), args.output_root.resolve()
    journal = root / "prefetch"
    journal.mkdir(exist_ok=True)
    with (journal / "runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = read_json(root / "plan.json")
        if plan["run_id"] != run.name:
            raise ValueError("KL plan belongs to another run")
        receipts = []
        for step in sorted({a for a, t in plan["cells"]} - set(args.exclude_steps), reverse=True):
            receipt = read_json(run / f"verl-run/checkpoint_archives/global_step_{step}.json")
            validate_receipt(run, receipt)
            if not model_scored(root, receipt):
                receipts.append(receipt)
        total_bytes = sum(f["bytes"] for r in receipts for f in r["public_export_files"])
        if shutil.disk_usage(root).free < total_bytes + 100 * 1024**3:
            raise RuntimeError("Not enough disk headroom for KL prefetch plus 100GiB reserve")
        states = {str(r["checkpoint_step"]): "pending" for r in receipts}
        mutex = threading.Lock()

        def report(step=None, state=None, error=None, overall="running"):
            with mutex:
                if step is not None:
                    states[str(step)] = state
                    write_json_atomic(
                        journal / f"step-{step}.json",
                        {
                            "step": step,
                            "state": state,
                            "error": error,
                            "time": time.time(),
                        },
                        immutable=False,
                    )
                write_json_atomic(
                    journal / "status.json",
                    {
                        "state": overall,
                        "models": dict(states),
                        "planned_bytes": total_bytes,
                        "excluded_steps": args.exclude_steps,
                        "workers": args.workers,
                        "gpu_used": False,
                        "updated_at": time.time(),
                    },
                    immutable=False,
                )

        def work(receipt):
            step = receipt["checkpoint_step"]
            report(step, "downloading")
            try:
                target = download_model(
                    run,
                    root,
                    receipt,
                    hf_cli=args.hf_cli,
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "HF_HUB_DISABLE_XET": "1"},
                    skip_if_scored=True,
                )
                report(step, "verified_ready" if target else "already_scored")
            except Exception as error:
                report(step, "failed", repr(error))
                raise

        report()
        failures = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(work, r) for r in receipts]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as error:
                    failures.append(repr(error))
        report(overall="failed" if failures else "complete")
        if failures:
            raise RuntimeError(f"{len(failures)} model downloads failed; see per-model status")


if __name__ == "__main__":
    main()
