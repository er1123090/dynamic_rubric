#!/usr/bin/env python3
"""Publish dense Medicine checkpoints sequentially without local cleanup."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil

from huggingface_hub import HfApi, hf_hub_download, save_torch_state_dict

from dynamic_rubric.artifacts import read_json, write_json_atomic
from dynamic_rubric.hashing import sha256_file
from scripts.phase1.hf_upload_watchdog import run_upload


ROOT = Path(__file__).resolve().parents[2] / "outputs/medicine"
RUNS = {
    "online": ROOT / "online_rubrics/seed-11/phase1-online-rubrics-medicine-full-dense-20260919-seed11",
    "static": ROOT / "static_r0_matched/seed-11/phase1-static-r0-medicine-qwen3-4b-matched-dense-20260928-seed11",
}
REPOS = {
    "online": "HYU-NLP-EVAL/qwen3-4b-rar-medicine-onlinerubrics-dense-seed11-step-{step:03d}",
    "static": "HYU-NLP-EVAL/qwen3-4b-rar-medicine-static-r0-matched-dense-seed11-step-{step:03d}",
}
PARAMETER_FILES = {
    "actor/fsdp_config.json",
    "actor/model_world_size_1_rank_0.pt",
    *(f"actor/huggingface/{name}" for name in (
        "chat_template.jinja", "config.json", "generation_config.json",
        "tokenizer.json", "tokenizer_config.json",
    )),
}
RESUME_FILES = {
    "actor/optim_world_size_1_rank_0.pt",
    "actor/extra_state_world_size_1_rank_0.pt",
    "data.pt",
}


def checkpoint(run: Path, step: int) -> tuple[Path, int, list[Path]]:
    root = run / "verl-run/checkpoints"
    latest = int((root / "latest_checkpointed_iteration.txt").read_text().strip())
    source = root / f"global_step_{step}"
    if root.is_symlink() or source.is_symlink() or not source.is_dir() or step > latest:
        raise ValueError(f"unsafe or absent checkpoint: {source}")
    files = sorted(path for path in source.rglob("*") if path.is_file())
    for path in source.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"checkpoint symlink: {path}")
    names = {path.relative_to(source).as_posix() for path in files}
    expected = PARAMETER_FILES | (RESUME_FILES if step == latest else set())
    if names != expected:
        raise ValueError(f"checkpoint file mismatch at step {step}: {names ^ expected}")
    return source, latest, files


def inventory(files: list[Path], base: Path, prefix: str = "") -> list[dict]:
    return [{
        "path": f"{prefix}{path.relative_to(base).as_posix()}",
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    } for path in files]


def actor_tree_hash(records: list[dict]) -> str:
    digest = hashlib.sha256()
    for row in records:
        name = row["path"]
        if name not in PARAMETER_FILES:
            continue
        digest.update(name.removeprefix("actor/").encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(row["sha256"]))
    return digest.hexdigest()


def online_export(run: Path, source: Path, step: int, actor_sha: str) -> Path:
    exported = run / "hf_dense_exports" / f"global_step_{step}"
    if exported.is_symlink() or not exported.is_dir():
        raise ValueError(f"missing online BF16 export: {exported}")
    manifest = read_json(exported / "audit_export_manifest.json")
    if not (
        manifest.get("artifact_kind") == "phase1_policy_checkpoint_export"
        and manifest.get("run_id") == run.name
        and manifest.get("checkpoint_step") == step
        and manifest.get("source_model_sha256") == actor_sha
        and manifest.get("source_model_bytes") == (source / "actor/model_world_size_1_rank_0.pt").stat().st_size
    ):
        raise ValueError(f"online export does not match checkpoint {step}")
    for artifact in manifest.get("artifacts", []):
        path = Path(artifact["path"])
        if path.parent.resolve() != exported.resolve() or path.is_symlink():
            raise ValueError(f"unsafe export artifact: {path}")
        if path.stat().st_size != artifact["bytes"] or sha256_file(path) != artifact["sha256"]:
            raise ValueError(f"export artifact mismatch: {path}")
    return exported


def static_export(run: Path, source: Path, step: int, actor_sha: str) -> Path:
    import torch

    exported = run / "hf_dense_exports" / f"global_step_{step}"
    marker = exported / "source_actor_sha256.json"
    if marker.is_file():
        if read_json(marker) != {"source_actor_sha256": actor_sha, "checkpoint_step": step}:
            raise ValueError(f"static export belongs to different source: {exported}")
        return exported
    if exported.exists():
        raise ValueError(f"incomplete static export needs inspection: {exported}")
    exported.mkdir(parents=True)
    for path in (source / "actor/huggingface").iterdir():
        if path.is_file():
            shutil.copy2(path, exported / path.name)
    config_path = exported / "config.json"
    config = read_json(config_path)
    config["dtype"] = "bfloat16"
    write_json_atomic(config_path, config, immutable=False)
    state = torch.load(source / "actor/model_world_size_1_rank_0.pt", map_location="cpu",
                       weights_only=False, mmap=True)
    if not isinstance(state, dict) or not state:
        raise ValueError("invalid static actor state")
    if not torch.equal(state["model.embed_tokens.weight"], state["lm_head.weight"]):
        raise ValueError("static actor tied weights differ")
    state.pop("lm_head.weight")
    bf16 = {name: tensor.detach().to(dtype=torch.bfloat16).contiguous()
            for name, tensor in state.items()}
    save_torch_state_dict(bf16, exported, max_shard_size="4GB", safe_serialization=True,
                          metadata={"format": "pt", "method": "static_r0_matched",
                                    "optimizer_step": str(step)})
    del bf16, state
    write_json_atomic(marker, {"source_actor_sha256": actor_sha, "checkpoint_step": step})
    return exported


def link_file(source: Path, target: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"unsafe source: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.is_symlink() or not os.path.samefile(source, target):
            raise ValueError(f"staging file differs: {target}")
    else:
        os.link(source, target)


def verify_remote(api: HfApi, repo_id: str, revision: str, rows: list[dict]) -> None:
    info = api.repo_info(repo_id, repo_type="model", revision=revision, files_metadata=True)
    if info.private or info.sha != revision:
        raise ValueError(f"repository is private or revision changed: {repo_id}")
    remote = {entry.rfilename: entry for entry in info.siblings}
    for row in rows:
        item = remote.get(row["path"])
        if item is None or item.size != row["bytes"]:
            raise ValueError(f"remote file missing or wrong size: {repo_id}/{row['path']}")
        if item.lfs is not None:
            sha = item.lfs.sha256 if hasattr(item.lfs, "sha256") else item.lfs["sha256"]
        else:
            cached = hf_hub_download(repo_id, row["path"], revision=revision)
            sha = sha256_file(Path(cached))
        if sha != row["sha256"]:
            raise ValueError(f"remote SHA256 mismatch: {repo_id}/{row['path']}")


def publish(family: str, step: int) -> dict:
    run = RUNS[family]
    repo_id = REPOS[family].format(step=step)
    source, latest, source_files = checkpoint(run, step)
    receipt_path = run / "verl-run/checkpoint_archives_dense" / f"global_step_{step}.json"
    api = HfApi()
    if receipt_path.is_file():
        prior = read_json(receipt_path)
        if prior.get("repo_id") == repo_id and prior.get("state") == "verified":
            verify_remote(api, repo_id, prior["revision"], prior["files"])
            print(json.dumps({"state": "already_verified", "family": family, "step": step}), flush=True)
            return prior
    source_rows = inventory(source_files, source, "original_checkpoint/")
    original_records = [{**row, "path": row["path"].removeprefix("original_checkpoint/")}
                        for row in source_rows]
    actor_sha = next(row["sha256"] for row in original_records
                     if row["path"] == "actor/model_world_size_1_rank_0.pt")
    if family == "online":
        commit = read_json(run / "verl-run/online_steps" / f"step-{step:06d}" / "commit.json")
        if actor_tree_hash(original_records) != commit["artifacts"]["actor_parameter_hash"]:
            raise ValueError(f"online commit lineage mismatch: step {step}")
        exported = online_export(run, source, step, actor_sha)
    else:
        exported = static_export(run, source, step, actor_sha)
    export_files = sorted(path for path in exported.iterdir()
                          if path.is_file() and path.name not in {
                              "audit_export_manifest.json", "source_actor_sha256.json"})
    if not any(path.name.endswith(".safetensors") for path in export_files):
        raise ValueError(f"missing BF16 model weights: {exported}")
    export_rows = inventory(export_files, exported)
    stage = run / "hf_dense_publication" / f"global_step_{step}" / "public"
    for row, path in zip(source_rows, source_files, strict=True):
        link_file(path, stage / row["path"])
    for row, path in zip(export_rows, export_files, strict=True):
        link_file(path, stage / row["path"])
    readme = stage / "README.md"
    description = ("OnlineRubrics-Every" if family == "online" else "static R0 matched")
    readme_text = (f"---\nlicense: apache-2.0\nbase_model: Qwen/Qwen3-4B-Instruct-2507\n"
                   f"library_name: transformers\npipeline_tag: text-generation\n---\n"
                   f"# RaR-Medicine {description} dense checkpoint, step {step}\n\n"
                   f"Run: `{run.name}`. The root contains a BF16 model for inference.\n"
                   f"`original_checkpoint/` contains the original veRL checkpoint files"
                   f"{' including optimizer and data state' if step == latest else ' (model parameters only)'}.\n"
                   f"The local checkpoint remains on disk. Research use only.\n")
    if readme.exists() and readme.read_text() != readme_text:
        raise ValueError(f"staging README differs: {readme}")
    readme.write_text(readme_text)
    rows = source_rows + export_rows
    manifest = {
        "schema_version": 1, "state": "prepared", "family": family, "run_id": run.name,
        "checkpoint_step": step, "repo_id": repo_id, "source_actor_sha256": actor_sha,
        "includes_resume_state": step == latest, "files": rows,
    }
    write_json_atomic(stage / "archive_manifest.json", manifest, immutable=False)
    for name in ("README.md", "archive_manifest.json"):
        path = stage / name
        rows.append({"path": name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    api.create_repo(repo_id, repo_type="model", private=False, exist_ok=True)
    if api.repo_info(repo_id, repo_type="model").private:
        raise ValueError(f"refusing to upload to private repository: {repo_id}")
    print(json.dumps({"state": "uploading", "family": family, "step": step,
                      "repo_id": repo_id, "file_count": len(rows)}), flush=True)
    env = os.environ.copy()
    env.setdefault("HF_HUB_DISABLE_XET", "1")
    hf = shutil.which("hf")
    if hf is None:
        raise ValueError("hf CLI is unavailable")
    run_upload([hf, "upload-large-folder", repo_id, str(stage), "--repo-type", "model",
                "--num-workers", "1", "--no-bars"], env=env,
               idle_timeout=float(env.get("HF_ARCHIVE_UPLOAD_IDLE_TIMEOUT_SECONDS", "900")),
               max_retries=int(env.get("HF_ARCHIVE_UPLOAD_MAX_RETRIES", "5")))
    revision = api.repo_info(repo_id, repo_type="model").sha
    verify_remote(api, repo_id, revision, rows)
    checkpoint(run, step)
    receipt = {**manifest, "state": "verified", "revision": revision, "files": rows,
               "verified_at": datetime.now(timezone.utc).isoformat(),
               "local_checkpoint_retained": True}
    write_json_atomic(receipt_path, receipt, immutable=False)
    print(json.dumps({"state": "verified", "family": family, "step": step,
                      "repo_id": repo_id, "revision": revision}), flush=True)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=RUNS, default="online")
    parser.add_argument("--step", type=int)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    if args.all == (args.step is not None):
        parser.error("choose exactly one of --all or --step")
    families = ("online", "static") if args.all else (args.family,)
    for family in families:
        steps = range(1, 61) if family == "online" else range(20)
        if args.step is not None:
            steps = (args.step,)
        for step in steps:
            publish(family, step)


if __name__ == "__main__":
    main()
