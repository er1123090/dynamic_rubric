#!/usr/bin/env python3
"""Publish a static-R0 matched RaR-Medicine checkpoint as a public HF model."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from huggingface_hub import HfApi, hf_hub_download, save_torch_state_dict
import torch


HF_METADATA = {
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
}
ORIGINAL_PARAMETER_FILES = {
    "actor/fsdp_config.json",
    "actor/model_world_size_1_rank_0.pt",
} | {f"actor/huggingface/{name}" for name in HF_METADATA}
FULL_RESUME_FILES = ORIGINAL_PARAMETER_FILES | {
    "actor/extra_state_world_size_1_rank_0.pt",
    "actor/optim_world_size_1_rank_0.pt",
    "data.pt",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_run(run: Path, step: int) -> tuple[Path, dict, int]:
    run = run.resolve()
    config_path = run / "config.resolved.json"
    checkpoint_root = run / "verl-run/checkpoints"
    source = checkpoint_root / f"global_step_{step}"
    if run.is_symlink() or not config_path.is_file() or not source.is_dir() or source.is_symlink():
        raise ValueError("run/config/checkpoint path is missing or unsafe")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if (
        config.get("method") != "static_r0_matched"
        or config.get("domain") != "medicine"
        or config.get("seed") != 11
        or config.get("training", {}).get("reward_source") != "rar_static_r0_only"
    ):
        raise ValueError("resolved config is not the matched static-R0 Medicine run")

    latest = int((checkpoint_root / "latest_checkpointed_iteration.txt").read_text().strip())
    if step > latest:
        raise ValueError(f"requested step {step} is newer than the saved latest checkpoint {latest}")

    files = {
        path.relative_to(source).as_posix()
        for path in source.rglob("*")
        if path.is_file()
    }
    expected = FULL_RESUME_FILES if step == latest else ORIGINAL_PARAMETER_FILES
    if files != expected:
        kind = "full resume" if step == latest else "historical parameter-only"
        raise ValueError(f"{kind} checkpoint allowlist mismatch: {files ^ expected}")
    for path in source.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"checkpoint symlink is forbidden: {path}")
    return source, config, latest


def hardlink_file(source: Path, target: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"invalid source file: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    os.link(source, target)


def model_card(config: dict, step: int, repo_id: str, actor_sha256: str) -> str:
    policy = config["models"]["policy"]
    expected_steps = config["training"]["expected_global_steps"]
    return f"""---
license: apache-2.0
base_model: {policy['model']}
library_name: transformers
pipeline_tag: text-generation
tags:
- qwen3
- grpo
- static-rubric
- r0
- rar-medicine
- research
---
# Static-R0 Matched GRPO on RaR-Medicine — step {step}

This is the policy after **{step} global optimizer updates** of the matched
static-rubric GRPO run (planned total: {expected_steps}). It is intentionally
separate from the OnlineRubrics/dynamic-rubric checkpoints.

## Experiment identity

- Method: `static_r0_matched`
- Reward source: `rar_static_r0_only`
- Domain: Medicine
- Training data: RaR-Medicine, 1,500 prompts
- Seed: 11
- Policy: `{policy['model']}`
- Base revision: `{policy['revision']}`
- Thinking: disabled
- GRPO global prompt batch: {config['training']['global_prompt_batch']}
- Rollouts per prompt: {config['training']['rollouts_per_prompt']}
- Learning rate: {config['training']['learning_rate']}

The root files are a BF16 Transformers export for inference. The
`original_checkpoint/` directory contains the exact original veRL/FSDP policy
parameter checkpoint and its tokenizer/configuration files. Optimizer,
trainer, and data-loader state are intentionally not published; the complete
resume checkpoint remains on Trainer.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo_id = "{repo_id}"
tokenizer = AutoTokenizer.from_pretrained(repo_id)
model = AutoModelForCausalLM.from_pretrained(
    repo_id, torch_dtype="bfloat16", device_map="auto"
)
```

This is an intermediate research checkpoint, not a clinical model. No medical
capability or safety claim is made.

Original actor parameter SHA256: `{actor_sha256}`
"""


def prepare_stage(run: Path, source: Path, config: dict, step: int, repo_id: str) -> Path:
    task_root = run / "hf_publish_staging" / f"global_step_{step}"
    stage = task_root / "public"
    complete = task_root / "prepared.json"
    actor = source / "actor/model_world_size_1_rank_0.pt"
    actor_sha256 = sha256_file(actor)

    if complete.is_file() and stage.is_dir():
        state = json.loads(complete.read_text(encoding="utf-8"))
        if state.get("actor_sha256") == actor_sha256 and state.get("repo_id") == repo_id:
            return stage
        raise ValueError("existing staging directory belongs to different model content")
    if task_root.exists():
        raise ValueError(f"incomplete staging directory needs inspection: {task_root}")

    task_root.parent.mkdir(parents=True, exist_ok=True)
    temp_root = Path(tempfile.mkdtemp(prefix=f"step-{step}-", dir=task_root.parent))
    temp_stage = temp_root / "public"
    temp_stage.mkdir()
    try:
        source_hf = source / "actor/huggingface"
        for name in sorted(HF_METADATA):
            shutil.copy2(source_hf / name, temp_stage / name)

        exported_config_path = temp_stage / "config.json"
        exported_config = json.loads(exported_config_path.read_text(encoding="utf-8"))
        exported_config["dtype"] = "bfloat16"
        write_json(exported_config_path, exported_config)

        print(json.dumps({"state": "loading_actor", "step": step}), flush=True)
        state_dict = torch.load(actor, map_location="cpu", weights_only=False, mmap=True)
        if not isinstance(state_dict, dict) or not state_dict:
            raise ValueError("actor parameter file is not a non-empty state dict")
        if "model.embed_tokens.weight" not in state_dict or "lm_head.weight" not in state_dict:
            raise ValueError("expected tied embedding keys are missing")
        if not torch.equal(state_dict["model.embed_tokens.weight"], state_dict["lm_head.weight"]):
            raise ValueError("tied embedding weights differ")
        state_dict.pop("lm_head.weight")
        converted = {
            name: tensor.detach().to(dtype=torch.bfloat16).contiguous()
            for name, tensor in state_dict.items()
        }
        print(json.dumps({"state": "writing_bf16_export", "step": step}), flush=True)
        save_torch_state_dict(
            converted,
            temp_stage,
            max_shard_size="4GB",
            safe_serialization=True,
            metadata={
                "format": "pt",
                "method": "static_r0_matched",
                "reward_source": "rar_static_r0_only",
                "optimizer_step": str(step),
            },
        )
        del converted
        del state_dict

        original_records = []
        for relative in sorted(ORIGINAL_PARAMETER_FILES):
            local = source / relative
            remote = f"original_checkpoint/{relative}"
            hardlink_file(local, temp_stage / remote)
            original_records.append(
                {
                    "path": relative,
                    "remote_path": remote,
                    "bytes": local.stat().st_size,
                    "sha256": sha256_file(local),
                }
            )

        policy = config["models"]["policy"]
        license_path = Path(
            hf_hub_download(policy["model"], "LICENSE", revision=policy["revision"])
        )
        shutil.copy2(license_path, temp_stage / "LICENSE")
        (temp_stage / "README.md").write_text(
            model_card(config, step, repo_id, actor_sha256), encoding="utf-8"
        )

        public_export_records = []
        for local in sorted(temp_stage.iterdir()):
            if not local.is_file() or local.name in {"README.md", "LICENSE"}:
                continue
            public_export_records.append(
                {
                    "path": local.name,
                    "remote_path": local.name,
                    "bytes": local.stat().st_size,
                    "sha256": sha256_file(local),
                }
            )
        manifest = {
            "schema_version": 1,
            "artifact_kind": "phase1_policy_checkpoint_publication",
            "method": "static_r0_matched",
            "domain": "medicine",
            "seed": 11,
            "checkpoint_step": step,
            "checkpoint_id": f"global_step_{step}",
            "repo_id": repo_id,
            "actor_parameter_sha256": actor_sha256,
            "optimizer_state_published": False,
            "local_full_resume_checkpoint_retained": True,
            "original_parameter_files": original_records,
            "public_export_files": public_export_records,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        write_json(temp_stage / "archive_manifest.json", manifest)
        write_json(
            temp_root / "prepared.json",
            {"repo_id": repo_id, "step": step, "actor_sha256": actor_sha256},
        )
        os.replace(temp_root, task_root)
    except BaseException:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise
    return stage


def verify_remote_stage(stage: Path, repo_id: str, revision: str | None = None) -> str:
    api = HfApi()
    info = api.repo_info(
        repo_id,
        repo_type="model",
        revision=revision,
        files_metadata=True,
    )
    if info.private:
        raise ValueError("uploaded repository is not public")
    if revision is not None and info.sha != revision:
        raise ValueError("repository revision did not resolve to the pinned commit")
    remote = {item.rfilename: item for item in info.siblings}
    local_files = [
        path
        for path in stage.rglob("*")
        if path.is_file()
        and path.relative_to(stage).parts[:2] != (".cache", "huggingface")
    ]
    for local in local_files:
        relative = local.relative_to(stage).as_posix()
        item = remote.get(relative)
        if item is None or item.size != local.stat().st_size:
            raise ValueError(f"remote file missing or wrong size: {relative}")
        expected = sha256_file(local)
        if item.lfs is not None:
            actual = item.lfs.sha256 if hasattr(item.lfs, "sha256") else item.lfs["sha256"]
        else:
            cached = hf_hub_download(
                repo_id,
                relative,
                revision=info.sha,
                repo_type="model",
            )
            actual = sha256_file(Path(cached))
        if actual != expected:
            raise ValueError(f"remote checksum mismatch: {relative}")
    return info.sha


def upload_and_verify(stage: Path, repo_id: str, workers: int) -> str:
    from scripts.phase1.hf_upload_watchdog import run_upload

    api = HfApi()
    api.create_repo(repo_id, repo_type="model", private=False, exist_ok=True)
    if api.repo_info(repo_id, repo_type="model").private:
        raise ValueError("repository is private; refusing implicit visibility change")
    hf = shutil.which("hf")
    if hf is None:
        raise ValueError("hf CLI is required")
    env = os.environ.copy()
    env.setdefault("HF_HUB_DISABLE_XET", "1")
    run_upload(
        [
            hf,
            "upload-large-folder",
            repo_id,
            str(stage),
            "--repo-type",
            "model",
            "--num-workers",
            str(workers),
        ],
        env=env,
        idle_timeout=float(env.get("HF_ARCHIVE_UPLOAD_IDLE_TIMEOUT_SECONDS", "600")),
        max_retries=int(env.get("HF_ARCHIVE_UPLOAD_MAX_RETRIES", "5")),
    )
    return verify_remote_stage(stage, repo_id)


def cleanup_historical_checkpoint(
    run: Path,
    step: int,
    stage: Path,
    receipt: dict,
    receipt_path: Path,
) -> None:
    """Delete one historical checkpoint only after pinned remote verification."""

    source, _, latest = validate_run(run, step)
    if step >= latest:
        raise ValueError("refusing to delete the latest full resume checkpoint")
    manifest = json.loads((stage / "archive_manifest.json").read_text(encoding="utf-8"))
    actor = source / "actor/model_world_size_1_rank_0.pt"
    actor_sha256 = sha256_file(actor)
    if (
        manifest.get("checkpoint_step") != step
        or manifest.get("repo_id") != receipt.get("repo_id")
        or manifest.get("actor_parameter_sha256") != actor_sha256
        or receipt.get("actor_parameter_sha256") != actor_sha256
    ):
        raise ValueError("local source, manifest, and receipt are not bound to the same actor")
    verify_remote_stage(stage, receipt["repo_id"], receipt["revision"])
    # Re-read the retention boundary immediately before changing local state.
    _, _, current_latest = validate_run(run, step)
    if current_latest != latest or step >= current_latest:
        raise ValueError("latest checkpoint changed before historical cleanup")
    original_bytes = sum(path.stat().st_size for path in source.rglob("*") if path.is_file())
    receipt["deletion_authorized_at"] = datetime.now(timezone.utc).isoformat()
    receipt["protected_resume_checkpoint"] = f"global_step_{latest}"
    write_json(receipt_path, receipt)
    task_root = stage.parent
    if task_root.is_symlink() or source.is_symlink():
        raise ValueError("refusing to delete a symlinked checkpoint or staging directory")
    receipt["source_deletion_started_at"] = datetime.now(timezone.utc).isoformat()
    write_json(receipt_path, receipt)
    shutil.rmtree(source)
    receipt["staging_deletion_started_at"] = datetime.now(timezone.utc).isoformat()
    write_json(receipt_path, receipt)
    shutil.rmtree(task_root)
    receipt["local_deleted_at"] = datetime.now(timezone.utc).isoformat()
    receipt["original_bytes_removed"] = original_bytes
    write_json(receipt_path, receipt)
    print(
        json.dumps(
            {
                "state": "local_deleted_verified_remote",
                "step": step,
                "repo_id": receipt["repo_id"],
                "revision": receipt["revision"],
                "original_bytes_removed": original_bytes,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--step", required=True, type=int)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--cleanup-after-verify", action="store_true")
    args = parser.parse_args()

    if args.prepare_only and args.cleanup_after_verify:
        parser.error("--prepare-only cannot be combined with --cleanup-after-verify")

    run = args.run.resolve()
    source, config, latest = validate_run(run, args.step)
    stage = prepare_stage(run, source, config, args.step, args.repo_id)
    print(json.dumps({"state": "prepared", "stage": str(stage)}), flush=True)
    if args.prepare_only:
        return
    receipt_path = run / "verl-run/checkpoint_archives" / f"static_global_step_{args.step}.json"
    previous = (
        json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt_path.is_file()
        else None
    )
    revision = None
    if previous and previous.get("state") == "verified" and previous.get("repo_id") == args.repo_id:
        try:
            revision = verify_remote_stage(stage, args.repo_id, previous["revision"])
            print(
                json.dumps(
                    {"state": "reused_verified_revision", "step": args.step, "revision": revision}
                ),
                flush=True,
            )
        except Exception:
            revision = None
    if revision is None:
        revision = upload_and_verify(stage, args.repo_id, args.workers)
    actor_sha256 = sha256_file(source / "actor/model_world_size_1_rank_0.pt")
    receipt = {
        "schema_version": 1,
        "state": "verified",
        "repo_id": args.repo_id,
        "revision": revision,
        "checkpoint_step": args.step,
        "run_id": run.name,
        "actor_parameter_sha256": actor_sha256,
        "public": True,
        "source_is_latest_resume_checkpoint": args.step == latest,
        "local_full_resume_checkpoint_retained": (
            run / "verl-run/checkpoints" / f"global_step_{latest}"
        ).is_dir(),
        "verified_at": datetime.now(timezone.utc).isoformat(),
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(receipt_path, receipt)
    print(json.dumps(receipt, sort_keys=True), flush=True)
    if args.cleanup_after_verify:
        cleanup_historical_checkpoint(run, args.step, stage, receipt, receipt_path)


if __name__ == "__main__":
    main()
