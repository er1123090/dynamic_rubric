#!/usr/bin/env python3
"""Publish exact historical checkpoints plus prebuilt BF16 audit exports."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil

from dynamic_rubric.artifacts import read_json, write_json_atomic
from dynamic_rubric.hashing import sha256_file


ALLOWED_STEPS = (0, 3, 6, 9, 12, 13, 15, 16, 18, 21, 24, 27, 30, 32, 33, 34, 36, 39, 40, 42)
# Keep ALLOWED_STEPS as the completed historical batch authorization. New dense
# runs may archive every non-final step once a newer checkpoint is sealed.
FINAL_STEP = 48
ARCHIVABLE_STEPS = tuple(range(FINAL_STEP))
HF_METADATA = {
    "config.json", "generation_config.json", "tokenizer.json",
    "tokenizer_config.json", "chat_template.jinja",
}
EXPORT_AUXILIARY = {"audit_export_manifest.json"}
PARAMETER_FILES = {"actor/model_world_size_1_rank_0.pt", "actor/fsdp_config.json"} | {
    f"actor/huggingface/{name}" for name in HF_METADATA
}
RESUME_ONLY_FILES = {
    "actor/optim_world_size_1_rank_0.pt", "actor/extra_state_world_size_1_rank_0.pt", "data.pt"
}


def check_source(run: Path, step: int, *, publish_latest: bool = False) -> Path:
    root = run / "verl-run" / "checkpoints"
    if root.is_symlink() or not root.is_dir():
        raise ValueError("checkpoint root must be an existing canonical directory")
    latest = int((root / "latest_checkpointed_iteration.txt").read_text().strip())
    if publish_latest:
        commit = read_json(run / "verl-run/latest_commit.json")
        if not (
            step == latest == 48
            and commit.get("optimizer_update_index") == step
            and commit.get("checkpoint_saved") is True
            and commit.get("checkpoint") == str(root / f"global_step_{step}")
        ):
            raise ValueError("upload-only authorization requires the saved final checkpoint 48")
    elif step not in ARCHIVABLE_STEPS or step >= latest:
        raise ValueError("only explicitly excluded, historical non-resume checkpoints may be archived")
    source = root / f"global_step_{step}"
    if source.is_symlink() or not source.is_dir():
        raise ValueError("source must be an existing canonical checkpoint directory")
    expected = PARAMETER_FILES | (RESUME_ONLY_FILES if publish_latest else set())
    files = set()
    for path in source.rglob("*"):
        if path.is_symlink():
            raise ValueError("checkpoint symlinks are forbidden")
        if path.is_file():
            files.add(path.relative_to(source).as_posix())
    if files != expected:
        raise ValueError(f"checkpoint file allowlist mismatch: {files ^ expected}")
    return source


def original_records(source: Path, *, parameters_only: bool = False) -> tuple[list[dict], str]:
    records = []
    digest = hashlib.sha256()
    for path in sorted(p for p in source.rglob("*") if p.is_file()):
        name = path.relative_to(source).as_posix()
        if parameters_only and name not in PARAMETER_FILES:
            continue
        checksum = sha256_file(path)
        records.append({"path": name, "remote_path": f"original_checkpoint/{name}",
                        "bytes": path.stat().st_size, "sha256": checksum})
        digest.update(path.relative_to(source / "actor").as_posix().encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(checksum))
    return records, digest.hexdigest()


def export_records(export_root: Path, run: Path, step: int, source: Path) -> tuple[Path, list[dict], dict]:
    """Validate and inventory an already-produced audit BF16 export."""

    export_root = export_root.resolve()
    exported = export_root / f"global_step_{step}"
    if export_root.is_symlink() or exported.is_symlink() or not exported.is_dir():
        raise ValueError("export must be an existing non-symlinked audit export directory")
    files: set[str] = set()
    for path in exported.rglob("*"):
        if path.is_symlink():
            raise ValueError("export symlinks are forbidden")
        if path.is_file():
            relative = path.relative_to(exported).as_posix()
            if "/" in relative:
                raise ValueError("audit export files must be at the export root")
            files.add(relative)
    model_files = {name for name in files if name.endswith(".safetensors")}
    if not HF_METADATA <= files or not model_files:
        raise ValueError("audit export is missing model/config/tokenizer files")
    allowed = HF_METADATA | model_files | EXPORT_AUXILIARY
    if "model.safetensors" not in model_files:
        allowed.add("model.safetensors.index.json")
        if "model.safetensors.index.json" not in files:
            raise ValueError("sharded audit export is missing model.safetensors.index.json")
    if files != allowed:
        raise ValueError(f"audit export file allowlist mismatch: {files ^ allowed}")

    manifest_path = exported / "audit_export_manifest.json"
    manifest = read_json(manifest_path)
    actor_model = source / "actor/model_world_size_1_rank_0.pt"
    if (
        manifest.get("schema_version") != 1
        or manifest.get("artifact_kind") != "phase1_policy_checkpoint_export"
        or manifest.get("checkpoint_step") != step
        or manifest.get("checkpoint_id") != f"global_step_{step}"
        or manifest.get("run_id") != run.name
        or manifest.get("source_model_sha256") != sha256_file(actor_model)
        or manifest.get("source_model_bytes") != actor_model.stat().st_size
    ):
        raise ValueError("audit export manifest is not bound to the source checkpoint")
    for record in manifest.get("artifacts", []):
        artifact = Path(str(record.get("path", ""))).resolve()
        if artifact.parent != exported or not artifact.is_file():
            raise ValueError("audit export manifest references an unsafe artifact")
        if artifact.stat().st_size != record.get("bytes") or sha256_file(artifact) != record.get("sha256"):
            raise ValueError("audit export manifest artifact checksum mismatch")

    records = []
    for name in sorted(files - EXPORT_AUXILIARY):
        path = exported / name
        records.append({
            "path": name,
            "remote_path": name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    binding = {
        "path": str(exported),
        "manifest": {
            "path": str(manifest_path),
            "bytes": manifest_path.stat().st_size,
            "sha256": sha256_file(manifest_path),
        },
        "config_sha256": manifest.get("config_sha256"),
        "launch_spec_sha256": manifest.get("launch_spec_sha256"),
    }
    return exported, records, binding


def initial_policy_provenance(export_root: Path, run: Path, source: Path, binding: dict) -> dict:
    provenance_path = export_root.resolve().parent / "responses/checkpoint-000000/provenance.json"
    if provenance_path.is_symlink() or not provenance_path.is_file():
        raise ValueError("step 0 requires fixed-probe initial-policy provenance")
    provenance = read_json(provenance_path)
    actor_model = source / "actor/model_world_size_1_rank_0.pt"
    if (
        provenance.get("schema_version") != 1
        or provenance.get("run_id") != run.name
        or provenance.get("global_step") != 0
        or provenance.get("checkpoint_hash") != sha256_file(actor_model)
        or provenance.get("config_sha256") != binding["config_sha256"]
        or provenance.get("launch_spec_sha256") != binding["launch_spec_sha256"]
    ):
        raise ValueError("initial-policy provenance does not match the step 0 export")
    return {
        "path": str(provenance_path.resolve()),
        "bytes": provenance_path.stat().st_size,
        "sha256": sha256_file(provenance_path),
    }


def stage_link(source: Path, target: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"invalid source file: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.is_symlink() or not os.path.samefile(source, target):
            raise ValueError(f"staging target differs from source: {target}")
    else:
        os.link(source, target)


def verify_remote_records(api, repo_id: str, revision: str, records: list[dict]) -> None:
    from huggingface_hub import hf_hub_download

    info = api.repo_info(repo_id, repo_type="model", revision=revision, files_metadata=True)
    if info.private or info.sha != revision:
        raise ValueError("archive is not public or pinned to the expected commit")
    remote = {item.rfilename: item for item in info.siblings}
    for record in records:
        name = record["remote_path"]
        item = remote.get(name)
        if item is None or item.size != record["bytes"]:
            raise ValueError(f"remote archive size mismatch: {name}")
        expected = record["sha256"]
        if item.lfs is not None:
            actual = item.lfs.sha256 if hasattr(item.lfs, "sha256") else item.lfs["sha256"]
        else:
            cached = hf_hub_download(repo_id, name, revision=revision, repo_type="model")
            actual = sha256_file(Path(cached))
        if actual != expected:
            raise ValueError(f"remote archive checksum mismatch: {name}")


def verify_remote(api, repo_id: str, revision: str, stage: Path, paths: list[str]) -> None:
    records = []
    for name in paths:
        local = stage / name
        records.append({
            "remote_path": name,
            "bytes": local.stat().st_size,
            "sha256": sha256_file(local),
        })
    verify_remote_records(api, repo_id, revision, records)


def archive(
    run: Path,
    step: int,
    *,
    export_root: Path,
    upload: bool,
    workers: int = 1,
    publish_latest: bool = False,
) -> dict:
    from huggingface_hub import HfApi, hf_hub_download
    from dynamic_rubric.phase1.audit_policy import load_run_contract

    run = run.resolve()
    source = check_source(run, step, publish_latest=publish_latest)
    task_root = run / "hf_archive_staging" / f"global_step_{step}"
    stage = task_root / "public"
    stage.mkdir(parents=True, exist_ok=True)
    repo_id = f"HYU-NLP-EVAL/qwen3-4b-rar-medicine-onlinerubrics-seed11-step-{step:03d}"
    print(json.dumps({"state": "hashing_original", "step": step}), flush=True)
    records, actor_hash = original_records(source, parameters_only=publish_latest)
    exported, public_export_files, export_binding = export_records(export_root, run, step, source)
    provenance_record = None
    if step == 0:
        provenance_record = initial_policy_provenance(export_root, run, source, export_binding)
    else:
        commit = read_json(run / "verl-run" / "online_steps" / f"step-{step:06d}" / "commit.json")
        if commit["artifacts"]["actor_parameter_hash"] != actor_hash:
            raise ValueError("source checkpoint does not match the committed training lineage")
    contract = load_run_contract(run)
    paths = []
    for record in records:
        stage_link(source / record["path"], stage / record["remote_path"])
        paths.append(record["remote_path"])
    for record in public_export_files:
        stage_link(exported / record["path"], stage / record["remote_path"])
        paths.append(record["remote_path"])
    license_path = Path(hf_hub_download(contract.model, "LICENSE", revision=contract.model_revision)).resolve()
    # HF's cache can be on a different filesystem; this small public text file
    # is copied while checkpoint tensors remain hardlinked on the data mount.
    shutil.copyfile(license_path, stage / "LICENSE")
    checkpoint_description = (
        "Final policy after 48 optimizer updates (3 epochs) of dynamic OnlineRubrics-Every GRPO."
        if publish_latest else "Intermediate policy from dynamic OnlineRubrics-Every GRPO training."
    )
    readme = f"""---
license: apache-2.0
base_model: {contract.model}
library_name: transformers
pipeline_tag: text-generation
tags:
- onlinerubrics
- grpo
- rar-medicine
- research
---
# OnlineRubrics RaR-Medicine: step {step}, seed 11

{checkpoint_description}
Distinct from static-rubric GRPO. Base model: {contract.model}; thinking disabled.
This checkpoint is a policy state used by the Phase-1 audit.
No downstream medical capability or safety claim is made. Research use only;
not validated for clinical decision-making.

Root files are the veRL-exported Hugging Face inference model (BF16).
`original_checkpoint/` preserves the exact original FSDP parameter checkpoint
and tokenizer/configuration files. Optimizer state, training data, responses,
rubrics, infrastructure configuration, and credentials are not included.
The original is retained because export precision/serialization differs.

Base model revision: {contract.model_revision}
Original actor tree SHA256: {actor_hash}
"""
    (stage / "README.md").write_text(readme, encoding="utf-8")
    public_manifest = {
        "schema_version": 1,
        "checkpoint_step": step,
        "method": "online_rubrics",
        "domain": "medicine",
        "seed": 11,
        "actor_parameter_hash": actor_hash,
        "files": records,
        "public_export_files": public_export_files,
    }
    write_json_atomic(stage / "archive_manifest.json", public_manifest, immutable=False)
    paths.extend(["LICENSE", "README.md", "archive_manifest.json"])
    staged_files = {
        path.relative_to(stage).as_posix()
        for path in stage.rglob("*")
        if path.is_file()
        and path.relative_to(stage).parts[:2] != (".cache", "huggingface")
    }
    if staged_files != set(paths):
        raise ValueError(f"staging file allowlist mismatch: {staged_files ^ set(paths)}")
    result = {"state": "prepared", "step": step, "repo_id": repo_id,
              "stage": str(stage), "file_count": len(paths),
              "original_bytes": sum(row["bytes"] for row in records),
              "public_export_bytes": sum(row["bytes"] for row in public_export_files)}
    print(json.dumps(result), flush=True)
    if not upload:
        return result
    api = HfApi()
    api.create_repo(repo_id, repo_type="model", private=False, exist_ok=True)
    if api.repo_info(repo_id, repo_type="model").private:
        raise ValueError("existing repository is private; refuse implicit visibility change")
    receipt_path = run / "verl-run" / "checkpoint_archives" / f"global_step_{step}.json"
    previous_receipt = read_json(receipt_path) if receipt_path.is_file() else None
    revision = None
    if previous_receipt and previous_receipt.get("repo_id") == repo_id:
        previous_revision = previous_receipt.get("revision")
        try:
            verify_remote(api, repo_id, previous_revision, stage, paths)
            revision = previous_revision
            print(json.dumps({"state": "reused_verified_revision", "step": step,
                              "revision": revision}), flush=True)
        except Exception:
            revision = None
    if revision is None:
        hf = shutil.which("hf")
        if hf is None:
            raise ValueError("hf CLI is required for resumable large-folder upload")
        from scripts.phase1.hf_upload_watchdog import run_upload

        upload_env = os.environ.copy()
        # Avoid the observed hf_xet timeout/stall; retain HF's resumable CLI cache.
        upload_env.setdefault("HF_HUB_DISABLE_XET", "1")
        print(json.dumps({"state": "upload_transport_selected", "step": step,
                          "xet_disabled": upload_env["HF_HUB_DISABLE_XET"],
                          "idle_timeout_seconds": float(upload_env.get(
                              "HF_ARCHIVE_UPLOAD_IDLE_TIMEOUT_SECONDS", "300"))}), flush=True)
        run_upload(
            [hf, "upload-large-folder", repo_id, str(stage), "--repo-type", "model",
             "--num-workers", str(workers)],
            env=upload_env,
            idle_timeout=float(upload_env.get("HF_ARCHIVE_UPLOAD_IDLE_TIMEOUT_SECONDS", "300")),
            max_retries=int(upload_env.get("HF_ARCHIVE_UPLOAD_MAX_RETRIES", "3")),
        )
        revision = api.repo_info(repo_id, repo_type="model").sha
    print(json.dumps({"state": "verifying_remote", "step": step, "revision": revision}), flush=True)
    verify_remote(api, repo_id, revision, stage, paths)
    check_source(run, step, publish_latest=publish_latest)
    receipt = {"schema_version": 1, "state": "verified", "checkpoint_step": step,
               "run_id": run.name, "repo_id": repo_id, "revision": revision,
               "actor_parameter_hash": actor_hash, "files": records,
               "public_export_files": public_export_files,
               "export_directory_binding": export_binding,
               "audit_export_manifest": read_json(exported / "audit_export_manifest.json"),
               "verified_at": datetime.now(timezone.utc).isoformat(),
               "verified_remote_file_count": len(paths)}
    if publish_latest:
        receipt["local_resume_checkpoint_retained"] = True
        receipt["upload_only"] = True
    if provenance_record is not None:
        receipt["initial_policy_provenance"] = provenance_record
    if previous_receipt and previous_receipt.get("revision") != revision:
        receipt["supersedes_revision"] = previous_receipt.get("revision")
    write_json_atomic(receipt_path, receipt, immutable=False)
    print(json.dumps({"state": "verified", "step": step, "receipt": str(receipt_path)}), flush=True)
    return receipt


def cleanup_verified(run: Path, step: int, *, protected_resume_step: int = 45) -> dict:
    """Delete only an unchanged, verified archive's explicit historical paths."""
    from dynamic_rubric.phase1.full_run import latest_full_checkpoint
    from dynamic_rubric.training.checkpoint_archive import verify_public_archive

    run = run.resolve()
    receipt_path = run / "verl-run/checkpoint_archives" / f"global_step_{step}.json"
    receipt = read_json(receipt_path)
    if receipt.get("run_id") != run.name or receipt.get("checkpoint_step") != step:
        raise ValueError("archive receipt is not bound to this run and step")
    source_path = run / "verl-run/checkpoints" / f"global_step_{step}"
    if not source_path.exists() and (
        receipt.get("local_deleted_at") or receipt.get("source_deletion_started_at")
    ):
        if not receipt.get("local_deleted_at"):
            receipt["local_deleted_at"] = datetime.now(timezone.utc).isoformat()
            write_json_atomic(receipt_path, receipt, immutable=False)
        return receipt
    source = check_source(run, step)
    protected = latest_full_checkpoint(run)
    if protected == source.resolve():
        raise ValueError("refusing to delete the protected full resume checkpoint")
    if protected.name != f"global_step_{protected_resume_step}" or step >= protected_resume_step:
        raise ValueError("protected resume checkpoint does not match cleanup authorization")
    remote_hash = verify_public_archive(receipt)
    records, local_hash = original_records(source)
    export_binding = receipt.get("export_directory_binding")
    if not isinstance(export_binding, dict) or not isinstance(export_binding.get("path"), str):
        raise ValueError("archive receipt has no bound audit export")
    exported = Path(export_binding["path"])
    export_exists = exported.is_dir() and not exported.is_symlink()
    if export_exists:
        exported, public_export_files, current_export_binding = export_records(
            exported.parent,
            run,
            step,
            source,
        )
        if (
            public_export_files != receipt.get("public_export_files")
            or current_export_binding != export_binding
            or read_json(exported / "audit_export_manifest.json")
            != receipt.get("audit_export_manifest")
        ):
            raise ValueError("archive no longer matches the bound audit export")
    elif receipt.get("export_deletion_started_at"):
        public_export_files = receipt.get("public_export_files")
        if not isinstance(public_export_files, list) or not public_export_files:
            raise ValueError("archive receipt has no public export inventory")
    else:
        raise ValueError("bound audit export disappeared before authorized cleanup")
    from huggingface_hub import HfApi

    verify_remote_records(
        HfApi(),
        receipt["repo_id"],
        receipt["revision"],
        public_export_files,
    )
    if step == 0:
        provenance = initial_policy_provenance(exported.parent, run, source, export_binding)
        lineage_matches = provenance == receipt.get("initial_policy_provenance")
    else:
        commit = read_json(run / "verl-run/online_steps" / f"step-{step:06d}/commit.json")
        lineage_matches = local_hash == commit["artifacts"]["actor_parameter_hash"]
    if (
        remote_hash != local_hash
        or not lineage_matches
        or records != receipt["files"]
    ):
        raise ValueError("archive no longer matches the original committed checkpoint")
    # Recheck the retention boundary immediately before changing local state.
    check_source(run, step)
    protected = latest_full_checkpoint(run)
    if protected == source.resolve() or protected.name != f"global_step_{protected_resume_step}":
        raise ValueError("protected resume checkpoint changed before cleanup")
    if export_exists:
        for record in public_export_files:
            path = exported / record["path"]
            if path.is_symlink() or path.stat().st_size != record["bytes"]:
                raise ValueError("audit export changed immediately before cleanup")
    staging = run / "hf_archive_staging" / f"global_step_{step}"
    if staging.is_symlink():
        raise ValueError("refusing to clean a symlinked staging directory")
    receipt["deletion_authorized_at"] = datetime.now(timezone.utc).isoformat()
    receipt["protected_resume_checkpoint"] = protected.name
    write_json_atomic(receipt_path, receipt, immutable=False)
    # Remove only this task's staging (including temporary hardlinks/exports),
    # then this audit's bound export and the canonical historical checkpoint.
    if staging.exists():
        receipt["staging_deletion_started_at"] = datetime.now(timezone.utc).isoformat()
        write_json_atomic(receipt_path, receipt, immutable=False)
        shutil.rmtree(staging)
    if export_exists:
        receipt["export_deletion_started_at"] = datetime.now(timezone.utc).isoformat()
        write_json_atomic(receipt_path, receipt, immutable=False)
        shutil.rmtree(exported)
        receipt["export_deleted_at"] = datetime.now(timezone.utc).isoformat()
        write_json_atomic(receipt_path, receipt, immutable=False)
    receipt["source_deletion_started_at"] = datetime.now(timezone.utc).isoformat()
    write_json_atomic(receipt_path, receipt, immutable=False)
    shutil.rmtree(source)
    receipt["local_deleted_at"] = datetime.now(timezone.utc).isoformat()
    receipt["original_bytes_removed"] = sum(row["bytes"] for row in records)
    write_json_atomic(receipt_path, receipt, immutable=False)
    print(json.dumps({"state": "local_deleted_verified_remote", "step": step,
                      "repo_id": receipt["repo_id"], "revision": receipt["revision"],
                      "original_bytes_removed": receipt["original_bytes_removed"]}), flush=True)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--step", required=True, type=int, choices=ARCHIVABLE_STEPS + (48,))
    parser.add_argument("--export-root", required=True, type=Path)
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--cleanup-verified-only", action="store_true")
    parser.add_argument("--protected-resume-step", type=int, default=45)
    parser.add_argument(
        "--publish-latest",
        action="store_true",
        help="Publish saved final step48 parameters only; never authorize cleanup",
    )
    args = parser.parse_args()
    if args.publish_latest and args.cleanup_verified_only:
        parser.error("latest checkpoint publication cannot be combined with cleanup")
    if args.cleanup_verified_only:
        if args.upload:
            parser.error("cleanup and upload must be separate verified phases")
        cleanup_verified(args.run, args.step, protected_resume_step=args.protected_resume_step)
    else:
        archive(
            args.run,
            args.step,
            export_root=args.export_root,
            upload=args.upload,
            workers=args.workers,
            publish_latest=args.publish_latest,
        )


if __name__ == "__main__":
    main()
