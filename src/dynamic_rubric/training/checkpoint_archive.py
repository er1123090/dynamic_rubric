"""Verification for immutable public archives of historical actor checkpoints."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from dynamic_rubric.hashing import sha256_file


class CheckpointArchiveError(RuntimeError):
    pass


_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_REVISION_RE = re.compile(r"[0-9a-f]{40}")
_REPO_RE = re.compile(r"HYU-NLP-EVAL/[A-Za-z0-9][A-Za-z0-9._-]*")
_MODEL_SHARD_RE = re.compile(r"actor/model_world_size_([1-9][0-9]*)_rank_([0-9]+)\.pt")
_REQUIRED_METADATA = {
    "actor/fsdp_config.json",
    "actor/huggingface/chat_template.jinja",
    "actor/huggingface/config.json",
    "actor/huggingface/generation_config.json",
    "actor/huggingface/tokenizer.json",
    "actor/huggingface/tokenizer_config.json",
}


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _validate_files(files: Any) -> list[dict[str, Any]]:
    if not isinstance(files, list) or not files:
        raise CheckpointArchiveError("checkpoint archive receipt has no files")
    validated: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    seen_remote: set[str] = set()
    shard_world_size: int | None = None
    shard_ranks: set[int] = set()
    for item in files:
        if not isinstance(item, Mapping):
            raise CheckpointArchiveError("checkpoint archive file entry is malformed")
        path = item.get("path")
        remote_path = item.get("remote_path")
        size = item.get("bytes")
        expected_sha = item.get("sha256")
        if not isinstance(path, str) or not isinstance(remote_path, str):
            raise CheckpointArchiveError("checkpoint archive file path is malformed")
        local_parts = PurePosixPath(path)
        remote_parts = PurePosixPath(remote_path)
        if (
            local_parts.is_absolute()
            or remote_parts.is_absolute()
            or ".." in local_parts.parts
            or ".." in remote_parts.parts
            or not path.startswith("actor/")
            or remote_path != f"original_checkpoint/{path}"
        ):
            raise CheckpointArchiveError(f"checkpoint archive file path is unsafe: {path!r}")
        if path in seen_paths or remote_path in seen_remote:
            raise CheckpointArchiveError("checkpoint archive contains duplicate files")
        if (
            not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
            or not isinstance(expected_sha, str)
            or _SHA256_RE.fullmatch(expected_sha) is None
        ):
            raise CheckpointArchiveError(f"checkpoint archive metadata is malformed: {path}")
        if "optim_world_size_" in path or "extra_state_world_size_" in path:
            raise CheckpointArchiveError("checkpoint archive receipt contains optimizer state")
        shard = _MODEL_SHARD_RE.fullmatch(path)
        if shard:
            world_size = int(shard.group(1))
            rank = int(shard.group(2))
            if shard_world_size is None:
                shard_world_size = world_size
            elif shard_world_size != world_size:
                raise CheckpointArchiveError("checkpoint archive mixes FSDP world sizes")
            if rank >= world_size or rank in shard_ranks:
                raise CheckpointArchiveError("checkpoint archive has invalid FSDP shard ranks")
            shard_ranks.add(rank)
        elif path not in _REQUIRED_METADATA:
            raise CheckpointArchiveError(f"checkpoint archive contains an unexpected actor file: {path}")
        seen_paths.add(path)
        seen_remote.add(remote_path)
        validated.append(
            {"path": path, "remote_path": remote_path, "bytes": size, "sha256": expected_sha}
        )
    if not _REQUIRED_METADATA.issubset(seen_paths) or shard_world_size is None:
        raise CheckpointArchiveError("checkpoint archive actor file set is incomplete")
    expected_ranks = set(range(shard_world_size))
    if shard_ranks != expected_ranks or len(seen_paths) != len(_REQUIRED_METADATA) + shard_world_size:
        raise CheckpointArchiveError("checkpoint archive FSDP shard set is incomplete")
    return validated


def _lfs_sha256(sibling: Any) -> str | None:
    lfs = _field(sibling, "lfs")
    if lfs is None:
        return None
    sha = _field(lfs, "sha256")
    return sha if isinstance(sha, str) and _SHA256_RE.fullmatch(sha) else None


def verify_public_archive(receipt: Mapping[str, Any], api: Any = None) -> str:
    """Verify an archived actor at a pinned public HF model commit and return its tree hash."""

    if not isinstance(receipt, Mapping):
        raise CheckpointArchiveError("checkpoint archive receipt is malformed")
    repo_id = receipt.get("repo_id")
    revision = receipt.get("revision")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("state") != "verified"
        or not isinstance(receipt.get("checkpoint_step"), int)
        or isinstance(receipt.get("checkpoint_step"), bool)
        or int(receipt["checkpoint_step"]) < 0
        or not isinstance(receipt.get("run_id"), str)
        or not receipt.get("run_id")
        or not isinstance(repo_id, str)
        or _REPO_RE.fullmatch(repo_id) is None
        or not isinstance(revision, str)
        or _REVISION_RE.fullmatch(revision) is None
        or not isinstance(receipt.get("actor_parameter_hash"), str)
        or _SHA256_RE.fullmatch(str(receipt["actor_parameter_hash"])) is None
        or not isinstance(receipt.get("verified_at"), str)
        or not receipt.get("verified_at")
    ):
        raise CheckpointArchiveError("checkpoint archive receipt identity is malformed")
    files = _validate_files(receipt.get("files"))

    if api is None:
        try:
            from huggingface_hub import HfApi
        except ImportError as error:
            raise CheckpointArchiveError("huggingface_hub is required to verify an archive") from error
        api = HfApi()
    try:
        info = api.repo_info(
            repo_id=repo_id,
            repo_type="model",
            revision=revision,
            files_metadata=True,
        )
    except Exception as error:
        raise CheckpointArchiveError("failed to inspect the pinned checkpoint archive") from error
    resolved_sha = _field(info, "sha")
    if resolved_sha != revision:
        raise CheckpointArchiveError("checkpoint archive revision did not resolve exactly")
    if _field(info, "private", False) is True:
        raise CheckpointArchiveError("checkpoint archive repository is not public")
    siblings = {_field(item, "rfilename"): item for item in (_field(info, "siblings", []) or [])}

    digest = hashlib.sha256()
    for item in sorted(files, key=lambda value: value["path"]):
        sibling = siblings.get(item["remote_path"])
        if sibling is None or _field(sibling, "size") != item["bytes"]:
            raise CheckpointArchiveError(f"checkpoint archive remote file mismatch: {item['path']}")
        actual_sha = _lfs_sha256(sibling)
        if actual_sha is None:
            try:
                if hasattr(api, "hf_hub_download"):
                    downloaded = api.hf_hub_download(
                        repo_id=repo_id,
                        filename=item["remote_path"],
                        repo_type="model",
                        revision=revision,
                    )
                else:
                    from huggingface_hub import hf_hub_download

                    downloaded = hf_hub_download(
                        repo_id=repo_id,
                        filename=item["remote_path"],
                        repo_type="model",
                        revision=revision,
                    )
                actual_sha = sha256_file(Path(downloaded))
            except Exception as error:
                raise CheckpointArchiveError(
                    f"failed to verify checkpoint archive file: {item['path']}"
                ) from error
        if actual_sha != item["sha256"]:
            raise CheckpointArchiveError(f"checkpoint archive SHA256 mismatch: {item['path']}")
        digest.update(item["path"].removeprefix("actor/").encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(actual_sha))
    actor_hash = digest.hexdigest()
    if actor_hash != receipt["actor_parameter_hash"]:
        raise CheckpointArchiveError("checkpoint archive actor parameter hash mismatch")
    return actor_hash
