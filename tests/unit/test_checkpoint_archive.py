from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from dynamic_rubric.hashing import sha256_file
from dynamic_rubric.training.checkpoint_archive import (
    CheckpointArchiveError,
    verify_public_archive,
)


_REVISION = "a" * 40
_REPO_ID = "HYU-NLP-EVAL/qwen3-4b-rar-medicine-onlinerubrics-seed11-step-013"


def _receipt(tmp_path: Path) -> tuple[dict, list[SimpleNamespace]]:
    actor = tmp_path / "actor"
    contents = {
        "fsdp_config.json": b"fsdp",
        "huggingface/chat_template.jinja": b"chat",
        "huggingface/config.json": b"config",
        "huggingface/generation_config.json": b"generation",
        "huggingface/tokenizer.json": b"tokenizer",
        "huggingface/tokenizer_config.json": b"tokenizer-config",
        "model_world_size_1_rank_0.pt": b"weights",
    }
    files = []
    siblings = []
    digest = hashlib.sha256()
    for relative, content in sorted(contents.items()):
        local = actor / relative
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(content)
        sha = sha256_file(local)
        path = f"actor/{relative}"
        remote_path = f"original_checkpoint/{path}"
        files.append(
            {"path": path, "remote_path": remote_path, "bytes": len(content), "sha256": sha}
        )
        siblings.append(
            SimpleNamespace(rfilename=remote_path, size=len(content), lfs={"sha256": sha})
        )
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha))
    actor_hash = digest.hexdigest()
    return (
        {
            "schema_version": 1,
            "state": "verified",
            "checkpoint_step": 13,
            "run_id": "phase1-run",
            "repo_id": _REPO_ID,
            "revision": _REVISION,
            "actor_parameter_hash": actor_hash,
            "files": files,
            "verified_at": "2026-09-08T00:00:00+00:00",
        },
        siblings,
    )


class _Api:
    def __init__(self, siblings: list[SimpleNamespace], *, revision: str = _REVISION) -> None:
        self.siblings = siblings
        self.revision = revision

    def repo_info(self, **kwargs: object) -> SimpleNamespace:
        assert kwargs == {
            "repo_id": _REPO_ID,
            "repo_type": "model",
            "revision": _REVISION,
            "files_metadata": True,
        }
        return SimpleNamespace(
            sha=self.revision,
            private=False,
            siblings=self.siblings,
        )


def test_verify_public_archive_reconstructs_actor_parameter_hash(tmp_path: Path) -> None:
    receipt, siblings = _receipt(tmp_path)

    assert verify_public_archive(receipt, api=_Api(siblings)) == receipt["actor_parameter_hash"]


def test_initial_checkpoint_archive_integrity_is_supported(tmp_path: Path) -> None:
    receipt, siblings = _receipt(tmp_path)
    receipt['checkpoint_step'] = 0
    assert verify_public_archive(receipt, api=_Api(siblings)) == receipt['actor_parameter_hash']


@pytest.mark.parametrize("field", ("revision", "repo_id"))
def test_verify_public_archive_rejects_malformed_identity(tmp_path: Path, field: str) -> None:
    receipt, siblings = _receipt(tmp_path)
    receipt[field] = "unsafe"

    with pytest.raises(CheckpointArchiveError, match="identity is malformed"):
        verify_public_archive(receipt, api=_Api(siblings))


def test_verify_public_archive_rejects_changed_remote_sha(tmp_path: Path) -> None:
    receipt, siblings = _receipt(tmp_path)
    siblings[-1].lfs = {"sha256": "0" * 64}

    with pytest.raises(CheckpointArchiveError, match="SHA256 mismatch"):
        verify_public_archive(receipt, api=_Api(siblings))


def test_verify_public_archive_rejects_changed_resolved_revision(tmp_path: Path) -> None:
    receipt, siblings = _receipt(tmp_path)

    with pytest.raises(CheckpointArchiveError, match="did not resolve exactly"):
        verify_public_archive(receipt, api=_Api(siblings, revision="b" * 40))


def test_verify_public_archive_requires_complete_actor_file_set(tmp_path: Path) -> None:
    receipt, siblings = _receipt(tmp_path)
    receipt["files"].pop()

    with pytest.raises(CheckpointArchiveError, match="incomplete"):
        verify_public_archive(receipt, api=_Api(siblings))
