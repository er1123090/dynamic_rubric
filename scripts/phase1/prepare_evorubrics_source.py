#!/usr/bin/env python3
"""Safely inspect or prepare the patched EvoRubrics source tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARCHIVE = ROOT / "docs/EvoRubrics-2155.zip"
DEFAULT_DESTINATION = ROOT / "environment/upstream/EvoRubrics"
DEFAULT_MANIFEST = ROOT / "environment/source-snapshots/EvoRubrics-2155-rq2-patch-manifest.json"


class SourcePreparationError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_member_path(info: zipfile.ZipInfo) -> PurePosixPath:
    name = info.filename
    path = PurePosixPath(name)
    if not name or "\\" in name or path.is_absolute() or ".." in path.parts:
        raise SourcePreparationError(f"unsafe archive member path: {name!r}")
    mode = info.external_attr >> 16
    kind = stat.S_IFMT(mode)
    if kind not in (0, stat.S_IFREG, stat.S_IFDIR):
        raise SourcePreparationError(f"archive member is not a regular file/directory: {name!r}")
    if info.flag_bits & 0x1:
        raise SourcePreparationError(f"encrypted archive member is unsupported: {name!r}")
    return path


def inspect_archive(archive: Path) -> list[tuple[zipfile.ZipInfo, PurePosixPath]]:
    if not archive.is_file():
        raise SourcePreparationError(f"missing EvoRubrics source archive: {archive}")
    try:
        with zipfile.ZipFile(archive) as source:
            result = [(info, _safe_member_path(info)) for info in source.infolist()]
            bad = source.testzip()
    except zipfile.BadZipFile as error:
        raise SourcePreparationError(f"invalid EvoRubrics ZIP: {archive}") from error
    if bad is not None:
        raise SourcePreparationError(f"corrupt archive member: {bad}")
    if not result:
        raise SourcePreparationError("EvoRubrics ZIP is empty")
    return result


def _load_manifest(
    path: Path | None, archive: Path
) -> tuple[Mapping[str, Any] | None, Path | None]:
    if path is None:
        return None, None
    if not path.is_file():
        raise SourcePreparationError(f"missing patch manifest: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise SourcePreparationError("patch manifest must be a JSON object")
    expected_archive_hash = str(raw.get("source_archive_sha256", ""))
    if expected_archive_hash and _sha256(archive) != expected_archive_hash:
        raise SourcePreparationError("source archive SHA-256 does not match the patch manifest")
    patch_value = raw.get("patch_file")
    if not isinstance(patch_value, str) or not patch_value:
        raise SourcePreparationError("patch manifest has no patch_file")
    patch = path.parent / Path(patch_value).name
    if not patch.is_file():
        raise SourcePreparationError(f"missing recorded patch: {patch}")
    if _sha256(patch) != str(raw.get("patch_sha256", "")):
        raise SourcePreparationError("patch SHA-256 does not match the manifest")
    return raw, patch


def _validate_patched_tree(root: Path, manifest: Mapping[str, Any]) -> None:
    modified = manifest.get("modified_files", [])
    if not isinstance(modified, list):
        raise SourcePreparationError("patch manifest modified_files must be a list")
    for record in modified:
        if not isinstance(record, Mapping):
            raise SourcePreparationError("invalid modified_files record")
        relative = PurePosixPath(str(record.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise SourcePreparationError(f"unsafe manifest path: {relative}")
        target = root.joinpath(*relative.parts)
        if not target.is_file():
            raise SourcePreparationError(f"patched source is missing: {relative}")
        if _sha256(target) != str(record.get("patched_sha256", "")):
            raise SourcePreparationError(f"patched source hash differs: {relative}")


def _extract(
    archive: Path,
    destination: Path,
    members: list[tuple[zipfile.ZipInfo, PurePosixPath]],
    manifest: Mapping[str, Any] | None,
    patch: Path | None,
) -> None:
    if destination.exists():
        raise SourcePreparationError(
            f"destination already exists and was left unchanged: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.prepare-", dir=destination.parent)
    )
    try:
        with zipfile.ZipFile(archive) as source:
            for info, relative in members:
                target = temporary.joinpath(*relative.parts)
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.open(info) as reader, target.open("xb") as writer:
                    shutil.copyfileobj(reader, writer)
                if info.external_attr >> 16 & stat.S_IXUSR:
                    target.chmod(target.stat().st_mode | stat.S_IXUSR)
        if patch is not None:
            result = subprocess.run(
                ["patch", "--batch", "--forward", "-p1", "-i", str(patch)],
                cwd=temporary,
                capture_output=True,
                text=True,
            )
            if result.returncode:
                raise SourcePreparationError(
                    "failed to apply recorded EvoRubrics patch:\n" + result.stdout + result.stderr
                )
        if manifest is not None:
            _validate_patched_tree(temporary, manifest)
        if destination.exists():
            raise SourcePreparationError(
                f"destination appeared during preparation and was left unchanged: {destination}"
            )
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true", help="inspect inputs without writing")
    action.add_argument(
        "--extract", action="store_true", help="extract and apply the recorded patch"
    )
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--patch-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--no-patch",
        action="store_true",
        help="extract a pristine tree (mainly for archive audits)",
    )
    args = parser.parse_args()
    try:
        archive = args.archive.expanduser().resolve()
        destination = args.destination.expanduser().resolve()
        members = inspect_archive(archive)
        manifest_path = None if args.no_patch else args.patch_manifest.expanduser().resolve()
        manifest, patch = _load_manifest(manifest_path, archive)
        if args.check:
            state = "existing source preserved" if destination.exists() else "ready to extract"
            if destination.exists() and manifest is not None:
                _validate_patched_tree(destination, manifest)
            print(
                json.dumps(
                    {
                        "archive": str(archive),
                        "archive_sha256": _sha256(archive),
                        "destination": str(destination),
                        "member_count": len(members),
                        "patch": str(patch) if patch else None,
                        "state": state,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            _extract(archive, destination, members, manifest, patch)
            print(destination)
    except (OSError, json.JSONDecodeError, SourcePreparationError) as error:
        print(f"EvoRubrics source preparation error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
