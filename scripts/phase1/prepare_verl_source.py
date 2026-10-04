#!/usr/bin/env python3
"""Prepare the pinned, locally patched veRL checkout without overwriting a tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPOSITORY = "https://github.com/verl-project/verl.git"
DEFAULT_COMMIT = "890dfc3ebdd5647f7ea9730375414b1e3fb4e9a6"
DEFAULT_DESTINATION = ROOT / "environment/upstream/verl"
DEFAULT_PATCH = ROOT / "patches/verl_training_handoff.patch"


class VerlSourceError(RuntimeError):
    pass


def _git(
    *args: str, cwd: Path | None = None, check: bool = True
) -> subprocess.CompletedProcess[bytes]:
    try:
        result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, check=False)
    except FileNotFoundError as error:
        raise VerlSourceError("git executable is required") from error
    if check and result.returncode:
        detail = (result.stderr or result.stdout).decode(errors="replace").strip()
        raise VerlSourceError(f"git {' '.join(args)} failed: {detail}")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_patch(path: Path) -> bytes:
    if not path.is_file():
        raise VerlSourceError(f"missing veRL handoff patch: {path}")
    content = path.read_bytes()
    if not content.startswith(b"diff --git "):
        raise VerlSourceError(f"invalid or empty veRL handoff patch: {path}")
    return content


def inspect_source(source: Path, commit: str, patch: Path) -> dict[str, object]:
    expected = _require_patch(patch)
    if not (source / ".git").exists():
        raise VerlSourceError(f"missing veRL Git checkout: {source}")
    head = _git("rev-parse", "HEAD", cwd=source).stdout.decode().strip()
    if head != commit:
        raise VerlSourceError(f"veRL HEAD is {head}; expected pinned commit {commit}")
    actual = _git("diff", "--binary", "--no-ext-diff", cwd=source).stdout
    if actual != expected:
        raise VerlSourceError(
            "veRL tracked changes differ from patches/verl_training_handoff.patch; "
            "the existing checkout was left unchanged"
        )
    reverse = _git("apply", "--reverse", "--check", str(patch), cwd=source, check=False)
    if reverse.returncode:
        raise VerlSourceError("the recorded veRL patch cannot be reversed from the checkout")
    untracked = (
        _git("ls-files", "--others", "--exclude-standard", cwd=source).stdout.decode().splitlines()
    )
    return {
        "commit": head,
        "patch": str(patch),
        "patch_sha256": _sha256(patch),
        "source": str(source),
        "state": "pinned commit with expected training patch",
        "untracked_files": untracked,
    }


def prepare_source(repository: str, destination: Path, commit: str, patch: Path) -> None:
    expected = _require_patch(patch)
    if destination.exists():
        raise VerlSourceError(f"destination already exists and was left unchanged: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.prepare-", dir=destination.parent)
    )
    shutil.rmtree(temporary)
    try:
        _git("clone", "--no-checkout", "--", repository, str(temporary))
        _git("checkout", "--detach", commit, cwd=temporary)
        if _git("diff", "--quiet", cwd=temporary, check=False).returncode:
            raise VerlSourceError("fresh veRL checkout is unexpectedly dirty")
        check = _git("apply", "--check", str(patch), cwd=temporary, check=False)
        if check.returncode:
            detail = (check.stderr or check.stdout).decode(errors="replace").strip()
            raise VerlSourceError(f"handoff patch does not apply to pinned commit: {detail}")
        _git("apply", str(patch), cwd=temporary)
        if _git("diff", "--binary", "--no-ext-diff", cwd=temporary).stdout != expected:
            raise VerlSourceError("applied veRL diff does not exactly match the handoff patch")
        if destination.exists():
            raise VerlSourceError(
                f"destination appeared during preparation and was left unchanged: {destination}"
            )
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true", help="verify an existing patched checkout")
    action.add_argument(
        "--prepare", action="store_true", help="clone, pin, and patch an absent target"
    )
    parser.add_argument("--repository", default=DEFAULT_REPOSITORY)
    parser.add_argument("--commit", default=DEFAULT_COMMIT)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--patch", type=Path, default=DEFAULT_PATCH)
    args = parser.parse_args()
    try:
        destination = args.destination.expanduser().resolve()
        patch = args.patch.expanduser().resolve()
        if args.check:
            print(
                json.dumps(
                    inspect_source(destination, args.commit, patch), indent=2, sort_keys=True
                )
            )
        else:
            prepare_source(args.repository, destination, args.commit, patch)
            print(
                json.dumps(
                    inspect_source(destination, args.commit, patch), indent=2, sort_keys=True
                )
            )
    except (OSError, VerlSourceError) as error:
        print(f"veRL source preparation error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
