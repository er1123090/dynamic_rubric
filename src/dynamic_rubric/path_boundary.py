"""Resolved filesystem boundaries for public and private experiment stages."""

from __future__ import annotations

from pathlib import Path


class PathBoundaryError(PermissionError):
    pass


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def resolve_project_path(
    project_root: Path,
    path: Path,
    *,
    allow_private: bool = False,
) -> Path:
    """Resolve symlinks and reject escapes or private mounts in public stages."""

    root = project_root.resolve()
    candidate = path if path.is_absolute() else root / path
    resolved = candidate.resolve()
    if not _within(resolved, root):
        raise PathBoundaryError(f"path escapes the project root: {path}")
    relative_parts = tuple(part.casefold() for part in resolved.relative_to(root).parts)
    if not allow_private and "private_gt" in relative_parts:
        raise PathBoundaryError(f"private-GT path is forbidden in a public stage: {path}")
    return resolved
