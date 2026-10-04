from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PREPARE = ROOT / "scripts/phase1/prepare_verl_source.py"
SETUP = ROOT / "scripts/phase1/setup_verl_runtime.sh"


def _command(*args: object, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(value) for value in args], cwd=ROOT, env=env, capture_output=True, text=True
    )


def _git(path: Path, *args: str) -> str:
    result = _command("git", "-C", path, *args)
    assert result.returncode == 0, result.stderr
    return result.stdout


def _fixture(tmp_path: Path) -> tuple[Path, str, Path]:
    repository = tmp_path / "origin"
    repository.mkdir()
    _git(repository, "init")
    _git(repository, "config", "user.name", "Test User")
    _git(repository, "config", "user.email", "test@example.invalid")
    source_file = repository / "verl/trainer.py"
    source_file.parent.mkdir(parents=True)
    source_file.write_text("mode = 'upstream'\n", encoding="utf-8")
    _git(repository, "add", "verl/trainer.py")
    _git(repository, "commit", "-m", "upstream")
    commit = _git(repository, "rev-parse", "HEAD").strip()
    source_file.write_text("mode = 'handoff'\n", encoding="utf-8")
    patch = tmp_path / "handoff.patch"
    patch.write_text(_git(repository, "diff", "--binary", "--no-ext-diff"), encoding="utf-8")
    _git(repository, "restore", "verl/trainer.py")
    return repository, commit, patch


def _prepare(
    repository: Path, commit: str, patch: Path, destination: Path
) -> subprocess.CompletedProcess[str]:
    return _command(
        sys.executable,
        PREPARE,
        "--prepare",
        "--repository",
        repository,
        "--commit",
        commit,
        "--patch",
        patch,
        "--destination",
        destination,
    )


def test_prepare_verl_clones_pinned_commit_and_applies_exact_patch(tmp_path: Path) -> None:
    repository, commit, patch = _fixture(tmp_path)
    destination = tmp_path / "prepared"

    prepared = _prepare(repository, commit, patch, destination)

    assert prepared.returncode == 0, prepared.stderr
    assert _git(destination, "rev-parse", "HEAD").strip() == commit
    assert (destination / "verl/trainer.py").read_text(encoding="utf-8") == "mode = 'handoff'\n"
    checked = _command(
        sys.executable,
        PREPARE,
        "--check",
        "--commit",
        commit,
        "--patch",
        patch,
        "--destination",
        destination,
    )
    assert checked.returncode == 0, checked.stderr


def test_prepare_verl_refuses_existing_or_drifted_destination(tmp_path: Path) -> None:
    repository, commit, patch = _fixture(tmp_path)
    destination = tmp_path / "prepared"
    assert _prepare(repository, commit, patch, destination).returncode == 0
    marker = destination / "local-note.txt"
    marker.write_text("preserve", encoding="utf-8")

    repeated = _prepare(repository, commit, patch, destination)

    assert repeated.returncode == 2
    assert "left unchanged" in repeated.stderr
    assert marker.read_text(encoding="utf-8") == "preserve"
    tracked = destination / "verl/trainer.py"
    tracked.write_text("mode = 'unexpected'\n", encoding="utf-8")
    checked = _command(
        sys.executable,
        PREPARE,
        "--check",
        "--commit",
        commit,
        "--patch",
        patch,
        "--destination",
        destination,
    )
    assert checked.returncode == 2
    assert "tracked changes differ" in checked.stderr


def test_setup_check_uses_overrides_without_install_or_network(tmp_path: Path) -> None:
    repository, commit, patch = _fixture(tmp_path)
    source = tmp_path / "source"
    assert _prepare(repository, commit, patch, source).returncode == 0
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    uv = fake_bin / "uv"
    uv.write_text("#!/bin/sh\nexit 99\n", encoding="utf-8")
    uv.chmod(0o755)
    nvidia_smi = fake_bin / "nvidia-smi"
    nvidia_smi.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    nvidia_smi.chmod(0o755)
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("example==1\n", encoding="utf-8")
    python310 = shutil.which("python3.10")
    assert python310 is not None
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "VERL_BASE_PYTHON": python310,
            "VERL_UV_BIN": str(uv),
            "VERL_SOURCE_ROOT": str(source),
            "VERL_SOURCE_PATCH": str(patch),
            "VERL_SOURCE_COMMIT": commit,
            "VERL_SOURCE_REPOSITORY": str(repository),
            "VERL_REQUIREMENTS": str(requirements),
            "VERL_VENV": str(tmp_path / "absent-venv"),
        }
    )

    result = _command("bash", SETUP, "--check", env=env)

    assert result.returncode == 0, result.stderr
    assert "runtime is not installed" in result.stdout
    assert not (tmp_path / "absent-venv").exists()
