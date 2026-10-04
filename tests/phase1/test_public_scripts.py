from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess

import pytest

from scripts.phase1 import launch_training, serve_training


ROOT = Path(__file__).resolve().parents[2]
PRIVATE_ABSOLUTE_PATH = re.compile(r"(?<!})/(?:home|data)/[A-Za-z0-9._-]+/")


def _tracked_scripts() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "scripts"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [ROOT / name for name in result.stdout.splitlines() if (ROOT / name).is_file()]


def test_tracked_scripts_do_not_publish_personal_absolute_paths() -> None:
    findings: list[str] = []
    for path in _tracked_scripts():
        text = path.read_text(encoding="utf-8", errors="replace")
        for match in PRIVATE_ABSOLUTE_PATH.finditer(text):
            findings.append(f"{path.relative_to(ROOT)}:{match.group(0)}")
    assert findings == []


@pytest.mark.parametrize("value", [None, "", "   "])
def test_launch_path_rejects_blank_configuration(value: object) -> None:
    with pytest.raises(launch_training.LaunchError, match="non-empty path"):
        launch_training._repo_path(value, label="models.policy.local_snapshot")


@pytest.mark.parametrize("value", [None, "", "   "])
def test_service_path_rejects_blank_configuration(value: object) -> None:
    with pytest.raises(ValueError, match="non-empty path"):
        serve_training._path(value, ROOT, label="models.judge.local_snapshot")


@pytest.mark.parametrize(
    ("script", "required_name"),
    [
        (
            "scripts/phase1/serve_gpt_oss_on_inference_a.sh",
            "PHASE1_INFERENCE_A_SSH_TARGET",
        ),
        (
            "scripts/phase1/serve_qwen32b_on_inference_b.sh",
            "PHASE1_INFERENCE_B_SSH_TARGET",
        ),
    ],
)
def test_optional_container_helpers_fail_fast_when_site_config_is_blank(
    script: str, required_name: str
) -> None:
    result = subprocess.run(
        ["bash", str(ROOT / script), "status"],
        cwd=ROOT,
        env={"PATH": os.environ["PATH"]},
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert required_name in result.stderr
