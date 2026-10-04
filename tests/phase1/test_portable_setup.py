from __future__ import annotations

import json
import stat
import subprocess
import sys
import zipfile
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
PRECOMPUTE = ROOT / "scripts/phase1/precompute_pi0.py"
PREPARE_SOURCE = ROOT / "scripts/phase1/prepare_evorubrics_source.py"


def _run(*args: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *(str(value) for value in args)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )


def test_precompute_check_renders_yaml_resource_plan_without_gpu(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    train = tmp_path / "train.jsonl"
    train.write_text('{"id": 1}\n{"id": 2}\n', encoding="utf-8")
    runtime = tmp_path / "python"
    runtime.touch()
    runtime.chmod(0o755)
    vllm = tmp_path / "vllm"
    vllm.touch()
    vllm.chmod(0o755)
    output = tmp_path / "cache"
    config = tmp_path / "online.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "domain": "science",
                "method": "online_rubrics",
                "seed": 17,
                "data": {"train_path": str(train), "train_prompt_count": 2},
                "models": {
                    "policy": {
                        "model": "example/policy",
                        "revision": "abc123",
                        "local_snapshot": str(model),
                    }
                },
                "infrastructure": {
                    "pi0_control": {
                        "gpus": [2, 3],
                        "tensor_parallel_size": 2,
                        "gpu_memory_utilization": 0.7,
                        "max_model_len": 4096,
                        "max_num_seqs": 8,
                        "max_num_batched_tokens": 8192,
                        "upstream_port": 19001,
                        "proxy_port": 19002,
                        "concurrency": 5,
                        "progress_every": 2,
                        "free_memory_threshold_mib": 512,
                        "vllm_bin": str(vllm),
                        "runtime_python": str(runtime),
                    }
                },
                "launch": {"environment": {"ONLINE_CONTROL_CACHE_DIR": str(output)}},
            }
        ),
        encoding="utf-8",
    )

    result = _run(PRECOMPUTE, "--config", config, "--check")

    assert result.returncode == 0, result.stderr
    rendered = json.loads(result.stdout)
    assert rendered["gpus"] == [2, 3]
    assert rendered["tensor_parallel_size"] == 2
    assert rendered["output_dir"] == str(output)
    assert rendered["prompt_count"] == 2
    assert "nvidia-smi" not in result.stderr

    document = yaml.safe_load(config.read_text(encoding="utf-8"))
    document["infrastructure"]["pi0_control"]["gpus"] = [1.5, 2]
    config.write_text(yaml.safe_dump(document), encoding="utf-8")
    invalid_gpu = _run(PRECOMPUTE, "--config", config, "--check")
    assert invalid_gpu.returncode == 2
    assert "integer indices" in invalid_gpu.stderr

    document["infrastructure"]["pi0_control"]["gpus"] = [2, 3]
    document["infrastructure"]["pi0_control"]["proxy_port"] = 19001
    config.write_text(yaml.safe_dump(document), encoding="utf-8")
    invalid_ports = _run(PRECOMPUTE, "--config", config, "--check")
    assert invalid_ports.returncode == 2
    assert "ports must be distinct" in invalid_ports.stderr


def test_source_check_does_not_write_and_extract_preserves_existing_destination(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("evorubric-main/main.py", "print('ok')\n")
        output.writestr("README.md", "upstream\n")
    destination = tmp_path / "EvoRubrics"

    checked = _run(
        PREPARE_SOURCE,
        "--check",
        "--no-patch",
        "--archive",
        archive,
        "--destination",
        destination,
    )
    assert checked.returncode == 0, checked.stderr
    assert not destination.exists()

    extracted = _run(
        PREPARE_SOURCE,
        "--extract",
        "--no-patch",
        "--archive",
        archive,
        "--destination",
        destination,
    )
    assert extracted.returncode == 0, extracted.stderr
    marker = destination / "local-change.txt"
    marker.write_text("keep", encoding="utf-8")

    repeated = _run(
        PREPARE_SOURCE,
        "--extract",
        "--no-patch",
        "--archive",
        archive,
        "--destination",
        destination,
    )
    assert repeated.returncode == 2
    assert "left unchanged" in repeated.stderr
    assert marker.read_text(encoding="utf-8") == "keep"


def test_source_extraction_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "traversal.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("../escaped.txt", "unsafe")
    destination = tmp_path / "EvoRubrics"

    result = _run(
        PREPARE_SOURCE,
        "--extract",
        "--no-patch",
        "--archive",
        archive,
        "--destination",
        destination,
    )

    assert result.returncode == 2
    assert "unsafe archive member path" in result.stderr
    assert not (tmp_path / "escaped.txt").exists()
    assert not destination.exists()


def test_source_extraction_rejects_symlinks(tmp_path: Path) -> None:
    archive = tmp_path / "symlink.zip"
    link = zipfile.ZipInfo("link")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr(link, "target")

    result = _run(
        PREPARE_SOURCE,
        "--check",
        "--no-patch",
        "--archive",
        archive,
        "--destination",
        tmp_path / "destination",
    )

    assert result.returncode == 2
    assert "not a regular file/directory" in result.stderr
