from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "phase1" / "export_evorubrics_sft_checkpoint.py"
SPEC = importlib.util.spec_from_file_location("export_evorubrics_sft_checkpoint", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _steps() -> list[dict]:
    return [
        {
            "event": "optimizer_step",
            "step": step,
            "examples": 32 if step < 47 else 28,
            "cumulative_exposures": min(step * 32, 1500),
            "token_mean_loss": 2.0 / step,
            "gradient_norm_before_clip": 1.0 / step,
        }
        for step in range(1, 48)
    ]


def test_optimizer_log_requires_contiguous_finite_47_steps_and_1500_exposures(
    tmp_path: Path,
) -> None:
    path = tmp_path / "train.log"
    path.write_text("noise\n" + "\n".join(json.dumps(row) for row in _steps()) + "\n")
    parsed = MODULE.parse_optimizer_log(path)
    assert parsed[-1]["step"] == 47
    assert sum(row["examples"] for row in parsed) == 1500

    broken = _steps()
    broken[20]["token_mean_loss"] = float("nan")
    path.write_text("\n".join(json.dumps(row) for row in broken))
    with pytest.raises(MODULE.ExportError, match="non-finite"):
        MODULE.parse_optimizer_log(path)


def test_optimizer_checkpoint_requires_every_parameter_at_step_47(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    path = tmp_path / "optimizer.pt"
    payload = {
        "steps": 47,
        "exposures": 1500,
        "optimizer": {"state": {0: {"step": torch.tensor(47)}, 1: {"step": 47}}},
    }
    torch.save(payload, path)
    proof = MODULE.validate_optimizer_checkpoint(path)
    assert proof["parameter_step_min"] == proof["parameter_step_max"] == 47

    payload["optimizer"]["state"][1]["step"] = 46
    torch.save(payload, path)
    with pytest.raises(MODULE.ExportError, match="every trained adapter parameter"):
        MODULE.validate_optimizer_checkpoint(path)


def test_cli_help_is_gpu_free() -> None:
    assert MODULE.build_parser().format_help().startswith("usage:")
