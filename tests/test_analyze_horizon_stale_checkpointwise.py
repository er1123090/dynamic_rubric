from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from dynamic_rubric.config import load_config

CHECKPOINTS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _variant(score: float) -> dict:
    separated = score != 1.0
    return {
        "exact_zar": not separated,
        "near_zero_zar": not separated,
        "pairwise": {
            "tie_rate": 0.8 if separated else 1.0,
            "separation_rate": 0.2 if separated else 0.0,
        },
        "spread": {
            "population_sd": 0.1 if separated else 0.0,
            "iqr": 0.05 if separated else 0.0,
            "unique_score_ratio": 0.125 if separated else 0.0625,
        },
    }


def _effectiveness(count: int) -> dict:
    return {
        "criterion_count": count,
        "counts": {"saturated": 0, "dead": 0, "effective": count},
        "ratios": {"saturated": 0.0, "dead": 0.0, "effective": 1.0},
    }


def _build_scores(root: Path, config_hash: str) -> None:
    source = root / "source.jsonl"
    source.parent.mkdir(parents=True)
    source.write_text("source\n", encoding="utf-8")
    for checkpoint_index, checkpoint in enumerate(CHECKPOINTS):
        directory = root / f"epoch-{checkpoint:.1f}"
        directory.mkdir(parents=True)
        summaries, scores = [], []
        for prompt_index in range(100):
            prompt_id = f"prompt-{prompt_index:03d}"
            is_na = checkpoint != 0.0 and prompt_index % 9 == checkpoint_index - 1
            variants = {"r0": _variant(1.0), "current": _variant(0.5)}
            if not is_na:
                variants["control"] = _variant(0.75)
            summaries.append(
                {
                    "seed_id": "11",
                    "checkpoint": checkpoint,
                    "policy_step": checkpoint_index,
                    "pool_family": "pool_b",
                    "prompt_id": prompt_id,
                    "response_count": 16,
                    "analysis_status": "na" if is_na else "valid",
                    "na_reason": "ineligible_count_matched_control" if is_na else None,
                    "variants": variants,
                    "criterion_effectiveness": {
                        "r0": _effectiveness(2),
                        "extension": _effectiveness(1 if checkpoint else 0),
                        "control_extension": _effectiveness(1 if checkpoint and not is_na else 0),
                    },
                    "online_criterion_count": 1 if checkpoint else 0,
                    "control_criterion_count": None if is_na else (1 if checkpoint else 0),
                }
            )
            for variant, numerator in (("r0", 10), ("current", 8), ("control", 9)):
                if variant == "control" and is_na:
                    continue
                for response_index in range(16):
                    value = numerator - int(response_index == 15 and variant != "r0")
                    scores.append(
                        {
                            "prompt_id": prompt_id,
                            "response_id": f"response-{response_index:02d}",
                            "variant": variant,
                            "score_numerator": value,
                            "score_denominator": 10,
                        }
                    )
        grades: list[dict] = []
        _write_jsonl(directory / "prompt_summary.jsonl", summaries)
        _write_jsonl(directory / "variant_scores.jsonl", scores)
        _write_jsonl(directory / "criterion_grades.jsonl", grades)
        seal = {
            "artifact_type": "horizon_score_seal",
            "seed_id": "11",
            "checkpoint": checkpoint,
            "policy_step": checkpoint_index,
            "pool_family": "pool_b",
            "comparison_scope": "full" if checkpoint else None,
            "config_hash": config_hash,
            "grader_model_revision": "9216db5781bf21249d130ec9da846c4624c16137",
            "tokenizer_revision": "9216db5781bf21249d130ec9da846c4624c16137",
            "target_encoding_version": "encoding",
            "prompt_count": 100,
            "response_count": 1600,
            "grade_count": 0,
            "sources": {"fixture": {"path": str(source), "sha256": _sha256(source)}},
            "outputs": {
                name: _sha256(directory / f"{name}.jsonl")
                for name in ("criterion_grades", "variant_scores", "prompt_summary")
            },
        }
        (directory / "score_seal.json").write_text(
            json.dumps(seal, sort_keys=True), encoding="utf-8"
        )


def test_real_command_is_deterministic_and_reports_zero_complete_case(tmp_path: Path) -> None:
    scores, output = tmp_path / "scores", tmp_path / "results"
    config_hash = load_config(Path("configs/horizon_medicine.yaml")).config_hash
    _build_scores(scores, config_hash)
    command = [
        sys.executable,
        "scripts/analyze_horizon_stale_checkpointwise.py",
        "--scores-root",
        str(scores),
        "--output-dir",
        str(output),
        "--bootstrap-replicates",
        "25",
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = f"src{os.pathsep}{env.get('PYTHONPATH', '')}"
    subprocess.run(command, check=True, cwd=Path(__file__).parents[1], env=env)
    first = {path.name: path.read_bytes() for path in output.iterdir()}
    subprocess.run(command, check=True, cwd=Path(__file__).parents[1], env=env)
    second = {path.name: path.read_bytes() for path in output.iterdir()}
    assert first == second
    result = json.loads(first["stale_checkpointwise.json"])
    assert result["complete_case"]["prompt_count"] == 0
    assert not result["complete_case"]["official_horizon_estimable"]
    assert result["checkpoint_results"][1]["coverage"]["na_reasons"] == {
        "ineligible_count_matched_control": 12
    }
    assert (
        result["checkpoint_results"][1]["g_count_control_minus_current"]["bootstrap_replicates"]
        == 25
    )


def test_rejects_digest_valid_but_wrong_config_lineage(tmp_path: Path) -> None:
    scores, output = tmp_path / "scores", tmp_path / "results"
    _build_scores(scores, "wrong-config")
    command = [
        sys.executable,
        "scripts/analyze_horizon_stale_checkpointwise.py",
        "--scores-root",
        str(scores),
        "--output-dir",
        str(output),
        "--bootstrap-replicates",
        "2",
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = f"src{os.pathsep}{env.get('PYTHONPATH', '')}"
    completed = subprocess.run(
        command,
        cwd=Path(__file__).parents[1],
        env=env,
        text=True,
        capture_output=True,
    )
    assert completed.returncode != 0
    assert "config hash differs" in completed.stderr
