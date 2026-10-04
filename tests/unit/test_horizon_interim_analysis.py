from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from dynamic_rubric.hashing import sha256_file
from dynamic_rubric.horizon.observations import ObservationSealError


SCRIPT_PATH = Path(__file__).parents[2] / "scripts" / "analyze_horizon_interim.py"
SPEC = importlib.util.spec_from_file_location("analyze_horizon_interim", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _row(prompt_id: str, *, pool: str, r0_zar: bool, current_zar: bool) -> dict:
    def variant(zar: bool, separation: float) -> dict:
        return {
            "advantage_degenerate": zar,
            "exact_zar": zar,
            "near_zero_zar": zar,
            "pairwise": {"tie_rate": 1.0 - separation, "separation_rate": separation},
            "spread": {"population_sd": separation, "iqr": 0.0, "unique_score_ratio": 0.25},
        }

    current = variant(current_zar, 0.3)
    current["pairwise_vs_r0"] = {
        "incremental_tie_resolution": 0.2,
        "conditional_tie_resolution": 0.25,
        "new_tie_rate": 0.01,
        "ordering_reversal_rate": 0.0,
    }
    current["ranking_vs_r0"] = {
        "kendall_tau_b": 0.8,
        "kendall_defined": True,
        "pairwise_ordering_agreement": 0.9,
        "top_set_jaccard": 0.75,
        "top_set_exact_match": False,
    }
    effectiveness = {
        "r0": {
            "criterion_count": 2,
            "counts": {"effective": 1, "saturated": 1, "dead": 0},
        },
        "extension": {
            "criterion_count": 1,
            "counts": {"effective": 1, "saturated": 0, "dead": 0},
        },
    }
    return {
        "analysis_status": "valid",
        "checkpoint": 0.2,
        "policy_step": 3,
        "pool_family": pool,
        "prompt_id": prompt_id,
        "response_count": 16 if pool == "pool_b" else 8,
        "variants": {"r0": variant(r0_zar, 0.1), "current": current},
        "criterion_effectiveness": effectiveness,
        "online_criterion_count": 1,
    }


def _sealed_summary(root: Path, rows: list[dict]) -> Path:
    root.mkdir(parents=True)
    path = root / "prompt_summary.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    seal = {
        "artifact_type": "horizon_score_seal",
        "comparison_scope": "r0_current",
        "prompt_count": len(rows),
        "outputs": {"prompt_summary": sha256_file(path)},
        "config_hash": "config-hash",
        "grader_model_revision": "grader-revision",
        "tokenizer_revision": "tokenizer-revision",
        "target_encoding_version": "encoding-v1",
        "seed_id": "11",
    }
    (root / "score_seal.json").write_text(json.dumps(seal), encoding="utf-8")
    return path


def test_build_report_uses_verified_summaries_and_paired_gains(tmp_path: Path) -> None:
    pool_b = tmp_path / "pool_b"
    pool_a = tmp_path / "pool_a"
    rows_b = [
        _row("p1", pool="pool_b", r0_zar=True, current_zar=False),
        _row("p2", pool="pool_b", r0_zar=False, current_zar=False),
    ]
    rows_a = [
        _row("p1", pool="pool_a", r0_zar=True, current_zar=True),
        _row("p2", pool="pool_a", r0_zar=False, current_zar=False),
    ]
    _sealed_summary(pool_b / "epoch-0.2", rows_b)
    _sealed_summary(pool_a / "epoch-0.2", rows_a)

    report = MODULE.build_report(
        pool_b_root=pool_b,
        pool_a_root=pool_a,
        checkpoints=(0.2,),
        baseline_path=None,
        kl_path=tmp_path / "missing-kl.json",
        rubric_root=None,
        bootstrap_iterations=100,
        bootstrap_seed=2718,
        expected_config_hash="config-hash",
        expected_seed_id="11",
        expected_grader_revision="grader-revision",
        expected_prompt_count=2,
    )

    result = report["pool_b"][0]
    assert result["gains"]["refresh_zar_r0_minus_current"] == pytest.approx(0.5)
    assert result["gains"]["separation_current_minus_r0"] == pytest.approx(0.2)
    assert result["criterion_effectiveness"]["r0"]["ratios"]["effective"] == 0.5
    assert result["bootstrap_95ci"]["refresh_zar_r0_minus_current"]["seed"] == 2718
    assert report["pool_a_vs_b_generalization"][0][
        "pool_b_minus_pool_a_refresh_gain"
    ] == pytest.approx(0.5)


def test_build_report_rejects_tampered_summary(tmp_path: Path) -> None:
    pool_b = tmp_path / "pool_b"
    path = _sealed_summary(
        pool_b / "epoch-0.2", [_row("p1", pool="pool_b", r0_zar=True, current_zar=False)]
    )
    path.write_text(path.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")

    with pytest.raises(ObservationSealError, match="digest mismatch"):
        MODULE._load_series(
            pool_b,
            (0.2,),
            bootstrap_iterations=10,
            bootstrap_seed=2718,
            expected_prompt_count=1,
        )


def test_build_report_rejects_pool_prompt_set_mismatch(tmp_path: Path) -> None:
    pool_b = tmp_path / "pool_b"
    pool_a = tmp_path / "pool_a"
    _sealed_summary(
        pool_b / "epoch-0.2",
        [_row("p1", pool="pool_b", r0_zar=True, current_zar=False)],
    )
    _sealed_summary(
        pool_a / "epoch-0.2",
        [_row("different", pool="pool_a", r0_zar=True, current_zar=False)],
    )

    with pytest.raises(ValueError, match="Pool A/B prompt ID sets differ"):
        MODULE.build_report(
            pool_b_root=pool_b,
            pool_a_root=pool_a,
            checkpoints=(0.2,),
            baseline_path=None,
            kl_path=tmp_path / "missing.json",
            rubric_root=None,
            bootstrap_iterations=10,
            bootstrap_seed=2718,
            expected_config_hash="config-hash",
            expected_seed_id="11",
            expected_grader_revision="grader-revision",
            expected_prompt_count=1,
        )


def test_rubric_structure_reports_distribution_rejections_and_churn(tmp_path: Path) -> None:
    root = tmp_path / "rubrics"
    root.mkdir()

    def write_step(step: int, criterion_hash: str, *, rejected: str) -> None:
        row = {
            "prompt_id": "p1",
            "r0": [],
            "extension": [
                {
                    "canonical_criterion_hash": criterion_hash,
                    "weight_units": 10,
                    "importance_class": "essential",
                    "criterion_type": "quality",
                }
            ],
            "rejected": [["candidate", rejected]],
        }
        (root / f"step-{step}.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

    write_step(3, "old", rejected="invalid_evidence")
    write_step(6, "new", rejected="not_atomic")
    analysis = MODULE.rubric_structural_analysis(root, (0.2, 0.4))

    assert analysis is not None
    assert analysis["checkpoints"][0]["weight_units_histogram"] == {"10": 1}
    assert analysis["checkpoints"][1]["rejection_reason_distribution"] == {"not_atomic": 1}
    assert analysis["adjacent_extension_hash_churn"][0]["mean_extension_hash_jaccard"] == 0.0
    assert analysis["adjacent_extension_hash_churn"][0]["mean_added_criteria"] == 1.0
