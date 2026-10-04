from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import pytest

from dynamic_rubric.artifacts import (
    read_json,
    read_jsonl,
    write_json_atomic,
    write_jsonl_atomic,
)
from dynamic_rubric.hashing import sha256_file
from dynamic_rubric.horizon.contracts import (
    CriterionType,
    ImportanceClass,
    WeightedCriterion,
    criterion_content_hash,
)
from dynamic_rubric.horizon.live_grading import (
    grade_horizon_pool_a,
    grade_horizon_pool_a_combined,
    grade_horizon_pool_b,
)
from dynamic_rubric.horizon.live_rubrics import build_live_horizon_rubrics
from dynamic_rubric.horizon.observations import (
    build_horizon_observations,
    verify_horizon_observation_seal,
)
from dynamic_rubric.providers.base import GenerationRequest, GenerationResult
from dynamic_rubric.providers.vllm import FullCriterionScore


def _criterion(identity: str, text: str, *, checkpoint: str) -> WeightedCriterion:
    return WeightedCriterion(
        criterion_instance_id=identity,
        canonical_criterion_hash=criterion_content_hash(text),
        text=text,
        importance_class=ImportanceClass.IMPORTANT,
        criterion_type=CriterionType.QUALITY,
        weight_units=7,
        source_checkpoint=checkpoint,
    )


def _prompt() -> dict[str, object]:
    text = "Answers correctly"
    return {
        "prompt_id": "p1",
        "messages": [{"role": "user", "content": "Question"}],
        "r0": {
            "criteria": [
                {
                    "criterion_id": "p1:r0:0",
                    "criterion": text,
                    "importance_class": "essential",
                    "criterion_type": "quality",
                    "weight_units": 10,
                    "canonical_criterion_hash": criterion_content_hash(text),
                }
            ]
        },
    }


class ScriptedExtractor:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, request: GenerationRequest) -> GenerationResult:
        if request.family == "horizon_extraction":
            value = (
                {
                    "analysis": "one grounded difference",
                    "new_criteria": [
                        {
                            "candidate_id": "candidate-1",
                            "quote": "alpha",
                            "criterion": "States diagnosis",
                            "weight": 5,
                            "importance_class": "important",
                            "criterion_type": "quality",
                        }
                    ],
                }
                if self.calls == 0
                else {"analysis": "none", "new_criteria": []}
            )
        else:
            value = {
                "analysis": "retain exact source wording",
                "final_criteria": [
                    {"criterion": "States diagnosis", "source_candidate_ids": ["candidate-1"]}
                ],
            }
        self.calls += 1
        return GenerationResult(
            text=json.dumps(value),
            requested_model="extractor",
            returned_model="extractor",
            request_id=f"request-{self.calls}",
            created_at=None,
            retry_count=0,
        )


def test_live_rubric_builder_runs_eight_extractions_and_count_matches_control(
    tmp_path: Path,
) -> None:
    schema_root = Path(__file__).resolve().parents[2] / "configs" / "schemas"
    rows = [{"prompt_id": "p1", "response_text": f"alpha response {index}"} for index in range(8)]
    control = _criterion("control-1", "Mentions uncertainty", checkpoint="step1")
    output = tmp_path / "rubrics.jsonl"
    result = build_live_horizon_rubrics(
        ScriptedExtractor(),
        prompts=[_prompt()],
        current_rows=rows,
        control_rows=rows,
        checkpoint_id="step3",
        pairing_seed=7,
        extraction_schema=read_json(schema_root / "horizon_extraction_v1.json"),
        dedup_schema=read_json(schema_root / "horizon_dedup_v1.json"),
        output_path=output,
        control_rubric_rows=[{"prompt_id": "p1", "extension": [asdict(control)]}],
    )
    rubric = read_jsonl(output)[0]
    assert result["request_count"] == 8
    assert result["candidate_count"] == 1
    assert len(rubric["extension"]) == 1
    assert len(rubric["control_extension"]) == 1
    assert rubric["control_match"]["eligible"] is True
    assert len(read_jsonl(tmp_path / "extraction_requests.jsonl")) == 8


class ScriptedGrader:
    def score_many_full(self, items, *, prompt_text_by_id=None):
        assert prompt_text_by_id == {"p1": '[{"content": "Question", "role": "user"}]'}
        rows = []
        for prompt_id, response_id, _, criterion_id, criterion_text in items:
            if criterion_text == "Distinguishes severity":
                present = int(response_id.rsplit("-", 1)[1]) % 2 == 0
            elif criterion_text == "Answers correctly":
                present = True
            else:
                present = False
            yes, no = (-0.1, -2.0) if present else (-2.0, -0.1)
            rows.append(
                FullCriterionScore(
                    prompt_id,
                    response_id,
                    criterion_id,
                    yes,
                    no,
                    0.9 if present else 0.1,
                    "ok",
                )
            )
        return tuple(rows)


class CountingScriptedGrader(ScriptedGrader):
    def __init__(self) -> None:
        self.item_counts: list[int] = []

    def score_many_full(self, items, *, prompt_text_by_id=None):
        self.item_counts.append(len(items))
        return super().score_many_full(items, prompt_text_by_id=prompt_text_by_id)


def _pool(step: int, family: str = "pool_b") -> list[dict[str, object]]:
    count = {"pool_a": 8, "pool_a_combined": 16, "pool_b": 16}[family]
    return [
        {
            "pool_family": family,
            "training_seed": 11,
            "policy_step": step,
            "prompt_id": "p1",
            "sample_index": index,
            "response_id": f"response-{family}-{step}-{index}",
            "response_text": f"answer {index}",
        }
        for index in range(count)
    ]


def test_live_grading_supports_auxiliary_pool_a_and_preserves_pool_b_contract(
    tmp_path: Path,
) -> None:
    extension = _criterion("current", "Distinguishes severity", checkpoint="step3")
    control = _criterion("control", "Mentions uncertainty", checkpoint="step1")
    rubric_rows = [
        {
            "prompt_id": "p1",
            "checkpoint_id": "step3",
            "extension": [asdict(extension)],
            "control_extension": [asdict(control)],
            "control_match": {"eligible": True},
        }
    ]
    output_dir = tmp_path / "pool-a"
    result = grade_horizon_pool_a(
        ScriptedGrader(),
        prompts=[_prompt()],
        pool_a_rows=_pool(3, "pool_a"),
        rubric_rows=rubric_rows,
        checkpoint=0.2,
        output_dir=output_dir,
        grader_model_revision="grader-rev",
        tokenizer_revision="grader-rev",
        epsilon_spread=0.01,
        delta_advantage=1e-8,
        expected_policy_step=3,
    )
    summary = read_jsonl(output_dir / "prompt_summary.jsonl")[0]
    assert result["pool_family"] == "pool_a"
    assert result["response_count"] == 8
    assert summary["pool_family"] == "pool_a"
    assert summary["response_count"] == 8
    assert summary["variants"]["current"]["pairwise"]["pair_count"] == 28

    combined_dir = tmp_path / "pool-a-combined"
    combined_result = grade_horizon_pool_a_combined(
        ScriptedGrader(),
        prompts=[_prompt()],
        pool_a_combined_rows=_pool(3, "pool_a_combined"),
        rubric_rows=rubric_rows,
        checkpoint=0.2,
        output_dir=combined_dir,
        grader_model_revision="grader-rev",
        tokenizer_revision="grader-rev",
        epsilon_spread=0.01,
        delta_advantage=1e-8,
        expected_policy_step=3,
    )
    combined_summary = read_jsonl(combined_dir / "prompt_summary.jsonl")[0]
    assert combined_result["pool_family"] == "pool_a_combined"
    assert combined_result["response_count"] == 16
    assert combined_summary["response_count"] == 16
    assert combined_summary["variants"]["current"]["pairwise"]["pair_count"] == 120

    with pytest.raises(ValueError, match="Pool B"):
        grade_horizon_pool_b(
            ScriptedGrader(),
            prompts=[_prompt()],
            pool_b_rows=_pool(3, "pool_a"),
            rubric_rows=rubric_rows,
            checkpoint=0.2,
            output_dir=tmp_path / "wrong-family",
            grader_model_revision="grader-rev",
            tokenizer_revision="grader-rev",
            epsilon_spread=0.01,
            delta_advantage=1e-8,
            expected_policy_step=3,
        )


def test_live_grading_reuses_current_grades_and_only_judges_added_control(
    tmp_path: Path,
) -> None:
    extension = _criterion("current", "Distinguishes severity", checkpoint="step3")
    control = _criterion("control", "Mentions uncertainty", checkpoint="step1")
    immediate_dir = tmp_path / "immediate"
    final_dir = tmp_path / "final"
    immediate_grader = CountingScriptedGrader()
    grade_horizon_pool_b(
        immediate_grader,
        prompts=[_prompt()],
        pool_b_rows=_pool(3),
        rubric_rows=[
            {
                "prompt_id": "p1",
                "checkpoint_id": "step3",
                "extension": [asdict(extension)],
            }
        ],
        checkpoint=0.2,
        output_dir=immediate_dir,
        grader_model_revision="grader-rev",
        tokenizer_revision="grader-rev",
        epsilon_spread=0.01,
        delta_advantage=1e-8,
        include_control=False,
        expected_policy_step=3,
    )
    final_grader = CountingScriptedGrader()
    result = grade_horizon_pool_b(
        final_grader,
        prompts=[_prompt()],
        pool_b_rows=_pool(3),
        rubric_rows=[
            {
                "prompt_id": "p1",
                "checkpoint_id": "step3",
                "extension": [asdict(extension)],
                "control_extension": [asdict(control)],
                "control_match": {"eligible": True},
            }
        ],
        checkpoint=0.2,
        output_dir=final_dir,
        grader_model_revision="grader-rev",
        tokenizer_revision="grader-rev",
        epsilon_spread=0.01,
        delta_advantage=1e-8,
        reused_grade_rows=read_jsonl(immediate_dir / "criterion_grades.jsonl"),
        expected_policy_step=3,
    )
    assert immediate_grader.item_counts == [32]
    assert read_jsonl(immediate_dir / "prompt_summary.jsonl")[0]["analysis_status"] == "valid"
    assert read_json(immediate_dir / "score_seal.json")["comparison_scope"] == "r0_current"
    assert final_grader.item_counts == [16]
    assert result["reused_grade_count"] == 32
    assert result["new_grade_count"] == 16


def test_live_grading_emits_full_metrics_and_observations_require_score_seals(
    tmp_path: Path,
) -> None:
    extension = _criterion("current", "Distinguishes severity", checkpoint="step3")
    control = _criterion("control", "Mentions uncertainty", checkpoint="step1")
    baseline_dir = tmp_path / "baseline"
    late_dir = tmp_path / "late"
    grade_horizon_pool_b(
        ScriptedGrader(),
        prompts=[_prompt()],
        pool_b_rows=_pool(0),
        rubric_rows=[{"prompt_id": "p1", "extension": []}],
        checkpoint=0.0,
        output_dir=baseline_dir,
        grader_model_revision="grader-rev",
        tokenizer_revision="grader-rev",
        epsilon_spread=0.01,
        delta_advantage=1e-8,
    )
    grade_horizon_pool_b(
        ScriptedGrader(),
        prompts=[_prompt()],
        pool_b_rows=_pool(3),
        rubric_rows=[
            {
                "prompt_id": "p1",
                "extension": [asdict(extension)],
                "control_extension": [asdict(control)],
                "control_match": {"eligible": True},
            }
        ],
        checkpoint=0.2,
        output_dir=late_dir,
        grader_model_revision="grader-rev",
        tokenizer_revision="grader-rev",
        epsilon_spread=0.01,
        delta_advantage=1e-8,
    )
    summary = read_jsonl(late_dir / "prompt_summary.jsonl")[0]
    assert summary["variants"]["r0"]["exact_zar"] is True
    assert summary["variants"]["current"]["exact_zar"] is False
    assert summary["variants"]["current"]["pairwise"]["pair_count"] == 120
    assert summary["criterion_effectiveness"]["extension"]["counts"]["effective"] == 1

    observations_path = tmp_path / "observations.jsonl"
    result = build_horizon_observations(
        [baseline_dir / "prompt_summary.jsonl", late_dir / "prompt_summary.jsonl"],
        output_path=observations_path,
        expected_seed_ids=["11"],
        expected_prompt_ids=["p1"],
        expected_checkpoints=[0.0, 0.2],
    )
    assert result["observation_count"] == 2
    assert verify_horizon_observation_seal(observations_path)["coverage"]["na"] == 0
    late = read_jsonl(observations_path)[1]
    assert (late["r0_zar"], late["current_zar"], late["control_zar"]) == (1, 0, 1)


def test_observation_builder_reports_na_and_excludes_the_full_prompt_trajectory(
    tmp_path: Path,
) -> None:
    summaries = []
    for checkpoint in (0.0, 0.2):
        directory = tmp_path / f"checkpoint-{checkpoint}"
        directory.mkdir()
        path = directory / "prompt_summary.jsonl"
        rows = []
        for prompt_id in ("p1", "p2"):
            is_na = checkpoint == 0.2 and prompt_id == "p2"
            variants = {
                "r0": {"exact_zar": False},
                "current": {"exact_zar": False},
            }
            if not is_na:
                variants["control"] = {"exact_zar": False}
            rows.append(
                {
                    "seed_id": "11",
                    "prompt_id": prompt_id,
                    "checkpoint": checkpoint,
                    "response_count": 16,
                    "analysis_status": "na" if is_na else "valid",
                    "variants": variants,
                }
            )
        write_jsonl_atomic(path, rows)
        write_json_atomic(
            directory / "score_seal.json",
            {
                "artifact_type": "horizon_score_seal",
                "config_hash": "config",
                "seed_id": "11",
                "checkpoint": checkpoint,
                "prompt_count": 2,
                "outputs": {"prompt_summary": sha256_file(path)},
            },
        )
        summaries.append(path)
    output = tmp_path / "observations.jsonl"
    result = build_horizon_observations(
        summaries,
        output_path=output,
        expected_seed_ids=["11"],
        expected_prompt_ids=["p1", "p2"],
        expected_checkpoints=[0.0, 0.2],
        expected_config_hash="config",
    )
    assert result["coverage"] == {
        "expected": 4,
        "valid": 2,
        "invalid": 1,
        "missing": 0,
        "na": 1,
        "paired_excluded": 1,
    }
    assert {row["prompt_id"] for row in read_jsonl(output)} == {"p1"}
