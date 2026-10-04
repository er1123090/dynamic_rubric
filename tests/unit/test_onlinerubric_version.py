from __future__ import annotations

from pathlib import Path
from typing import Any

from dynamic_rubric.artifacts import read_json, read_jsonl, write_json_atomic, write_jsonl_atomic
from dynamic_rubric.batch_dynamic import DYNAMIC_CRITERION_INSTRUCTIONS
from dynamic_rubric.config import config_from_mapping
from dynamic_rubric.initial_batch_dynamic import initial_rubric_batch_stage
from dynamic_rubric.onlinerubric_batch import (
    ONLINERUBRIC_PAIRWISE_COMPARISONS,
    onlinerubric_dedup_stage,
    onlinerubric_extraction_stage,
    prepare_onlinerubric_dedup_batch,
    prepare_onlinerubric_extraction_batch,
)
from dynamic_rubric.onlinerubric_bootstrap import (
    ONLINERUBRIC_R0_DEDUP_STAGE,
    ONLINERUBRIC_R0_EXTRACTION_STAGE,
    _project_rl_compatible_r0,
    prepare_onlinerubric_r0_dedup_batch,
    prepare_onlinerubric_r0_extraction_batch,
)
from dynamic_rubric.pipeline import PipelineContext
from dynamic_rubric.prompt_versions.initial_rubric_prompt import (
    INITIAL_RUBRIC_CRITERION_INSTRUCTIONS,
)
from dynamic_rubric.prompt_versions.onlinerubric_prompt import (
    ONLINERUBRIC_DEDUP_SYSTEM_PROMPT,
    build_onlinerubric_dedup_messages,
    ONLINERUBRIC_EXTRACTOR_SYSTEM_PROMPT,
    build_onlinerubric_extractor_messages,
)
from dynamic_rubric.rubrics.static import UNIVERSAL_CRITERIA

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODE = "dynamic_prev_budgeted"
PROMPT_ID = "prompt-1"


def _context(tmp_path: Path, stage: str) -> PipelineContext:
    config = config_from_mapping(
        {
            "experiment": "test-onlinerubric",
            "paths": {
                "public_data": "data/public",
                "artifacts": "artifacts",
                "results": "results",
            },
            "models": {
                "rubric_generator": {
                    "requested_model": "gpt-5-mini",
                    "reasoning_effort": "medium",
                }
            },
            "training": {"reward_source": "static_r0_only"},
        },
        stage=stage,
    )
    return PipelineContext(
        root=tmp_path,
        config_path=tmp_path / "configs" / "pilot.yaml",
        stage=stage,
        run_id="paper-test",
        config=config,
    )


def _criterion(index: int) -> dict[str, Any]:
    return {
        "criterion_id": f"task-{index:02d}",
        "text": f"Satisfies medical requirement number {index}",
        "weight": 0.125,
        "source": "task_specific",
        "created_step": 0,
    }


def _write_fixture(tmp_path: Path) -> None:
    public_root = tmp_path / "data" / "public"
    write_jsonl_atomic(
        public_root / "pilot_probe.jsonl",
        [
            {
                "prompt_id": PROMPT_ID,
                "source": "test",
                "messages": [{"role": "user", "content": "What should the patient do?"}],
            }
        ],
    )
    for schema in (
        "configs/schemas/onlinerubric_extraction_v1.json",
        "configs/schemas/onlinerubric_dedup_v1.json",
    ):
        write_json_atomic(tmp_path / schema, read_json(PROJECT_ROOT / schema))

    run_root = tmp_path / "artifacts" / "runs" / "paper-test"
    write_jsonl_atomic(
        run_root / "generate-static" / "static_rubrics.jsonl",
        [
            {
                "prompt_id": PROMPT_ID,
                "rubric_id": f"{PROMPT_ID}:R_0",
                "criteria": [_criterion(index) for index in range(1, 9)],
            }
        ],
    )
    write_jsonl_atomic(
        run_root / "train-static" / "reference_responses.jsonl",
        [
            {
                "prompt_id": PROMPT_ID,
                "policy_step": 0,
                "family": "reference_discovery",
                "sample_index": index,
                "response_id": f"reference-{index}",
                "response_text": f"Reference response {index}",
            }
            for index in range(8)
        ],
    )
    probe_rows = []
    for family in ("trajectory_discovery", "trajectory_validation"):
        for index in range(4):
            probe_rows.append(
                {
                    "prompt_id": PROMPT_ID,
                    "policy_step": 1,
                    "family": family,
                    "sample_index": index,
                    "response_id": f"{family}-{index}",
                    "output": f"{family} response {index}",
                }
            )
    write_jsonl_atomic(
        run_root / "train-static" / "verl-run" / "probes" / "1.jsonl",
        probe_rows,
    )


def test_prompt_versions_are_explicit_and_paper_prompt_has_required_inputs() -> None:
    assert DYNAMIC_CRITERION_INSTRUCTIONS == INITIAL_RUBRIC_CRITERION_INSTRUCTIONS
    assert initial_rubric_batch_stage(MODE) == "initial_dynamic_prev_batch"
    assert onlinerubric_extraction_stage(MODE) == "onlinerubric_extraction_prev_batch"
    assert onlinerubric_dedup_stage(MODE) == "onlinerubric_dedup_prev_batch"

    messages = build_onlinerubric_extractor_messages(
        prompt=[{"role": "user", "content": "Question text"}],
        existing_rubric=[{"criterion": "Existing criterion", "weight": 1}],
        response_a="Candidate A",
        response_b="Candidate B",
    )
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "reward hacking" in messages[0]["content"]
    assert "Do not use your own knowledge" in messages[0]["content"]
    assert '"new_criteria"' in messages[0]["content"]
    assert "Prompt:" in messages[1]["content"]
    assert "Question text" in messages[1]["content"]
    assert "Existing Rubric:" in messages[1]["content"]
    assert "Existing criterion" in messages[1]["content"]
    assert "Response A:\nCandidate A" in messages[1]["content"]
    assert "Response B:\nCandidate B" in messages[1]["content"]
    assert "ONLY to deduplicate and aggregate" in ONLINERUBRIC_DEDUP_SYSTEM_PROMPT
    assert "pair of responses" in ONLINERUBRIC_EXTRACTOR_SYSTEM_PROMPT

def test_phase1_dedup_prompt_treats_existing_rubric_as_exclusion_list() -> None:
    messages = build_onlinerubric_dedup_messages(
        prompt=[{"role": "user", "content": "Question text"}],
        existing_rubric=[{"criterion": "Existing criterion", "weight": 1}],
        candidate_criteria=[
            {"quote": "Grounded quote", "criterion": "Candidate criterion", "weight": 2}
        ],
        exclude_existing=True,
    )

    assert "Existing Rubric is an exclusion list" in messages[0]["content"]
    assert "Never copy, paraphrase, merge" in messages[0]["content"]
    assert "empty `final_criteria` list" in messages[0]["content"]



def test_onlinerubric_preparation_uses_eight_pair_calls_then_one_dedup_call(
    tmp_path: Path,
) -> None:
    _write_fixture(tmp_path)
    extraction_context = _context(tmp_path, onlinerubric_extraction_stage(MODE))
    extraction = prepare_onlinerubric_extraction_batch(
        extraction_context,
        max_step=1,
        mode=MODE,
    )
    assert extraction["generation_method"] == "onlinerubric"
    assert extraction["pairwise_comparisons_per_instance"] == ONLINERUBRIC_PAIRWISE_COMPARISONS
    assert extraction["requests"] == 8

    extraction_inputs = [
        row for record in extraction["input_files"] for row in read_jsonl(record["path"])
    ]
    assert len(extraction_inputs) == 8
    assert all(len(row["body"]["input"]) == 2 for row in extraction_inputs)
    assert all(
        row["body"]["input"][0]["content"] == ONLINERUBRIC_EXTRACTOR_SYSTEM_PROMPT
        for row in extraction_inputs
    )
    assert all(
        "What should the patient do?" in row["body"]["input"][1]["content"]
        and "Existing Rubric:" in row["body"]["input"][1]["content"]
        and "Response A:" in row["body"]["input"][1]["content"]
        and "Response B:" in row["body"]["input"][1]["content"]
        for row in extraction_inputs
    )

    request_map = read_jsonl(extraction["request_map"]["path"])
    extracted_rows = []
    for identity in request_map:
        extracted_rows.append(
            {
                **identity,
                "analysis": "Pair-level analysis",
                "new_criteria": [
                    {
                        "quote": f"quote-{identity['pair_index']}",
                        "criterion": f"Criterion from pair {identity['pair_index']}",
                        "weight": 2,
                    }
                ],
                "provider_call": {"returned_model": "gpt-5-mini-test"},
            }
        )
    write_jsonl_atomic(
        extraction_context.stage_root() / "onlinerubric_extracted_criteria.jsonl",
        extracted_rows,
    )

    dedup_context = _context(tmp_path, onlinerubric_dedup_stage(MODE))
    dedup = prepare_onlinerubric_dedup_batch(dedup_context, mode=MODE)
    assert dedup["generation_method"] == "onlinerubric"
    assert dedup["requests"] == 1
    dedup_input = read_jsonl(dedup["input_files"][0]["path"])[0]
    assert dedup_input["body"]["input"][0]["content"] == ONLINERUBRIC_DEDUP_SYSTEM_PROMPT
    user_content = dedup_input["body"]["input"][1]["content"]
    assert "Candidate Criteria From Pairwise Comparisons:" in user_content
    assert user_content.count('"source_pair_id"') == 8
    assert "What should the patient do?" in user_content
    assert "Existing Rubric:" in user_content


def _write_probe(tmp_path: Path, step: int, prompt_ids: tuple[str, ...]) -> None:
    rows = []
    for prompt_id in prompt_ids:
        for family in ("trajectory_discovery", "trajectory_validation"):
            for index in range(4):
                rows.append(
                    {
                        "prompt_id": prompt_id,
                        "policy_step": step,
                        "family": family,
                        "sample_index": index,
                        "response_id": f"{prompt_id}-{step}-{family}-{index}",
                        "output": f"{prompt_id} step {step} {family} response {index}",
                    }
                )
    write_jsonl_atomic(
        tmp_path
        / "artifacts"
        / "runs"
        / "paper-test"
        / "train-static"
        / "verl-run"
        / "probes"
        / f"{step}.jsonl",
        rows,
    )


def test_onlinerubric_preparation_supports_sparse_steps_and_exact_prompt_subset(
    tmp_path: Path,
) -> None:
    _write_fixture(tmp_path)
    for step in (2, 3, 9, 10):
        _write_probe(tmp_path, step, (PROMPT_ID,))

    context = _context(tmp_path, onlinerubric_extraction_stage(MODE))
    manifest = prepare_onlinerubric_extraction_batch(
        context,
        max_step=10,
        policy_steps=(3, 10),
        selected_prompt_ids=(PROMPT_ID,),
        mode=MODE,
    )

    assert manifest["policy_steps"] == [3, 10]
    assert manifest["prompt_ids"] == [PROMPT_ID]
    assert manifest["requests"] == 16
    request_map = read_jsonl(manifest["request_map"]["path"])
    assert {row["policy_step"] for row in request_map} == {3, 10}
    assert {(row["policy_step"], row["control_policy"]) for row in request_map} == {
        (3, "pi_2"),
        (10, "pi_9"),
    }


def test_onlinerubric_bootstrap_r0_uses_only_universal_seed(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    run_root = tmp_path / "artifacts" / "runs" / "paper-test"
    write_jsonl_atomic(
        run_root / "generate-static" / "static_candidates.jsonl",
        [
            {
                "prompt_id": PROMPT_ID,
                "sample_index": index,
                "response_id": f"candidate-{index}",
                "response_text": f"Independent pi0 candidate {index}",
            }
            for index in range(8)
        ],
    )

    extraction_context = _context(tmp_path, ONLINERUBRIC_R0_EXTRACTION_STAGE)
    extraction = prepare_onlinerubric_r0_extraction_batch(
        extraction_context, selected_prompt_ids=(PROMPT_ID,)
    )
    assert extraction["requests"] == 8
    assert extraction["gold_access"] is False
    assert extraction["original_prompt_specific_r0_access"] is False
    identities = read_jsonl(extraction["request_map"]["path"])
    assert {row["control_policy"] for row in identities} == {"pi_0_independent_reference"}
    inputs = read_jsonl(extraction["input_files"][0]["path"])
    user_prompt = inputs[0]["body"]["input"][1]["content"]
    assert all(text in user_prompt for text in UNIVERSAL_CRITERIA)
    assert "Satisfies medical requirement" not in user_prompt

    write_jsonl_atomic(
        extraction_context.stage_root() / "onlinerubric_extracted_criteria.jsonl",
        [
            {
                **identity,
                "analysis": "Pair analysis",
                "new_criteria": [
                    {
                        "quote": f"quote-{identity['pair_index']}",
                        "criterion": f"Criterion from pair {identity['pair_index']}",
                        "weight": identity["pair_index"] + 1,
                    }
                ],
                "provider_call": {"returned_model": "gpt-5-mini-test"},
            }
            for identity in identities
        ],
    )
    dedup_context = _context(tmp_path, ONLINERUBRIC_R0_DEDUP_STAGE)
    dedup = prepare_onlinerubric_r0_dedup_batch(dedup_context)
    assert dedup["requests"] == 1
    assert dedup["original_prompt_specific_r0_access"] is False
    dedup_input = read_jsonl(dedup["input_files"][0]["path"])[0]
    assert dedup_input["body"]["input"][0]["content"] == ONLINERUBRIC_DEDUP_SYSTEM_PROMPT

    projected = _project_rl_compatible_r0(
        {
            "prompt_id": PROMPT_ID,
            "rubric_id": f"{PROMPT_ID}:onlinerubric_bootstrap_full_R_0",
            "provider_call": {"returned_model": "gpt-5-mini-test"},
            "criteria": [
                {
                    "criterion_id": f"online-{index}",
                    "text": f"Criterion {index}",
                    "weight": index,
                }
                for index in range(1, 9)
            ],
        }
    )
    assert len(projected["criteria"]) == 8
    assert [item["source"] for item in projected["criteria"]] == ["task_specific"] * 6 + [
        "universal"
    ] * 2
    assert projected["provenance"]["gold_access"] is False
    assert projected["provenance"]["source_rubric_id"] == (
        f"{PROMPT_ID}:onlinerubric_bootstrap_full_R_0"
    )
