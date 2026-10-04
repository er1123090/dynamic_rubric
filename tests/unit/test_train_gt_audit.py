from __future__ import annotations

from pathlib import Path

import pytest

from dynamic_rubric.artifacts import read_jsonl, write_json_atomic, write_jsonl_atomic
from dynamic_rubric.train_gt_audit import (
    APPROVAL_DESTINATION,
    APPROVAL_PURPOSE,
    APPROVED_PAYLOAD_CATEGORIES,
    TrainGoldAuditError,
    prepare_train_gold_batch,
    stage_root,
    summarize_train_gold_scores,
)


def _schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "criterion_scores": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "criterion_id": {"type": "string"},
                        "score": {"type": "integer", "enum": [0, 1]},
                    },
                    "required": ["criterion_id", "score"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["criterion_scores"],
        "additionalProperties": False,
    }


def _approval(model: str = "gpt-5-mini") -> dict[str, object]:
    return {
        "approved": True,
        "destination": APPROVAL_DESTINATION,
        "endpoint": "/v1/responses",
        "purpose": APPROVAL_PURPOSE,
        "requested_model": model,
        "payload_categories": list(APPROVED_PAYLOAD_CATEGORIES),
    }


def _prepare_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    run_root = tmp_path / "artifacts" / "runs" / "test-run"
    public = tmp_path / "data" / "public" / "pilot_train.jsonl"
    write_jsonl_atomic(
        public,
        [
            {"prompt_id": "p1", "messages": [{"role": "user", "content": "question 1"}]},
            {"prompt_id": "p2", "messages": [{"role": "user", "content": "question 2"}]},
        ],
    )
    private_gt = tmp_path / "data" / "private_gt" / "gold.jsonl"
    write_jsonl_atomic(
        private_gt,
        [
            {
                "prompt_id": prompt_id,
                "gold_rubric": [{"criterion": "Be correct", "points": 1}],
            }
            for prompt_id in ("p1", "p2")
        ],
    )
    schema = tmp_path / "schema.json"
    write_json_atomic(schema, _schema())
    approval = stage_root(run_root) / "egress-approval.json"
    write_json_atomic(approval, _approval())
    rollout = run_root / "train-static" / "verl-run" / "rollouts" / "1.jsonl"
    rows = []
    for prompt_id in ("p1", "p2"):
        for replicate in range(2):
            rows.append(
                {
                    "policy_step": 1,
                    "prompt_id": prompt_id,
                    "output": f"{prompt_id} response {replicate}",
                    "static_reward": 0.2 + 0.2 * replicate,
                    "response_id": f"shared-{prompt_id}",
                    "sample_index": 7,
                    "logical_seed": 11,
                }
            )
    write_jsonl_atomic(rollout, rows)
    return run_root, private_gt, schema, approval


def test_prepare_train_gt_assigns_unique_ids_despite_source_collisions(
    tmp_path: Path,
) -> None:
    run_root, private_gt, schema, approval = _prepare_inputs(tmp_path)

    manifest = prepare_train_gold_batch(
        run_root,
        private_gt,
        schema,
        approval,
        max_step=1,
        expected_rows_per_step=4,
        expected_outputs_per_prompt=2,
    )

    assert manifest["requests"] == 4
    assert manifest["unique_source_response_ids"] == 2
    assert manifest["evaluation_response_ids"] == 4
    mapping = read_jsonl(Path(str(manifest["request_map"]["path"])))
    assert len({row["evaluation_response_id"] for row in mapping}) == 4
    assert len({row["source_response_id"] for row in mapping}) == 2
    requests = []
    for record in manifest["input_files"]:
        requests.extend(read_jsonl(Path(str(record["path"]))))
    assert {row["body"]["model"] for row in requests} == {"gpt-5-mini"}
    assert all(row["body"]["text"]["format"]["strict"] is True for row in requests)


def test_prepare_train_gt_rejects_model_approval_drift(tmp_path: Path) -> None:
    run_root, private_gt, schema, approval = _prepare_inputs(tmp_path)
    write_json_atomic(approval, _approval("gpt-5"), immutable=False)

    with pytest.raises(TrainGoldAuditError, match="approval scope"):
        prepare_train_gold_batch(
            run_root,
            private_gt,
            schema,
            approval,
            max_step=1,
            expected_rows_per_step=4,
            expected_outputs_per_prompt=2,
        )


def test_summarize_train_gt_reports_paired_step_statistics(tmp_path: Path) -> None:
    rows = []
    for step, proxy_values, gold_values in (
        (1, (0.2, 0.4), (0.0, 0.5)),
        (2, (0.6, 0.8), (0.5, 1.0)),
    ):
        for index, prompt_id in enumerate(("p1", "p2")):
            rows.append(
                {
                    "policy_step": step,
                    "prompt_id": prompt_id,
                    "proxy_reward": proxy_values[index],
                    "gold_score": gold_values[index],
                }
            )
    output = tmp_path / "summary.csv"

    result = summarize_train_gold_scores(rows, output)

    assert result["rows"] == 4
    assert result["steps"] == 2
    assert result["response_level_pearson"] > 0.9
    assert result["step_mean_pearson"] == pytest.approx(1.0)
    lines = output.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert "train_proxy_mean" in lines[0]
    assert "train_gt_mean" in lines[0]
