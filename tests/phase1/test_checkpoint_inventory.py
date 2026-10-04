from pathlib import Path

from dynamic_rubric.phase1.checkpoint_inventory import (
    comparison_plan,
    discover_committed_policy_checkpoints,
)
from scripts.phase1.plan_all_saved_checkpoint_analysis import _read_jsonl_rows, build_plan


def _checkpoint(root: Path, step: int, *, model: bool = True) -> None:
    actor = root / "verl-run" / "checkpoints" / f"global_step_{step}" / "actor"
    actor.mkdir(parents=True)
    (actor / "fsdp_config.json").write_text('{"world_size": 1}\n')
    if model:
        (actor / "model_world_size_1_rank_0.pt").write_bytes(b"parameters")


def test_discovers_all_committed_checkpoints_in_numeric_order(tmp_path: Path) -> None:
    for step in (40, 3, 34, 0, 12):
        _checkpoint(tmp_path, step)
    tracker = tmp_path / "verl-run" / "checkpoints" / "latest_checkpointed_iteration.txt"
    tracker.write_text("40\n")

    inventory = discover_committed_policy_checkpoints(tmp_path)

    assert inventory["steps"] == [0, 3, 12, 34, 40]


def test_excludes_uncommitted_and_missing_parameter_directories(tmp_path: Path) -> None:
    _checkpoint(tmp_path, 0)
    _checkpoint(tmp_path, 3, model=False)
    _checkpoint(tmp_path, 6)
    tracker = tmp_path / "verl-run" / "checkpoints" / "latest_checkpointed_iteration.txt"
    tracker.write_text("3\n")

    inventory = discover_committed_policy_checkpoints(tmp_path)

    assert inventory["steps"] == [0]
    assert [(row["step"], row["exclusion_reason"]) for row in inventory["excluded"]] == [
        (3, "missing_nonempty_model_parameter_shards"),
        (6, "newer_than_commit_tracker"),
    ]


def test_comparison_plan_uses_all_adjacent_and_triangular_cells() -> None:
    plan = comparison_plan([0, 3, 34, 40])

    assert plan["adjacent"] == [
        {"stale_evaluator_step": 0, "fresh_policy_evaluator_step": 3},
        {"stale_evaluator_step": 3, "fresh_policy_evaluator_step": 34},
        {"stale_evaluator_step": 34, "fresh_policy_evaluator_step": 40},
    ]
    assert len(plan["reuse_horizon_triangle"]) == 10
    assert {"evaluator_step": 34, "policy_step": 40} in plan["reuse_horizon_triangle"]


def test_plan_marks_missing_probe_coverage_instead_of_omitting(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    audit_root = tmp_path / "audit"
    for step in (0, 34, 40):
        _checkpoint(run_root, step)
    tracker = run_root / "verl-run" / "checkpoints" / "latest_checkpointed_iteration.txt"
    tracker.write_text("40\n")
    complete = audit_root / "responses" / "checkpoint-000034"
    complete.mkdir(parents=True)
    (complete / "probe_A.jsonl").write_text("{}\n")
    (complete / "probe_B.jsonl").write_text("{}\n")

    plan = build_plan(run_root, audit_root)

    assert [row["step"] for row in plan["coverage"]] == [0, 34, 40]
    statuses = {row["step"]: row["response_generation_status"] for row in plan["coverage"]}
    assert statuses == {
        0: "planned_not_completed",
        34: "files_present_unvalidated",
        40: "planned_not_completed",
    }
    assert plan["policy_outcome_evaluation"]["timing_analysis_role"].startswith("forbidden")
    export = plan["policy_audit_commands"]["export_all_discovered_checkpoints"]
    assert export[export.index("--steps") + 1 :] == ["0", "34", "40"]


def test_jsonl_reader_preserves_unicode_line_separator_inside_response(tmp_path: Path) -> None:
    path = tmp_path / "responses.jsonl"
    path.write_text('{"response_text":"left\u2028right"}\n')

    assert _read_jsonl_rows(path) == [{"response_text": "left\u2028right"}]
