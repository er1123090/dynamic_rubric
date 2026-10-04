from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_medicine_audit_is_a_hard_gate_before_science() -> None:
    script = (ROOT / "scripts" / "supervise_medicine_audit_then_science.sh").read_text()
    audit = script.index("run_horizon_audit.sh")
    audit_marker = script.index('[[ ! -s "${MEDICINE_AUDIT_MARKER}" ]]')
    science = script.index("DOMAIN=science")
    assert audit < audit_marker < science
    assert "audit_and_science_not_started" in script


def test_audit_uses_100_prompts_and_all_ten_checkpoints() -> None:
    script = (ROOT / "scripts" / "run_horizon_audit.sh").read_text()
    assert "expected_prompt_count=100" in script
    assert "checkpoint_steps=(0 3 6 9 13 16 24 32 40 48)" in script
    assert "JUDGE_BASE_URL=${JUDGE_BASE_URL:-http://127.0.0.1:8102}" in script
    assert "OPENAI_API_KEY is required for GPT-5-mini rubric extraction" in script


def test_pool_a_auxiliary_is_checkpoint_addressable_and_stays_separate() -> None:
    audit = (ROOT / "scripts" / "run_horizon_audit.sh").read_text()
    auxiliary = (ROOT / "scripts" / "run_horizon_pool_a_auxiliary.sh").read_text()

    assert audit.rindex("finalize_audit") < audit.rindex("run_pool_a_auxiliary")
    assert "RUN_POOL_A_AUXILIARY=${RUN_POOL_A_AUXILIARY:-1}" in audit
    assert "checkpoint_steps=(3 6 9 13 16 24 32 40 48)" in auxiliary
    assert "checkpoint_epochs=(0.2 0.4 0.6 0.8 1.0 1.5 2.0 2.5 3.0)" in auxiliary
    assert "TARGET_STEP=${TARGET_STEP:-}" in auxiliary
    assert "--pool-a" in auxiliary
    assert "--r0-current-only" in auxiliary
    assert "scores_pool_a_auxiliary" in auxiliary


def test_parallel_rubric_runner_is_completion_driven() -> None:
    script = (ROOT / "scripts" / "run_horizon_parallel_rubric_audit.sh").read_text()
    assert "checkpoint_steps=(3 6 9 13 16 24 32 40 48)" in script
    assert "launching ${#checkpoint_steps[@]} checkpoint rubric builders concurrently" in script
    assert "step ${step} rubric ready; launched Pool A then immediate Pool B R0-vs-Rt grading" in script
    assert "attach-horizon-controls" in script
    assert "--reuse-score-dir" in script
    assert "JUDGE_WORKERS=${JUDGE_WORKERS:-2}" in script


def test_stale_control_runner_reuses_current_grades_across_three_judges() -> None:
    script = (ROOT / "scripts" / "run_horizon_stale_control_parallel.sh").read_text()

    assert '"${INFERENCE_A_TP2_URL:-http://127.0.0.1:8102}"' in script
    assert '"${INFERENCE_A_GPU2_URL:-http://127.0.0.1:8104}"' in script
    assert '"${INFERENCE_C_URL:-http://127.0.0.1:8103}"' in script
    assert '"6 13 32"' in script
    assert '"9 24"' in script
    assert '"48"' in script
    assert "--reuse-score-dir" in script
    assert "--r0-current-only" not in script
    assert "stale-control.complete" in script
