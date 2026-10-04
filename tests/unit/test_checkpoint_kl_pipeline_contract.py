from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_checkpoint_kl_wrapper_seals_kl_before_existing_audit() -> None:
    script = (ROOT / "scripts/run_horizon_checkpoint_kl_then_audit.sh").read_text()
    analyze = script.index('compute_horizon_checkpoint_kl.py" analyze')
    seal = script.index('[[ -s "${KL_SEAL}" ]]')
    audit = script.index('bash "${PROJECT_ROOT}/scripts/run_horizon_audit.sh"')

    assert analyze < seal < audit
    assert "checkpoint_steps=(0 3 6 9 13 16 24 32 40 48)" in script
    assert '--responses-per-prompt 16' in script


def test_medicine_supervisor_routes_through_checkpoint_kl_wrapper() -> None:
    script = (ROOT / "scripts/supervise_medicine_kl_audit_then_science.sh").read_text()

    assert 'run_horizon_checkpoint_kl_then_audit.sh' in script
    assert script.index("DOMAIN=medicine") < script.index("DOMAIN=science")
