from __future__ import annotations

import pytest

from dynamic_rubric.training.control_policy import (
    ControlContractError,
    ControlMode,
    ControlPolicyLedger,
    PolicySnapshot,
)


def snapshot(index: int) -> PolicySnapshot:
    return PolicySnapshot(update_index=index, version=f"A{index}", content_hash=f"hash-{index}")


def test_pi_ref_always_selects_a0() -> None:
    ledger = ControlPolicyLedger(ControlMode.PI_REF, snapshot(0))
    assert ledger.select(1).control.update_index == 0
    ledger.commit_actor(snapshot(1))
    assert ledger.select(2).control.update_index == 0


def test_pi_old_is_exactly_one_committed_update_behind_current() -> None:
    ledger = ControlPolicyLedger(ControlMode.PI_OLD, snapshot(0))
    first = ledger.select(1)
    assert first.current.update_index == first.control.update_index == 0
    ledger.commit_actor(snapshot(1))
    second = ledger.select(2)
    assert second.current.update_index == 1
    assert second.control.update_index == 0
    ledger.commit_actor(snapshot(2))
    third = ledger.select(3)
    assert third.current.update_index == 2
    assert third.control.update_index == 1


def test_lookahead_is_bounded_and_identity_bound() -> None:
    ledger = ControlPolicyLedger(ControlMode.PI_OLD, snapshot(0))
    pending = ledger.bind_lookahead(
        optimizer_update_index=1,
        batch_uid="batch-1",
        control_snapshot_hash="hash-0",
        response_hashes=("r0", "r1"),
    )
    assert pending.batch_uid == "batch-1"
    with pytest.raises(ControlContractError, match="only one"):
        ledger.bind_lookahead(
            optimizer_update_index=1,
            batch_uid="batch-2",
            control_snapshot_hash="hash-0",
            response_hashes=("r2",),
        )
    with pytest.raises(ControlContractError, match="identity"):
        ledger.consume_lookahead(optimizer_update_index=1, batch_uid="wrong")
    assert ledger.consume_lookahead(
        optimizer_update_index=1, batch_uid="batch-1"
    ) == pending


def test_wrong_control_snapshot_fails_closed() -> None:
    ledger = ControlPolicyLedger(ControlMode.PI_REF, snapshot(0))
    with pytest.raises(ControlContractError, match="wrong control"):
        ledger.bind_lookahead(
            optimizer_update_index=1,
            batch_uid="batch-1",
            control_snapshot_hash="not-a0",
            response_hashes=("r0",),
        )
