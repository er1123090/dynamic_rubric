"""Policy-version contracts for paper-faithful OnlineRubrics controls.

This module does not own model serving. It makes the reference/lagged snapshot
selection and the single pending lookahead transaction explicit so a runtime
cannot silently substitute current-policy or stale responses.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence


class ControlContractError(RuntimeError):
    pass


class ControlMode(str, Enum):
    PI_REF = "pi_ref"
    PI_OLD = "pi_old"


@dataclass(frozen=True)
class PolicySnapshot:
    update_index: int
    version: str
    content_hash: str

    def __post_init__(self) -> None:
        if self.update_index < 0:
            raise ValueError("policy update_index must be non-negative")
        if not self.version or not self.content_hash:
            raise ValueError("policy snapshot requires version and content_hash")


@dataclass(frozen=True)
class ControlSelection:
    optimizer_update_index: int
    current: PolicySnapshot
    control: PolicySnapshot
    mode: ControlMode

    def __post_init__(self) -> None:
        expected_current = self.optimizer_update_index - 1
        if self.optimizer_update_index < 1 or self.current.update_index != expected_current:
            raise ControlContractError(
                "current snapshot must be A_(t-1) for optimizer update t"
            )
        expected_control = (
            0 if self.mode is ControlMode.PI_REF else max(0, self.optimizer_update_index - 2)
        )
        if self.control.update_index != expected_control:
            raise ControlContractError(
                f"{self.mode.value} control for update {self.optimizer_update_index} "
                f"must be A_{expected_control}, got A_{self.control.update_index}"
            )


@dataclass(frozen=True)
class LookaheadBatch:
    optimizer_update_index: int
    batch_uid: str
    control_snapshot_hash: str
    response_hashes: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.optimizer_update_index < 1:
            raise ValueError("lookahead update index must be positive")
        if not self.batch_uid or not self.control_snapshot_hash:
            raise ValueError("lookahead requires batch and control identities")
        if not self.response_hashes or any(not value for value in self.response_hashes):
            raise ValueError("lookahead requires non-empty response hashes")


class ControlPolicyLedger:
    """Tracks committed actors and at most one bound next-batch control inventory."""

    def __init__(self, mode: ControlMode | str, initial: PolicySnapshot):
        self.mode = ControlMode(mode)
        if initial.update_index != 0:
            raise ControlContractError("initial control snapshot must be A_0")
        self._snapshots: dict[int, PolicySnapshot] = {0: initial}
        self._pending: LookaheadBatch | None = None

    @property
    def pending(self) -> LookaheadBatch | None:
        return self._pending

    def commit_actor(self, snapshot: PolicySnapshot) -> None:
        expected = max(self._snapshots) + 1
        if snapshot.update_index != expected:
            raise ControlContractError(
                f"actor commits must be sequential: expected A_{expected}, got A_{snapshot.update_index}"
            )
        self._snapshots[snapshot.update_index] = snapshot
        # Only A0 plus the two newest actors are needed for pi_old selection/resume.
        for index in sorted(self._snapshots):
            if index != 0 and index < snapshot.update_index - 1:
                del self._snapshots[index]

    def select(self, optimizer_update_index: int) -> ControlSelection:
        current_index = optimizer_update_index - 1
        control_index = (
            0 if self.mode is ControlMode.PI_REF else max(0, optimizer_update_index - 2)
        )
        try:
            current = self._snapshots[current_index]
            control = self._snapshots[control_index]
        except KeyError as exc:
            raise ControlContractError(
                f"required policy snapshot A_{exc.args[0]} is not committed"
            ) from exc
        return ControlSelection(
            optimizer_update_index=optimizer_update_index,
            current=current,
            control=control,
            mode=self.mode,
        )

    def bind_lookahead(
        self,
        *,
        optimizer_update_index: int,
        batch_uid: str,
        control_snapshot_hash: str,
        response_hashes: Sequence[str],
    ) -> LookaheadBatch:
        if self._pending is not None:
            raise ControlContractError("only one lookahead batch may be pending")
        selection = self.select(optimizer_update_index)
        if selection.control.content_hash != control_snapshot_hash:
            raise ControlContractError("lookahead responses are bound to the wrong control snapshot")
        pending = LookaheadBatch(
            optimizer_update_index=optimizer_update_index,
            batch_uid=batch_uid,
            control_snapshot_hash=control_snapshot_hash,
            response_hashes=tuple(response_hashes),
        )
        self._pending = pending
        return pending

    def consume_lookahead(
        self, *, optimizer_update_index: int, batch_uid: str
    ) -> LookaheadBatch:
        pending = self._pending
        if pending is None:
            raise ControlContractError("no lookahead batch is pending")
        if (
            pending.optimizer_update_index != optimizer_update_index
            or pending.batch_uid != batch_uid
        ):
            raise ControlContractError("lookahead batch identity does not match the current step")
        self._pending = None
        return pending

    def restore_pending(self, value: Mapping[str, object]) -> LookaheadBatch:
        if self._pending is not None:
            raise ControlContractError("cannot restore over an existing pending lookahead")
        response_hashes = value.get("response_hashes")
        if not isinstance(response_hashes, (list, tuple)):
            raise ControlContractError("restored lookahead response_hashes must be a sequence")
        pending = LookaheadBatch(
            optimizer_update_index=int(value["optimizer_update_index"]),
            batch_uid=str(value["batch_uid"]),
            control_snapshot_hash=str(value["control_snapshot_hash"]),
            response_hashes=tuple(str(item) for item in response_hashes),
        )
        selection = self.select(pending.optimizer_update_index)
        if pending.control_snapshot_hash != selection.control.content_hash:
            raise ControlContractError("restored lookahead uses the wrong control snapshot")
        self._pending = pending
        return pending
