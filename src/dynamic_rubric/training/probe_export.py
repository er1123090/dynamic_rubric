from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from ..artifacts import write_jsonl_atomic


@dataclass(frozen=True)
class ProbeRecord:
    run_id: str
    prompt_id: str
    policy_step: int
    timing: str
    family: str
    sample_index: int
    seed: int
    response_id: str
    response_text: str
    checkpoint_hash: str
    config_hash: str
    base_policy: str
    static_proxy_reward: float | None = None
    kl_from_pi0: float | None = None
    response_embedding_distance: float | None = None

    def validate(self) -> None:
        if self.policy_step < 1:
            raise ValueError("pi_t probe records require t >= 1")
        if self.timing != "after_optimizer_update":
            raise ValueError("pi_t is defined only after optimizer update t")
        if self.family not in {"trajectory_discovery", "trajectory_validation"}:
            raise ValueError("probe record has an invalid response family")


def validate_probe_inventory(
    records: Iterable[ProbeRecord], prompt_count: int, step_count: int, samples_per_family: int
) -> None:
    records = list(records)
    seen_ids: set[str] = set()
    by_key: dict[tuple[int, str, str], int] = {}
    for record in records:
        record.validate()
        if record.response_id in seen_ids:
            raise ValueError(f"response_id reused: {record.response_id}")
        seen_ids.add(record.response_id)
        key = record.policy_step, record.prompt_id, record.family
        by_key[key] = by_key.get(key, 0) + 1
    expected_keys = step_count * prompt_count * 2
    if len(by_key) != expected_keys or any(
        value != samples_per_family for value in by_key.values()
    ):
        raise ValueError("probe inventory is incomplete or has duplicate samples")


def append_probe_records(path: Path, records: Sequence[ProbeRecord]) -> None:
    for record in records:
        record.validate()
    write_jsonl_atomic(path, (asdict(record) for record in records), immutable=True)


def _state_bytes(value: bytes | str | int | float) -> bytes:
    return value if isinstance(value, bytes) else str(value).encode()


def prove_probe_side_effect_free(
    train_three_steps: Callable[[bool], Mapping[str, bytes | str | int | float]],
) -> dict[str, str]:
    """Run paired 3-step smoke and compare every supplied state component."""

    without_probe = train_three_steps(False)
    with_probe = train_three_steps(True)
    if without_probe.keys() != with_probe.keys():
        raise AssertionError("probe changed the training-state inventory")
    hashes: dict[str, str] = {}
    for key in without_probe:
        left = _state_bytes(without_probe[key])
        right = _state_bytes(with_probe[key])
        if left != right:
            raise AssertionError(f"probe changed training state: {key}")
        hashes[key] = hashlib.sha256(left).hexdigest()
    return hashes
