"""Deterministic, source-blind response pairing for criterion extraction."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import random
from typing import Sequence


MAX_EXTRACTED_CANDIDATES = 3


@dataclass(frozen=True, slots=True)
class BlindPair:
    pair_id: str
    response_a: str
    response_b: str


@dataclass(frozen=True, slots=True)
class PairAssignment:
    """Audit-only assignment kept separate from generator-facing ``BlindPair``."""

    pair_id: str
    current_label: str
    control_label: str
    current_index: int
    control_index: int


@dataclass(frozen=True, slots=True)
class PairingPlan:
    blind_pairs: tuple[BlindPair, ...]
    assignments: tuple[PairAssignment, ...]

    def generator_payload(self) -> tuple[BlindPair, ...]:
        """Return data that contains no current/control source labels."""

        return self.blind_pairs


def _derive_seed(seed: int | str, prompt_id: str, step: int) -> int:
    material = f"{seed}\x1f{prompt_id}\x1f{step}\x1fpairing-v1".encode()
    return int.from_bytes(sha256(material).digest()[:16], "big")


def make_blind_pairing(
    current_responses: Sequence[str],
    control_responses: Sequence[str],
    *,
    seed: int | str,
    prompt_id: str,
    step: int,
) -> PairingPlan:
    """Pair two equal-sized response sets and deterministically blind every A/B side."""

    current = tuple(current_responses)
    control = tuple(control_responses)
    if not current or len(current) != len(control):
        raise ValueError("current and control responses must have the same non-zero length")
    if step < 0:
        raise ValueError("step must be non-negative")
    rng = random.Random(_derive_seed(seed, prompt_id, step))
    control_order = list(range(len(control)))
    rng.shuffle(control_order)
    if len(current) == 8:
        current_is_a_order = [True] * 4 + [False] * 4
        rng.shuffle(current_is_a_order)
    else:
        current_is_a_order = [bool(rng.getrandbits(1)) for _ in current]
    blind_pairs: list[BlindPair] = []
    assignments: list[PairAssignment] = []
    for current_index, (control_index, current_is_a) in enumerate(
        zip(control_order, current_is_a_order, strict=True)
    ):
        pair_id = sha256(
            f"{prompt_id}\x1f{step}\x1f{current_index}\x1f{control_index}".encode()
        ).hexdigest()[:20]
        if current_is_a:
            response_a, response_b = current[current_index], control[control_index]
            current_label, control_label = "A", "B"
        else:
            response_a, response_b = control[control_index], current[current_index]
            current_label, control_label = "B", "A"
        blind_pairs.append(BlindPair(pair_id, response_a, response_b))
        assignments.append(
            PairAssignment(pair_id, current_label, control_label, current_index, control_index)
        )
    return PairingPlan(tuple(blind_pairs), tuple(assignments))


pair_responses_blind = make_blind_pairing
