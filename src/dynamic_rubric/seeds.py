"""Deterministic RNG namespaces and globally stable response identifiers."""

from __future__ import annotations

import hashlib
from enum import Enum

from .hashing import canonical_json_bytes


class SeedFamily(str, Enum):
    STATIC_CANDIDATE = "static_candidate"
    REFERENCE_DISCOVERY = "reference_discovery"
    REFERENCE_VALIDATION = "reference_validation"
    TRAJECTORY_DISCOVERY = "trajectory_discovery"
    TRAJECTORY_VALIDATION = "trajectory_validation"
    AUDIT_BON = "audit_bon"
    TRAINING = "training"
    BOOTSTRAP = "bootstrap"
    PAIRING = "pairing"
    HORIZON_FIXED_CONTROL = "horizon_fixed_control"
    HORIZON_SHAM_CONTROL = "horizon_sham_control"
    HORIZON_POOL_A = "horizon_pool_a"
    HORIZON_POOL_B = "horizon_pool_b"


RESPONSE_FAMILIES = tuple(
    family for family in SeedFamily if family.value not in {"training", "bootstrap", "pairing"}
)
_NAMESPACE_INDEX = {family: index + 1 for index, family in enumerate(SeedFamily)}
_DIGEST_BITS = 128
SEED_NAMESPACE_SIZE = 1 << _DIGEST_BITS
SEED_RANGES = {
    family: (index * SEED_NAMESPACE_SIZE, (index + 1) * SEED_NAMESPACE_SIZE - 1)
    for family, index in _NAMESPACE_INDEX.items()
}

# vLLM forwards seeds through implementations whose portable supported range is
# the non-negative unsigned 32-bit interval. Logical seeds remain full-width.
VLLM_SEED_MAX = (1 << 32) - 1


def _family(value: SeedFamily | str) -> SeedFamily:
    try:
        return value if isinstance(value, SeedFamily) else SeedFamily(value)
    except ValueError as exc:
        raise ValueError(f"unknown seed family: {value!r}") from exc


def derive_seed(
    run_id: str,
    family: SeedFamily | str,
    prompt_id: str,
    policy_step: int,
    sample_index: int,
) -> int:
    """Derive a deterministic integer whose high bits identify its family."""

    family = _family(family)
    if not run_id or not prompt_id:
        raise ValueError("run_id and prompt_id must be non-empty")
    if policy_step < 0 or sample_index < 0:
        raise ValueError("policy_step and sample_index must be non-negative")
    payload = [run_id, family.value, prompt_id, policy_step, sample_index]
    suffix = int.from_bytes(hashlib.sha256(canonical_json_bytes(payload)).digest()[:16], "big")
    return _NAMESPACE_INDEX[family] * SEED_NAMESPACE_SIZE + suffix


def seed_namespace(seed: int) -> SeedFamily:
    if seed < 0:
        raise ValueError("seed must be non-negative")
    index = seed // SEED_NAMESPACE_SIZE
    for family, candidate in _NAMESPACE_INDEX.items():
        if candidate == index:
            return family
    raise ValueError("seed is outside registered namespaces")


def vllm_seed(logical_seed: int) -> int:
    """Fold a logical seed into vLLM [0, 2**32 - 1] range while callers retain both values."""

    if logical_seed < 0:
        raise ValueError("logical_seed must be non-negative")
    return logical_seed % (VLLM_SEED_MAX + 1)


def response_id(
    run_id: str,
    family: SeedFamily | str,
    prompt_id: str,
    policy_step: int,
    sample_index: int,
) -> str:
    family = _family(family)
    seed = derive_seed(run_id, family, prompt_id, policy_step, sample_index)
    digest = hashlib.sha256(
        canonical_json_bytes([run_id, family.value, prompt_id, policy_step, sample_index, seed])
    ).hexdigest()
    return f"resp_{family.value}_{digest}"


deterministic_seed = derive_seed
deterministic_response_id = response_id
