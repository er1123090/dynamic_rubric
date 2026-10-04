"""Pure replay state transitions for all canonical rubric trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
import json
from typing import Iterable

from .admission import AdmissionDecision, AdmissionEvidence, evaluate_admission
from .pool import DynamicPoolEntry, RubricPool, make_budgeted_pool, make_cumulative_pool
from .static import Criterion, StaticRubric, validate_criterion_text


class ReplayMode(str, Enum):
    STATIC = "static"
    DYNAMIC_FIXED_BUDGETED = "dynamic_fixed_budgeted"
    DYNAMIC_PREV_BUDGETED = "dynamic_prev_budgeted"
    REFRESH_ONLY_BUDGETED = "refresh_only_budgeted"
    DYNAMIC_FIXED_CUMULATIVE = "dynamic_fixed_cumulative"


DYNAMIC_REPLAY_MODES = frozenset(
    {
        ReplayMode.DYNAMIC_FIXED_BUDGETED,
        ReplayMode.DYNAMIC_PREV_BUDGETED,
        ReplayMode.REFRESH_ONLY_BUDGETED,
        ReplayMode.DYNAMIC_FIXED_CUMULATIVE,
    }
)


@dataclass(frozen=True, slots=True)
class ReplayCandidate:
    criterion_id: str
    text: str
    evidence: AdmissionEvidence


@dataclass(frozen=True, slots=True)
class CandidateResult:
    criterion_id: str
    decision: AdmissionDecision


@dataclass(frozen=True, slots=True)
class ReplaySnapshot:
    prompt_id: str
    step: int
    mode: ReplayMode
    pool: RubricPool
    candidate_results: tuple[CandidateResult, ...] = ()
    admitted_id: str | None = None
    evicted_id: str | None = None

    @property
    def criteria(self) -> tuple[Criterion, ...]:
        return self.pool.criteria

    @property
    def content_hash(self) -> str:
        payload = [(item.criterion_id, item.text, item.source) for item in self.criteria]
        return sha256(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
        ).hexdigest()


def initial_replay_snapshot(static_rubric: StaticRubric, mode: ReplayMode | str) -> ReplaySnapshot:
    parsed_mode = ReplayMode(mode)
    if parsed_mode is ReplayMode.DYNAMIC_FIXED_CUMULATIVE:
        pool = make_cumulative_pool(static_rubric.criteria)
    else:
        pool = make_budgeted_pool(static_rubric.criteria)
    return ReplaySnapshot(static_rubric.prompt_id, 0, parsed_mode, pool)


def replay_step(
    previous: ReplaySnapshot,
    *,
    step: int,
    candidates: Iterable[ReplayCandidate] = (),
) -> ReplaySnapshot:
    """Apply one replay step and always return a new immutable snapshot.

    At most three candidates may be proposed.  Among candidates passing every
    gate, exactly one highest-utility candidate is offered to the configured
    pool.  Equal-utility ties use lexical criterion ID.  A rejection/no proposal
    records the step while preserving the previous content hash.
    """

    if step <= previous.step:
        raise ValueError("replay steps must increase monotonically")
    proposed = tuple(candidates)
    if len(proposed) > 3:
        raise ValueError("extractor may propose at most 3 candidates")
    if previous.mode is ReplayMode.STATIC:
        if proposed:
            raise ValueError("static replay cannot accept dynamic candidates")
        return ReplaySnapshot(previous.prompt_id, step, previous.mode, previous.pool)

    ids = [candidate.criterion_id for candidate in proposed]
    if len(ids) != len(set(ids)):
        raise ValueError("candidate criterion IDs must be unique within a step")
    evaluated: list[tuple[ReplayCandidate, AdmissionDecision]] = []
    for candidate in proposed:
        validate_criterion_text(candidate.text)
        evaluated.append((candidate, evaluate_admission(candidate.evidence)))
    results = tuple(
        CandidateResult(candidate.criterion_id, decision) for candidate, decision in evaluated
    )
    admissible = [(candidate, decision) for candidate, decision in evaluated if decision.admitted]
    if not admissible:
        return ReplaySnapshot(
            previous.prompt_id,
            step,
            previous.mode,
            previous.pool,
            candidate_results=results,
        )

    chosen, decision = min(admissible, key=lambda item: (-item[1].utility, item[0].criterion_id))
    criterion = Criterion(
        criterion_id=chosen.criterion_id,
        text=validate_criterion_text(chosen.text),
        source="dynamic",
        created_step=step,
    )
    update = previous.pool.add(DynamicPoolEntry(criterion, decision.utility, step))
    return ReplaySnapshot(
        previous.prompt_id,
        step,
        previous.mode,
        update.pool,
        candidate_results=results,
        admitted_id=update.admitted_id,
        evicted_id=update.evicted_id,
    )


advance_replay = replay_step
