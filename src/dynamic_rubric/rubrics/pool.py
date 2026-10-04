"""Immutable budgeted and cumulative rubric pools."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .static import Criterion, STATIC_CRITERIA_COUNT


MAX_DYNAMIC_CRITERIA = 4
MAX_BUDGETED_CRITERIA = STATIC_CRITERIA_COUNT + MAX_DYNAMIC_CRITERIA


@dataclass(frozen=True, slots=True)
class DynamicPoolEntry:
    criterion: Criterion
    utility: float
    last_validated_step: int

    def __post_init__(self) -> None:
        if self.criterion.source != "dynamic":
            raise ValueError("dynamic pool entries must have dynamic provenance")
        if not math.isfinite(self.utility):
            raise ValueError("utility must be finite")
        if self.last_validated_step < 0:
            raise ValueError("last_validated_step must be non-negative")


@dataclass(frozen=True, slots=True)
class PoolUpdate:
    pool: "RubricPool"
    admitted_id: str | None
    evicted_id: str | None


@dataclass(frozen=True, slots=True)
class RubricPool:
    static_criteria: tuple[Criterion, ...]
    dynamic_entries: tuple[DynamicPoolEntry, ...] = ()
    cumulative: bool = False

    def __post_init__(self) -> None:
        if len(self.static_criteria) != STATIC_CRITERIA_COUNT:
            raise ValueError("pool must permanently preserve exactly 8 static criteria")
        if any(item.source == "dynamic" for item in self.static_criteria):
            raise ValueError("static criteria cannot have dynamic provenance")
        ids = [item.criterion_id for item in self.static_criteria]
        ids.extend(entry.criterion.criterion_id for entry in self.dynamic_entries)
        if len(ids) != len(set(ids)):
            raise ValueError("criterion IDs must be unique")
        if not self.cumulative and len(self.dynamic_entries) > MAX_DYNAMIC_CRITERIA:
            raise ValueError("budgeted pool cannot exceed 12 total criteria")

    @property
    def criteria(self) -> tuple[Criterion, ...]:
        return self.static_criteria + tuple(entry.criterion for entry in self.dynamic_entries)

    def add(self, entry: DynamicPoolEntry) -> PoolUpdate:
        """Return a new pool; budgeted eviction is deterministic and static-safe."""

        if any(item.criterion_id == entry.criterion.criterion_id for item in self.static_criteria):
            raise ValueError("dynamic criterion ID collides with a static criterion")
        if any(
            item.criterion.criterion_id == entry.criterion.criterion_id
            for item in self.dynamic_entries
        ):
            raise ValueError("dynamic criterion IDs are immutable and cannot be reused")
        candidates = self.dynamic_entries + (entry,)
        if self.cumulative or len(candidates) <= MAX_DYNAMIC_CRITERIA:
            return PoolUpdate(
                RubricPool(self.static_criteria, candidates, self.cumulative),
                entry.criterion.criterion_id,
                None,
            )

        evicted = min(
            candidates,
            key=lambda item: (item.utility, item.last_validated_step, item.criterion.criterion_id),
        )
        retained = tuple(item for item in candidates if item is not evicted)
        admitted_id = None if evicted is entry else entry.criterion.criterion_id
        return PoolUpdate(
            RubricPool(self.static_criteria, retained, cumulative=False),
            admitted_id,
            evicted.criterion.criterion_id,
        )


def make_budgeted_pool(static_criteria: tuple[Criterion, ...]) -> RubricPool:
    return RubricPool(static_criteria=tuple(static_criteria), cumulative=False)


def make_cumulative_pool(static_criteria: tuple[Criterion, ...]) -> RubricPool:
    return RubricPool(static_criteria=tuple(static_criteria), cumulative=True)
