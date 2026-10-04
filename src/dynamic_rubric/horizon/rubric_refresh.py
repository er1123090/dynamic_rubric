"""Pure construction helpers for non-cumulative checkpoint rubric refreshes."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import re
from typing import Iterable, Mapping, Sequence

from dynamic_rubric.horizon.contracts import (
    CriterionType,
    HorizonContractError,
    ImportanceClass,
    WEIGHT_UNITS,
    WeightedCriterion,
    criterion_content_hash,
    make_criterion_instance_id,
)


MAX_ONLINE_CRITERIA = 8
_SPACE_RE = re.compile(r"\s+")
_POLICY_IDENTITY_RE = re.compile(
    r"\b(?:current|baseline|trained|control)\s+(?:policy|model|response)|"
    r"\bcheckpoint\b|\bpi[_ ]?\d+\b",
    re.IGNORECASE,
)
_NEGATIVE_PREFIX_RE = re.compile(
    r"^(?:no\b|not\b|never\b|avoid\b|omit\b|fail(?:s)?\s+to\b|"
    r"must\s+not\b|should\s+not\b|does\s+not\b)",
    re.IGNORECASE,
)
_NON_ATOMIC_RE = re.compile(r"(?:;|\n|\band\b|\bor\b|\bas well as\b)", re.IGNORECASE)


class RubricRefreshError(HorizonContractError):
    pass


@dataclass(frozen=True, slots=True)
class ExtractionCandidate:
    candidate_id: str
    criterion: str
    evidence_quote: str
    source_pair_id: str
    source_checkpoint: str
    raw_paper_weight: int
    importance_class: ImportanceClass
    criterion_type: CriterionType
    response_a: str
    response_b: str

    def __post_init__(self) -> None:
        if any(
            not value.strip()
            for value in (
                self.candidate_id,
                self.criterion,
                self.evidence_quote,
                self.source_pair_id,
                self.source_checkpoint,
            )
        ):
            raise RubricRefreshError("candidate identity, text, evidence, and lineage are required")
        if (
            isinstance(self.raw_paper_weight, bool)
            or not isinstance(self.raw_paper_weight, int)
            or self.raw_paper_weight < 1
        ):
            raise RubricRefreshError("raw_paper_weight must be a positive integer")
        if self.criterion_type is CriterionType.PITFALL:
            if self.importance_class is ImportanceClass.PITFALL:
                raise RubricRefreshError(
                    "extractor importance_class excludes pitfall; criterion_type carries semantics"
                )
        elif self.importance_class is ImportanceClass.PITFALL:
            raise RubricRefreshError("quality candidates cannot declare pitfall importance")


@dataclass(frozen=True, slots=True)
class DedupCluster:
    criterion: str
    source_candidate_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.criterion.strip() or not self.source_candidate_ids:
            raise RubricRefreshError("dedup cluster requires criterion and source candidates")
        if len(set(self.source_candidate_ids)) != len(self.source_candidate_ids):
            raise RubricRefreshError("dedup source_candidate_ids must be unique")


@dataclass(frozen=True, slots=True)
class Resolution:
    accepted: bool
    criterion: WeightedCriterion | None
    reject_reason: str | None
    class_support: tuple[tuple[str, int], ...] = ()
    tie_break_trace: tuple[str, ...] = ()


def _normalized_evidence(value: str) -> str:
    return _SPACE_RE.sub(" ", value).strip().casefold()


def evidence_is_grounded(candidate: ExtractionCandidate) -> bool:
    quote = _normalized_evidence(candidate.evidence_quote)
    return bool(quote) and any(
        quote in _normalized_evidence(response)
        for response in (candidate.response_a, candidate.response_b)
    )


def _class_tie_hash(importance: ImportanceClass, candidates: Sequence[ExtractionCandidate]) -> str:
    ids = sorted(
        candidate.candidate_id
        for candidate in candidates
        if candidate.importance_class is importance
    )
    return sha256("\x1f".join(ids).encode("utf-8")).hexdigest()


def resolve_dedup_cluster(
    cluster: DedupCluster,
    candidates_by_id: Mapping[str, ExtractionCandidate],
    *,
    prompt_id: str,
    checkpoint_id: str,
) -> Resolution:
    """Resolve class/type/weight from source candidates, never from dedup output metadata."""

    missing = set(cluster.source_candidate_ids).difference(candidates_by_id)
    if missing:
        raise RubricRefreshError(f"unknown source candidate IDs: {sorted(missing)}")
    sources = tuple(candidates_by_id[item] for item in cluster.source_candidate_ids)
    if any(source.source_checkpoint != checkpoint_id for source in sources):
        return Resolution(False, None, "prior_checkpoint_lineage")

    # Dedup may select/copy a source criterion but may not invent a new one.
    cluster_hash = criterion_content_hash(cluster.criterion)
    if cluster_hash not in {criterion_content_hash(source.criterion) for source in sources}:
        return Resolution(False, None, "dedup_new_criterion")

    types = {source.criterion_type for source in sources}
    if len(types) != 1:
        return Resolution(False, None, "mixed_semantics")

    support_counter: Counter[ImportanceClass] = Counter()
    for importance in {source.importance_class for source in sources}:
        support_counter[importance] = len(
            {source.source_pair_id for source in sources if source.importance_class is importance}
        )

    trace: list[str] = []
    criterion_type = next(iter(types))
    if criterion_type is CriterionType.PITFALL:
        if _NEGATIVE_PREFIX_RE.search(cluster.criterion.strip()):
            return Resolution(False, None, "not_positive_form")
        importance = ImportanceClass.PITFALL
        trace.append("all_sources_pitfall:weight=9")
    else:
        ranked = sorted(
            support_counter,
            key=lambda item: (
                -support_counter[item],
                -WEIGHT_UNITS[item],
                _class_tie_hash(item, sources),
            ),
        )
        importance = ranked[0]
        trace.extend(
            (
                f"support={support_counter[importance]}",
                f"weight_tiebreak={WEIGHT_UNITS[importance]}",
                f"candidate_hash_tiebreak={_class_tie_hash(importance, sources)}",
            )
        )

    raw_weight = max(source.raw_paper_weight for source in sources)
    criterion = WeightedCriterion(
        criterion_instance_id=make_criterion_instance_id(
            prompt_id=prompt_id,
            checkpoint_id=checkpoint_id,
            canonical_criterion_hash=cluster_hash,
        ),
        canonical_criterion_hash=cluster_hash,
        text=_SPACE_RE.sub(" ", cluster.criterion).strip(),
        importance_class=importance,
        criterion_type=criterion_type,
        weight_units=WEIGHT_UNITS[importance],
        source_candidate_ids=tuple(sorted(cluster.source_candidate_ids)),
        source_checkpoint=checkpoint_id,
        raw_paper_weight=raw_weight,
        distinct_source_pair_support=len({source.source_pair_id for source in sources}),
    )
    return Resolution(
        accepted=True,
        criterion=criterion,
        reject_reason=None,
        class_support=tuple(sorted((key.value, value) for key, value in support_counter.items())),
        tie_break_trace=tuple(trace),
    )


def criterion_filter_reason(
    resolution: Resolution,
    candidates_by_id: Mapping[str, ExtractionCandidate],
    *,
    r0_texts: Iterable[str] = (),
) -> str | None:
    if not resolution.accepted or resolution.criterion is None:
        return resolution.reject_reason or "unresolved"
    criterion = resolution.criterion
    sources = tuple(candidates_by_id[item] for item in criterion.source_candidate_ids)
    if any(not evidence_is_grounded(source) for source in sources):
        return "invalid_evidence"
    if criterion.canonical_criterion_hash in {criterion_content_hash(text) for text in r0_texts}:
        return "duplicate_r0"
    text = criterion.text.strip()
    if len(text.split()) < 2 or len(text) > 500:
        return "not_binary_judgeable"
    if _NON_ATOMIC_RE.search(text):
        return "not_atomic"
    if _NEGATIVE_PREFIX_RE.search(text):
        return "not_positive_form"
    if _POLICY_IDENTITY_RE.search(text):
        return "policy_identity_leak"
    return None


def cap_criteria(
    criteria: Iterable[WeightedCriterion], *, max_count: int = MAX_ONLINE_CRITERIA
) -> tuple[tuple[WeightedCriterion, ...], tuple[WeightedCriterion, ...]]:
    if max_count < 0:
        raise ValueError("max_count must be non-negative")
    values = tuple(criteria)

    def priority(item: WeightedCriterion) -> tuple[int, int, int, str]:
        support = item.distinct_source_pair_support
        return (
            -support,
            -item.weight_units,
            -(item.raw_paper_weight or 0),
            item.canonical_criterion_hash,
        )

    ranked = tuple(sorted(values, key=priority))
    return ranked[:max_count], ranked[max_count:]


def validate_current_checkpoint_lineage(
    criteria: Iterable[WeightedCriterion],
    *,
    checkpoint_id: str,
    current_candidate_ids: Iterable[str],
) -> None:
    allowed = set(current_candidate_ids)
    for criterion in criteria:
        if criterion.source_checkpoint != checkpoint_id:
            raise RubricRefreshError("prior_checkpoint_lineage")
        if not criterion.source_candidate_ids or not set(criterion.source_candidate_ids) <= allowed:
            raise RubricRefreshError("current criterion lineage is outside current Pool A")


def build_current_extension(
    resolutions: Iterable[Resolution],
    candidates_by_id: Mapping[str, ExtractionCandidate],
    *,
    checkpoint_id: str,
    r0_texts: Iterable[str] = (),
    max_count: int = MAX_ONLINE_CRITERIA,
) -> tuple[tuple[WeightedCriterion, ...], tuple[tuple[str, str], ...]]:
    accepted: list[WeightedCriterion] = []
    rejected: list[tuple[str, str]] = []
    seen_hashes: set[str] = set()
    frozen_r0_texts = tuple(r0_texts)
    for resolution in resolutions:
        criterion = resolution.criterion
        identity = criterion.criterion_instance_id if criterion is not None else "unresolved"
        reason = criterion_filter_reason(resolution, candidates_by_id, r0_texts=frozen_r0_texts)
        if reason is None and criterion is not None:
            if criterion.canonical_criterion_hash in seen_hashes:
                rejected.append((identity, "duplicate_candidate"))
                continue
            seen_hashes.add(criterion.canonical_criterion_hash)
            accepted.append(criterion)
        else:
            rejected.append((identity, reason or "unresolved"))
    kept, overflow = cap_criteria(accepted, max_count=max_count)
    rejected.extend((criterion.criterion_instance_id, "over_cap") for criterion in overflow)
    validate_current_checkpoint_lineage(
        kept,
        checkpoint_id=checkpoint_id,
        current_candidate_ids=candidates_by_id,
    )
    return kept, tuple(rejected)
