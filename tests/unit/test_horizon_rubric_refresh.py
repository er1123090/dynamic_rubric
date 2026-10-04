from __future__ import annotations

import pytest

from dynamic_rubric.horizon.contracts import CriterionType, ImportanceClass
from dynamic_rubric.horizon.rubric_refresh import (
    DedupCluster,
    ExtractionCandidate,
    RubricRefreshError,
    build_current_extension,
    cap_criteria,
    criterion_filter_reason,
    resolve_dedup_cluster,
    validate_current_checkpoint_lineage,
)


def candidate(
    identity: str,
    *,
    text: str = "States a concrete contraindication",
    checkpoint: str = "step3",
    pair: str = "pair0",
    importance: ImportanceClass = ImportanceClass.IMPORTANT,
    criterion_type: CriterionType = CriterionType.QUALITY,
    weight: int = 2,
    quote: str = "kidney disease",
) -> ExtractionCandidate:
    return ExtractionCandidate(
        candidate_id=identity,
        criterion=text,
        evidence_quote=quote,
        source_pair_id=pair,
        source_checkpoint=checkpoint,
        raw_paper_weight=weight,
        importance_class=importance,
        criterion_type=criterion_type,
        response_a="The patient has kidney disease.",
        response_b="No medical history was supplied.",
    )


def resolve(*items: ExtractionCandidate, text: str | None = None, checkpoint: str = "step3"):
    mapping = {item.candidate_id: item for item in items}
    cluster = DedupCluster(text or items[0].criterion, tuple(mapping))
    return resolve_dedup_cluster(cluster, mapping, prompt_id="p", checkpoint_id=checkpoint), mapping


def test_resolver_uses_distinct_pair_support_then_weight_and_keeps_paper_weight_provenance() -> (
    None
):
    items = (
        candidate("a", pair="pair0", importance=ImportanceClass.IMPORTANT, weight=9),
        candidate("b", pair="pair0", importance=ImportanceClass.IMPORTANT, weight=5),
        candidate("c", pair="pair1", importance=ImportanceClass.OPTIONAL, weight=12),
        candidate("d", pair="pair2", importance=ImportanceClass.OPTIONAL, weight=4),
    )
    result, _ = resolve(*items)
    assert result.accepted and result.criterion is not None
    assert result.criterion.importance_class is ImportanceClass.OPTIONAL
    assert result.criterion.weight_units == 3
    assert result.criterion.raw_paper_weight == 12
    assert result.criterion.distinct_source_pair_support == 3
    assert dict(result.class_support) == {"important": 1, "optional": 2}


def test_quality_class_support_tie_prefers_higher_weight_units() -> None:
    result, _ = resolve(
        candidate("a", pair="pair0", importance=ImportanceClass.OPTIONAL),
        candidate("b", pair="pair1", importance=ImportanceClass.ESSENTIAL),
    )
    assert result.criterion is not None
    assert result.criterion.importance_class is ImportanceClass.ESSENTIAL


def test_mixed_semantics_is_rejected_and_positive_pitfall_maps_to_nine() -> None:
    mixed, _ = resolve(
        candidate("quality"),
        candidate("pitfall", criterion_type=CriterionType.PITFALL),
    )
    assert not mixed.accepted and mixed.reject_reason == "mixed_semantics"

    pitfall, mapping = resolve(
        candidate(
            "pitfall",
            text="Avoids recommending a contraindicated drug",
            criterion_type=CriterionType.PITFALL,
        )
    )
    assert pitfall.criterion is not None
    assert pitfall.criterion.importance_class is ImportanceClass.PITFALL
    assert pitfall.criterion.weight_units == 9
    assert criterion_filter_reason(pitfall, mapping) is None


def test_dedup_cannot_invent_criterion_or_import_prior_lineage() -> None:
    item = candidate("a")
    invented, _ = resolve(item, text="Introduces an unseen criterion")
    assert invented.reject_reason == "dedup_new_criterion"
    stale, _ = resolve(candidate("old", checkpoint="step2"))
    assert stale.reject_reason == "prior_checkpoint_lineage"


def test_evidence_filter_and_deterministic_cap_do_not_read_pool_b() -> None:
    bad = candidate("bad", quote="not in either response")
    result, mapping = resolve(bad)
    assert criterion_filter_reason(result, mapping) == "invalid_evidence"

    resolved = []
    all_candidates = {}
    for index in range(10):
        text = f"States concrete finding {index}"
        item = candidate(
            f"c{index}",
            text=text,
            pair=f"pair{index}",
            importance=ImportanceClass.ESSENTIAL if index == 9 else ImportanceClass.OPTIONAL,
            weight=index + 1,
        )
        value, mapping = resolve(item)
        resolved.append(value)
        all_candidates.update(mapping)
    accepted, rejected = build_current_extension(
        resolved, all_candidates, checkpoint_id="step3", max_count=8
    )
    assert len(accepted) == 8
    assert sum(reason == "over_cap" for _, reason in rejected) == 2
    assert accepted[0].importance_class is ImportanceClass.ESSENTIAL
    assert cap_criteria(reversed(accepted), max_count=8)[0] == tuple(accepted)


def test_same_content_rediscovery_is_allowed_but_lineage_must_be_current() -> None:
    old, _ = resolve(candidate("old", checkpoint="step2"), checkpoint="step2")
    current_item = candidate("new", checkpoint="step3")
    current, mapping = resolve(current_item, checkpoint="step3")
    assert old.criterion is not None and current.criterion is not None
    assert old.criterion.canonical_criterion_hash == current.criterion.canonical_criterion_hash
    assert old.criterion.criterion_instance_id != current.criterion.criterion_instance_id
    accepted, rejected = build_current_extension((current,), mapping, checkpoint_id="step3")
    assert len(accepted) == 1 and not rejected
    with pytest.raises(RubricRefreshError, match="prior_checkpoint_lineage"):
        validate_current_checkpoint_lineage(
            (old.criterion,), checkpoint_id="step3", current_candidate_ids=("new",)
        )
