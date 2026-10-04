"""Regression checks for offline report aggregation, no inference dependencies."""
import importlib.util
from pathlib import Path


path = Path(__file__).resolve().parents[2] / "scripts/phase1/analyze_design_alignment.py"
spec = importlib.util.spec_from_file_location("design_alignment", path)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def test_binary_criterion_categories_partition_all_criteria():
    criteria = [{"criterion_id": name} for name in ("mixed", "saturated", "dead")]
    rewards = [dict(grades=[("mixed", 0), ("saturated", 1), ("dead", 0)]),
               dict(grades=[("mixed", 1), ("saturated", 1), ("dead", 0)])]
    counts = analysis.criterion_counts(rewards, criteria)
    assert counts == {"effective": 1, "saturated": 1, "dead": 1}
    assert analysis.summarize_counts(counts)["effective_ratio"] == 1 / 3


def test_design_criterion_pooling_is_not_mean_of_prompt_ratios():
    first = analysis.Counter(effective=1, saturated=0, dead=0)
    second = analysis.Counter(effective=0, saturated=3, dead=0)
    first.update(second)
    assert analysis.summarize_counts(first)["effective_ratio"] == 0.25
    assert analysis.summarize_counts(first)["effective_ratio"] != (1 + 0) / 2


def test_empty_component_is_missing_not_zero():
    result = analysis.summarize_counts(analysis.Counter(effective=0, saturated=0, dead=0))
    assert result["criterion_occurrences"] == 0
    assert result["effective_ratio"] is None
