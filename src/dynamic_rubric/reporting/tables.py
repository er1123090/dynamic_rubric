from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..artifacts import write_text_atomic


REQUIRED_COMPARISONS = (
    "static",
    "dynamic_fixed_budgeted",
    "dynamic_prev_budgeted",
    "refresh_only_budgeted",
    "dynamic_fixed_cumulative",
)
REQUIRED_METRICS = (
    "gt_auc",
    "stale_rubric_regret",
    "top1_agreement",
    "kendall_tau_b",
    "reward_resolution",
    "rubric_churn",
    "policy_distance",
)
INTERPRETATIONS = (
    "local_redundancy",
    "policy_adaptive_gain",
    "general_rubric_improvement",
    "textual_only_churn",
    "frequent_update_need",
    "weak_policy_drift",
    "updater_miss",
    "refresh_noise",
)


def validate_report(report: Mapping[str, Any]) -> None:
    comparisons = report.get("comparisons")
    if not isinstance(comparisons, Mapping):
        raise ValueError("report.comparisons must be an object")
    missing_comparisons = set(REQUIRED_COMPARISONS) - set(comparisons)
    if missing_comparisons:
        raise ValueError(f"report is missing comparisons: {sorted(missing_comparisons)}")
    metrics = report.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("report.metrics must be an object")
    missing_metrics = set(REQUIRED_METRICS) - set(metrics)
    if missing_metrics:
        raise ValueError(f"report is missing metrics: {sorted(missing_metrics)}")
    interpretation = report.get("interpretation")
    if interpretation not in INTERPRETATIONS:
        raise ValueError("report must choose one preliminary interpretation")
    statistical_rules = report.get("statistical_rules", {})
    if statistical_rules.get("equivalence") != "ci_contained_in_[-0.015,+0.015]":
        raise ValueError("equivalence must use CI containment")
    if statistical_rules.get("meaningful_difference") != "abs(point)>=0.03_and_ci_excludes_zero":
        raise ValueError("meaningful difference rule is invalid")


def build_report(
    run_id: str,
    manifest_hash: str,
    comparisons: Mapping[str, Any],
    metrics: Mapping[str, Any],
    interpretation: str,
    evidence: Sequence[str],
    recommendation: str,
    *,
    bootstrap_samples: int,
    confidence: float,
) -> dict[str, Any]:
    report = {
        "schema_version": 1,
        "run_id": run_id,
        "manifest_hash": manifest_hash,
        "comparisons": dict(comparisons),
        "metrics": dict(metrics),
        "statistical_rules": {
            "bootstrap_unit": "prompt_id",
            "bootstrap_samples": bootstrap_samples,
            "confidence": confidence,
            "equivalence": "ci_contained_in_[-0.015,+0.015]",
            "meaningful_difference": "abs(point)>=0.03_and_ci_excludes_zero",
        },
        "interpretation": interpretation,
        "evidence": list(evidence),
        "go_no_go_recommendation": recommendation,
    }
    validate_report(report)
    return report


def write_report_json(path: Path, report: Mapping[str, Any]) -> None:
    validate_report(report)
    payload = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    write_text_atomic(path, payload)


def render_markdown(report: Mapping[str, Any]) -> str:
    validate_report(report)
    lines = [
        "# Dynamic Rubric Staleness Audit",
        "",
        f"Run: `{report['run_id']}`",
        "",
        "## Primary comparisons",
        "",
        "| Condition | Summary |",
        "| --- | --- |",
    ]
    for name in REQUIRED_COMPARISONS:
        lines.append(f"| `{name}` | {json.dumps(report['comparisons'][name], sort_keys=True)} |")
    lines.extend(
        [
            "",
            "## Metric families",
            "",
            "| Metric | Result |",
            "| --- | --- |",
        ]
    )
    for name in REQUIRED_METRICS:
        lines.append(f"| `{name}` | {json.dumps(report['metrics'][name], sort_keys=True)} |")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"Interpretation: `{report['interpretation']}`",
            "",
            f"Recommendation: {report['go_no_go_recommendation']}",
            "",
        ]
    )
    return "\n".join(lines)
