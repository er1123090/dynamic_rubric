from __future__ import annotations

from typing import Any, Mapping, Sequence


def trajectory_plot_spec(
    rows: Sequence[Mapping[str, Any]], x: str = "policy_step", y: str = "gt_auc"
) -> dict[str, Any]:
    """Dependency-free Vega-Lite-compatible plot specification."""

    required = {x, y, "condition"}
    if any(not required.issubset(row) for row in rows):
        raise ValueError(f"plot rows require {sorted(required)}")
    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "description": "Dynamic rubric staleness trajectory",
        "data": {"values": [dict(row) for row in rows]},
        "mark": {"type": "line", "point": True},
        "encoding": {
            "x": {"field": x, "type": "quantitative"},
            "y": {"field": y, "type": "quantitative", "scale": {"domain": [0, 1]}},
            "color": {"field": "condition", "type": "nominal"},
            "detail": {"field": "policy", "type": "nominal"},
        },
    }
