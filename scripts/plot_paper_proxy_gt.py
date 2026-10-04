#!/usr/bin/env python3
"""Plot paper-style Qwen proxy and GPT-5 GT BoN curves side by side."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from plot_interim_proxy_gt import _curve_rows


def _load_metadata(interim_root: Path) -> dict[str, Any]:
    for name in ("result.json", "manifest.json"):
        path = interim_root / name
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"no result.json or manifest.json under {interim_root}")


def _write_outputs(output_root: Path, rows: list[dict[str, Any]]) -> tuple[Path, Path, Path]:
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["svg.hashsalt"] = "dynamic-rubric-paper-proxy-gt-v1"
    import matplotlib.pyplot as plt

    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "combined_proxy_gt_curves.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    policies = sorted({str(row["policy_id"]) for row in rows}, key=lambda value: int(value[3:]))
    figure, axes = plt.subplots(
        2, 2, figsize=(13.5, 9.0), squeeze=False, sharey=True
    )
    styles = (
        ("static_proxy", "Static · Qwen proxy", "#2563eb", "-", "o"),
        ("dynamic_proxy", "Dynamic · Qwen proxy", "#ea580c", "-", "o"),
        ("static_gold", "Static · GPT-5 GT", "#2563eb", "--", "s"),
        ("dynamic_gold", "Dynamic · GPT-5 GT", "#ea580c", "--", "s"),
    )
    for axis, policy in zip(axes.flat, policies):
        current = [row for row in rows if row["policy_id"] == policy]
        x = [math.log2(int(row["n"])) for row in current]
        for field, label, color, linestyle, marker in styles:
            axis.plot(
                x,
                [float(row[field]) for row in current],
                color=color,
                linestyle=linestyle,
                marker=marker,
                linewidth=2.1,
                markersize=5,
                label=label,
            )
        step = int(policy[3:])
        axis.set_title(rf"$\pi_{{{step}}}$: $R_0$ vs $R_{{{step}}}$")
        axis.set_xticks(x, [str(row["n"]) for row in current], rotation=45)
        axis.set_xlabel("BoN size N")
        axis.set_ylim(0.4, 1.01)
        axis.grid(alpha=0.22)

    for axis in axes[:, 0]:
        axis.set_ylabel("Mean score")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=4,
        frameon=False,
    )
    prompt_count = int(rows[0]["prompts"])
    permutations = int(rows[0]["permutations"])
    figure.suptitle(
        f"Static vs Dynamic BoN: selection proxy vs hidden GT "
        f"({prompt_count} prompts × {permutations} permutations)",
        y=0.995,
    )
    figure.text(
        0.5,
        0.015,
        "Each proxy/GT point evaluates the same selected responses from the same candidate pools.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0.02, 0.05, 1.0, 0.88))

    png_path = output_root / "combined_proxy_gt_curves.png"
    svg_path = output_root / "combined_proxy_gt_curves.svg"
    figure.savefig(png_path, dpi=200, metadata={"Date": None})
    figure.savefig(svg_path, metadata={"Date": None})
    plt.close(figure)
    return csv_path, png_path, svg_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--interim-root", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+", required=True)
    args = parser.parse_args()

    metadata = _load_metadata(args.interim_root)
    prompt_ids = [str(value) for value in metadata["prompt_ids"]]
    rows = _curve_rows(
        args.run_root,
        prompt_ids,
        [int(value) for value in metadata["n_grid"]],
        int(metadata["permutations"]),
    )
    requested = {f"pi_{step}" for step in args.steps}
    rows = [row for row in rows if str(row["policy_id"]) in requested]
    present = {str(row["policy_id"]) for row in rows}
    if present != requested:
        raise ValueError(f"requested policies are incomplete: {sorted(requested - present)}")

    paths = _write_outputs(args.interim_root, rows)
    print(
        json.dumps(
            {
                "policies": sorted(present, key=lambda value: int(value[3:])),
                "rows": len(rows),
                "outputs": [str(path) for path in paths],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
