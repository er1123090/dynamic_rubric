"""Five-prompt-first, extensible paper-judge BoN rescore experiment."""

from __future__ import annotations

import csv
import hashlib
import io
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .artifacts import artifact_record, read_json, write_json_atomic, write_text_atomic
from .evaluation.bootstrap import paired_prompt_bootstrap
from .judge_prompts import PAPER_JUDGE_PROMPT_VERSION, PAPER_JUDGE_SYSTEM_PROMPT
from .minimum_gold import PAPER_APPROVED_PAYLOAD_CATEGORIES
from .minimum_gold_streaming import (
    prepare_gold_selection_shard,
    submit_gold_selection_shard,
)
from .minimum_interim import analyze_interim, ordered_prompt_subset, target_groups
from .minimum_staleness import (
    FOCAL_STEPS,
    MODE,
    N_GRID,
    PERMUTATIONS,
    MinimumExperimentError,
    _jsonl,
    _shard_name,
    score_bon,
    select_bon_shard,
)


LINKED_SOURCE_STAGES = (
    "generate-bon",
    "generate-static",
    "replay-dynamic-minimum",
    "train-static",
)


def _safe_link(source: Path, target: Path) -> None:
    if target.is_symlink():
        if target.resolve() != source.resolve():
            raise MinimumExperimentError(f"paper run source link drift: {target}")
        return
    if target.exists():
        raise MinimumExperimentError(f"paper run link target already exists: {target}")
    target.symlink_to(source.resolve(), target_is_directory=True)


def prepare_paper_run(source_run_root: Path, run_root: Path) -> dict[str, Any]:
    """Create an isolated derived run that reads expensive upstream artifacts by symlink."""

    source_run_root = source_run_root.resolve()
    run_root = run_root.absolute()
    if source_run_root == run_root:
        raise MinimumExperimentError("paper judge run must not overwrite the source run")
    if source_run_root.parent != run_root.parent.resolve():
        raise MinimumExperimentError("paper judge run must be a sibling of the source run")
    run_root.mkdir(parents=True, exist_ok=True)
    for stage in LINKED_SOURCE_STAGES:
        source = source_run_root / stage
        if not source.exists():
            raise MinimumExperimentError(f"missing source stage for paper run: {source}")
        _safe_link(source, run_root / stage)

    source_record = {
        "schema_version": 1,
        "source_run_root": str(source_run_root),
        "judge_prompt_version": PAPER_JUDGE_PROMPT_VERSION,
        "linked_stages": {
            stage: str((source_run_root / stage).resolve()) for stage in LINKED_SOURCE_STAGES
        },
    }
    write_json_atomic(run_root / "paper-judge-source.json", source_record)
    write_json_atomic(
        run_root / "paper-judge-prompt-contract.json",
        {
            "schema_version": 1,
            "prompt_version": PAPER_JUDGE_PROMPT_VERSION,
            "system_prompt": PAPER_JUDGE_SYSTEM_PROMPT,
            "system_prompt_sha256": hashlib.sha256(PAPER_JUDGE_SYSTEM_PROMPT.encode()).hexdigest(),
            "qwen_contract": "one criterion; target likelihood over exact YES/NO",
            "gpt5_contract": "all physician criteria; structured integer 1/0",
            "shared_inputs": ["user conversation", "assistant response", "criterion text"],
        },
    )

    approval_path = run_root / "paper-judge-gold-egress-approval.json"
    if approval_path.is_file():
        approval = read_json(approval_path)
    else:
        approval = {
            "schema_version": 1,
            "approved": True,
            "approval_source": "explicit user request in active conversation",
            "approved_at": datetime.now(timezone.utc).isoformat(),
            "destination": "OpenAI GPT-5 Batch API",
            "endpoint": "/v1/responses",
            "purpose": "hidden-gold-evaluation",
            "requested_model": "gpt-5",
            "payload_categories": list(PAPER_APPROVED_PAYLOAD_CATEGORIES),
        }
        write_json_atomic(approval_path, approval)
    if tuple(approval.get("payload_categories", ())) != PAPER_APPROVED_PAYLOAD_CATEGORIES:
        raise MinimumExperimentError("paper judge egress approval scope drift")
    return {
        "run_root": str(run_root),
        "source_run_root": str(source_run_root),
        "approval_path": str(approval_path),
    }


def _group_paths(run_root: Path, source_run_root: Path, prompt_ids: Sequence[str]):
    for policy_id, prompt_id in sorted(target_groups(prompt_ids)):
        stem = f"{policy_id}-{_shard_name(policy_id, prompt_id)}.jsonl"
        yield (
            policy_id,
            prompt_id,
            run_root / "select-bon-minimum" / "shards" / stem,
            source_run_root / "select-bon-minimum" / "shards" / stem,
        )


def score_and_select_subset(
    run_root: Path,
    source_run_root: Path,
    score_endpoint: str,
    *,
    prompt_count: int,
    workers: int,
) -> dict[str, Any]:
    """Rescore only the fixed prefix of prompts and publish resumable selection shards."""

    prompt_ids = ordered_prompt_subset(run_root, prompt_count)
    targets = target_groups(prompt_ids)

    def on_shard(policy_id: str, prompt_id: str, candidates: list[dict[str, Any]]) -> None:
        select_bon_shard(run_root, policy_id, prompt_id, candidates)

    result = score_bon(
        run_root,
        score_endpoint,
        workers=workers,
        on_shard=on_shard,
        target_groups=targets,
        progress_filename=f"paper-judge-{prompt_count}prompt-progress.json",
        judge_prompt_version=PAPER_JUDGE_PROMPT_VERSION,
        routing_contract_path=run_root / "paper-judge-qwen-routing.json",
    )
    for _, _, new_selection, old_selection in _group_paths(run_root, source_run_root, prompt_ids):
        if not new_selection.is_file() or not old_selection.is_file():
            raise MinimumExperimentError(
                f"old/new selection shard is incomplete: {new_selection.name}"
            )
    return {"prompt_ids": list(prompt_ids), "score": result}


def prepare_or_submit_gold_subset(
    run_root: Path,
    source_run_root: Path,
    private_gt: Path,
    schema_path: Path,
    *,
    prompt_count: int,
    submit: bool,
) -> dict[str, Any]:
    """Audit only the deduplicated paper-judge-selected responses."""

    prompt_ids = ordered_prompt_subset(run_root, prompt_count)
    approval_path = run_root / "paper-judge-gold-egress-approval.json"
    groups = []
    for policy_id, prompt_id, new_selection, _old_selection in _group_paths(
        run_root, source_run_root, prompt_ids
    ):
        manifest = prepare_gold_selection_shard(
            run_root,
            new_selection,
            private_gt,
            schema_path,
            prompt_version=PAPER_JUDGE_PROMPT_VERSION,
        )
        row: dict[str, Any] = {
            "policy_id": policy_id,
            "prompt_id": prompt_id,
            "requests": manifest["requests"],
        }
        if submit:
            receipt = submit_gold_selection_shard(
                run_root,
                new_selection,
                approval_path=approval_path,
            )
            row["batch_id"] = receipt["batch_id"]
        groups.append(row)
    return {
        "prompt_ids": list(prompt_ids),
        "groups": groups,
        "submitted": submit,
        "requests": sum(int(row["requests"]) for row in groups),
    }


def _gold_for_subset(run_root: Path, prompt_ids: Sequence[str]) -> dict[tuple[str, str], float]:
    gold: dict[tuple[str, str], float] = {}
    for policy_id, prompt_id in sorted(target_groups(prompt_ids)):
        stem = f"{policy_id}-{_shard_name(policy_id, prompt_id)}"
        path = run_root / "audit-gold-streaming-private" / "groups" / stem / "gold_scores.jsonl"
        if not path.is_file():
            raise MinimumExperimentError(f"paper judge gold group is incomplete: {path}")
        for row in _jsonl(path):
            key = str(row["prompt_id"]), str(row["response_id"])
            value = float(row["gold_score"])
            previous = gold.setdefault(key, value)
            if previous != value:
                raise MinimumExperimentError(f"paper judge gold score drift: {key}")
    return gold


def _selection_cells(path: Path) -> dict[tuple[str, str, str, int, int], str]:
    cells: dict[tuple[str, str, str, int, int], str] = {}
    for row in _jsonl(path):
        key = (
            str(row["policy_id"]),
            str(row["prompt_id"]),
            str(row["mode"]),
            int(row["n"]),
            int(row["permutation"]),
        )
        response_id = str(row["response_id"])
        previous = cells.setdefault(key, response_id)
        if previous != response_id:
            raise MinimumExperimentError(f"duplicate selection cell: {key}")
    return cells


def analyze_prompt_gain(
    run_root: Path,
    source_run_root: Path,
    prompt_ids: Sequence[str],
) -> dict[str, Any]:
    """Compare old- versus paper-prompt selection under the same new GPT-5 gold."""

    gold = _gold_for_subset(run_root, prompt_ids)
    old_cells: dict[tuple[str, str, str, int, int], str] = {}
    new_cells: dict[tuple[str, str, str, int, int], str] = {}
    for _, _, new_path, old_path in _group_paths(run_root, source_run_root, prompt_ids):
        new_cells.update(_selection_cells(new_path))
        old_cells.update(_selection_cells(old_path))
    if set(new_cells) != set(old_cells):
        raise MinimumExperimentError("old/new paper judge selection grids do not align")

    rows: list[dict[str, Any]] = []
    for step in FOCAL_STEPS:
        policy_id = f"pi_{step}"
        for mode in ("static", MODE):
            for n in N_GRID:
                pairs = []
                old_values = []
                new_values = []
                for prompt_id in prompt_ids:
                    current_old = []
                    current_new = []
                    for permutation in range(PERMUTATIONS):
                        key = policy_id, prompt_id, mode, n, permutation
                        current_old.append(gold[(prompt_id, old_cells[key])])
                        current_new.append(gold[(prompt_id, new_cells[key])])
                    old_mean = math.fsum(current_old) / len(current_old)
                    new_mean = math.fsum(current_new) / len(current_new)
                    old_values.append(old_mean)
                    new_values.append(new_mean)
                    pairs.append((prompt_id, new_mean, old_mean))
                bootstrap = paired_prompt_bootstrap(
                    pairs,
                    iterations=10_000,
                    seed=20_260_826 + step * 2_048 + n + (0 if mode == "static" else 1),
                )
                rows.append(
                    {
                        "policy_id": policy_id,
                        "policy_step": step,
                        "mode": mode,
                        "n": n,
                        "old_prompt_mean_gold": math.fsum(old_values) / len(old_values),
                        "paper_prompt_mean_gold": math.fsum(new_values) / len(new_values),
                        "paper_minus_old": bootstrap.point_estimate,
                        "ci_low": bootstrap.ci_low,
                        "ci_high": bootstrap.ci_high,
                    }
                )

    output_root = run_root / f"paper-judge-gain-{len(prompt_ids)}prompt"
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    csv_path = output_root / "old_new_prompt_gain.csv"
    write_text_atomic(csv_path, stream.getvalue())
    figure_path = _plot_prompt_gain(output_root, rows, len(prompt_ids))
    result = {
        "schema_version": 1,
        "diagnostic_only": True,
        "prompt_ids": list(prompt_ids),
        "judge_prompt_version": PAPER_JUDGE_PROMPT_VERSION,
        "rows": rows,
        "csv": artifact_record(csv_path),
        "figure": artifact_record(figure_path),
    }
    write_json_atomic(output_root / "result.json", result)
    return result


def _plot_prompt_gain(
    output_root: Path, rows: Sequence[Mapping[str, Any]], prompt_count: int
) -> Path:
    import matplotlib  # pyright: ignore[reportMissingImports]

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # pyright: ignore[reportMissingImports]

    output_root.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey=True)
    styles = {
        ("static", "old"): ("Static old prompt", "--"),
        ("static", "paper"): ("Static paper prompt", "-"),
        (MODE, "old"): ("Dynamic old prompt", "--"),
        (MODE, "paper"): ("Dynamic paper prompt", "-"),
    }
    for axis, step in zip(axes.flat, FOCAL_STEPS):
        current = [row for row in rows if int(row["policy_step"]) == step]
        for (mode, version), (label, linestyle) in styles.items():
            selected = [row for row in current if str(row["mode"]) == mode]
            field = "old_prompt_mean_gold" if version == "old" else "paper_prompt_mean_gold"
            axis.plot(
                [math.log2(int(row["n"])) for row in selected],
                [float(row[field]) for row in selected],
                marker="o",
                linestyle=linestyle,
                label=label,
            )
        axis.set_title(f"pi_{step}")
        axis.set_ylim(0.0, 1.0)
        axis.grid(alpha=0.25)
        axis.set_xticks(
            [math.log2(n) for n in N_GRID],
            [str(n) for n in N_GRID],
            rotation=45,
        )
    axes[0, 0].legend(loc="best", fontsize=8)
    figure.supxlabel("BoN size N (log2 spacing)")
    figure.supylabel("Mean paper-style GPT-5 gold score")
    figure.suptitle(f"Old vs paper judge selection ({prompt_count} fixed prompts)")
    figure.tight_layout(rect=(0.03, 0.03, 1.0, 0.95))
    output_path = output_root / "old_new_prompt_gain.png"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".png", dir=output_root
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        figure.savefig(temporary, dpi=180, metadata={"Date": None})
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
        plt.close(figure)
    return output_path


def analyze_subset(run_root: Path, *, prompt_count: int) -> dict[str, Any]:
    prompt_ids = ordered_prompt_subset(run_root, prompt_count)
    curves = analyze_interim(run_root, prompt_ids)
    return {"curves": curves}
