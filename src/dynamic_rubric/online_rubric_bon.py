"""Paper-style BoN comparison for the two completed OnlineRubric variants."""

from __future__ import annotations

import csv
import hashlib
import io
import math
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .artifacts import artifact_record, read_jsonl, write_json_atomic, write_text_atomic
from .evaluation.bon import fixed_candidate_permutations, select_best_of_n
from .hashing import canonical_json_bytes, sha256_file
from .judge_prompts import (
    PAPER_JUDGE_PROMPT_VERSION,
    PAPER_JUDGE_SYSTEM_PROMPT,
)
from .minimum_gold import PAPER_APPROVED_PAYLOAD_CATEGORIES
from .minimum_gold_streaming import (
    prepare_gold_selection_shard,
    submit_gold_selection_shard,
)
from .minimum_interim import ordered_prompt_subset, target_groups
from .minimum_staleness import (
    FOCAL_STEPS,
    N_GRID,
    PERMUTATIONS,
    MinimumExperimentError,
    TargetScoreClient,
    _audit_conversations,
    _bon_groups,
    _jsonl,
    _publish_gzip_jsonl,
    _publish_jsonl,
    _score_prompt,
    _shard_name,
)


PI_REF_MODE = "pi_ref"
PI_OLD_MODE = "pi_old"
ONLINE_MODES = (PI_REF_MODE, PI_OLD_MODE)
ONLINE_STAGES = {
    PI_REF_MODE: "onlinerubric_dedup_fixed_batch",
    PI_OLD_MODE: "onlinerubric_dedup_prev_batch",
}
SCORE_STAGE = "score-proxy-minimum"
SELECTION_STAGE = "select-bon-minimum"
PERMUTATION_SEED_PREFIX = "pilot-static-r0-100step-20260821"


def weighted_proxy_score(
    probabilities: Mapping[str, float], criteria: Sequence[Mapping[str, Any]]
) -> float:
    """Return the positive-integer-weighted mean criterion probability."""

    if not criteria:
        raise MinimumExperimentError("OnlineRubric has no criteria")
    numerator = 0.0
    denominator = 0
    for criterion in criteria:
        criterion_key = str(criterion["criterion_key"])
        weight = criterion.get("weight")
        if isinstance(weight, bool) or not isinstance(weight, int) or weight < 1:
            raise MinimumExperimentError(f"invalid OnlineRubric weight: {weight!r}")
        if criterion_key not in probabilities:
            raise MinimumExperimentError(f"missing OnlineRubric criterion score: {criterion_key}")
        numerator += weight * float(probabilities[criterion_key])
        denominator += weight
    return numerator / denominator


def _safe_link(source: Path, target: Path) -> None:
    if target.is_symlink():
        if target.resolve() != source.resolve():
            raise MinimumExperimentError(f"OnlineRubric source link drift: {target}")
        return
    if target.exists():
        raise MinimumExperimentError(f"OnlineRubric link target already exists: {target}")
    target.symlink_to(source.resolve(), target_is_directory=True)


def prepare_online_run(source_run_root: Path, run_root: Path) -> dict[str, Any]:
    """Create an isolated run linked to immutable candidates and OnlineRubrics."""

    source_run_root = source_run_root.resolve()
    run_root = run_root.absolute()
    if source_run_root == run_root:
        raise MinimumExperimentError("OnlineRubric BoN run must not overwrite its source")
    if source_run_root.parent != run_root.parent.resolve():
        raise MinimumExperimentError("OnlineRubric BoN run must be a source-run sibling")
    run_root.mkdir(parents=True, exist_ok=True)
    linked_stages = ("generate-bon", *ONLINE_STAGES.values())
    for stage in linked_stages:
        source = source_run_root / stage
        if not source.exists():
            raise MinimumExperimentError(f"missing OnlineRubric source stage: {source}")
        _safe_link(source, run_root / stage)

    source_record = {
        "schema_version": 1,
        "experiment": "onlinerubric-pi-ref-vs-pi-old-paper-judge-bon",
        "source_run_root": str(source_run_root),
        "linked_stages": {
            stage: str((source_run_root / stage).resolve()) for stage in linked_stages
        },
        "modes": dict(ONLINE_STAGES),
        "judge_prompt_version": PAPER_JUDGE_PROMPT_VERSION,
        "rubric_aggregation": "positive-integer-weighted-mean",
    }
    write_json_atomic(run_root / "online-rubric-source.json", source_record)
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
    if not approval_path.is_file():
        write_json_atomic(
            approval_path,
            {
                "schema_version": 1,
                "approved": True,
                "approval_source": "explicit user request in active conversation",
                "approved_at": datetime.now(timezone.utc).isoformat(),
                "destination": "OpenAI GPT-5 Batch API",
                "endpoint": "/v1/responses",
                "purpose": "hidden-gold-evaluation",
                "requested_model": "gpt-5",
                "payload_categories": list(PAPER_APPROVED_PAYLOAD_CATEGORIES),
            },
        )
    return {
        "run_root": str(run_root),
        "source_run_root": str(source_run_root),
        "approval_path": str(approval_path),
    }


def _criterion_key(prompt_id: str, text: str) -> str:
    return hashlib.sha256(f"{prompt_id}\0{text}".encode()).hexdigest()


def load_online_rubrics(run_root: Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    """Load both completed OnlineRubric collections with strict weights."""

    rubrics: dict[tuple[str, str, int], dict[str, Any]] = {}
    for mode, stage in ONLINE_STAGES.items():
        path = run_root / stage / "onlinerubric_rubrics.jsonl"
        if not path.is_file():
            raise MinimumExperimentError(f"missing OnlineRubric collection: {path}")
        for source in _jsonl(path):
            prompt_id = str(source["prompt_id"])
            step = int(source["policy_step"])
            key = mode, prompt_id, step
            if key in rubrics:
                raise MinimumExperimentError(f"duplicate OnlineRubric: {key}")
            criteria = []
            for item in source.get("criteria", ()):
                criterion = dict(item)
                text = str(criterion["text"])
                criterion["criterion_key"] = _criterion_key(prompt_id, text)
                weight = criterion.get("weight")
                if isinstance(weight, bool) or not isinstance(weight, int) or weight < 1:
                    raise MinimumExperimentError(
                        f"invalid OnlineRubric weight for {key}: {weight!r}"
                    )
                criteria.append(criterion)
            if not criteria:
                raise MinimumExperimentError(f"empty OnlineRubric: {key}")
            row = dict(source)
            row["criteria"] = criteria
            row["experiment_mode"] = mode
            rubrics[key] = row
    return rubrics


def _selection_path(run_root: Path, policy_id: str, prompt_id: str) -> Path:
    stem = f"{policy_id}-{_shard_name(policy_id, prompt_id)}.jsonl"
    return run_root / SELECTION_STAGE / "shards" / stem


def _score_path(run_root: Path, policy_id: str, prompt_id: str) -> Path:
    return run_root / SCORE_STAGE / "shards" / _selection_path(run_root, policy_id, prompt_id).name


def select_online_bon_shard(
    run_root: Path,
    policy_id: str,
    prompt_id: str,
    candidates: Sequence[dict[str, Any]],
    *,
    expected_groups: int,
) -> Path:
    """Select both OnlineRubric N-curves from the same candidate permutations."""

    stage_root = run_root / SELECTION_STAGE
    bon_path = run_root / "generate-bon" / "bon_pool.jsonl"
    write_json_atomic(
        stage_root / "manifest.json",
        {
            "schema_version": 1,
            "comparison": list(ONLINE_MODES),
            "n_grid": list(N_GRID),
            "permutations": PERMUTATIONS,
            "tie_break": "lowest_global_candidate_id",
            "candidate_permutation_seed_prefix": PERMUTATION_SEED_PREFIX,
            "bon_sha256": sha256_file(bon_path),
            "score_manifest_sha256": sha256_file(run_root / SCORE_STAGE / "manifest.json"),
        },
    )
    output_path = _selection_path(run_root, policy_id, prompt_id)
    if output_path.is_file():
        return output_path
    score_path = _score_path(run_root, policy_id, prompt_id)
    if not score_path.is_file():
        raise MinimumExperimentError(f"missing OnlineRubric proxy shard: {score_path}")
    score_rows = read_jsonl(score_path)
    by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in score_rows:
        by_mode[str(row["mode"])].append(row)
    if set(by_mode) != set(ONLINE_MODES):
        raise MinimumExperimentError(f"OnlineRubric score modes mismatch: {sorted(by_mode)}")

    candidate_by_id = {int(row["global_candidate_id"]): row for row in candidates}
    permutations = fixed_candidate_permutations(
        list(candidate_by_id),
        seed=f"{PERMUTATION_SEED_PREFIX}:{policy_id}:{prompt_id}",
        count=PERMUTATIONS,
    )
    pool_hash = hashlib.sha256(
        canonical_json_bytes(
            [
                [candidate_id, candidate_by_id[candidate_id]["response_text"]]
                for candidate_id in sorted(candidate_by_id)
            ]
        )
    ).hexdigest()
    selections = []
    for mode in ONLINE_MODES:
        current = by_mode[mode]
        score_by_id = {int(row["global_candidate_id"]): float(row["score"]) for row in current}
        if set(score_by_id) != set(candidate_by_id):
            raise MinimumExperimentError(f"candidate score inventory mismatch: {mode}")
        sample = current[0]
        for permutation_index, permutation in enumerate(permutations):
            for n in N_GRID:
                selected_id = int(select_best_of_n(score_by_id, permutation, n))
                selected = candidate_by_id[selected_id]
                selections.append(
                    {
                        "policy_id": policy_id,
                        "policy_step": sample["policy_step"],
                        "prompt_id": prompt_id,
                        "rubric_id": sample["rubric_id"],
                        "mode": mode,
                        "rubric_step": sample["rubric_step"],
                        "n": n,
                        "permutation": permutation_index,
                        "pool_hash": pool_hash,
                        "global_candidate_id": selected_id,
                        "response_id": selected["response_id"],
                        "response_text": selected["response_text"],
                    }
                )
    _publish_jsonl(output_path, selections)
    completed = len(list((stage_root / "shards").glob("*.jsonl")))
    write_json_atomic(
        stage_root / "progress.json",
        {
            "completed_prompt_policy_shards": completed,
            "expected_prompt_policy_shards": expected_groups,
        },
        immutable=False,
    )
    return output_path


def score_and_select_online_subset(
    run_root: Path,
    score_endpoint: str,
    *,
    prompt_count: int,
    workers: int,
) -> dict[str, Any]:
    """Score and select five-prompt-first OnlineRubric BoN shards resumably."""

    prompt_ids = ordered_prompt_subset(run_root, prompt_count)
    targets = target_groups(prompt_ids)
    client = TargetScoreClient(score_endpoint, workers=workers)
    rubrics = load_online_rubrics(run_root)
    conversations = _audit_conversations(run_root)
    bon_path = run_root / "generate-bon" / "bon_pool.jsonl"
    rubric_paths = [
        run_root / stage / "onlinerubric_rubrics.jsonl" for stage in ONLINE_STAGES.values()
    ]
    write_json_atomic(
        run_root / SCORE_STAGE / "manifest.json",
        {
            "schema_version": 1,
            "comparison": list(ONLINE_MODES),
            "focal_steps": list(FOCAL_STEPS),
            "pool_size": 1024,
            "judge_prompt_version": PAPER_JUDGE_PROMPT_VERSION,
            "rubric_aggregation": "positive-integer-weighted-mean",
            "shared_exact_text_criterion_cache": True,
            "score_identity": client.identity(),
            "score_routing": {
                "strategy": "deterministic-length-balanced-largest-first",
                "upstream_weights": list(client.routing_weights()),
            },
            "inputs": {str(path): sha256_file(path) for path in (bon_path, *rubric_paths)},
        },
    )

    completed = 0
    total_pairs = 0
    seen: set[tuple[str, str]] = set()
    for source_index, ((policy_id, prompt_id), candidates) in enumerate(_bon_groups(bon_path), 1):
        key = policy_id, prompt_id
        if key not in targets:
            continue
        seen.add(key)
        step = int(candidates[0]["policy_step"])
        if policy_id != f"pi_{step}" or step not in FOCAL_STEPS or len(candidates) != 1024:
            raise MinimumExperimentError(f"BoN group inventory mismatch: {key}")
        score_path = _score_path(run_root, policy_id, prompt_id)
        trace_path = (
            run_root
            / SCORE_STAGE
            / "criterion-shards"
            / f"{policy_id}-{_shard_name(policy_id, prompt_id)}.jsonl.gz"
        )
        if score_path.is_file() != trace_path.is_file():
            raise MinimumExperimentError(f"partial OnlineRubric score shard: {key}")
        if score_path.is_file():
            completed += 1
            select_online_bon_shard(
                run_root,
                policy_id,
                prompt_id,
                candidates,
                expected_groups=len(targets),
            )
            continue

        conversation = conversations.get(prompt_id)
        if conversation is None:
            raise MinimumExperimentError(f"paper judge conversation is absent: {prompt_id}")
        group_rubrics = [rubrics[(mode, prompt_id, step)] for mode in ONLINE_MODES]
        criteria_by_key: dict[str, dict[str, Any]] = {}
        for rubric in group_rubrics:
            for criterion in rubric["criteria"]:
                criterion_key = str(criterion["criterion_key"])
                previous = criteria_by_key.setdefault(criterion_key, dict(criterion))
                if previous["text"] != criterion["text"]:
                    raise MinimumExperimentError(f"OnlineRubric criterion hash collision: {key}")
        tasks = []
        for candidate in candidates:
            response_id = str(candidate["response_id"])
            for criterion_key, criterion in criteria_by_key.items():
                tasks.append(
                    (
                        (criterion_key, response_id),
                        _score_prompt(
                            str(criterion["text"]),
                            str(candidate["response_text"]),
                            conversation=conversation,
                            prompt_version=PAPER_JUDGE_PROMPT_VERSION,
                        ),
                    )
                )
        scores = client.score(tasks)
        total_pairs += len(scores)
        candidate_by_response = {str(row["response_id"]): row for row in candidates}

        def criterion_output() -> Iterator[dict[str, Any]]:
            for (criterion_key, response_id), score in sorted(scores.items()):
                yield {
                    "policy_id": policy_id,
                    "policy_step": step,
                    "prompt_id": prompt_id,
                    "response_id": response_id,
                    "global_candidate_id": candidate_by_response[response_id][
                        "global_candidate_id"
                    ],
                    "criterion_key": criterion_key,
                    **score,
                    "parse_success": True,
                }

        _publish_gzip_jsonl(trace_path, criterion_output())
        score_rows = []
        for candidate in candidates:
            response_id = str(candidate["response_id"])
            probabilities = {
                criterion_key: scores[(criterion_key, response_id)]["probability_yes"]
                for criterion_key in criteria_by_key
            }
            for mode, rubric in zip(ONLINE_MODES, group_rubrics):
                score = weighted_proxy_score(probabilities, rubric["criteria"])
                score_rows.append(
                    {
                        "policy_id": policy_id,
                        "policy_step": step,
                        "prompt_id": prompt_id,
                        "global_candidate_id": candidate["global_candidate_id"],
                        "response_id": response_id,
                        "rubric_id": f"{rubric['rubric_id']}:{mode}",
                        "mode": mode,
                        "rubric_step": step,
                        "criterion_count": len(rubric["criteria"]),
                        "criterion_weight_total": sum(
                            int(item["weight"]) for item in rubric["criteria"]
                        ),
                        "score": score,
                        "judge_repeat_score": score,
                        "judge_repeat_method": "deterministic_temperature_zero_cache_identity",
                    }
                )
        _publish_jsonl(score_path, score_rows)
        completed += 1
        write_json_atomic(
            run_root / SCORE_STAGE / f"online-rubric-{prompt_count}prompt-progress.json",
            {
                "completed_prompt_policy_shards": completed,
                "expected_prompt_policy_shards": len(targets),
                "last_policy_id": policy_id,
                "last_prompt_id": prompt_id,
                "source_group_index": source_index,
                "criterion_pairs_scored_this_process": total_pairs,
            },
            immutable=False,
        )
        select_online_bon_shard(
            run_root,
            policy_id,
            prompt_id,
            candidates,
            expected_groups=len(targets),
        )
    if seen != targets:
        raise MinimumExperimentError(f"target BoN groups are absent: {sorted(targets - seen)}")
    return {
        "prompt_ids": list(prompt_ids),
        "prompt_policy_shards": completed,
        "expected_prompt_policy_shards": len(targets),
        "criterion_pairs_scored_this_process": total_pairs,
    }


def prepare_or_submit_available_gold(
    run_root: Path,
    private_gt: Path,
    schema_path: Path,
    *,
    prompt_count: int,
    submit: bool,
) -> dict[str, Any]:
    """Prepare/submit the union of π_ref and π_old selections in every ready shard."""

    prompt_ids = ordered_prompt_subset(run_root, prompt_count)
    groups = []
    for policy_id, prompt_id in sorted(target_groups(prompt_ids)):
        selection_path = _selection_path(run_root, policy_id, prompt_id)
        if not selection_path.is_file():
            continue
        manifest = prepare_gold_selection_shard(
            run_root,
            selection_path,
            private_gt,
            schema_path,
            prompt_version=PAPER_JUDGE_PROMPT_VERSION,
        )
        row: dict[str, Any] = {
            "policy_id": policy_id,
            "prompt_id": prompt_id,
            "requests": int(manifest["requests"]),
        }
        if submit:
            receipt = submit_gold_selection_shard(
                run_root,
                selection_path,
                approval_path=run_root / "paper-judge-gold-egress-approval.json",
            )
            row["batch_id"] = receipt["batch_id"]
        groups.append(row)
    return {
        "prompt_ids": list(prompt_ids),
        "groups": groups,
        "ready_groups": len(groups),
        "expected_groups": len(target_groups(prompt_ids)),
        "requests": sum(int(row["requests"]) for row in groups),
        "submitted": submit,
    }


def _completed_policies(run_root: Path, prompt_ids: Sequence[str]) -> tuple[str, ...]:
    completed = []
    for step in FOCAL_STEPS:
        policy_id = f"pi_{step}"
        if all(
            (
                run_root
                / "audit-gold-streaming-private"
                / "groups"
                / _selection_path(run_root, policy_id, prompt_id).stem
                / "gold_scores.jsonl"
            ).is_file()
            for prompt_id in prompt_ids
        ):
            completed.append(policy_id)
    return tuple(completed)


def online_curve_rows(
    run_root: Path, prompt_ids: Sequence[str], policies: Sequence[str]
) -> list[dict[str, Any]]:
    """Compute proxy and hidden-GT curves over the same selected responses."""

    rows = []
    for policy_id in policies:
        selections = []
        proxy: dict[tuple[str, str, str], float] = {}
        gold: dict[tuple[str, str], float] = {}
        for prompt_id in prompt_ids:
            selection_path = _selection_path(run_root, policy_id, prompt_id)
            score_path = _score_path(run_root, policy_id, prompt_id)
            gold_path = (
                run_root
                / "audit-gold-streaming-private"
                / "groups"
                / selection_path.stem
                / "gold_scores.jsonl"
            )
            for path in (selection_path, score_path, gold_path):
                if not path.is_file():
                    raise MinimumExperimentError(f"missing OnlineRubric curve input: {path}")
            selections.extend(_jsonl(selection_path))
            for row in _jsonl(score_path):
                proxy[prompt_id, str(row["mode"]), str(row["response_id"])] = float(row["score"])
            for row in _jsonl(gold_path):
                gold[prompt_id, str(row["response_id"])] = float(row["gold_score"])
        grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
        for row in selections:
            grouped[str(row["mode"]), int(row["n"])].append(row)
        for n in N_GRID:
            output: dict[str, Any] = {
                "policy_id": policy_id,
                "policy_step": int(policy_id[3:]),
                "n": n,
                "prompts": len(prompt_ids),
                "permutations": PERMUTATIONS,
            }
            expected = len(prompt_ids) * PERMUTATIONS
            for mode in ONLINE_MODES:
                current = grouped[mode, n]
                if len(current) != expected:
                    raise MinimumExperimentError(
                        f"OnlineRubric selection inventory mismatch: "
                        f"{(policy_id, mode, n)}={len(current)} != {expected}"
                    )
                output[f"{mode}_proxy"] = (
                    math.fsum(
                        proxy[
                            str(row["prompt_id"]),
                            mode,
                            str(row["response_id"]),
                        ]
                        for row in current
                    )
                    / expected
                )
                output[f"{mode}_gold"] = (
                    math.fsum(
                        gold[str(row["prompt_id"]), str(row["response_id"])] for row in current
                    )
                    / expected
                )
            rows.append(output)
    return rows


def _plot_online_curves(output_root: Path, rows: Sequence[Mapping[str, Any]]) -> tuple[Path, Path]:
    import matplotlib  # pyright: ignore[reportMissingImports]

    matplotlib.use("Agg")
    matplotlib.rcParams["svg.hashsalt"] = "onlinerubric-paper-proxy-gt-v1"
    import matplotlib.pyplot as plt  # pyright: ignore[reportMissingImports]

    policies = sorted({str(row["policy_id"]) for row in rows}, key=lambda value: int(value[3:]))
    figure, axes = plt.subplots(1, len(policies), figsize=(7 * len(policies), 5.4), squeeze=False)
    styles = (
        ("pi_ref_proxy", "π_ref · Qwen proxy", "#2563eb", "-", "o"),
        ("pi_old_proxy", "π_old · Qwen proxy", "#ea580c", "-", "o"),
        ("pi_ref_gold", "π_ref · GPT-5 GT", "#2563eb", "--", "s"),
        ("pi_old_gold", "π_old · GPT-5 GT", "#ea580c", "--", "s"),
    )
    for axis, policy_id in zip(axes.flat, policies):
        current = [row for row in rows if str(row["policy_id"]) == policy_id]
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
        step = int(policy_id[3:])
        axis.set_title(rf"$\pi_{{{step}}}$: OnlineRubric $\pi_{{ref}}$ vs $\pi_{{old}}$")
        axis.set_xticks(x, [str(row["n"]) for row in current], rotation=45)
        axis.set_xlabel("BoN size N")
        axis.set_ylim(0.0, 1.01)
        axis.grid(alpha=0.22)
    axes[0, 0].set_ylabel("Mean score")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=4,
        frameon=False,
    )
    figure.suptitle(
        "OnlineRubric BoN: weighted Qwen proxy vs paper-style GPT-5 hidden GT "
        f"({rows[0]['prompts']} prompts × {rows[0]['permutations']} permutations)",
        y=0.99,
    )
    figure.text(
        0.5,
        0.015,
        "Same candidate pools; solid = selection proxy, dashed = independent hidden GT.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0.02, 0.05, 1.0, 0.83))
    output_root.mkdir(parents=True, exist_ok=True)
    png_path = output_root / "combined_proxy_gt_curves.png"
    svg_path = output_root / "combined_proxy_gt_curves.svg"
    figure.savefig(png_path, dpi=200, metadata={"Date": None})
    figure.savefig(svg_path, metadata={"Date": None})
    plt.close(figure)
    return png_path, svg_path


def analyze_online_subset(run_root: Path, *, prompt_count: int) -> dict[str, Any]:
    """Publish a combined graph for every policy complete on all fixed prompts."""

    prompt_ids = ordered_prompt_subset(run_root, prompt_count)
    policies = _completed_policies(run_root, prompt_ids)
    if not policies:
        raise MinimumExperimentError("no complete OnlineRubric policy has GPT-5 GT yet")
    rows = online_curve_rows(run_root, prompt_ids, policies)
    output_root = run_root / f"interim-bon-{prompt_count}prompt-online-rubric"
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    csv_path = output_root / "combined_proxy_gt_curves.csv"
    write_text_atomic(csv_path, stream.getvalue(), immutable=False)
    png_path, svg_path = _plot_online_curves(output_root, rows)
    result = {
        "schema_version": 1,
        "diagnostic_only": True,
        "prompt_ids": list(prompt_ids),
        "policies": list(policies),
        "n_grid": list(N_GRID),
        "permutations": PERMUTATIONS,
        "modes": list(ONLINE_MODES),
        "rubric_aggregation": "positive-integer-weighted-mean",
        "rows": rows,
        "csv": artifact_record(csv_path),
        "figures": [artifact_record(png_path), artifact_record(svg_path)],
    }
    write_json_atomic(output_root / "result.json", result, immutable=False)
    return result
