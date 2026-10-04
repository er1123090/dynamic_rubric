#!/usr/bin/env python3
"""Run a paper-style Qwen BoN selection with the private human rubric fixed as R0."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from dynamic_rubric.artifacts import read_jsonl, write_json_atomic
from dynamic_rubric.evaluation.bon import fixed_candidate_permutations, select_best_of_n
from dynamic_rubric.evaluation.gold_score import weighted_gold_score
from dynamic_rubric.hashing import canonical_json_bytes, sha256_file
from dynamic_rubric.judge_prompts import PAPER_JUDGE_PROMPT_VERSION
from dynamic_rubric.minimum_interim import ordered_prompt_subset, target_groups
from dynamic_rubric.minimum_staleness import (
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


MODE = "human_gt_r0"
SEED_PREFIX = "pilot-static-r0-100step-20260821"
SCORE_STAGE = "score-human-gt-r0-paper-qwen"
SELECTION_STAGE = "select-human-gt-r0-bon"


def _safe_link(source: Path, target: Path) -> None:
    if target.is_symlink():
        if target.resolve() != source.resolve():
            raise MinimumExperimentError(f"human-GT run source link drift: {target}")
        return
    if target.exists():
        raise MinimumExperimentError(f"human-GT run link target exists: {target}")
    target.symlink_to(source.resolve(), target_is_directory=True)


def prepare_run(source_run_root: Path, run_root: Path) -> dict[str, Any]:
    source_run_root = source_run_root.resolve()
    run_root = run_root.absolute()
    if source_run_root == run_root or source_run_root.parent != run_root.parent.resolve():
        raise MinimumExperimentError("derived run must be a distinct sibling of its source")
    run_root.mkdir(parents=True, exist_ok=True)
    source = source_run_root / "generate-bon"
    if not source.is_dir():
        raise MinimumExperimentError(f"missing candidate source: {source}")
    _safe_link(source, run_root / "generate-bon")
    record = {
        "schema_version": 1,
        "experiment": "human-gt-r0-paper-qwen-bon",
        "source_run_root": str(source_run_root),
        "linked_candidate_stage": str(source.resolve()),
        "selection_rubric": "private human HealthBench rubric fixed as R0",
        "judge_prompt_version": PAPER_JUDGE_PROMPT_VERSION,
        "mode": MODE,
    }
    write_json_atomic(run_root / "human-gt-r0-source.json", record)
    return record


def _load_human_rubrics(
    private_gt: Path, prompt_ids: Sequence[str]
) -> dict[str, list[dict[str, Any]]]:
    wanted = set(prompt_ids)
    rubrics: dict[str, list[dict[str, Any]]] = {}
    for row in _jsonl(private_gt):
        prompt_id = str(row["prompt_id"])
        if prompt_id not in wanted:
            continue
        raw = row.get("gold_rubric")
        if not isinstance(raw, list) or not raw:
            raise MinimumExperimentError(f"empty human rubric: {prompt_id}")
        criteria = []
        for index, item in enumerate(raw):
            if not isinstance(item, Mapping):
                raise MinimumExperimentError(f"malformed human criterion: {prompt_id}")
            text = item.get("criterion")
            points = item.get("points")
            if (
                not isinstance(text, str)
                or not text.strip()
                or isinstance(points, bool)
                or not isinstance(points, (int, float))
            ):
                raise MinimumExperimentError(f"malformed human criterion: {prompt_id}")
            criterion_id = f"human-gold-{index:03d}"
            criteria.append(
                {
                    "criterion_id": criterion_id,
                    "criterion_key": hashlib.sha256(
                        f"{prompt_id}\0{text}".encode()
                    ).hexdigest(),
                    "text": text,
                    "points": float(points),
                }
            )
        rubrics[prompt_id] = criteria
    if set(rubrics) != wanted:
        raise MinimumExperimentError(
            f"human rubric prompts missing: {sorted(wanted - set(rubrics))}"
        )
    return rubrics


def _output_path(root: Path, stage: str, policy_id: str, prompt_id: str) -> Path:
    stem = f"{policy_id}-{_shard_name(policy_id, prompt_id)}.jsonl"
    return root / stage / "shards" / stem


def _select_shard(
    run_root: Path,
    policy_id: str,
    prompt_id: str,
    candidates: Sequence[dict[str, Any]],
) -> Path:
    output_path = _output_path(run_root, SELECTION_STAGE, policy_id, prompt_id)
    if output_path.is_file():
        return output_path
    score_path = _output_path(run_root, SCORE_STAGE, policy_id, prompt_id)
    scores = read_jsonl(score_path)
    score_by_id = {int(row["global_candidate_id"]): float(row["score"]) for row in scores}
    candidate_by_id = {int(row["global_candidate_id"]): row for row in candidates}
    if set(score_by_id) != set(candidate_by_id):
        raise MinimumExperimentError(f"human-GT score inventory mismatch: {policy_id, prompt_id}")
    permutations = fixed_candidate_permutations(
        list(candidate_by_id),
        seed=f"{SEED_PREFIX}:{policy_id}:{prompt_id}",
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
    for permutation_index, permutation in enumerate(permutations):
        for n in N_GRID:
            selected_id = int(select_best_of_n(score_by_id, permutation, n))
            selected = candidate_by_id[selected_id]
            selections.append(
                {
                    "policy_id": policy_id,
                    "policy_step": int(selected["policy_step"]),
                    "prompt_id": prompt_id,
                    "rubric_id": f"{prompt_id}:human_gt:R_0",
                    "mode": MODE,
                    "rubric_step": 0,
                    "n": n,
                    "permutation": permutation_index,
                    "pool_hash": pool_hash,
                    "global_candidate_id": selected_id,
                    "response_id": selected["response_id"],
                    "response_text": selected["response_text"],
                }
            )
    _publish_jsonl(output_path, selections)
    return output_path


def score_and_select(
    run_root: Path,
    private_gt: Path,
    score_endpoint: str,
    *,
    prompt_count: int,
    policy_step: int,
    workers: int,
) -> dict[str, Any]:
    prompt_ids = ordered_prompt_subset(run_root, prompt_count)
    targets = target_groups(prompt_ids, (policy_step,))
    rubrics = _load_human_rubrics(private_gt, prompt_ids)
    conversations = _audit_conversations(run_root)
    client = TargetScoreClient(score_endpoint, workers=workers)
    bon_path = run_root / "generate-bon" / "bon_pool.jsonl"
    write_json_atomic(
        run_root / SCORE_STAGE / "manifest.json",
        {
            "schema_version": 1,
            "comparison": [MODE],
            "policy_steps": [policy_step],
            "prompt_ids": list(prompt_ids),
            "pool_size": 1024,
            "n_grid": list(N_GRID),
            "permutations": PERMUTATIONS,
            "judge_prompt_version": PAPER_JUDGE_PROMPT_VERSION,
            "aggregation": "HealthBench signed-point normalized weighted score",
            "score_identity": client.identity(),
            "score_routing": {
                "strategy": "deterministic-length-balanced-largest-first",
                "upstream_weights": list(client.routing_weights()),
            },
            "inputs": {
                str(bon_path): sha256_file(bon_path),
                str(private_gt): sha256_file(private_gt),
            },
        },
    )
    write_json_atomic(
        run_root / SELECTION_STAGE / "manifest.json",
        {
            "schema_version": 1,
            "comparison": [MODE],
            "policy_steps": [policy_step],
            "prompt_ids": list(prompt_ids),
            "n_grid": list(N_GRID),
            "permutations": PERMUTATIONS,
            "tie_break": "lowest_global_candidate_id",
            "candidate_permutation_seed_prefix": SEED_PREFIX,
            "bon_sha256": sha256_file(bon_path),
        },
    )

    completed = 0
    pairs = 0
    seen: set[tuple[str, str]] = set()
    for (policy_id, prompt_id), candidates in _bon_groups(bon_path):
        key = policy_id, prompt_id
        if key not in targets:
            continue
        seen.add(key)
        if len(candidates) != 1024:
            raise MinimumExperimentError(f"candidate pool mismatch: {key}")
        score_path = _output_path(run_root, SCORE_STAGE, policy_id, prompt_id)
        trace_path = (
            run_root
            / SCORE_STAGE
            / "criterion-shards"
            / f"{policy_id}-{_shard_name(policy_id, prompt_id)}.jsonl.gz"
        )
        if score_path.is_file() != trace_path.is_file():
            raise MinimumExperimentError(f"partial human-GT score shard: {key}")
        if not score_path.is_file():
            criteria = rubrics[prompt_id]
            conversation = conversations[prompt_id]
            tasks = [
                (
                    (str(criterion["criterion_key"]), str(candidate["response_id"])),
                    _score_prompt(
                        str(criterion["text"]),
                        str(candidate["response_text"]),
                        conversation=conversation,
                        prompt_version=PAPER_JUDGE_PROMPT_VERSION,
                    ),
                )
                for candidate in candidates
                for criterion in criteria
            ]
            scores = client.score(tasks)
            pairs += len(scores)
            candidate_by_response = {
                str(candidate["response_id"]): candidate for candidate in candidates
            }

            def criterion_rows() -> Iterator[dict[str, Any]]:
                for (criterion_key, response_id), score in sorted(scores.items()):
                    yield {
                        "policy_id": policy_id,
                        "policy_step": policy_step,
                        "prompt_id": prompt_id,
                        "response_id": response_id,
                        "global_candidate_id": candidate_by_response[response_id][
                            "global_candidate_id"
                        ],
                        "criterion_key": criterion_key,
                        **score,
                        "parse_success": True,
                    }

            _publish_gzip_jsonl(trace_path, criterion_rows())
            score_rows = []
            point_weights = {
                str(criterion["criterion_id"]): float(criterion["points"])
                for criterion in criteria
            }
            for candidate in candidates:
                response_id = str(candidate["response_id"])
                criterion_scores = {
                    str(criterion["criterion_id"]): scores[
                        (str(criterion["criterion_key"]), response_id)
                    ]["probability_yes"]
                    for criterion in criteria
                }
                proxy_score = weighted_gold_score(criterion_scores, point_weights)
                score_rows.append(
                    {
                        "policy_id": policy_id,
                        "policy_step": policy_step,
                        "prompt_id": prompt_id,
                        "global_candidate_id": candidate["global_candidate_id"],
                        "response_id": response_id,
                        "rubric_id": f"{prompt_id}:human_gt:R_0",
                        "mode": MODE,
                        "rubric_step": 0,
                        "criterion_count": len(criteria),
                        "criterion_positive_points": sum(
                            max(float(criterion["points"]), 0.0) for criterion in criteria
                        ),
                        "score": proxy_score,
                    }
                )
            _publish_jsonl(score_path, score_rows)
        _select_shard(run_root, policy_id, prompt_id, candidates)
        completed += 1
        write_json_atomic(
            run_root / SCORE_STAGE / "progress.json",
            {
                "completed_prompt_policy_shards": completed,
                "expected_prompt_policy_shards": len(targets),
                "last_policy_id": policy_id,
                "last_prompt_id": prompt_id,
                "criterion_pairs_scored_this_process": pairs,
            },
            immutable=False,
        )
    if seen != targets:
        raise MinimumExperimentError(f"target groups absent: {sorted(targets - seen)}")
    return {
        "policy_id": f"pi_{policy_step}",
        "prompt_ids": list(prompt_ids),
        "completed_prompt_policy_shards": completed,
        "criterion_pairs_scored_this_process": pairs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("prepare", "score"), required=True)
    parser.add_argument("--source-run-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--private-gt", type=Path, required=True)
    parser.add_argument("--score-endpoint", default="http://127.0.0.1:8104")
    parser.add_argument("--prompt-count", type=int, default=5)
    parser.add_argument("--policy-step", type=int, default=3)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    prepared = prepare_run(args.source_run_root, args.run_root)
    if args.phase == "prepare":
        result: dict[str, Any] = prepared
    else:
        result = score_and_select(
            args.run_root,
            args.private_gt,
            args.score_endpoint,
            prompt_count=args.prompt_count,
            policy_step=args.policy_step,
            workers=args.workers,
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
