"""Human-GT R0 versus one OnlineRubric control on the shared BoN pools."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterator, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .artifacts import write_json_atomic
from .evaluation.gold_score import weighted_gold_score
from .hashing import sha256_file
from .judge_prompts import PAPER_JUDGE_PROMPT_VERSION, PAPER_JUDGE_SYSTEM_PROMPT
from .minimum_gold import PAPER_APPROVED_PAYLOAD_CATEGORIES
from .minimum_interim import ordered_prompt_subset, target_groups
from .minimum_staleness import (
    FOCAL_STEPS,
    MODE,
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
    select_bon_shard,
)
from .static_online_bon import CONTROL_STAGES, _online_rows, weighted_expanded_criteria


HUMAN_STATIC_R0 = "human_gt"
LINKED_STAGES = ("generate-bon", "train-static")


def _safe_link(source: Path, target: Path) -> None:
    if target.is_symlink():
        if target.resolve() != source.resolve():
            raise MinimumExperimentError(f"human/static OnlineRubric source link drift: {target}")
        return
    if target.exists():
        raise MinimumExperimentError(f"human/static OnlineRubric link target exists: {target}")
    target.symlink_to(source.resolve(), target_is_directory=True)


def load_human_rubrics(
    private_gt: Path, prompt_ids: Sequence[str]
) -> dict[str, tuple[dict[str, Any], ...]]:
    """Load the exact signed-point HealthBench criteria for the requested prompts."""

    wanted = {str(prompt_id) for prompt_id in prompt_ids}
    rubrics: dict[str, tuple[dict[str, Any], ...]] = {}
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
        rubrics[prompt_id] = tuple(criteria)
    if set(rubrics) != wanted:
        raise MinimumExperimentError(
            f"human rubric prompts missing: {sorted(wanted - set(rubrics))}"
        )
    return rubrics


def prepare_human_static_online_run(
    source_run_root: Path,
    run_root: Path,
    control: str,
    private_gt: Path,
) -> dict[str, Any]:
    """Create an isolated Human-GT-R0 versus OnlineRubric derived run."""

    if control not in CONTROL_STAGES:
        raise MinimumExperimentError(f"unsupported OnlineRubric control: {control}")
    source_run_root = source_run_root.resolve()
    run_root = run_root.absolute()
    if source_run_root == run_root or source_run_root.parent != run_root.parent.resolve():
        raise MinimumExperimentError("derived run must be a sibling distinct from its source")
    if not private_gt.is_file():
        raise MinimumExperimentError(f"missing private human rubric: {private_gt}")
    run_root.mkdir(parents=True, exist_ok=True)
    linked_stages = (*LINKED_STAGES, CONTROL_STAGES[control])
    for stage in linked_stages:
        source = source_run_root / stage
        if not source.exists():
            raise MinimumExperimentError(f"missing source stage: {source}")
        _safe_link(source, run_root / stage)

    source_record = {
        "schema_version": 1,
        "experiment": "human-gt-r0-vs-dynamic-onlinerubric-paper-judge-bon",
        "static_r0": HUMAN_STATIC_R0,
        "static_rubric": "private human HealthBench rubric with signed point weights",
        "online_control": control,
        "source_run_root": str(source_run_root),
        "private_gt_path": str(private_gt.resolve()),
        "private_gt_sha256": sha256_file(private_gt),
        "linked_stages": {
            stage: str((source_run_root / stage).resolve()) for stage in linked_stages
        },
        "judge_prompt_version": PAPER_JUDGE_PROMPT_VERSION,
        "dynamic_rubric_aggregation": "positive-integer-weighted-mean",
    }
    write_json_atomic(run_root / "human-static-online-rubric-source.json", source_record)
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
        "online_control": control,
        "static_r0": HUMAN_STATIC_R0,
    }


def score_and_select_human_static_online_subset(
    run_root: Path,
    source_run_root: Path,
    private_gt: Path,
    control: str,
    score_endpoint: str,
    *,
    prompt_count: int,
    workers: int,
) -> dict[str, Any]:
    """Score Human-GT R0 and one OnlineRubric, then publish resumable selections."""

    prompt_ids = ordered_prompt_subset(run_root, prompt_count)
    targets = target_groups(prompt_ids)
    human = load_human_rubrics(private_gt, prompt_ids)
    online = _online_rows(source_run_root, control)
    conversations = _audit_conversations(run_root)
    client = TargetScoreClient(score_endpoint, workers=workers)
    bon_path = run_root / "generate-bon" / "bon_pool.jsonl"
    routing_contract = run_root / "paper-judge-qwen-routing.json"
    if not routing_contract.is_file():
        raise MinimumExperimentError(f"missing judge routing contract: {routing_contract}")
    write_json_atomic(
        run_root / "score-proxy-minimum" / "manifest.json",
        {
            "schema_version": 1,
            "comparison": ["human_gt_r0", f"onlinerubric_{control}"],
            "static_r0": HUMAN_STATIC_R0,
            "online_control": control,
            "focal_steps": list(FOCAL_STEPS),
            "prompt_ids": list(prompt_ids),
            "pool_size": 1024,
            "n_grid": list(N_GRID),
            "permutations": PERMUTATIONS,
            "judge_prompt_version": PAPER_JUDGE_PROMPT_VERSION,
            "human_aggregation": "HealthBench signed-point normalized weighted score",
            "dynamic_aggregation": "positive-integer-weighted-mean",
            "score_identity": client.identity(),
            "score_routing": {
                "strategy": "deterministic-length-balanced-largest-first",
                "upstream_weights": list(client.routing_weights()),
                "contract_path": str(routing_contract.resolve()),
                "contract_sha256": sha256_file(routing_contract),
            },
            "inputs": {
                str(bon_path): sha256_file(bon_path),
                str(private_gt): sha256_file(private_gt),
                str(source_run_root / CONTROL_STAGES[control] / "onlinerubric_rubrics.jsonl"):
                    sha256_file(source_run_root / CONTROL_STAGES[control] / "onlinerubric_rubrics.jsonl"),
            },
        },
    )

    completed = 0
    total_pairs = 0
    seen: set[tuple[str, str]] = set()
    for (policy_id, prompt_id), candidates in _bon_groups(bon_path):
        group_key = policy_id, prompt_id
        if group_key not in targets:
            continue
        seen.add(group_key)
        step = int(candidates[0]["policy_step"])
        if policy_id != f"pi_{step}" or len(candidates) != 1024:
            raise MinimumExperimentError(f"BoN group inventory mismatch: {group_key}")
        shard_id = _shard_name(policy_id, prompt_id)
        score_shard = run_root / "score-proxy-minimum" / "shards" / f"{policy_id}-{shard_id}.jsonl"
        trace_shard = (
            run_root
            / "score-proxy-minimum"
            / "criterion-shards"
            / f"{policy_id}-{shard_id}.jsonl.gz"
        )
        if score_shard.is_file() != trace_shard.is_file():
            raise MinimumExperimentError(f"partial human/static score shard: {group_key}")
        if score_shard.is_file():
            select_bon_shard(run_root, policy_id, prompt_id, candidates)
            completed += 1
            continue

        human_criteria = human[prompt_id]
        dynamic_row = online[(prompt_id, step)]
        dynamic_criteria = weighted_expanded_criteria(
            prompt_id, step, control, tuple(dynamic_row["criteria"])
        )
        unique_criteria: dict[str, dict[str, Any]] = {}
        for criterion in human_criteria:
            unique_criteria[str(criterion["criterion_key"])] = {
                "criterion_key": str(criterion["criterion_key"]),
                "text": str(criterion["text"]),
                "source": "human_gt",
            }
        for criterion in dynamic_criteria:
            criterion_key = str(criterion["criterion_id"])
            previous = unique_criteria.setdefault(
                criterion_key,
                {
                    "criterion_key": criterion_key,
                    "text": str(criterion["text"]),
                    "source": "onlinerubric_pairwise",
                },
            )
            if previous["text"] != str(criterion["text"]):
                raise MinimumExperimentError(f"criterion key collision: {prompt_id} {criterion_key}")
        conversation = conversations[prompt_id]
        tasks = [
            (
                (criterion_key, str(candidate["response_id"])),
                _score_prompt(
                    str(criterion["text"]),
                    str(candidate["response_text"]),
                    conversation=conversation,
                    prompt_version=PAPER_JUDGE_PROMPT_VERSION,
                ),
            )
            for candidate in candidates
            for criterion_key, criterion in unique_criteria.items()
        ]
        scores = client.score(tasks)
        total_pairs += len(scores)
        candidate_by_response = {
            str(candidate["response_id"]): candidate for candidate in candidates
        }

        def criterion_rows() -> Iterator[dict[str, Any]]:
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
                    "criterion_source": unique_criteria[criterion_key]["source"],
                    **score,
                    "parse_success": True,
                }

        _publish_gzip_jsonl(trace_shard, criterion_rows())
        human_weights = {
            str(criterion["criterion_id"]): float(criterion["points"])
            for criterion in human_criteria
        }
        dynamic_ids = [str(criterion["criterion_id"]) for criterion in dynamic_criteria]
        score_rows = []
        for candidate in candidates:
            response_id = str(candidate["response_id"])
            human_scores = {
                str(criterion["criterion_id"]): scores[
                    (str(criterion["criterion_key"]), response_id)
                ]["probability_yes"]
                for criterion in human_criteria
            }
            dynamic_values = [
                scores[(criterion_id, response_id)]["probability_yes"]
                for criterion_id in dynamic_ids
            ]
            human_score = weighted_gold_score(human_scores, human_weights)
            dynamic_score = math.fsum(dynamic_values) / len(dynamic_values)
            common = {
                "policy_id": policy_id,
                "policy_step": step,
                "prompt_id": prompt_id,
                "global_candidate_id": candidate["global_candidate_id"],
                "response_id": response_id,
            }
            score_rows.extend(
                (
                    {
                        **common,
                        "rubric_id": f"{prompt_id}:human_gt:R_0",
                        "mode": "static",
                        "rubric_step": 0,
                        "criterion_count": len(human_criteria),
                        "criterion_positive_points": math.fsum(
                            max(float(criterion["points"]), 0.0)
                            for criterion in human_criteria
                        ),
                        "score": human_score,
                        "judge_repeat_score": human_score,
                        "judge_repeat_method": "deterministic_temperature_zero_cache_identity",
                        "static_r0": HUMAN_STATIC_R0,
                    },
                    {
                        **common,
                        "rubric_id": f"{prompt_id}:{MODE}:R_{step}",
                        "mode": MODE,
                        "rubric_step": step,
                        "criterion_count": len(dynamic_ids),
                        "score": dynamic_score,
                        "judge_repeat_score": dynamic_score,
                        "judge_repeat_method": "deterministic_temperature_zero_cache_identity",
                        "online_control": control,
                    },
                )
            )
        _publish_jsonl(score_shard, score_rows)
        select_bon_shard(run_root, policy_id, prompt_id, candidates)
        completed += 1
        write_json_atomic(
            run_root / "score-proxy-minimum" / "paper-judge-5prompt-progress.json",
            {
                "completed_prompt_policy_shards": completed,
                "expected_prompt_policy_shards": len(targets),
                "last_policy_id": policy_id,
                "last_prompt_id": prompt_id,
                "criterion_pairs_scored_this_process": total_pairs,
                "static_r0": HUMAN_STATIC_R0,
            },
            immutable=False,
        )
    if seen != targets:
        raise MinimumExperimentError(f"target groups absent: {sorted(targets - seen)}")
    return {
        "static_r0": HUMAN_STATIC_R0,
        "online_control": control,
        "prompt_ids": list(prompt_ids),
        "prompt_policy_shards": completed,
        "expected_prompt_policy_shards": len(targets),
        "criterion_pairs_scored_this_process": total_pairs,
    }
