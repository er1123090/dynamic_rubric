"""CPU-only RQ2 design audit; missing counterfactuals remain null, never zero.

Uses the validated committed-training export, verifies reward/rubric hashes,
recomputes criterion-pooled ratios, and indexes SAME-prompt historical rubrics.
It does not launch inference, change training, or relabel training as probe B.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import statistics


def read_jsonl(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def criterion_counts(rewards, criteria):
    counts = Counter(effective=0, saturated=0, dead=0)
    grade_maps = [dict(row["grades"]) for row in rewards]
    for criterion in criteria:
        values = [grades[criterion["criterion_id"]] for grades in grade_maps]
        assert set(values) <= {0, 1}
        counts["saturated" if all(values) else "dead" if not any(values) else "effective"] += 1
    return counts


def summarize_counts(counts):
    total = sum(counts.values())
    return dict(counts=counts, criterion_occurrences=total,
                **{f"{kind}_ratio": count / total if total else None for kind, count in counts.items()})


def write_csv(path, records):
    if records:
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--source-analysis", type=Path, required=True)
    parser.add_argument("--probe-manifest", type=Path, required=True)
    parser.add_argument("--through-step", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = args.source_analysis / "training_by_prompt.csv"
    with source.open() as handle:
        records = list(csv.DictReader(handle))
    assert {int(r["global_step"]) for r in records} == set(range(1, args.through_step + 1))
    lookup = {(int(r["global_step"]), r["prompt_occurrence_id"]): r for r in records}
    probe = set(json.loads(args.probe_manifest.read_text())["prompt_ids"])
    cfg = json.loads((args.run / "config.resolved.json").read_text())
    totals = defaultdict(Counter)
    history, pending, step_rows = {}, [], []
    exposures = 0
    verified = []
    for step in range(1, args.through_step + 1):
        directory = args.run / "verl-run/online_steps" / f"step-{step:06d}"
        commit = json.loads((directory / "commit.json").read_text())
        assert commit["state"] == "committed" and commit["optimizer_update_index"] == step
        for name in ("rubric_unions.jsonl", "rewards.jsonl", "batch.json"):
            actual = hashlib.sha256((directory / name).read_bytes()).hexdigest()
            assert actual == commit["artifacts"][name], (step, name)
            verified.append(dict(step=step, path=str(directory / name), sha256=actual))
        batch = json.loads((directory / "batch.json").read_text())
        assert batch["current_policy"]["policy_version"] == step - 1
        grouped = defaultdict(list)
        for row in read_jsonl(directory / "rewards.jsonl"):
            grouped[row["prompt_occurrence_id"]].append(row)
        step_counts, eligible = Counter(), 0
        for union in read_jsonl(directory / "rubric_unions.jsonl"):
            occurrence = union["prompt_occurrence_id"]
            record = lookup[step, occurrence]
            pid = record["prompt_id"]
            rewards = sorted(grouped[occurrence], key=lambda r: r["rollout_index"])
            response_ids = [r["response_id"] for r in rewards]
            assert len(response_ids) == len(set(response_ids)) == 16
            assert response_ids == record["response_id"].split("|")
            assert all(r["rubric_hash"] == union["content_hash"] for r in rewards)
            for component, criteria in (("offline", union["offline_criteria"]),
                                        ("online", union["online_criteria"]),
                                        ("union", union["offline_criteria"] + union["online_criteria"])):
                counts = criterion_counts(rewards, criteria)
                totals[component].update(counts)
                if component == "union":
                    step_counts.update(counts)
            fresh_zar = int(len({r["reward"] for r in rewards}) == 1)
            assert fresh_zar == int(float(record["fresh_exact_zero_advantage"]))
            if pid in history:
                stale = history[pid]
                assert stale["step"] < step
                eligible += 1
                pending.append(dict(domain=cfg["domain"], method=cfg["method"], seed=cfg["seed"],
                    global_step=step, checkpoint_id=f"update-{step}", prompt_id=pid,
                    response_id="|".join(response_ids), pool="train_batch", policy_checkpoint=f"policy-version-{step-1}",
                    evaluator_checkpoint=union["content_hash"], fresh_or_stale="paired_manifest_unscored",
                    stale_step=stale["step"], stale_policy_version=stale["step"]-1,
                    stale_evaluator_checkpoint=stale["rubric_hash"], evaluator_age_updates=step-stale["step"],
                    prompt_exposures_since_stale_creation=exposures-stale["pre_exposures"],
                    completions_since_stale_creation=16*(exposures-stale["pre_exposures"]),
                    exposure_clock="whole committed batches; evaluator creation assigned pre-update boundary",
                    stale_rubric_file=stale["path"], stale_prompt_occurrence_id=stale["occurrence"],
                    fresh_rubric_file=str(directory / "rubric_unions.jsonl"),
                    current_responses_file=str(directory / "current_responses.jsonl"),
                    fresh_rewards_file=str(directory / "rewards.jsonl"),
                    fixed_probe_member=pid in probe, primary_probe_eligible=False,
                    fresh_zar=fresh_zar, stale_zar=None, update_value=None,
                    fresh_tie=float(record["fresh_pairwise_tie_rate"]),
                    fresh_margin=float(record["fresh_top_median_margin"]),
                    scoring_status="pending_same_prompt_stale_on_identical_current_responses"))
            history[pid] = dict(step=step, rubric_hash=union["content_hash"], occurrence=occurrence,
                                path=str(directory / "rubric_unions.jsonl"), pre_exposures=exposures)
        exposures += len(grouped)
        step_rows.append(dict(update=step, response_policy_version=step-1, groups=len(grouped),
                              stale_rubric_available=eligible, same_pool_scored_pairs=0,
                              cumulative_prompt_exposures=exposures,
                              **{k:v for k,v in summarize_counts(step_counts).items() if k != "counts"}))
    ages = [r["evaluator_age_updates"] for r in pending]
    eligible_zar = sum(r["fresh_zar"] for r in pending)
    summary = dict(through_update=args.through_step, scope="committed_actual_training_only",
        source_export_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), verified_input_files=len(verified),
        prompt_visits=exposures, unique_prompts=len(history), primary_probe_scored_pairs=0,
        training_stale_scored_pairs=0, criterion_pooled={k:summarize_counts(v) for k,v in totals.items()},
        training_stale_rubric_available=len(pending), responses_requiring_stale_grading=16*len(pending),
        prior_same_prompt_rubric_unavailable=exposures-len(pending),
        fixed_probe_members_among_eligible=sum(r["fixed_probe_member"] for r in pending),
        stale_age_updates=dict(min=min(ages), median=statistics.median(ages), max=max(ages),
                              distribution=dict(sorted(Counter(ages).items()))),
        eligible_fresh_zar_count=eligible_zar, eligible_fresh_zar=eligible_zar/len(pending),
        logically_possible_v_batch_zar=[-eligible_zar/len(pending), 1-eligible_zar/len(pending)],
        bound_interpretation="Logical bounds without stale scores, NOT confidence intervals or measured update values",
        realized_v_adj_zar=None, reuse_horizon=None, predictor_target=None)
    write_csv(args.output / "pending_same_prompt_pairs.csv", pending)
    write_csv(args.output / "design_alignment_by_step.csv", step_rows)
    write_csv(args.output / "verified_inputs.csv", verified)
    (args.output / "design_alignment_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
