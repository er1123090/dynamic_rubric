"""CPU-only committed-run diagnostics; never relabel training data as probe B.

The offline-component ablation reuses grades from the UNION judge context.
It is neither a separate R0 judge call nor a fresh/stale update-value estimate.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from dynamic_rubric.phase1.metrics import ScoreGroup, group_metrics, kendall_tau_b


def rows(path):
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def csv_out(path, data):
    if not data:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)


def mean(data, key):
    vals = [r[key] for r in data if r.get(key) is not None]
    return float(np.mean(vals)) if vals else None


def paired_ci(values, seed=11, replicates=5000):
    if not values or replicates <= 0:
        raise ValueError("nonempty paired values and positive replicates required")
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    sampled = np.concatenate([x[rng.integers(0, len(x), (min(200, replicates-start), len(x)))].mean(1)
                              for start in range(0, replicates, 200)])
    return [float(x.mean()), *np.quantile(sampled, [0.025, 0.975]).tolist()]


def pooled_criteria(records, prefix):
    counts = {kind: sum(r[f"{prefix}_{kind}_criterion_count"] for r in records)
              for kind in ("effective", "saturated", "dead")}
    total = sum(counts.values())
    return {"count": total, **counts,
            **{kind + "_ratio": value / total if total else None for kind, value in counts.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--through-step", type=int, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--bootstrap-replicates", type=int, default=5000)
    args = ap.parse_args()
    args.run = args.run.resolve()
    args.output = args.output.resolve()
    cfg = json.loads((args.run / "config.resolved.json").read_text())
    probe_path = ROOT / "outputs/medicine/shared/seed-11/manifests/fixed_train_probe.json"
    probe = set(json.loads(probe_path.read_text())["prompt_ids"])
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    all_groups, steps, provenance, hashes = [], [], [], []
    seen_ids = set()
    visits = Counter()
    cumulative_chars = 0
    for step in range(1, args.through_step + 1):
        d = args.run / "verl-run/online_steps" / f"step-{step:06d}"
        commit = json.loads((d / "commit.json").read_text())
        assert commit["state"] == "committed" and commit["optimizer_update_index"] == step
        for name, expected in commit["artifacts"].items():
            if "." not in name:
                continue
            p = d / name
            actual = hashlib.sha256(p.read_bytes()).hexdigest()
            assert actual == expected, (step, name)
            hashes.append({"step": step, "path": str(p.relative_to(ROOT)), "sha256": actual})
        batch = json.loads((d / "batch.json").read_text())
        assert batch["current_policy"]["policy_version"] == step - 1
        rewards = rows(d / "rewards.jsonl")
        rubrics = {r["prompt_occurrence_id"]: r for r in rows(d / "rubric_unions.jsonl")}
        current = {r["response_id"]: r for r in rows(d / "current_responses.jsonl")}
        controls = rows(d / "control_responses.jsonl")
        judge = rows(d / "grader_receipts.jsonl")
        assert {r["response_id"] for r in rewards} == set(current)
        assert not seen_ids.intersection(current)
        seen_ids.update(current)
        pid = {r["metadata"]["prompt_occurrence_id"]: r["prompt_id"] for r in judge}
        assert all(r["returned_model"] == cfg["models"]["judge"]["model"] for r in judge)
        assert all(r["policy"]["policy_version"] == step - 1 for r in current.values())
        assert all(r["policy"]["policy_version"] == 0 for r in controls)
        assert not set(current).intersection(r["response_id"] for r in controls)
        assert len(judge) == len(current) == len(rewards)
        extraction = rows(d / "extraction_receipts.jsonl")
        candidate_n = Counter()
        for r in extraction:
            candidate_n[r["metadata"]["prompt_occurrence_id"]] += len(json.loads(r["result_text"])["new_criteria"])
        elicited = defaultdict(set)
        for r in rows(d / "blind_pairs.jsonl"):
            elicited[r["prompt_occurrence_id"]].add(r["assignment"]["current_index"])
        grouped = defaultdict(list)
        for r in rewards:
            grouped[r["prompt_occurrence_id"]].append(r)
        per_step = []
        for occurrence, group in grouped.items():
            group.sort(key=lambda r: r["rollout_index"])
            assert [r["rollout_index"] for r in group] == list(range(16))
            assert len(elicited[occurrence]) == 8
            union = rubrics[occurrence]
            offline, online = union["offline_criteria"], union["online_criteria"]
            criteria = offline + online
            cid = [c["criterion_id"] for c in criteria]
            assert len(cid) == len(set(cid))
            weights = {c["criterion_id"]: c["weight"] for c in criteria}
            base_weight = sum(c["weight"] for c in offline)
            new_weight = sum(c["weight"] for c in online)
            raw_grades = [dict(r["grades"]) for r in group]
            assert all(set(g) == set(cid) for g in raw_grades)
            for r, g in zip(group, raw_grades):
                assert r["rubric_hash"] == union["content_hash"]
                assert r["numerator"] == sum(weights[k] * v for k, v in g.items())
                assert r["denominator"] == base_weight + new_weight
                assert abs(r["reward"] - r["numerator"] / r["denominator"]) < 1e-12
            response_ids = tuple(r["response_id"] for r in group)
            fresh_rewards = tuple(r["reward"] for r in group)
            rational_rewards = tuple(Fraction(r["numerator"], r["denominator"]) for r in group)
            assert (len(set(rational_rewards)) == 1) == (len(set(fresh_rewards)) == 1)
            base_rewards = tuple(sum(g[c["criterion_id"]] * c["weight"] for c in offline) / base_weight for g in raw_grades)
            grades = {k: tuple(g[k] for g in raw_grades) for k in cid}
            args_m = dict(epsilon_z=cfg["analysis"]["epsilon_z"], epsilon_t=cfg["analysis"]["epsilon_t"])
            fresh = group_metrics(ScoreGroup(pid[occurrence], union["content_hash"], str(step-1), response_ids, fresh_rewards, grades), **args_m)
            base = group_metrics(ScoreGroup(pid[occurrence], "offline_component_same_union_context", str(step-1), response_ids, base_rewards,
                                           {c["criterion_id"]: grades[c["criterion_id"]] for c in offline}), **args_m)
            new = group_metrics(ScoreGroup(pid[occurrence], "online_component", str(step-1), response_ids, fresh_rewards,
                                          {c["criterion_id"]: grades[c["criterion_id"]] for c in online}), **args_m)
            prompt_id = pid[occurrence]
            visits[prompt_id] += 1
            chars = [len(current[rid]["text"]) for rid in response_ids]
            cumulative_chars += sum(chars)
            record = dict(domain=cfg["domain"], method=cfg["method"], seed=cfg["seed"], global_step=step,
                          checkpoint_id=f"update-{step}", prompt_id=prompt_id, prompt_occurrence_id=occurrence,
                          response_id="|".join(response_ids), pool="train_batch", policy_checkpoint=f"policy-version-{step-1}",
                          evaluator_checkpoint=union["content_hash"], fresh_or_stale="fresh", visit=visits[prompt_id],
                          fixed_probe_member=prompt_id in probe, candidate_count=candidate_n[occurrence],
                          offline_count=len(offline), online_count=len(online), online_weight_fraction=new_weight/(base_weight+new_weight),
                          reward_mean=float(np.mean(fresh_rewards)), reward_one_fraction=float(np.mean(np.asarray(fresh_rewards)==1)),
                          reward_zero_fraction=float(np.mean(np.asarray(fresh_rewards)==0)), response_chars_mean=float(np.mean(chars)),
                          recovered_by_online=int(base["exact_zero_advantage"] == 1 and fresh["exact_zero_advantage"] == 0),
                          lost_by_online=int(base["exact_zero_advantage"] == 0 and fresh["exact_zero_advantage"] == 1),
                          component_kendall_tau_b=kendall_tau_b(base_rewards, fresh_rewards))
            for prefix, metrics in (("fresh",fresh),("offline_component",base),("online",new)):
                for kind in ("effective", "saturated", "dead"):
                    ratio = metrics[f"{kind}_criterion_ratio"]
                    record[f"{prefix}_{kind}_criterion_count"] = round(ratio * metrics["criterion_count"]) if ratio is not None else 0
                for k in ("exact_zero_advantage","near_zero_advantage","reward_std","pairwise_tie_rate","pairwise_separation_rate",
                          "effective_criterion_ratio","saturated_criterion_ratio","dead_criterion_ratio","top_median_margin"):
                    if prefix == "online" and "criterion_ratio" not in k:
                        continue
                    record[f"{prefix}_{k}"] = metrics[k]
            # The eight responses not used for elicitation still belong to training.
            idx = [i for i in range(16) if i not in elicited[occurrence]]
            record["non_elicitation_train8_zar"] = int(len({fresh_rewards[i] for i in idx}) == 1)
            all_groups.append(record)
            per_step.append(record)
            for r in group:
                provenance.append(dict(domain=cfg["domain"], method=cfg["method"], seed=cfg["seed"], global_step=step,
                                       checkpoint_id=f"update-{step}", prompt_id=prompt_id, response_id=r["response_id"], pool="train_batch",
                                       policy_checkpoint=f"policy-version-{step-1}", evaluator_checkpoint=union["content_hash"],
                                       fresh_or_stale="fresh", used_for_elicitation=r["rollout_index"] in elicited[occurrence],
                                       used_for_gradient=True, source=str(d.relative_to(ROOT)/"current_responses.jsonl")))
        numeric = [k for k,v in per_step[0].items() if isinstance(v,(int,float)) and k not in ("global_step","seed","visit")]
        summary = dict(step=step, policy_version=step-1, prompts=len(per_step), responses=len(rewards),
                       cumulative_prompt_exposures=len(all_groups), cumulative_completions=len(provenance),
                       commit_time_utc=datetime.fromtimestamp((d/"commit.json").stat().st_mtime, timezone.utc).isoformat())
        summary.update({k: mean(per_step,k) for k in numeric})
        summary["component_kendall_tau_b"] = mean(per_step,"component_kendall_tau_b")
        summary["judge_wall_minutes_proxy"] = ((d/"grader_receipts.jsonl").stat().st_mtime - (d/"rubric_unions.jsonl").stat().st_mtime)/60
        steps.append(summary)

    numeric = [k for k,v in all_groups[0].items() if isinstance(v,(int,float)) and k not in ("global_step","seed","visit")]
    epochs=[]
    for start in range(1, args.through_step + 1, 16):
        end = min(start + 15, args.through_step)
        subset=[r for r in all_groups if start<=r["global_step"]<=end]
        if subset:
            epochs.append(dict(steps=f"{start}-{end}",prompts=len(subset),**{k:mean(subset,k) for k in numeric}))
    by_pid=defaultdict(list)
    for r in all_groups:by_pid[r["prompt_id"]].append(r)
    paired=[]
    for pid,rs in by_pid.items():
        if len(rs)>=2:
            a,b=rs[0],rs[1]
            paired.append(dict(prompt_id=pid,step_first=a["global_step"],step_second=b["global_step"],
                               **{k:b[k]-a[k] for k in ("fresh_exact_zero_advantage","fresh_pairwise_tie_rate",
                                                       "fresh_effective_criterion_ratio","online_count","response_chars_mean","reward_mean")}))
    paired_summary={k:paired_ci([r[k] for r in paired], replicates=args.bootstrap_replicates) for k in paired[0] if k not in ("prompt_id","step_first","step_second")} if paired else {}
    summary=dict(cutoff_utc=datetime.now(timezone.utc).isoformat(),through_step=args.through_step,
                 committed_prompt_exposures=len(all_groups),unique_training_prompts=len(by_pid),current_response_count=len(provenance),
                 control_response_uses=len(provenance)//2,elicitation_comparisons=len(provenance)//2,
                 unique_response_ids=len(seen_ids),sha256_validated_files=len(hashes),
                 fixed_probe_ids=len(probe),fixed_probe_ids_seen_in_training=len(probe.intersection(by_pid)),
                 probe_B_score_groups_found=None,stale_score_groups_found=None,
                 counterfactual_scope="Not inspected by this training-only analyzer; null is not zero coverage.",
                 bootstrap_replicates=args.bootstrap_replicates, bootstrap_seed=11,
                 scope="Committed actual-training data only. Offline component ablation uses union-context grades, not old-evaluator scores.",
                 epsilon_z=cfg["analysis"]["epsilon_z"],epsilon_t=cfg["analysis"]["epsilon_t"],
                 overall={k:mean(all_groups,k) for k in numeric},epochs=epochs,
                 paired_visit_n=len(paired),paired_visit_deltas_95ci=paired_summary,
                 component_recovered_groups=sum(r["recovered_by_online"] for r in all_groups),
                 component_lost_groups=sum(r["lost_by_online"] for r in all_groups),
                 total_online_criteria=sum(r["online_count"] for r in all_groups),
                 total_raw_candidates=sum(r["candidate_count"] for r in all_groups),
                 visit_count_distribution=dict(Counter(visits.values())))
    summary["criterion_pooled"] = {p: pooled_criteria(all_groups,p) for p in ("fresh", "offline_component", "online")}
    summary["exact_rational_groups_verified"] = len(all_groups)
    criterion_steps = []
    for step in range(1, args.through_step + 1):
        selected = [r for r in all_groups if r["global_step"] == step]
        for prefix in ("fresh", "offline_component", "online"):
            criterion_steps.append({"step":step,"component":prefix,**pooled_criteria(selected,prefix)})
    csv_out(out/"criterion_pooled_by_step.csv",criterion_steps)
    csv_out(out/"training_by_step.csv",steps)
    csv_out(out/"training_by_prompt.csv",all_groups)
    csv_out(out/"paired_prompt_visits.csv",paired)
    csv_out(out/"response_provenance.csv",provenance)
    csv_out(out/"source_hashes.csv",hashes)
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
    plt.rcParams.update({"font.size":10,"svg.fonttype":"none","axes.spines.top":False,"axes.spines.right":False})
    x=np.array([r["step"] for r in steps])
    def y(key):return np.array([r[key] for r in steps],dtype=float)
    def save(fig,name):
        fig.tight_layout()
        fig.savefig(out/(name+".svg"),bbox_inches="tight")
        fig.savefig(out/(name+".png"),dpi=160,bbox_inches="tight")
        plt.close(fig)
    fig,ax=plt.subplots(1,3,figsize=(14,3.8))
    ax[0].plot(x,100*y("fresh_exact_zero_advantage"),"o-",label="Exact ZAR")
    ax[0].plot(x,100*y("fresh_near_zero_advantage"),"s--",label="Near ZAR (SD <= 0.01)")
    ax[0].set(ylabel="Prompt groups (%)",title="A. Fresh reward usability",ylim=(-1,25));ax[0].legend()
    ax[1].plot(x,100*y("fresh_pairwise_tie_rate"),"o-",color="#d55e00")
    ax[1].set(ylabel="Response pairs (%)",title="B. Pairwise ties (epsilon = 0.01)")
    ax[2].plot(x,y("fresh_reward_std"),"o-",label="Reward SD")
    ax[2].plot(x,y("fresh_top_median_margin"),"s--",label="Top-median margin")
    ax[2].set(title="C. Within-group reward spread");ax[2].legend()
    for a in ax:a.set_xlabel("Optimizer update (responses from policy s-1)");a.axvline(16.5,color="gray",ls=":")
    fig.suptitle(f"OnlineRubrics / Medicine / seed 11 - TRAINING diagnostics, through update {args.through_step}",y=1.04)
    save(fig,"figure1_fresh_training_discriminability")
    fig,ax=plt.subplots(1,3,figsize=(14,3.8))
    ax[0].plot(x,y("candidate_count"),label="Raw candidates / prompt")
    ax[0].plot(x,y("online_count"),label="Retained online criteria / prompt")
    ax[0].set(title="A. Elicitation and deduplication",ylabel="Count");ax[0].legend(fontsize=8)
    for prefix,label,c in (("offline_component","Offline component","#0072b2"),("online","New online criteria","#d55e00")):
        ax[1].plot(x,100*y(prefix+"_effective_criterion_ratio"),label=label,color=c)
    ax[1].set(title="B. Effective criterion ratio",ylabel="Prompt-balanced mixed criteria (%)");ax[1].legend(fontsize=8)
    ax[2].plot(x,100*y("offline_component_pairwise_tie_rate"),label="Offline component only")
    ax[2].plot(x,100*y("fresh_pairwise_tie_rate"),label="Full fresh union")
    ax[2].set(title="C. Same-grades component ablation",ylabel="Pairwise tie rate (%)");ax[2].legend(fontsize=8)
    for a in ax:a.set_xlabel("Optimizer update");a.axvline(16.5,color="gray",ls=":")
    fig.suptitle("Component diagnostics only: NOT fresh-vs-stale, NOT an independent R0 judgment",y=1.04)
    save(fig,"figure2_online_criterion_diagnostics")
    fig,ax=plt.subplots(1,3,figsize=(14,3.8))
    ax[0].plot(x,y("reward_mean"),"o-",label="Mean fresh reward")
    ax[0].plot(x,y("reward_one_fraction"),"s--",label="Reward = 1 mass")
    ax[0].plot(x,y("reward_zero_fraction"),"^--",label="Reward = 0 mass")
    ax[0].set(title="A. Reward / ceiling / floor",ylim=(0,1));ax[0].legend(fontsize=8)
    ax[1].plot(x,100*y("fresh_exact_zero_advantage"),"o-",label="Exact ZAR")
    ax[1].plot(x,100*y("fresh_near_zero_advantage"),"s--",label="Near ZAR (SD <= 0.01)")
    ax[1].set(title="B. All-tie groups",ylabel="Prompt groups (%)");ax[1].legend(fontsize=8)
    pooled = [r for r in criterion_steps if r["component"] == "fresh"]
    for kind in ("effective","saturated","dead"):
        ax[2].plot(x,[100*r[kind+"_ratio"] for r in pooled],"o-",label=kind)
    ax[2].set(title="C. Criterion states (pooled)",ylabel="Criterion occurrences (%)");ax[2].legend(fontsize=8)
    for a in ax:a.set_xlabel("Optimizer update (response policy s-1)")
    fig.suptitle(f"RQ1 Figure 1 counterpart: DYNAMIC fresh TRAINING diagnostics through {args.through_step}",y=1.04)
    save(fig,"figure3_rq1_style_training_saturation")
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
