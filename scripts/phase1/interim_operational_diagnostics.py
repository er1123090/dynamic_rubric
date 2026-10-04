"""Supplement interim CPU analysis with reward-validated log joins and sensitivity."""
from collections import defaultdict
import csv
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/phase1/medicine/online_rubrics/seed-11/interim-step30-20260907"
RUN = ROOT / "outputs/medicine/online_rubrics/seed-11/phase1-online-rubrics-medicine-full-20260905-seed11-final"


def main():
    steps={int(r["step"]):r for r in csv.DictReader((OUT/"training_by_step.csv").open())}
    groups=list(csv.DictReader((OUT/"training_by_prompt.csv").open()))
    logs={}
    for path in sorted((RUN.parent/"supervisor").glob("*.log"),key=lambda p:p.stat().st_mtime):
        for line in path.read_text(errors="replace").splitlines():
            if "training/global_step:" not in line or "critic/score/mean:" not in line:
                continue
            vals={k:float(v) for k,v in re.findall(r"([A-Za-z0-9_/]+):(-?[0-9]+(?:\.[0-9]+)?(?:e[+-]?[0-9]+)?)",line)}
            step=int(vals["training/global_step"])
            if step not in steps or abs(vals["critic/score/mean"]-float(steps[step]["reward_mean"]))>1e-6:
                continue
            n=int(steps[step]["responses"])
            logs[step]=dict(step=step,policy_version=step-1,source=str(path.relative_to(ROOT)),
                            reward_mean_check=vals["critic/score/mean"],response_tokens_mean=vals["response_length/mean"],
                            length_cap_ratio=vals["response_length/clip_ratio"],
                            approximate_response_tokens=round(vals["response_length/mean"]*n),
                            step_minutes=vals["perf/time_per_step"]/60,
                            actor_update_minutes=vals["timing_s/update_actor"]/60,
                            kl_penalty_loss_NOT_policy_distance=vals.get("actor/kl_loss"))
    with (OUT/"operational_log_metrics.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(next(iter(logs.values()))));w.writeheader();w.writerows(logs[s] for s in sorted(logs))
    eps=[0,0.001,0.005,0.01,0.02]
    sensitivity={e:dict(tie=[],near=[]) for e in eps}
    i,j=np.triu_indices(16,1)
    for step in steps:
        by=defaultdict(list)
        p=RUN/"verl-run/online_steps"/f"step-{step:06d}"/"rewards.jsonl"
        for line in p.open():
            r=json.loads(line);by[r["prompt_occurrence_id"]].append(r["reward"])
        for scores in by.values():
            x=np.array(scores);diff=np.abs(x[i]-x[j]);sd=np.std(x)
            for e in eps:
                sensitivity[e]["tie"].append(float((diff<=e).mean()))
                sensitivity[e]["near"].append(int(len(set(scores))==1) if e==0 else int(sd<=e))
    # Repeated prompt visits are resampled together; target is exposure-weighted.
    clusters=defaultdict(lambda:[0,0.0,0])
    for r in groups:
        c=clusters[r["prompt_id"]]
        c[0]+=int(r["recovered_by_online"])-int(r["lost_by_online"])
        c[1]+=float(r["offline_component_pairwise_tie_rate"])-float(r["fresh_pairwise_tie_rate"])
        c[2]+=1
    arr=np.array(list(clusters.values()));rng=np.random.default_rng(11)
    bs=[]
    for _ in range(25):
        selected=arr[rng.integers(0,len(arr),(200,len(arr)))].sum(1)
        bs.extend((selected[:,:2]/selected[:,2,None]).tolist())
    bs=np.array(bs)
    summary=dict(log_join="Only same step plus reward mean agreeing within 1e-6; no asynchronous-line offset inference.",
                 log_steps=sorted(logs),missing_log_steps=sorted(set(steps)-set(logs)),
                 approximate_cumulative_response_tokens=sum(x["approximate_response_tokens"] for x in logs.values()),
                 component_cluster_bootstrap_zar_gain_95ci=np.quantile(bs[:,0],[.025,.975]).tolist(),
                 component_cluster_bootstrap_tie_reduction_95ci=np.quantile(bs[:,1],[.025,.975]).tolist(),
                 epsilon_sensitivity=[dict(epsilon=e,tie_rate=float(np.mean(v["tie"])),near_zar=float(np.mean(v["near"]))) for e,v in sensitivity.items()],
                 token_counts_note="Reconstructed from rounded console mean times responses; not exact per-response token artifacts.")
    (OUT/"operational_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    plt.rcParams.update({"font.size":10,"svg.fonttype":"none","axes.spines.top":False,"axes.spines.right":False})
    fig,ax=plt.subplots(1,3,figsize=(14,3.8))
    x=sorted(logs)
    ax[0].plot(x,[logs[s]["response_tokens_mean"] for s in x],"o-",color="#0072b2")
    ax[0].set(title="A. Response length (verified console)",ylabel="Mean tokens / response")
    ax[1].plot(x,[100*logs[s]["length_cap_ratio"] for s in x],"o-",color="#d55e00")
    ax[1].set(title="B. Responses reaching 3584-token cap",ylabel="Responses (%)")
    x2=sorted(steps)
    ax[2].plot(x2,[float(steps[s]["reward_mean"]) for s in x2],"o-",label="Fresh union mean")
    ax[2].set(title="C. Training reward (not accuracy)",ylabel="Mean normalized score")
    for a in ax:a.set_xlabel("Optimizer update");a.axvline(16.5,color="gray",ls=":")
    fig.suptitle("Operational warning: longer outputs and late score decline can coexist with low ZAR",y=1.04)
    fig.tight_layout()
    for ext in ("png","svg"):fig.savefig(OUT/("figure3_length_and_training_score."+ext),dpi=160,bbox_inches="tight")
    print(json.dumps(summary,indent=2))
    for s in [1,16,24,25,27,28,29,30]:
        if s in logs:print(logs[s])


if __name__=="__main__":main()
