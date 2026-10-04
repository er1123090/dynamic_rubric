"""Offline current34-to-stale32 KL-style audit, not exact distribution KL."""

import argparse
import base64
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
from collections import defaultdict

import numpy as np


def read_rows(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def token_estimators(current, stale):
    d = np.asarray(current, dtype=np.float64) - np.asarray(stale, dtype=np.float64)
    ratio_log = -d
    clipped = np.clip(ratio_log, -20, 20)
    return d, np.expm1(clipped) - clipped, int(np.count_nonzero(clipped != ratio_log))


def paired_ci(values, replicates, seed):
    x = np.asarray(values)
    rng = np.random.default_rng(seed)
    means = np.concatenate(
        [
            x[rng.integers(len(x), size=(min(200, replicates - i), len(x)))].mean(1)
            for i in range(0, replicates, 200)
        ]
    )
    return np.quantile(means, [0.025, 0.975]).tolist()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--bootstrap-replicates", type=int, default=20000)
    p.add_argument("--seed", type=int, default=11)
    args = p.parse_args()
    files = [
        args.source / "kl_scores" / f"policy-step-{step}_pool-step-34.jsonl" for step in (32, 34)
    ]
    old, new = [{r["response_id"]: r for r in read_rows(path)} for path in files]
    assert set(old) == set(new) and len(new) == 1600
    prompt = defaultdict(lambda: defaultdict(list))
    total = [0.0, 0.0]
    count = 0
    clipped = 0
    for rid, a in new.items():
        b = old[rid]
        for field in ("prompt_id", "sample_index", "response_token_count", "response_token_hash"):
            assert a[field] == b[field], (rid, field)
        arrays = [
            np.frombuffer(base64.b64decode(r["response_token_logprobs_f32le_b64"]), dtype="<f4")
            for r in (a, b)
        ]
        assert all(len(x) == a["response_token_count"] for x in arrays)
        k1, k3, nclip = token_estimators(*arrays)
        for name, values, i in (("k1", k1, 0), ("k3", k3, 1)):
            prompt[a["prompt_id"]][name].append(float(values.mean()))
            total[i] += float(values.sum())
        count += len(k1)
        clipped += nclip
    assert len(prompt) == 100 and all(len(x["k1"]) == 16 for x in prompt.values())
    summary = {
        "direction": "current34 || stale32; sampled on pi34 Pool B",
        "estimator_scope": "nucleus-sampled, retokenized response-only raw-softmax KL-style proxy; not unbiased exact KL",
        "responses": len(new),
        "prompts": len(prompt),
        "tokens": count,
        "clipped_tokens": clipped,
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seed": args.seed,
    }
    for name, i in (("k1", 0), ("k3", 1)):
        x = [statistics.mean(r[name]) for r in prompt.values()]
        summary[name] = {
            "prompt_balanced_mean": statistics.mean(x),
            "prompt_balanced_se": statistics.stdev(x) / math.sqrt(len(x)),
            "token_weighted_mean": total[i] / count,
        }
    inventory = {}
    pool_ids = {}
    for step in (32, 34):
        for pool, size in (("A", 8), ("B", 16)):
            path = args.source / "responses" / f"checkpoint-{step:06d}" / f"probe_{pool}.jsonl"
            files.append(path)
            rows = read_rows(path)
            key = f"step{step}_pool{pool}"
            ids = {r["response_id"] for r in rows}
            pids = {r["prompt_id"] for r in rows}
            assert len(ids) == len(rows) == 100 * size and pids == set(prompt)
            pool_ids[key] = ids
            for r in rows:
                prompt[r["prompt_id"]][key].append(r["usage"]["completion_tokens"])
            inventory[key] = {
                "responses": len(rows),
                "prompts": len(pids),
                "mean_completion_tokens": statistics.mean(
                    r["usage"]["completion_tokens"] for r in rows
                ),
                "cap3584_fraction": sum(r["usage"]["completion_tokens"] >= 3584 for r in rows)
                / len(rows),
            }
    assert pool_ids["step34_poolB"] == set(new)
    for i, key in enumerate(pool_ids):
        for other in list(pool_ids)[i + 1 :]:
            assert not pool_ids[key].intersection(pool_ids[other])
    length = {}
    for pool in ("A", "B"):
        changes = [
            statistics.mean(r[f"step34_pool{pool}"]) - statistics.mean(r[f"step32_pool{pool}"])
            for r in prompt.values()
        ]
        length[pool] = {
            "mean_delta_tokens_34_minus_32": statistics.mean(changes),
            "paired_prompt_ci95": paired_ci(changes, args.bootstrap_replicates, args.seed),
        }
    summary.update(
        pool_inventory=inventory,
        length_shift=length,
        all_pool_id_intersections=0,
        source_hashes={str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
    )
    args.output.mkdir(parents=True, exist_ok=True)
    records = [
        {"prompt_id": pid, **{k: statistics.mean(v) for k, v in r.items()}}
        for pid, r in sorted(prompt.items())
    ]
    with (args.output / "per_prompt.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
