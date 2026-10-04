"""Join canonical rubric weights to same-response operational audit groups."""

import argparse
import csv
import json
from pathlib import Path
import statistics


def rows(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--inference_b", type=Path, required=True)
    p.add_argument("--analysis", type=Path, required=True)
    args = p.parse_args()
    weights = {}
    for step in range(1, 35):
        for r in rows(
            args.run / "verl-run/online_steps" / f"step-{step:06d}" / "rubric_unions.jsonl"
        ):
            signature = tuple(
                sorted(c["weight"] for c in r["offline_criteria"] + r["online_criteria"])
            )
            key = r["content_hash"]
            if key in weights:
                assert weights[key] == signature
            weights[key] = signature
    matches = {}
    for file in sorted((args.inference_b / "groups").rglob("*.json")):
        g = json.loads(file.read_text())
        fresh = weights[g["fresh"][0]["rubric_hash"]]
        stale = weights[g["stale"][0]["rubric_hash"]]
        assert g["fresh"][0]["prompt_id"] == g["stale"][0]["prompt_id"] == g["prompt_id"]
        matches[int(g["global_step"]), g["prompt_id"]] = {
            "count_matched": len(fresh) == len(stale),
            "count_and_weight_matched": fresh == stale,
        }
    assert len(matches) == 1692
    output = {
        "scope": "Operational same-response sensitivity; filtering not rubric modification or GT",
        "matching": "exact sorted multiset of all initial+online criterion weights",
        "groups": len(matches),
        "strata": {},
    }
    for stratum in ("inference_b", "trainer"):
        with (args.analysis / f"per_group_{stratum}.csv").open() as f:
            all_rows = list(csv.DictReader(f))
        groups = {}
        for mode in ("count_matched", "count_and_weight_matched"):
            selected = [r for r in all_rows if matches[int(r["global_step"]), r["prompt_id"]][mode]]
            stats = {
                "groups": len(selected),
                "unique_prompts": len({r["prompt_id"] for r in selected}),
            }
            for name in (
                "fresh_zar",
                "stale_zar",
                "v_adj_zar",
                "delta_tie_rate",
                "delta_margin",
                "kendall_tau_b",
            ):
                values = [float(r[name]) for r in selected if r[name] != ""]
                stats[name] = statistics.mean(values) if values else None
            groups[mode] = stats
        output["strata"][stratum] = groups
    (args.analysis / "count_weight_sensitivity.json").write_text(
        json.dumps(output, indent=2) + "\n"
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
