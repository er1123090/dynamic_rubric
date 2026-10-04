#!/usr/bin/env python3
"""Read-only evidence checks, with a generated JSON verification receipt."""
import csv
import hashlib
import json
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/horizon/medicine/metric_framework_reanalysis_20260908"
OLD = ROOT / "results/horizon/medicine/additional_discriminability_20260907"
records = json.loads((OLD / "artifact_manifest.json").read_text())["files"]
checks = []
for item in records:
    path = ROOT / item["path"]
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    checks.append({"path": item["path"], "expected": item["sha256"],
                   "actual": actual, "matches": actual == item["sha256"]})
assert len(checks) == 42
assert all(item["matches"] for item in checks)

def rows(name, folder=OUT):
    with (folder / name).open(newline="") as handle:
        return list(csv.DictReader(handle))

assert len(rows("trajectory_training.csv")) == 46
assert len(rows("trajectory_heldout_base100.csv")) == 10
summary = {row["cohort"]: row for row in rows("transition_summary.csv")}
pooled = summary["pooled300"]
assert int(pooled["n_prompt_checkpoint_events"]) == 2700
assert int(pooled["recovered_events"]) == 34
assert int(pooled["unique_prompts_ever_recovered"]) == 21
assert sum(int(pooled[key]) for key in (
    "stable_usable_events", "new_collapse_events", "recovered_events",
    "persistent_collapse_events")) == 2700
assert abs(float(pooled["egr_gain"]) - float(pooled["zar_reduction"])) < 1e-15
fresh = rows("transition_fresh_gains.csv")
metric_map = {"mad": "group_mad", "tie": "pairwise_tie", "zar": "exact_zar"}
errors = []
for row in rows("pool_b_contrasts.csv", OLD):
    if row["contrast"] != "mean_fresh_gain_steps3_48" or row["metric"] not in metric_map:
        continue
    cohort = "pooled300" if row["cohort"] == "total300" else row["cohort"]
    values = [float(item["estimate"]) for item in fresh
              if item["cohort"] == cohort
              and item["metric"] == metric_map[row["metric"]]]
    assert len(values) == 9
    errors.append(abs(sum(values) / len(values) - float(row["estimate"])))
assert max(errors) < 1e-15
images = sorted(OUT.glob("figure*.png"))
assert len(images) == 7
for path in images:
    with Image.open(path) as im:
        im.verify()
    assert path.with_suffix(".svg").is_file()
assert "{{" not in (OUT / "core_metric_report.md").read_text()
receipt = {
    "original_manifest_files_checked": len(checks),
    "original_manifest_all_match": True,
    "original_file_checks": checks,
    "figures_verified": [p.name for p in images],
    "training_rows": 46, "heldout_rows": 10,
    "fresh_core_mean_max_abs_error": max(errors),
    "pooled_transition_counts_verified": True,
    "fresh_gain_identity": "EGR gain = ZAR(old) - ZAR(fresh) = recovered - new collapse",
    "new_model_or_judge_calls": 0,
}
(OUT / "report_verification.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps({k: v for k, v in receipt.items() if k != "original_file_checks"}))


