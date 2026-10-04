#!/usr/bin/env python3
"""Plot actual-training MAD, PTR, and ECR for the latest Online/Static runs.

The analysis reads only reward artifacts that were used during training. It does
not generate new responses, rubrics, or grades.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from fractions import Fraction
from itertools import combinations
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ONLINE_RUN = (
    ROOT
    / "outputs/medicine/online_rubrics/seed-11/"
    "phase1-online-rubrics-medicine-full-dense-20260919-seed11"
)
STATIC_RUN = (
    ROOT
    / "outputs/medicine/static_r0_matched/seed-11/"
    "phase1-static-r0-medicine-qwen3-4b-matched-20260914"
)
OUTPUT = ROOT / "results/latest_dense_online_vs_static_actual_training_metrics_20260926"

ONLINE_STEPS = tuple(range(1, 49))
STATIC_STEPS = tuple(range(1, 43))
RESPONSES_PER_PROMPT = 16
FULL_PROMPTS_PER_STEP = 96
REMAINDER_PROMPTS_PER_STEP = 60
STEPS_PER_EPOCH = 16
PTR_THRESHOLD = Fraction(1, 100)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_group(rewards: list[Fraction], grade_matrix: list[list[int]]) -> dict[str, float]:
    if len(rewards) != RESPONSES_PER_PROMPT:
        raise RuntimeError(f"expected {RESPONSES_PER_PROMPT} responses, found {len(rewards)}")
    if len(grade_matrix) != RESPONSES_PER_PROMPT:
        raise RuntimeError("criterion grade matrix does not match response count")
    criterion_count = len(grade_matrix[0])
    if criterion_count == 0 or any(len(row) != criterion_count for row in grade_matrix):
        raise RuntimeError("criterion inventory is empty or changes within a prompt group")

    reward_mean = sum(rewards, Fraction()) / len(rewards)
    pairs = list(combinations(rewards, 2))
    mixed_criteria = sum(
        0 < sum(response[index] for response in grade_matrix) < RESPONSES_PER_PROMPT
        for index in range(criterion_count)
    )
    return {
        "mad": float(sum(abs(value - reward_mean) for value in rewards) / len(rewards)),
        "ptr": sum(abs(left - right) <= PTR_THRESHOLD for left, right in pairs) / len(pairs),
        "ecr": mixed_criteria / criterion_count,
        "criterion_count": criterion_count,
    }


def aggregate_step(method: str, step: int, groups: Iterable[dict[str, float]]) -> dict:
    rows = list(groups)
    expected_prompts = (
        REMAINDER_PROMPTS_PER_STEP
        if step % STEPS_PER_EPOCH == 0
        else FULL_PROMPTS_PER_STEP
    )
    if len(rows) != expected_prompts:
        raise RuntimeError(
            f"{method} step {step}: expected {expected_prompts} prompts, found {len(rows)}"
        )
    return {
        "method": method,
        "training_step": step,
        "prompt_groups": len(rows),
        "responses": len(rows) * RESPONSES_PER_PROMPT,
        "MAD": sum(row["mad"] for row in rows) / len(rows),
        "PTR": sum(row["ptr"] for row in rows) / len(rows),
        "ECR": sum(row["ecr"] for row in rows) / len(rows),
        "mean_criterion_count": sum(row["criterion_count"] for row in rows) / len(rows),
    }


def online_metrics() -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    sources: list[dict] = []
    step_root = ONLINE_RUN / "verl-run/online_steps"
    for step in ONLINE_STEPS:
        directory = step_root / f"step-{step:06d}"
        reward_path = directory / "rewards.jsonl"
        batch_path = directory / "batch.json"
        commit_path = directory / "commit.json"
        for path in (reward_path, batch_path, commit_path):
            if not path.is_file():
                raise RuntimeError(f"missing Online training artifact: {path}")

        batch = json.loads(batch_path.read_text(encoding="utf-8"))
        commit = json.loads(commit_path.read_text(encoding="utf-8"))
        if int(batch["current_policy"]["policy_version"]) != step - 1:
            raise RuntimeError(f"Online step {step}: response policy must be step-1")
        if commit.get("state") != "committed" or int(commit["optimizer_update_index"]) != step:
            raise RuntimeError(f"Online step {step}: invalid commit record")

        grouped: dict[str, list[dict]] = defaultdict(list)
        for reward in read_jsonl(reward_path):
            grouped[str(reward["prompt_occurrence_id"])].append(reward)

        group_metrics: list[dict[str, float]] = []
        for prompt_id, prompt_rows in grouped.items():
            prompt_rows.sort(key=lambda row: int(row["rollout_index"]))
            if [int(row["rollout_index"]) for row in prompt_rows] != list(range(RESPONSES_PER_PROMPT)):
                raise RuntimeError(f"Online step {step}, {prompt_id}: invalid rollout indices")
            criterion_ids = [str(item[0]) for item in prompt_rows[0]["grades"]]
            grade_matrix = []
            rewards = []
            for row in prompt_rows:
                if [str(item[0]) for item in row["grades"]] != criterion_ids:
                    raise RuntimeError(f"Online step {step}, {prompt_id}: criterion order changed")
                rewards.append(Fraction(int(row["numerator"]), int(row["denominator"])))
                grade_matrix.append([int(item[1]) for item in row["grades"]])
            group_metrics.append(summarize_group(rewards, grade_matrix))

        rows.append(aggregate_step("Online Rubrics", step, group_metrics))
        sources.append(
            {
                "method": "Online Rubrics",
                "training_step": step,
                "path": str(reward_path.relative_to(ROOT)),
                "sha256": sha256(reward_path),
                "bytes": reward_path.stat().st_size,
            }
        )
    return rows, sources


def static_metrics() -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    sources: list[dict] = []
    rollout_root = STATIC_RUN / "verl-run/rollouts"
    for step in STATIC_STEPS:
        path = rollout_root / f"{step}.jsonl"
        if not path.is_file():
            raise RuntimeError(f"missing Static training artifact: {path}")
        grouped: dict[str, list[dict]] = defaultdict(list)
        for rollout in read_jsonl(path):
            if int(rollout["policy_step"]) != step:
                raise RuntimeError(f"Static step {step}: policy_step mismatch")
            grouped[str(rollout["prompt_id"])].append(rollout)

        group_metrics: list[dict[str, float]] = []
        for prompt_id, prompt_rows in grouped.items():
            prompt_rows.sort(key=lambda row: int(row["sample_index"]))
            if [int(row["sample_index"]) for row in prompt_rows] != list(range(RESPONSES_PER_PROMPT)):
                raise RuntimeError(f"Static step {step}, {prompt_id}: invalid sample indices")
            rewards: list[Fraction] = []
            grade_matrix: list[list[int]] = []
            criterion_count = int(prompt_rows[0]["criterion_count"])
            for row in prompt_rows:
                probabilities = [
                    float(value) for value in json.loads(str(row["criterion_probabilities_json"]))
                ]
                if len(probabilities) != criterion_count or int(row["criterion_count"]) != criterion_count:
                    raise RuntimeError(f"Static step {step}, {prompt_id}: criterion layout changed")
                reward = Fraction(int(row["score_num"]), int(row["score_den"]))
                if abs(float(reward) - float(row["static_reward"])) > 1e-12:
                    raise RuntimeError(f"Static step {step}, {prompt_id}: exact reward mismatch")
                rewards.append(reward)
                grade_matrix.append([int(probability > 0.5) for probability in probabilities])
            group_metrics.append(summarize_group(rewards, grade_matrix))

        rows.append(aggregate_step("Static R0 matched", step, group_metrics))
        sources.append(
            {
                "method": "Static R0 matched",
                "training_step": step,
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    return rows, sources


def plot(metrics: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.8), constrained_layout=True)
    styles = {
        "Online Rubrics": {"color": "#3366cc", "marker": "o", "linestyle": "-"},
        "Static R0 matched": {"color": "#dd7711", "marker": "s", "linestyle": "--"},
    }
    labels = {
        "Online Rubrics": "Online Rubrics actual training (steps 1–48)",
        "Static R0 matched": "Static R0 actual training (steps 1–42)",
    }
    for axis, metric in zip(axes, ("MAD", "PTR", "ECR"), strict=True):
        for method in ("Online Rubrics", "Static R0 matched"):
            frame = metrics[metrics["method"] == method]
            style = styles[method]
            axis.plot(
                frame["training_step"],
                frame[metric],
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                markersize=3.2,
                markeredgewidth=0,
                linewidth=1.7,
                label=labels[method],
            )
        axis.set_title(metric)
        axis.set_xlabel("Training step")
        axis.set_ylabel(metric)
        axis.set_xlim(0.5, 48.5)
        axis.set_xticks([1, 6, 12, 18, 24, 30, 36, 42, 48])
        axis.grid(alpha=0.22)
    axes[0].legend(frameon=False, fontsize=8.2, loc="best")
    fig.suptitle("Actual training reward diagnostics: latest Online Rubrics vs matched Static R0")
    fig.savefig(OUTPUT / "online_dense_vs_static_actual_training_mad_ptr_ecr.svg", bbox_inches="tight")
    fig.savefig(
        OUTPUT / "online_dense_vs_static_actual_training_mad_ptr_ecr.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    online_rows, online_sources = online_metrics()
    static_rows, static_sources = static_metrics()
    metrics = pd.DataFrame(online_rows + static_rows)
    metrics.to_csv(OUTPUT / "actual_training_mad_ptr_ecr_by_step.csv", index=False)
    pd.DataFrame(online_sources + static_sources).to_csv(
        OUTPUT / "source_artifact_hashes.csv", index=False
    )
    plot(metrics)

    summaries = []
    for method, frame in metrics.groupby("method", sort=False):
        summary = {
            "method": method,
            "steps": frame["training_step"].astype(int).tolist(),
            "prompt_groups": int(frame["prompt_groups"].sum()),
            "responses": int(frame["responses"].sum()),
        }
        for metric in ("MAD", "PTR", "ECR"):
            summary[metric] = {
                "mean_across_steps": float(frame[metric].mean()),
                "first_step": float(frame.iloc[0][metric]),
                "last_available_step": float(frame.iloc[-1][metric]),
                "min": float(frame[metric].min()),
                "max": float(frame[metric].max()),
            }
        summaries.append(summary)

    payload = {
        "schema_version": 1,
        "definition": {
            "unit": "equal-weight mean over prompt groups within each training step",
            "MAD": "mean absolute deviation of the 16 exact normalized rewards from their group mean",
            "PTR": "fraction of the 120 response pairs with absolute reward difference <= 0.01",
            "ECR": "fraction of rubric criteria whose hard binary grade varies across the 16 responses",
            "static_hard_grade": "criterion probability_yes > 0.5",
        },
        "runs": {
            "online": str(ONLINE_RUN.relative_to(ROOT)),
            "static": str(STATIC_RUN.relative_to(ROOT)),
        },
        "coverage_note": (
            "Online has committed actual-training rewards for steps 1-48. The matched Static R0 run "
            "has actual-training rollout/reward artifacts only for steps 1-42 because training stopped "
            "after step 42; steps 43-48 are not imputed."
        ),
        "summaries": summaries,
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
