"""Verify the configured judge using a synthetic arithmetic example."""

from __future__ import annotations

import argparse
from pathlib import Path

from dynamic_rubric.artifacts import read_json, write_json_atomic
from dynamic_rubric.phase1.evorubrics_probe import UpstreamJudge
from dynamic_rubric.phase1.evorubrics_run import resolve_judge_identity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--judge-base-url", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    spec = read_json(args.run_root / "launch_spec.json")
    endpoint = args.judge_base_url.rstrip("/")
    if not endpoint.endswith("/v1"):
        endpoint += "/v1"
    identity = resolve_judge_identity(endpoint, spec["judge_model"])
    judge = UpstreamJudge(root, base_url=endpoint, model=spec["judge_model"])
    results = judge.grade(
        question="What is 2 + 2?",
        answers=["2 + 2 equals 4.", "2 + 2 equals 5."],
        criteria=[{"criterion": "States that 2 + 2 equals 4.", "weight": 1}],
    )
    if len(results) != 2:
        raise RuntimeError("Judge omitted an answer")
    for result, expected in zip(results, (True, False)):
        grades = result.get("details", {}).get("rubric_scores", [])
        if len(grades) != 1 or grades[0].get("criteria_met") is not expected:
            raise RuntimeError("Synthetic criterion judgment did not match the expected boolean")
    path = args.run_root / "judge-preflight.json"
    write_json_atomic(
        path,
        {
            "status": "passed",
            "model": spec["judge_model"],
            "endpoint": endpoint,
            "judge_identity": identity,
            "actual_remote_call": True,
            "results": results,
        },
    )
    print(path)


if __name__ == "__main__":
    main()
