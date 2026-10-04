"""Semantic inventory validation beyond aggregate row counts."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .artifacts import read_json, read_jsonl
from .evaluation.bon import shared_pool_hash


class SemanticInventoryError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SemanticInventoryError(message)


def _groups(
    rows: Sequence[Mapping[str, Any]], *keys: str
) -> dict[tuple[Any, ...], list[Mapping[str, Any]]]:
    result: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        result[tuple(row[key] for key in keys)].append(row)
    return result


def _expected_prompt_sets(context: Any) -> dict[str, set[str]]:
    manifest = read_json(context.public_root / "split_manifest.json")
    configured = {str(name): int(count) for name, count in context.raw.get("splits", {}).items()}
    entries = manifest.get("splits", {})
    _require(set(entries) == set(configured), "split manifest names differ from config")
    _require(int(manifest.get("seed", -1)) == context.config.split_seed, "split seed differs")
    ownership: dict[str, str] = {}
    result = {"all": set(), "probe": set(), "audit": set()}
    for split, count in configured.items():
        entry = entries[split]
        path = context.public_root / f"{split}.jsonl"
        rows = read_jsonl(path)
        prompt_ids = [str(row["prompt_id"]) for row in rows]
        manifest_ids = [str(value) for value in entry.get("prompt_ids", [])]
        _require(len(prompt_ids) == count, f"{split} prompt count differs from config")
        _require(prompt_ids == manifest_ids, f"{split} prompt IDs differ from manifest")
        _require(len(prompt_ids) == len(set(prompt_ids)), f"duplicate prompt in {split}")
        _require(
            hashlib.sha256(path.read_bytes()).hexdigest() == entry.get("sha256"),
            f"{split} bytes differ from manifest",
        )
        for prompt_id in prompt_ids:
            _require(prompt_id not in ownership, f"prompt {prompt_id} appears in multiple splits")
            ownership[prompt_id] = split
        result["all"].update(prompt_ids)
        if "probe" in split:
            result["probe"].update(prompt_ids)
        if "audit" in split:
            result["audit"].update(prompt_ids)
    return result


def _validate_static(
    static: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    expected_prompts: set[str],
) -> None:
    static_prompts = {str(row["prompt_id"]) for row in static}
    _require(len(static_prompts) == len(static), "duplicate static rubric")
    _require(static_prompts == expected_prompts, "static rubric prompt set differs")
    for row in static:
        criteria = row["criteria"]
        _require(len(criteria) == 8, "static rubric must have exactly eight criteria")
        _require(
            all(math.isclose(float(item["weight"]), 1 / 8, abs_tol=1e-12) for item in criteria),
            "static criteria must have equal weights",
        )
        sources = [item["source"] for item in criteria]
        _require(
            sources.count("task_specific") == 6,
            "static rubric must preserve six generated criteria",
        )
        _require(
            sources.count("universal") == 2, "static rubric must preserve two universal criteria"
        )
        provenance = row.get("provenance", {})
        _require(
            provenance.get("gold_access") is False, "static rubric provenance must deny gold access"
        )
    candidate_groups = _groups(candidates, "prompt_id")
    _require(
        {str(key[0]) for key in candidate_groups} == expected_prompts,
        "static candidate prompt set differs",
    )
    for _prompt, rows in candidate_groups.items():
        _require(len(rows) == 12, "each prompt requires twelve static candidates")
        _require(
            {int(row["sample_index"]) for row in rows} == set(range(12)),
            "static candidate slots differ",
        )
        temperatures = {int(row["sample_index"]): float(row["temperature"]) for row in rows}
        _require(
            all(temperatures[index] == 0.7 for index in range(6)), "low-temperature group differs"
        )
        _require(
            all(temperatures[index] == 1.1 for index in range(6, 12)),
            "high-temperature group differs",
        )


def _validate_response_families(
    context: Any,
    references: Sequence[Mapping[str, Any]],
    development: Sequence[Mapping[str, Any]],
    final: Sequence[Mapping[str, Any]],
    expected_reference_prompts: set[str],
    expected_development_prompts: set[str],
    expected_final_prompts: set[str],
) -> None:
    family_config = context.raw.get("response_families", {})
    reference_counts = {
        "reference_discovery": max(8, int(family_config.get("reference_discovery", 8))),
        "reference_validation": max(4, int(family_config.get("reference_validation", 4))),
    }
    reference_groups = _groups(references, "prompt_id", "family")
    expected_reference_cells = {
        (prompt_id, family)
        for prompt_id in expected_reference_prompts
        for family in reference_counts
    }
    _require(
        set(reference_groups) == expected_reference_cells,
        "reference prompt/family Cartesian coverage differs",
    )
    for (_prompt, family), rows in reference_groups.items():
        _require(len(rows) == reference_counts[str(family)], f"wrong {family} count")
        _require(
            {int(row["sample_index"]) for row in rows} == set(range(reference_counts[str(family)])),
            f"wrong {family} sample slots",
        )
        _require({row["policy_id"] for row in rows} == {"pi_0"}, "reference policy must be pi_0")
        _require({int(row["policy_step"]) for row in rows} == {0}, "reference step must be zero")
    probe_config = context.raw.get("probe", {})
    for split_name, rows, count, expected_prompts in (
        (
            "development",
            development,
            int(probe_config.get("development_samples_per_family", 4)),
            expected_development_prompts,
        ),
        (
            "final",
            final,
            int(probe_config.get("final_samples_per_family", 4)),
            expected_final_prompts,
        ),
    ):
        groups = _groups(rows, "prompt_id", "policy_step", "family")
        expected_cells = {
            (prompt_id, step, family)
            for prompt_id in expected_prompts
            for step in range(1, context.config.training.max_steps + 1)
            for family in ("trajectory_discovery", "trajectory_validation")
        }
        _require(set(groups) == expected_cells, f"{split_name} response Cartesian coverage differs")
        for (prompt_id, step, family), values in groups.items():
            _require(
                family in {"trajectory_discovery", "trajectory_validation"},
                f"unexpected {split_name} response family",
            )
            _require(len(values) == count, f"wrong {split_name} response count")
            _require(
                {int(row["sample_index"]) for row in values} == set(range(count)),
                f"wrong {split_name} sample slots",
            )
            _require(
                {row["policy_id"] for row in values} == {f"pi_{step}"},
                "policy-step identity mismatch",
            )
            _require(
                all(row["timing"] == "after_optimizer_update" for row in values),
                "trajectory response is not post-update",
            )
            _require(
                all(row["config_hash"] == context.config.config_hash for row in values),
                f"trajectory config drift for {prompt_id}",
            )


def _validate_replay(
    context: Any, replay: Sequence[Mapping[str, Any]], expected_prompts: set[str]
) -> None:
    modes = tuple(context.raw.get("replay", {}).get("modes", ()))
    expected_steps = set(range(1, context.config.training.max_steps + 1))
    groups = _groups(replay, "prompt_id", "policy_step", "mode")
    _require(all(len(rows) == 1 for rows in groups.values()), "duplicate replay snapshot cell")
    expected_cells = {
        (prompt_id, step, mode)
        for prompt_id in expected_prompts
        for step in expected_steps
        for mode in modes
    }
    _require(set(groups) == expected_cells, "replay Cartesian prompt/step/mode coverage differs")
    for row in replay:
        criteria = row["criteria"]
        _require(
            sum(item["source"] != "dynamic" for item in criteria) == 8, "static criteria were lost"
        )
        if row["mode"] != "dynamic_fixed_cumulative":
            _require(len(criteria) <= 12, "budgeted replay exceeds twelve criteria")
        if row["mode"] == "static":
            _require(row["admitted_id"] is None, "static replay admitted a dynamic criterion")
            continue
        _require(
            row.get("generator_payload_source_blind") is True,
            "generator pairing leaked source labels",
        )
        _require(
            all(item["evidence"]["independent_validation"] for item in row["candidate_evidence"]),
            "replay used non-independent validation evidence",
        )
        mode = str(row["mode"])
        step = int(row["policy_step"])
        if mode == "refresh_only_budgeted":
            _require(
                row["current_response_used"] is False, "refresh-only consumed current responses"
            )
            _require(not row["current_response_ids"], "refresh-only records current response IDs")
        elif mode == "dynamic_prev_budgeted":
            expected = "pi_0" if step == 1 else f"pi_{step - 1}"
            _require(row["control_policy"] == expected, "previous-policy control step differs")
        else:
            _require(row["control_policy"] == "pi_0", "fixed replay control must remain pi_0")
    by_prompt_step = _groups(replay, "prompt_id", "policy_step")
    for rows in by_prompt_step.values():
        by_mode = {str(row["mode"]): row for row in rows}
        fixed = by_mode["dynamic_fixed_budgeted"]
        cumulative = by_mode["dynamic_fixed_cumulative"]
        for key in ("discovery_pool_hash", "validation_pool_hash", "pairing_hash"):
            _require(fixed[key] == cumulative[key], f"fixed/cumulative {key} differs")
    cumulative_groups = _groups(
        [row for row in replay if row["mode"] == "dynamic_fixed_cumulative"], "prompt_id"
    )
    for rows in cumulative_groups.values():
        counts = [
            int(row["criterion_count"])
            for row in sorted(rows, key=lambda item: int(item["policy_step"]))
        ]
        _require(counts == sorted(counts), "cumulative rubric size is not monotonic")


def _expected_evaluation_rubrics(context: Any, expected_prompts: set[str]) -> dict[str, set[str]]:
    bon_config = context.raw.get("bon", {})
    dynamic_modes = {
        str(mode) for mode in context.raw.get("replay", {}).get("modes", []) if mode != "static"
    }
    dynamic_steps = {int(step) for step in bon_config.get("rubric_steps", [])} - {0}
    return {
        prompt_id: {f"{prompt_id}:R_0"}
        | {f"{prompt_id}:{mode}:R_{step}" for mode in dynamic_modes for step in dynamic_steps}
        for prompt_id in expected_prompts
    }


def _validate_bon_and_selection(
    context: Any,
    bon: Sequence[Mapping[str, Any]],
    scores: Sequence[Mapping[str, Any]],
    selections: Sequence[Mapping[str, Any]],
    expected_prompts: set[str],
) -> None:
    bon_config = context.raw.get("bon", {})
    pool_size = int(bon_config.get("pool_size", 64))
    sizes = {int(value) for value in bon_config.get("sizes", [1, 2, 4, 8, 16, 32, 64])}
    permutations = set(range(int(bon_config.get("permutations", 5))))
    candidates = _groups(bon, "policy_id", "prompt_id")
    policy_ids = {f"pi_{int(step)}" for step in bon_config.get("focal_steps", [])}
    expected_candidate_groups = {
        (policy_id, prompt_id) for policy_id in policy_ids for prompt_id in expected_prompts
    }
    _require(set(candidates) == expected_candidate_groups, "BoN policy/prompt groups differ")
    pool_hashes: dict[tuple[str, str], str] = {}
    candidate_ids: dict[tuple[str, str], set[int]] = {}
    for key, rows in candidates.items():
        _require(len(rows) == pool_size, "BoN pool size differs")
        _require(
            {int(row["sample_index"]) for row in rows} == set(range(pool_size)),
            "BoN sample slots differ",
        )
        candidate_ids[(str(key[0]), str(key[1]))] = {
            int(row["global_candidate_id"]) for row in rows
        }
        pool_hashes[(str(key[0]), str(key[1]))] = shared_pool_hash(
            [
                {"candidate_id": row["global_candidate_id"], "response_text": row["response_text"]}
                for row in rows
            ]
        )
    expected_rubrics = _expected_evaluation_rubrics(context, expected_prompts)
    expected_score_groups = {
        (policy_id, prompt_id, rubric_id)
        for policy_id in policy_ids
        for prompt_id, rubric_ids in expected_rubrics.items()
        for rubric_id in rubric_ids
    }
    score_groups = _groups(scores, "policy_id", "prompt_id", "rubric_id")
    _require(set(score_groups) == expected_score_groups, "proxy score rubric groups differ")
    for key, rows in score_groups.items():
        pool_key = str(key[0]), str(key[1])
        _require(
            {int(row["global_candidate_id"]) for row in rows} == candidate_ids[pool_key],
            "rubric score candidate IDs do not equal the shared BoN pool",
        )
    selection_groups = _groups(selections, "policy_id", "prompt_id", "rubric_id")
    _require(set(selection_groups) == expected_score_groups, "selection rubric groups differ")
    for key, rows in selection_groups.items():
        cells = {(int(row["n"]), int(row["permutation"])) for row in rows}
        _require(
            cells == {(n, p) for n in sizes for p in permutations if n <= pool_size},
            "selection grid differs",
        )
        pool_key = str(key[0]), str(key[1])
        _require(
            {str(row["pool_hash"]) for row in rows} == {pool_hashes[pool_key]},
            "rubrics did not share pool hash",
        )
        _require(
            all(int(row["global_candidate_id"]) in candidate_ids[pool_key] for row in rows),
            "selection references a candidate outside its pool",
        )


def validate_semantic_inventory(context: Any, paths: Mapping[str, Path]) -> list[str]:
    static = read_jsonl(paths["static"])
    candidates = read_jsonl(paths["candidates"])
    references = read_jsonl(paths["references"])
    development = read_jsonl(paths["development"])
    final = read_jsonl(paths["final"])
    replay = read_jsonl(paths["replay_final"])
    bon = read_jsonl(paths["bon"])
    scores = read_jsonl(paths["scores"])
    selections = read_jsonl(paths["selections"])
    package = read_jsonl(paths["audit_package"])
    gold = read_jsonl(paths["gold"])

    expected_prompts = _expected_prompt_sets(context)
    _validate_static(static, candidates, expected_prompts["all"])
    _validate_response_families(
        context,
        references,
        development,
        final,
        expected_prompts["probe"] | expected_prompts["audit"],
        expected_prompts["probe"],
        expected_prompts["audit"],
    )
    _validate_replay(context, replay, expected_prompts["audit"])
    _validate_bon_and_selection(context, bon, scores, selections, expected_prompts["audit"])
    selected_ids = {(str(row["prompt_id"]), str(row["response_id"])) for row in selections}
    package_ids = {(str(row["prompt_id"]), str(row["response_id"])) for row in package}
    gold_ids = {(str(row["prompt_id"]), str(row["response_id"])) for row in gold}
    _require(package_ids == selected_ids, "audit package is not the exact unique selection set")
    _require(gold_ids == package_ids, "hidden-gold scores do not exactly cover the audit package")
    _require(
        all(
            row["response_text_hash"]
            == hashlib.sha256(str(row["response_text"]).encode()).hexdigest()
            for row in package
        ),
        "audit response hash differs from the selected response bytes",
    )
    prepare_result = read_json(context.run_root / "prepare-data" / "result.json")
    audit_result = read_json(context.run_root / "audit-gold" / "result.json")
    _require(
        prepare_result.get("public_leakage_scan") == "passed",
        "public leakage scan evidence is absent",
    )
    _require(
        audit_result.get("public_artifact_leakage_scan") == "passed",
        "post-pipeline public artifact leakage scan evidence is absent",
    )
    lock = read_json(context.run_root / "updater_lock.json")
    receipt = read_json(context.run_root / "trajectory" / "final_sealed" / "unseal_receipt.json")
    _require(
        lock["operator_hash"] == receipt["operator_hash"],
        "unseal operator differs from updater lock",
    )
    return [
        "static_weights_and_provenance",
        "response_family_cartesian_coverage",
        "replay_modes_steps_pairing_and_monotonicity",
        "shared_bon_candidate_and_selection_grids",
        "exact_selected_package_and_gold_ids",
        "updater_unseal_operator_compatibility",
        "public_leakage_scan_evidence",
    ]
