from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from dynamic_rubric.hashing import sha256_file
from dynamic_rubric.providers.base import GenerationResult
from scripts.phase1 import evaluate_final_policies as evaluation
from scripts.phase1 import run_policy_checkpoint_trajectory as trajectory


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _rar_row(prompt_id: str = "rar-1") -> dict:
    return {
        "prompt_id": prompt_id,
        "messages": [{"role": "user", "content": "medical question"}],
        "reference_answer": "private reference that must not reach policy generation",
        "r0": {
            "criteria": [
                {"criterion_id": "r1", "criterion": "is correct", "weight_units": 10},
                {"criterion_id": "r2", "criterion": "is concise", "weight_units": 3},
            ]
        },
    }


def _health_row(prompt_id: str = "hb-1") -> dict:
    return {
        "prompt_id": prompt_id,
        "prompt": [{"role": "user", "content": "health question"}],
        "rubrics": [
            {"criterion": "helpful", "points": 2, "tags": ["helpfulness"]},
            {"criterion": "dangerous", "points": -3, "tags": ["harm"]},
        ],
    }


def _config(tmp_path: Path) -> Path:
    rar = tmp_path / "rar.jsonl"
    health = tmp_path / "health.jsonl"
    _write_jsonl(rar, [_rar_row(), _rar_row("rar-2")])
    _write_jsonl(health, [_health_row(), _health_row("hb-2")])
    config = {
        "schema_version": 1,
        "output_root": str(tmp_path / "outputs"),
        "seed": 11,
        "bootstrap_replicates": 50,
        "healthbench_upstream": {
            "repository": str(tmp_path),
            "commit": "pinned",
            "source_path": "healthbench_eval.py",
        },
        "datasets": {
            "rar_medicine": {
                "kind": "rar",
                "path": str(rar),
                "expected_count": 2,
                "sha256": sha256_file(rar),
            },
            "healthbench": {
                "kind": "healthbench",
                "path": str(health),
                "expected_count": 2,
                "sha256": sha256_file(health),
            },
        },
        "models": {
            "static_base": {"method": "static", "role": "base", "served_model": "static_base", "artifact": "a"},
            "static_final": {"method": "static", "role": "final", "served_model": "static_final", "artifact": "b"},
            "online_base": {"method": "online", "role": "base", "served_model": "online_base", "artifact": "c"},
            "online_final": {"method": "online", "role": "final", "served_model": "online_final", "artifact": "d"},
        },
        "generation": {"base_url": "http://policy", "workers": 2, "max_retries": 1},
        "grading": {
            "base_url": "http://judge",
            "served_model": "Qwen3-32B",
            "workers": 3,
            "max_retries": 1,
            "parse_retries": 1,
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


class FakeAdapter:
    requests = []

    def __init__(self, base_url, model, cache_dir, **kwargs):
        self.model = model
        self.base_url = base_url

    def request_provenance(self, request):
        return {"selected_base_url": self.base_url, "provider_cache_path": "fake"}

    def generate(self, request):
        self.requests.append(request)
        if request.family.endswith("generation"):
            assert "reference" not in request.messages[0]["content"]
            text = f"answer from {request.metadata['model_name']}"
            usage = {
                "finish_reason": "stop",
                "prompt_tokens": 3,
                "completion_tokens": 4,
                "total_tokens": 7,
            }
        else:
            if request.metadata["parse_attempt"] == 0:
                text = '{"explanation":"bad type","criteria_met":"yes"}'
            else:
                text = '{"explanation":"mock grade","criteria_met":true}'
            usage = {"finish_reason": "stop", "completion_tokens": 5}
        return GenerationResult(
            text=text,
            requested_model=self.model,
            returned_model=self.model,
            request_id="fake-request",
            created_at=1,
            retry_count=0,
            usage=usage,
            raw_response_hash="fake-hash",
        )


@pytest.fixture
def patched_runner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config_path = _config(tmp_path)
    FakeAdapter.requests = []
    monkeypatch.setattr(evaluation, "VLLMChatAdapter", FakeAdapter)
    monkeypatch.setattr(
        evaluation,
        "load_official_grader_template",
        lambda config, root: (
            "Conversation:\n<<conversation>>\nRubric:\n<<rubric_item>>",
            {"commit": "pinned", "grader_template_sha256": "template-hash"},
        ),
    )
    return config_path


def test_ast_literal_extracts_stripped_official_template() -> None:
    source = 'GRADER_TEMPLATE = """ grader text """.strip()\n'
    assert evaluation._literal_assignment(source, "GRADER_TEMPLATE") == "grader text"


def test_signed_score_is_not_clipped_per_example() -> None:
    criteria = evaluation.normalize_dataset_row("healthbench", _health_row(), 0)["criteria"]
    grades = [
        {"criterion_id": "0", "criteria_met": False},
        {"criterion_id": "1", "criteria_met": True},
    ]
    assert evaluation.score_prompt(criteria, grades) == -1.5


def test_smoke_namespace_is_separate_and_source_hash_fails_closed(
    patched_runner: Path,
) -> None:
    smoke = evaluation.load_config(patched_runner, limit=1)
    full = evaluation.load_config(patched_runner, limit=None)
    assert evaluation.run_directory(smoke) != evaluation.run_directory(full)
    evaluation.prepare(smoke)
    manifest = json.loads((evaluation.run_directory(smoke) / "manifest.json").read_text())
    assert manifest["run_kind"] == "smoke"
    assert {item["selected_count"] for item in manifest["sources"].values()} == {1}
    broken = yaml.safe_load(patched_runner.read_text())
    broken["datasets"]["rar_medicine"]["sha256"] = "0" * 64
    patched_runner.write_text(yaml.safe_dump(broken))
    with pytest.raises(evaluation.EvaluationError, match="source hash mismatch"):
        evaluation.prepare(evaluation.load_config(patched_runner, limit=None))


def test_mocked_end_to_end_is_resumable_and_paired(patched_runner: Path) -> None:
    config = evaluation.load_config(patched_runner, limit=1)
    output = evaluation.prepare(config)
    for model_name in sorted(evaluation.REQUIRED_MODEL_NAMES):
        evaluation.generate(config, model_name)
        evaluation.generate(config, model_name)  # valid immutable response is reused
    evaluation.grade(config)
    first_call_count = len(FakeAdapter.requests)
    evaluation.grade(config)  # valid immutable criterion grades are reused
    assert len(FakeAdapter.requests) == first_call_count
    summary = evaluation.summarize(config)
    assert summary["run_kind"] == "smoke"
    assert summary["comparisons"]["rar_medicine"]["static"]["n_prompts"] == 1
    assert summary["comparisons"]["healthbench"]["online"]["aggregate_clip_0_1"] is True
    failures = list((output / "grade_failures").rglob("*.json"))
    assert failures  # malformed boolean was retained, never converted to grade zero
    generation_requests = [request for request in FakeAdapter.requests if request.family.endswith("generation")]
    assert len(generation_requests) == 8
    assert all(request.temperature == 0 and request.metadata["single_answer"] for request in generation_requests)


def test_generation_sampling_contract_is_configurable(patched_runner: Path) -> None:
    raw = yaml.safe_load(patched_runner.read_text())
    raw["generation"].update({"seed": 101, "temperature": 0.7, "top_p": 0.8})
    patched_runner.write_text(yaml.safe_dump(raw))
    config = evaluation.load_config(patched_runner, limit=1)
    output = evaluation.prepare(config)
    evaluation.generate(config, "online_base")
    requests = [
        request for request in FakeAdapter.requests if request.family.endswith("generation")
    ]
    assert len(requests) == 2
    assert all(request.seed == 101 for request in requests)
    assert all(request.temperature == pytest.approx(0.7) for request in requests)
    assert all(request.top_p == pytest.approx(0.8) for request in requests)
    response = next((output / "responses" / "online_base").rglob("*.json"))
    contract = json.loads(response.read_text())["generation_contract"]
    assert contract == {
        "seed": 101,
        "temperature": pytest.approx(0.7),
        "top_p": pytest.approx(0.8),
        "thinking": False,
        "n": 1,
    }


def test_summarize_rejects_missing_grade(patched_runner: Path) -> None:
    config = evaluation.load_config(patched_runner, limit=1)
    evaluation.prepare(config)
    for model_name in sorted(evaluation.REQUIRED_MODEL_NAMES):
        evaluation.generate(config, model_name)
    with pytest.raises(evaluation.EvaluationError, match="missing criterion grade"):
        evaluation.summarize(config)


def test_arbitrary_checkpoint_trajectory_is_paired_to_own_base(
    patched_runner: Path,
) -> None:
    raw = yaml.safe_load(patched_runner.read_text())
    raw["models"]["static_step_003"] = {
        "method": "static",
        "role": "checkpoint",
        "step": 3,
        "served_model": "static_step_003",
        "artifact": "static-three",
    }
    raw["models"]["online_step_003"] = {
        "method": "online",
        "role": "checkpoint",
        "step": 3,
        "served_model": "online_step_003",
        "artifact": "online-three",
    }
    patched_runner.write_text(yaml.safe_dump(raw))
    config = evaluation.load_config(patched_runner, limit=1)
    evaluation.prepare(config)
    for model_name in evaluation.ordered_model_names(config):
        evaluation.generate(config, model_name)
    evaluation.grade(config)
    summary = evaluation.summarize(config)
    assert [
        row["global_step"] for row in summary["trajectories"]["rar_medicine"]["static"]
    ] == [0, 3, 48]
    assert [
        row["global_step"] for row in summary["trajectories"]["healthbench"]["online"]
    ] == [0, 3, 48]
    assert summary["comparisons"]["rar_medicine"]["static"]["global_step"] == 48


def test_seeded_dataset_sample_and_dataset_specific_grading(
    patched_runner: Path,
) -> None:
    raw = yaml.safe_load(patched_runner.read_text())
    raw["datasets"]["healthbench"]["sample_count"] = 1
    raw["datasets"]["healthbench"]["sample_seed"] = 11
    patched_runner.write_text(yaml.safe_dump(raw))
    config = evaluation.load_config(patched_runner, limit=None)
    output = evaluation.prepare(config)
    prepared = [json.loads(line) for line in (output / "prepared" / "prompts.jsonl").read_text().splitlines()]
    assert sum(row["dataset"] == "healthbench" for row in prepared) == 1
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["sources"]["healthbench"]["sampling"]["kind"] == "seeded_random_without_replacement"
    for model_name in evaluation.ordered_model_names(config):
        evaluation.generate(config, model_name)
    evaluation.grade(config, "static_base", dataset_name="rar_medicine")
    assert list((output / "grades" / "static_base" / "rar_medicine").rglob("*.json"))
    assert not list((output / "grades" / "static_base" / "healthbench").rglob("*.json"))

def test_seeded_sample_can_require_existing_prompt_ids(patched_runner: Path) -> None:
    raw = yaml.safe_load(patched_runner.read_text())
    health_path = Path(raw["datasets"]["healthbench"]["path"])
    health_rows = [_health_row(f"hb-{index}") for index in range(6)]
    _write_jsonl(health_path, health_rows)
    required_path = health_path.parent / "existing-prompts.jsonl"
    _write_jsonl(
        required_path,
        [
            {"dataset": "healthbench", "prompt_id": "hb-1"},
            {"dataset": "healthbench", "prompt_id": "hb-4"},
            {"dataset": "rar_medicine", "prompt_id": "rar-1"},
        ],
    )
    raw["datasets"]["healthbench"].update(
        {
            "expected_count": 6,
            "sha256": sha256_file(health_path),
            "sample_count": 4,
            "sample_seed": 11,
            "include_prompt_ids_from": str(required_path),
        }
    )
    patched_runner.write_text(yaml.safe_dump(raw))

    config = evaluation.load_config(patched_runner, limit=None)
    output = evaluation.prepare(config)
    prepared = [
        json.loads(line)
        for line in (output / "prepared" / "prompts.jsonl").read_text().splitlines()
    ]
    selected = {row["prompt_id"] for row in prepared if row["dataset"] == "healthbench"}
    assert len(selected) == 4
    assert {"hb-1", "hb-4"} <= selected
    sampling = json.loads((output / "manifest.json").read_text())["sources"]["healthbench"]["sampling"]
    assert sampling["kind"] == "seeded_random_without_replacement_with_required_ids"
    assert sampling["required_ids"]["required_count"] == 2
    assert sampling["additional_random_count"] == 2

def test_parallel_prefetch_skips_models_with_complete_responses(
    patched_runner: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = evaluation.load_config(patched_runner, limit=1)
    config["runtime"] = {
        "temporary_download_root": str(tmp_path / "downloads"),
        "download_workers": 2,
    }
    output = evaluation.prepare(config)
    evaluation.generate(config, "static_base")
    downloaded = []

    def fake_download(model_name, spec, staging_root):
        downloaded.append(model_name)
        destination = staging_root / model_name
        destination.mkdir(parents=True, exist_ok=True)
        return destination, True

    monkeypatch.setattr(trajectory, "_download_model", fake_download)
    trajectory.download_all(config)
    assert set(downloaded) == set(config["models"]) - {"static_base"}
    status = json.loads((output / "status" / "prefetch.json").read_text())
    assert status["state"] == "complete"
    assert status["download_workers"] == 2

