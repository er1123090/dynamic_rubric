from __future__ import annotations

import copy
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

from dynamic_rubric.artifacts import ImmutableArtifactError
from dynamic_rubric.phase1.config import (
    Phase1ConfigError,
    load_phase1_config,
    validate_phase1_mapping,
)
from dynamic_rubric.phase1.evorubrics_data import (
    EvoRubricsDataError,
    convert_rar_record,
    prepare_evorubrics_data,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "phase1" / "medicine_evorubrics.yaml"
PROBE_MANIFEST = (
    REPO_ROOT
    / "outputs"
    / "medicine"
    / "shared"
    / "seed-11"
    / "manifests"
    / "fixed_train_probe.json"
)
TRAIN_PATH = REPO_ROOT / "data" / "rar" / "medicine" / "eval300" / "train.jsonl"


def _first_train_record() -> dict:
    return json.loads(TRAIN_PATH.read_text(encoding="utf-8").splitlines()[0])


def test_evo_config_combines_rq2_and_public_training_contracts() -> None:
    config = load_phase1_config(CONFIG_PATH)
    evo = config.method_config
    assert evo["policy_responses_m"] == 4
    assert evo["rubric_sets_n"] == 4
    assert evo["pool_b_count"] == 16
    assert evo["dataset_mode"] == "open_rubrics"
    assert evo["training_flow"] == "unified"
    assert evo["lora_rank"] == 32
    assert evo["lora_alpha"] == 64
    assert evo["policy_learning_rate"] == 2.0e-5
    assert evo["rubric_generator_learning_rate"] == 5.0e-6
    assert evo["kl_loss_coefficient"] == 1.0e-4
    assert evo["generation_temperature"] == 0.7
    assert evo["reward_weights"] == {
        "similarity": 0.25,
        "discrimination": 0.25,
        "diversity": 0.25,
        "reflect": 0.25,
    }
    assert evo["reflect_use_golden_rubrics"] is True


def test_evo_config_rejects_training_pool_as_probe_pool() -> None:
    config = load_phase1_config(CONFIG_PATH)
    raw = copy.deepcopy(config.raw)
    raw["evorubrics"]["pool_b_count"] = 4
    with pytest.raises(Phase1ConfigError, match="Evo Pool-B must be 16"):
        validate_phase1_mapping(raw, source_path=CONFIG_PATH)


def test_convert_rar_record_matches_public_open_rubrics_shape_without_answer() -> None:
    source = _first_train_record()
    converted = convert_rar_record(source, row_number=1)
    assert converted["question"] == source["messages"][-1]["content"]
    assert converted["prompt"] == source["messages"]
    assert converted["prompt_id"] == source["prompt_id"]
    assert converted["prompt_hash"] == source["prompt_hash"]
    assert converted["source_index"] == 0
    assert converted["rubrics"][0] == {
        "criterion": source["r0"]["criteria"][0]["criterion"],
        "points": source["r0"]["criteria"][0]["weight_units"],
        "tags": [],
    }
    assert converted["metadata"]["criteria"][0]["criterion_id"] == (
        source["r0"]["criteria"][0]["criterion_id"]
    )
    serialized = json.dumps(converted, ensure_ascii=False)
    assert "reference_answer" not in serialized
    assert source["reference_answer"] not in serialized
    assert "golden_answer" not in converted


def test_convert_rejects_changed_prompt_under_preserved_hash() -> None:
    source = _first_train_record()
    source["messages"][0]["content"] += " changed"
    with pytest.raises(EvoRubricsDataError, match="prompt_hash"):
        convert_rar_record(source, row_number=1)


def test_prepare_real_splits_is_idempotent_and_source_bound(tmp_path: Path) -> None:
    config = load_phase1_config(CONFIG_PATH)
    first = prepare_evorubrics_data(
        config,
        repo_root=REPO_ROOT,
        output_dir=tmp_path,
        fixed_probe_manifest_path=PROBE_MANIFEST,
    )
    second = prepare_evorubrics_data(
        config,
        repo_root=REPO_ROOT,
        output_dir=tmp_path,
        fixed_probe_manifest_path=PROBE_MANIFEST,
    )
    assert first.manifest == second.manifest
    train = json.loads(first.train_path.read_text(encoding="utf-8"))
    heldout = json.loads(first.heldout_path.read_text(encoding="utf-8"))
    assert len(train) == 1500
    assert len(heldout) == 300
    assert first.manifest["answer_fields_exported"] is False
    assert first.manifest["fixed_train_probe"]["prompt_count"] == 100
    assert first.manifest["fixed_train_probe"]["remains_in_training"] is True
    assert (
        first.manifest["fixed_train_probe"]["extra_responses_used_for_gradient"]
        is False
    )
    assert not ({row["prompt_id"] for row in train} & {row["prompt_id"] for row in heldout})

    first.train_path.write_bytes(b"[]\n")
    with pytest.raises(ImmutableArtifactError, match="refusing to overwrite"):
        prepare_evorubrics_data(
            config,
            repo_root=REPO_ROOT,
            output_dir=tmp_path,
            fixed_probe_manifest_path=PROBE_MANIFEST,
        )


class _FakeTensor:
    shape = (1, 4)

    def squeeze(self, _dimension: int):
        return self

    def unsqueeze(self, _dimension: int):
        return self


class _FakeTokenizer:
    chat_template = "available"

    def __init__(self) -> None:
        self.template_kwargs = []

    def apply_chat_template(self, _messages, **kwargs):
        self.template_kwargs.append(kwargs)
        return "rendered prompt"

    def __call__(self, *_args, **_kwargs):
        return {"input_ids": _FakeTensor(), "attention_mask": _FakeTensor()}


def test_upstream_dataset_preserves_identity_and_uses_hf_thinking_keyword(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    torch = types.ModuleType("torch")
    torch_utils = types.ModuleType("torch.utils")
    torch_data = types.ModuleType("torch.utils.data")
    torch_data.Dataset = object
    torch_utils.data = torch_data
    torch.utils = torch_utils
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.utils", torch_utils)
    monkeypatch.setitem(sys.modules, "torch.utils.data", torch_data)

    verl = types.ModuleType("verl")
    verl_utils = types.ModuleType("verl.utils")
    verl_model = types.ModuleType("verl.utils.model")
    verl_model.compute_position_id_with_mask = lambda mask: mask
    verl_utils.model = verl_model
    verl.utils = verl_utils
    monkeypatch.setitem(sys.modules, "verl", verl)
    monkeypatch.setitem(sys.modules, "verl.utils", verl_utils)
    monkeypatch.setitem(sys.modules, "verl.utils.model", verl_model)

    models = types.ModuleType("models")
    models.POLICY_LLM_SYSTEM_PROMPT = "policy"
    models.RUBRICS_GENERATOR_SYSTEM_PROMPT = "rubrics"
    models.UNIVERSAL_RUBRICS_GENERATOR_SYSTEM_PROMPT = "universal"
    monkeypatch.setitem(sys.modules, "models", models)

    source = convert_rar_record(_first_train_record(), row_number=1)
    data_path = tmp_path / "data.json"
    data_path.write_text(json.dumps([source]), encoding="utf-8")
    upstream_path = (
        REPO_ROOT
        / "environment"
        / "upstream"
        / "EvoRubrics"
        / "evorubric-main"
        / "adversarial_dataset.py"
    )
    spec = importlib.util.spec_from_file_location("tested_adversarial_dataset", upstream_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tokenizer = _FakeTokenizer()
    dataset = module.AdversarialDataset(
        str(data_path),
        tokenizer,
        processor=None,
        config={"dataset_mode": "open_rubrics", "max_prompt_length": 32},
    )
    item = dataset[0]
    assert item["prompt_id"] == source["prompt_id"]
    assert item["prompt_hash"] == source["prompt_hash"]
    assert item["source_index"] == 0
    assert tokenizer.template_kwargs[-1]["enable_thinking"] is False
    assert "extra_body" not in tokenizer.template_kwargs[-1]
