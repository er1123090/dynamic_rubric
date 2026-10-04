#!/usr/bin/env python3
"""Export the static-R0 VERL checkpoints as loadable Hugging Face models."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from huggingface_hub import save_torch_state_dict
import torch


RUN_ID = "pilot-static-r0-100step-20260821"
BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
BASE_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_card(step: int) -> str:
    status = (
        "This is the initialization checkpoint (pi_0), before any RL optimizer update."
        if step == 0
        else f"This checkpoint is pi_{step}, after {step} static-rubric RL optimizer updates."
    )
    return f"""---
license: apache-2.0
library_name: transformers
pipeline_tag: text-generation
base_model: {BASE_MODEL}
tags:
- qwen3
- healthbench
- reinforcement-learning
- static-rubric
- r0
---

# Qwen3-4B HealthBench Static-Rubric R0 — Step {step}

{status}

This model belongs to the **static-rubric R0** experiment, not a dynamic-rubric
training run. Training rewards use a frozen, prompt-specific rubric bank: each
training prompt is scored with its own `R0(x)` throughout optimization.

## Provenance

- Run ID: `{RUN_ID}`
- Optimizer step: `{step}`
- Base model: `{BASE_MODEL}`
- Base revision: `{BASE_REVISION}`
- Reward source: `static_r0_only`
- Policy training split: 256 HealthBench prompts
- Parameterization: full-model RL
- Export dtype: BF16
- Original format: VERL FSDP v1, world size 1, FP32 state dict

## Loading

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo_id = "HYU-NLP-EVAL/qwen3-4b-healthbench-static-r0-step-{step:03d}"
tokenizer = AutoTokenizer.from_pretrained(repo_id)
model = AutoModelForCausalLM.from_pretrained(repo_id, torch_dtype="bfloat16")
```

## Intended use and limitations

This is a research checkpoint for studying proxy-rubric staleness during policy
optimization. It is not a medical device and must not be used as a substitute
for professional medical advice. Static-rubric reward improvement does not by
itself establish improvement against independent HealthBench ground truth.
"""


def _validate_existing(output_dir: Path, step: int) -> bool:
    manifest_path = output_dir / "export_manifest.json"
    index_path = output_dir / "model.safetensors.index.json"
    if not manifest_path.is_file() or not index_path.is_file():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("optimizer_step") != step:
        raise RuntimeError(f"existing export has the wrong step: {output_dir}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    shard_names = sorted(set(index.get("weight_map", {}).values()))
    if not shard_names or any(not (output_dir / name).is_file() for name in shard_names):
        return False
    return True


def export_checkpoint(checkpoint_root: Path, export_root: Path, step: int) -> Path:
    source_root = checkpoint_root / f"global_step_{step}" / "actor"
    source_model = source_root / "model_world_size_1_rank_0.pt"
    source_hf = source_root / "huggingface"
    if not source_model.is_file() or not source_hf.is_dir():
        raise FileNotFoundError(f"incomplete source checkpoint: {source_root}")

    output_dir = export_root / f"qwen3-4b-healthbench-static-r0-step-{step:03d}"
    if output_dir.exists():
        if _validate_existing(output_dir, step):
            print(f"validated existing export: {output_dir}", flush=True)
            return output_dir
        raise RuntimeError(f"incomplete existing export must be inspected: {output_dir}")

    export_root.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f"step-{step:03d}-", dir=export_root))
    try:
        for source in source_hf.iterdir():
            destination = temp_dir / source.name
            if source.is_dir():
                shutil.copytree(source, destination)
            else:
                shutil.copy2(source, destination)

        config_path = temp_dir / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["dtype"] = "bfloat16"
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        (temp_dir / "README.md").write_text(_model_card(step), encoding="utf-8")

        state_dict = torch.load(source_model, map_location="cpu", weights_only=False, mmap=True)
        if not isinstance(state_dict, dict) or not state_dict:
            raise RuntimeError(f"invalid state dict: {source_model}")
        if "model.embed_tokens.weight" not in state_dict or "lm_head.weight" not in state_dict:
            raise RuntimeError("expected tied embedding keys are missing")
        if not torch.equal(state_dict["model.embed_tokens.weight"], state_dict["lm_head.weight"]):
            raise RuntimeError("tied embedding weights differ; refusing a lossy export")
        state_dict.pop("lm_head.weight")
        converted = {
            name: tensor.detach().to(dtype=torch.bfloat16).contiguous()
            for name, tensor in state_dict.items()
        }
        save_torch_state_dict(
            converted,
            temp_dir,
            max_shard_size="4GB",
            safe_serialization=True,
            metadata={
                "format": "pt",
                "run_id": RUN_ID,
                "reward_source": "static_r0_only",
                "optimizer_step": str(step),
            },
        )
        del converted
        del state_dict

        index_path = temp_dir / "model.safetensors.index.json"
        index = json.loads(index_path.read_text(encoding="utf-8"))
        shard_names = sorted(set(index["weight_map"].values()))
        manifest = {
            "schema_version": 1,
            "run_id": RUN_ID,
            "optimizer_step": step,
            "reward_source": "static_r0_only",
            "base_model": BASE_MODEL,
            "base_revision": BASE_REVISION,
            "source_model_path": str(source_model),
            "source_model_size": source_model.stat().st_size,
            "source_model_sha256": _sha256(source_model),
            "export_dtype": "bfloat16",
            "weight_shards": [
                {
                    "name": name,
                    "size": (temp_dir / name).stat().st_size,
                    "sha256": _sha256(temp_dir / name),
                }
                for name in shard_names
            ],
        }
        (temp_dir / "export_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temp_dir, output_dir)
    except BaseException:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    print(f"exported: {output_dir}", flush=True)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--export-root", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+", required=True)
    args = parser.parse_args()
    for step in args.steps:
        export_checkpoint(args.checkpoint_root, args.export_root, step)


if __name__ == "__main__":
    main()
