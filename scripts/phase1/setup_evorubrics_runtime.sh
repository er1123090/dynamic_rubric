#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
project_root=$(cd "${script_dir}/../.." && pwd)
venv_root=${EVORUBRICS_VENV:-${project_root}/.venvs/evorubrics}
lock_file=${EVORUBRICS_LOCK_FILE:-${project_root}/environment/evorubrics-runtime-lock.txt}
upstream_root=${EVORUBRICS_SOURCE_ROOT:-${project_root}/environment/upstream/EvoRubrics}
source_archive=${EVORUBRICS_SOURCE_ARCHIVE:-${project_root}/docs/EvoRubrics-2155.zip}
patch_manifest=${EVORUBRICS_PATCH_MANIFEST:-${project_root}/environment/source-snapshots/EvoRubrics-2155-rq2-patch-manifest.json}

usage() {
  echo "usage: $0 --check | --prepare-source | --install" >&2
}

[[ $# = 1 ]] || { usage; exit 2; }
action=$1
case "${action}" in
  --check|--prepare-source|--install) ;;
  *) usage; exit 2 ;;
esac

if [[ "$(uname -s)" != Linux ]]; then
  echo "EvoRubrics runtime requires Linux" >&2
  exit 2
fi

if [[ -n "${EVORUBRICS_BASE_PYTHON:-}" ]]; then
  base_python=${EVORUBRICS_BASE_PYTHON}
else
  base_python=$(command -v python3.10 || true)
fi

[[ -n "${base_python}" && -x "${base_python}" ]] || {
  echo "missing Python 3.10; install it or set EVORUBRICS_BASE_PYTHON" >&2
  exit 2
}
python_version=$("${base_python}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
[[ "${python_version}" = 3.10 ]] || {
  echo "EvoRubrics requires Python 3.10, got ${python_version}: ${base_python}" >&2
  exit 2
}
[[ -f "${lock_file}" ]] || { echo "missing runtime lock: ${lock_file}" >&2; exit 2; }

source_args=(
  --archive "${source_archive}"
  --destination "${upstream_root}"
  --patch-manifest "${patch_manifest}"
)
if [[ "${action}" = --prepare-source ]]; then
  exec "${base_python}" "${script_dir}/prepare_evorubrics_source.py" --extract "${source_args[@]}"
fi

if [[ -n "${EVORUBRICS_UV_BIN:-}" ]]; then
  uv_bin=${EVORUBRICS_UV_BIN}
else
  uv_bin=$(command -v uv || true)
fi
[[ -n "${uv_bin}" && -x "${uv_bin}" ]] || {
  echo "missing uv; install it or set EVORUBRICS_UV_BIN" >&2
  exit 2
}
"${base_python}" "${script_dir}/prepare_evorubrics_source.py" --check "${source_args[@]}"
[[ -f "${upstream_root}/evorubric-main/main_shared_base.py" ]] || {
  echo "EvoRubrics source is not prepared; run $0 --prepare-source first" >&2
  exit 2
}
command -v nvidia-smi >/dev/null || {
  echo "CUDA-capable NVIDIA runtime is required (nvidia-smi is unavailable)" >&2
  exit 2
}

if [[ "${action}" = --check ]]; then
  echo "EvoRubrics prerequisites are present. Runtime installation was not started."
  exit 0
fi

if [[ ! -x "${venv_root}/bin/python" ]]; then
  "${uv_bin}" venv --python "${base_python}" "${venv_root}"
fi
"${uv_bin}" pip install \
  --python "${venv_root}/bin/python" \
  --torch-backend cu124 \
  --requirement "${lock_file}"
"${uv_bin}" pip check --python "${venv_root}/bin/python"

export PYTHONPATH="${project_root}/src:${upstream_root}/evorubric-main:${upstream_root}/third_party/verl:${upstream_root}/rubric_guidance${PYTHONPATH:+:${PYTHONPATH}}"
"${venv_root}/bin/python" - <<'PYVERIFY'
import datasets
import flash_attn
import main_shared_base
import peft
import pyarrow
import torch
import vllm
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from transformers import Qwen3Config, Qwen3ForCausalLM

assert torch.__version__.startswith("2.6.0")
assert peft.__version__ == "0.17.1"
assert vllm.__version__ == "0.8.5"
assert pyarrow.__version__ == "20.0.0"
assert datasets.__version__ == "2.14.4"
assert flash_attn.__version__ == "2.7.4.post1"
config = Qwen3Config(vocab_size=128, hidden_size=32, intermediate_size=64,
                     num_hidden_layers=1, num_attention_heads=4,
                     num_key_value_heads=2, head_dim=8, max_position_embeddings=64)
model = get_peft_model(
    Qwen3ForCausalLM(config),
    LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"),
    adapter_name="policy_llm",
)
model.add_adapter("rubrics_generator", model.peft_config["policy_llm"])
assert get_peft_model_state_dict(model, adapter_name="policy_llm")
assert get_peft_model_state_dict(model, adapter_name="rubrics_generator")
print("EvoRubrics runtime import and dual-LoRA check passed")
PYVERIFY

