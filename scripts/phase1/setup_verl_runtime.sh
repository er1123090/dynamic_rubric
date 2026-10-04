#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
project_root=$(cd "${script_dir}/../.." && pwd)
venv_root=${VERL_VENV:-${project_root}/.venvs/verl}
source_root=${VERL_SOURCE_ROOT:-${project_root}/environment/upstream/verl}
requirements=${VERL_REQUIREMENTS:-${project_root}/environment/verl-runtime-requirements.txt}
source_patch=${VERL_SOURCE_PATCH:-${project_root}/patches/verl_training_handoff.patch}
source_commit=${VERL_SOURCE_COMMIT:-890dfc3ebdd5647f7ea9730375414b1e3fb4e9a6}
source_repository=${VERL_SOURCE_REPOSITORY:-https://github.com/verl-project/verl.git}

usage() {
  echo "usage: $0 --check | --prepare-source | --install" >&2
}

[[ $# = 1 ]] || { usage; exit 2; }
action=$1
case "${action}" in
  --check|--prepare-source|--install) ;;
  *) usage; exit 2 ;;
esac

[[ "$(uname -s)" = Linux ]] || { echo "veRL runtime requires Linux" >&2; exit 2; }
if [[ -n "${VERL_BASE_PYTHON:-}" ]]; then
  base_python=${VERL_BASE_PYTHON}
else
  base_python=$(command -v python3.10 || true)
fi
[[ -n "${base_python}" && -x "${base_python}" ]] || {
  echo "missing Python 3.10; install it or set VERL_BASE_PYTHON" >&2
  exit 2
}
python_version=$("${base_python}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
[[ "${python_version}" = 3.10 ]] || {
  echo "veRL requires Python 3.10, got ${python_version}: ${base_python}" >&2
  exit 2
}
[[ -f "${source_patch}" ]] || { echo "missing veRL handoff patch: ${source_patch}" >&2; exit 2; }

source_args=(
  --repository "${source_repository}"
  --commit "${source_commit}"
  --destination "${source_root}"
  --patch "${source_patch}"
)
if [[ "${action}" = --prepare-source ]]; then
  exec "${base_python}" "${script_dir}/prepare_verl_source.py" --prepare "${source_args[@]}"
fi

if [[ -n "${VERL_UV_BIN:-}" ]]; then
  uv_bin=${VERL_UV_BIN}
else
  uv_bin=$(command -v uv || true)
fi
[[ -n "${uv_bin}" && -x "${uv_bin}" ]] || {
  echo "missing uv; install it or set VERL_UV_BIN" >&2
  exit 2
}
[[ -f "${requirements}" ]] || {
  echo "missing runtime constraints: ${requirements}" >&2
  exit 2
}

"${base_python}" "${script_dir}/prepare_verl_source.py" --check "${source_args[@]}"
command -v nvidia-smi >/dev/null || {
  echo "CUDA-capable NVIDIA runtime is required (nvidia-smi is unavailable)" >&2
  exit 2
}

verify_runtime() {
  local runtime_python=$1
  "${runtime_python}" - "${source_root}" <<'PYVERIFY'
import importlib.metadata as metadata
import json
import sys
from pathlib import Path

import torch
import transformers
import verl
import vllm

source = Path(sys.argv[1]).resolve()
actual = {
    "python": f"{sys.version_info.major}.{sys.version_info.minor}",
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "transformers": transformers.__version__,
    "vllm_distribution": metadata.version("vllm"),
    "vllm_import": vllm.__version__,
    "verl": getattr(verl, "__version__", "unknown"),
    "verl_file": str(Path(verl.__file__).resolve()),
}
expected = {
    "python": "3.10",
    "torch": "2.11.0+cu129",
    "torch_cuda": "12.9",
    "transformers": "5.10.4",
    "vllm_distribution": "0.20.1+cu129",
}
problems = [f"{name}: expected {value}, got {actual[name]}" for name, value in expected.items()
            if actual[name] != value]
if source not in Path(verl.__file__).resolve().parents:
    problems.append(f"verl import is not from the pinned checkout: {actual['verl_file']}")
if problems:
    raise SystemExit("runtime pin verification failed:\n  " + "\n  ".join(problems))
print(json.dumps(actual, indent=2, sort_keys=True))
PYVERIFY
}

if [[ "${action}" = --check ]]; then
  if [[ -x "${venv_root}/bin/python" ]]; then
    verify_runtime "${venv_root}/bin/python"
  else
    echo "veRL source/prerequisites are valid; runtime is not installed at ${venv_root}."
  fi
  exit 0
fi

if [[ ! -x "${venv_root}/bin/python" ]]; then
  "${uv_bin}" venv --python "${base_python}" "${venv_root}"
fi
"${uv_bin}" pip install --python "${venv_root}/bin/python" --requirement "${requirements}"
"${uv_bin}" pip install --python "${venv_root}/bin/python" --editable "${project_root}"
"${uv_bin}" pip install --python "${venv_root}/bin/python" --no-deps --editable "${source_root}"
"${uv_bin}" pip check --python "${venv_root}/bin/python"
verify_runtime "${venv_root}/bin/python"


