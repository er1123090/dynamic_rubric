# 여섯 RaR 학습의 환경 자료

설치 순서는 [프로젝트 README](../README.md#2-환경-준비)를 따릅니다. GPU 학습 환경을 실행기 `.venv`와 구분합니다.

| 용도 | 기본 환경 | 재현 자료 | 준비 명령 |
| --- | --- | --- | --- |
| YAML 실행기 | `.venv` | `pyproject.toml` | `python3.10 -m venv .venv` 후 `pip install -e .` |
| Static / Online | `.venvs/verl` | `verl-runtime-requirements.txt`, `../patches/verl_training_handoff.patch` | `scripts/phase1/setup_verl_runtime.sh` |
| judge / extractor / pi₀ vLLM | `.venvs/judge` | 같은 최신 veRL GPU 패키지 사용 가능 | `VERL_VENV="$PWD/.venvs/judge" bash scripts/phase1/setup_verl_runtime.sh --install` |
| Evo 학습 | `.venvs/evorubrics` | `evorubrics-runtime-lock.txt`, `source-snapshots/EvoRubrics-2155-rq2-patch-manifest.json` | `scripts/phase1/setup_evorubrics_runtime.sh` |

두 setup SH는 `--prepare-source`, `--install`, `--check`를 구분합니다. 처음에는 source를 준비하고 설치합니다. `--check`는 설치나 학습을 수행하지 않습니다. 기존 upstream checkout을 자동 덮어쓰지 않으므로, 이미 있다면 먼저 확인 결과를 읽으세요.

Static/Online의 현재 실행 환경에서 확인한 주요 import 버전은 Python 3.10, Torch `2.11.0+cu129`, Transformers `5.10.4`, vLLM `0.20.1`(distribution `0.20.1+cu129`)입니다. requirements는 기존 Conda 환경 전체를 복사하지 않고 직접 의존성을 제한합니다. vLLM wheel URL/해시는 설치된 배포 기록에서 가져왔습니다. **새 환경의 resolver 및 전체 GPU 학습은 별도 검증 대상**입니다. CUDA Torch wheel index가 필요한 경우 `UV_EXTRA_INDEX_URL`을 지정합니다.

Evo의 lock은 Python 3.10/Linux x86_64, Torch 2.6/CUDA 12.4, vLLM 0.8.5, CPython 3.10 전용 FlashAttention wheel을 전제로 합니다. 최신 judge 패키지를 이 환경에 섞지 마세요. 설치 과정에서 source hash, package/import, 두 LoRA 구성 가능 여부를 확인합니다.

`upstream-lock.json`, `SOURCES.md`, 이름에 특정 서버가 들어간 JSON은 **이전 실험 시점의 이력**입니다. 예를 들어 과거 lock의 Transformers 5.14.1은 현재 실제 import 버전과 다릅니다. 이전 기록을 변경해 과거 결과를 재해석하지 않고, 신규 여섯 학습의 환경은 위 requirements/setup 경로를 기준으로 준비합니다.

GitHub에는 requirements/lock/패치/소스 manifest를 포함합니다. `upstream/`, 가상환경, 모델 캐시는 `.gitignore`로 제외됩니다. 원본 Evo ZIP도 별도로 전달되어야 합니다. 설치 및 준비 명령은 실행자가 요청했을 때만 다운로드/파일 생성을 수행합니다.
