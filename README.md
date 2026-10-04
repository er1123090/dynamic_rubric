# Dynamic Rubric

RaR-Medicine / RaR-Science에서 **Static, OnlineRubrics, EvoRubrics**를 학습합니다. 실험별 **YAML 하나로 설정하고 SH 하나로 실행**합니다. Docker는 필요하지 않습니다.

## 1. 실험 선택

모든 명령은 저장소 루트에서 실행합니다.

| 데이터 | 방법 | 설정 (`configs/launch/`) | 실행 (`scripts/phase1/`) |
| --- | --- | --- | --- |
| Medicine | Static | [medicine_static_rubric.yaml](configs/launch/medicine_static_rubric.yaml) | [train_medicine_static_rubric.sh](scripts/phase1/train_medicine_static_rubric.sh) |
| Medicine | Online | [medicine_online_rubric.yaml](configs/launch/medicine_online_rubric.yaml) | [train_medicine_online_rubric.sh](scripts/phase1/train_medicine_online_rubric.sh) |
| Medicine | Evo | [medicine_evorubric.yaml](configs/launch/medicine_evorubric.yaml) | [train_medicine_evo_rubric.sh](scripts/phase1/train_medicine_evo_rubric.sh) |
| Science | Static | [science_static_rubric.yaml](configs/launch/science_static_rubric.yaml) | [train_science_static_rubric.sh](scripts/phase1/train_science_static_rubric.sh) |
| Science | Online | [science_online_rubric.yaml](configs/launch/science_online_rubric.yaml) | [train_science_online_rubric.sh](scripts/phase1/train_science_online_rubric.sh) |
| Science | Evo | [science_evorubric.yaml](configs/launch/science_evorubric.yaml) | [train_science_evo_rubric.sh](scripts/phase1/train_science_evo_rubric.sh) |

## 2. 환경·모델 준비

**Linux x86_64, Python 3.10, NVIDIA GPU·드라이버, `git`, `patch`, `uv`**가 필요합니다.

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install -e .
nvidia-smi -L

# Static/Online 학습 환경 + 모든 방법의 judge용 소스 준비
bash scripts/phase1/setup_verl_runtime.sh --prepare-source
bash scripts/phase1/setup_verl_runtime.sh --install
bash scripts/phase1/setup_verl_runtime.sh --check

# 추론 머신에서 judge 환경 설치 (해당 머신에도 소스와 모델 필요)
VERL_VENV="$PWD/.venvs/judge" bash scripts/phase1/setup_verl_runtime.sh --install
```

Evo는 원본 ZIP을 **별도로 받아 `docs/EvoRubrics-2155.zip`에 둔 후**, 학습 환경을 추가로 준비합니다.

```bash
bash scripts/phase1/setup_evorubrics_runtime.sh --prepare-source
bash scripts/phase1/setup_evorubrics_runtime.sh --install
bash scripts/phase1/setup_evorubrics_runtime.sh --check
```

`.venv`는 실행기, `.venvs/verl`·`.venvs/evorubrics`는 학습, `.venvs/judge`는 추론용입니다. **Evo 학습 환경의 vLLM으로 GPT-OSS judge를 실행하지 마세요.** 패키지 버전은 [veRL requirements](environment/verl-runtime-requirements.txt), [Evo lock](environment/evorubrics-runtime-lock.txt)을 따릅니다. 새 머신의 드라이버·GPU 호환성은 별도 확인이 필요합니다.

모델은 YAML의 revision에 맞춰 별도 준비하고 `models.<역할>.local_snapshot`에 경로를 지정합니다.

| 역할 | 모델 |
| --- | --- |
| 정책 | `Qwen/Qwen3-4B-Instruct-2507` |
| Static/Online judge | `Qwen/Qwen3-32B` |
| Online extractor / Evo judge | `openai/gpt-oss-120b` |

## 3. 데이터 준비

**모델·데이터·체크포인트·로그·원본 ZIP은 GitHub에 포함하지 않습니다.** 준비된 RaR 분할과 `split_manifest.json`, `rar_manifest.json`을 별도로 받아 아래 경로에 둡니다. 각 도메인은 train / development / final **1,500 / 150 / 300개**입니다.

| 도메인 | JSONL 디렉터리 | Static용 parquet |
| --- | --- | --- |
| Medicine | `data/rar/medicine/eval300/` | `data/rar/medicine/verl/{train,development}.parquet` |
| Science | `data/rar/science/public/` | `data/rar/science/verl/{train,development}.parquet` |

## 4. YAML 수정

| 설정할 항목 | YAML 위치 |
| --- | --- |
| 학습 GPU | `infrastructure.optimizer.gpus` |
| 추론 GPU·TP·메모리·포트·실행 파일 | `infrastructure.services.<서비스>.instances[]` |
| Online 제어 응답 생성 GPU | `infrastructure.pi0_control` |
| epochs·batch·step·LR | `training` (Evo LoRA·LR 등은 `evorubrics`) |
| 학습 Python | `launch.entry_python` (Online은 `launch.environment.RUNTIME_PYTHON`도 확인) |
| 서비스 URL·요청 동시성 | `launch.environment` |
| 저장 위치·실험 이름 | `output.root`, `launch.run_id` |

- 새 실험은 새 `run_id`를 사용합니다(Evo는 `launch.smoke_run_id`도 변경). 경로는 저장소 루트 기준이며, `host`/`code_host`는 비워도 됩니다.
- 학습값을 바꾸려면 `launch.tuning_mode: custom`을 사용합니다. step·seed 변경 시 연관 설정은 [상세 안내](docs/phase1_training_entrypoints.md)를 확인하세요.
- GPU 번호는 YAML에서 지정합니다. **Evo 학습은 GPU 한 장**, judge는 여러 장을 사용할 수 있습니다. 학습·추론 GPU가 겹치지 않게 배치하고 VRAM에 맞춰 메모리를 조정하세요.

## 5. 실행

아래는 **Science Online** 예시입니다. 다른 실험은 1절의 YAML/SH로 바꾸고, Static/Evo에서는 Online 전용 단계를 생략합니다.

### Online만: 제어 응답 사전 생성

학습 머신에서 지정 GPU가 비어 있을 때 실행합니다. Medicine은 `precompute_medicine_pi0.sh`를 사용합니다.

```bash
bash scripts/phase1/precompute_science_pi0.sh --check
bash scripts/phase1/precompute_science_pi0.sh
```

### 추론 서비스 시작

**추론 머신의 별도 터미널에서 각각 실행**합니다. 아래 `--check`는 설정 검사만 하므로, 확인 후 제거해야 서버가 시작됩니다. 서버를 원격 배치하거나 SSH로 자동 접속하지 않습니다.

```bash
# 모든 방법: judge (Static은 점수 프록시도 함께 시작)
.venv/bin/python scripts/phase1/serve_training.py \
  --config configs/launch/science_online_rubric.yaml --service judge --check

# Online만: extractor
.venv/bin/python scripts/phase1/serve_training.py \
  --config configs/launch/science_online_rubric.yaml --service extractor --check
```

다른 머신의 서비스는 SSH 터널 또는 접근 가능한 주소로 YAML의 URL을 맞추세요. Static judge URL은 raw vLLM이 아닌 점수 프록시 주소입니다. 인증 없는 API를 공개 인터넷에 노출하지 마세요.

### 학습 시작

학습 머신에서 실행합니다. `--check`는 설정·경로, `--check-services`는 서버 연결·모델까지 검사하고 종료합니다.

```bash
bash scripts/phase1/train_science_online_rubric.sh --check
bash scripts/phase1/train_science_online_rubric.sh --check-services
bash scripts/phase1/train_science_online_rubric.sh
```

## 6. 체크포인트·로그

기본 저장 위치는 `outputs/{domain}/{method}/seed-{seed}/{run_id}/`입니다.

- **Static/Online:** 매 step 모델 저장. 이전 step은 파라미터만, 최신 완료 step은 optimizer 등 재개 상태도 보관합니다. 같은 설정/run ID에 `--resume`을 붙여 재개합니다.
- **Evo:** 매 step policy/generator LoRA 저장. 통합 SH의 `--resume`은 아직 지원하지 않습니다.
- **콘솔 로그:** run 디렉터리와 같은 상위 폴더의 `_launch_logs/`에 저장합니다.

세부 파라미터·다중 추론 인스턴스·재개 조건은 [학습 설정과 실행 경로](docs/phase1_training_entrypoints.md)를 참고하세요.
