# Dynamic Rubric — RaR 학습 실험

RaR-Medicine / RaR-Science에서 **Static R0, OnlineRubrics, EvoRubrics**를 실행하는 저장소입니다. 각 실험은 **YAML 하나를 편집하고 SH 하나로 학습**합니다. General domain은 포함하지 않습니다. 학습 알고리즘은 GRPO이며, veRL의 `main_ppo`라는 모듈 이름이 PPO 실험을 의미하지는 않습니다.

동료에게 처음 전달한다면 **환경 준비 → YAML 수정 → 추론 서비스 준비 → Online 제어 응답 생성 → 실행 전 확인 → 학습** 순서로 진행하세요. 여섯 canonical 진입점은 **Docker 없이 Linux + Python 가상환경에서 직접 실행**할 수 있습니다. 학습 SH가 컨테이너를 만들거나 원격 서버를 자동 배치하거나 SSH로 접속하지는 않습니다.

## 1. 실행할 실험 선택

모든 명령은 `dynamic_rubric` 루트에서 실행합니다. 새 실험마다 YAML의 `launch.run_id`를 바꾸세요.

| 데이터 | 방법 | 편집할 YAML | 최종 학습 SH |
| --- | --- | --- | --- |
| Medicine | Static | [medicine_static_rubric.yaml](configs/launch/medicine_static_rubric.yaml) | [train_medicine_static_rubric.sh](scripts/phase1/train_medicine_static_rubric.sh) |
| Medicine | Online | [medicine_online_rubric.yaml](configs/launch/medicine_online_rubric.yaml) | [train_medicine_online_rubric.sh](scripts/phase1/train_medicine_online_rubric.sh) |
| Medicine | Evo | [medicine_evorubric.yaml](configs/launch/medicine_evorubric.yaml) | [train_medicine_evo_rubric.sh](scripts/phase1/train_medicine_evo_rubric.sh) |
| Science | Static | [science_static_rubric.yaml](configs/launch/science_static_rubric.yaml) | [train_science_static_rubric.sh](scripts/phase1/train_science_static_rubric.sh) |
| Science | Online | [science_online_rubric.yaml](configs/launch/science_online_rubric.yaml) | [train_science_online_rubric.sh](scripts/phase1/train_science_online_rubric.sh) |
| Science | Evo | [science_evorubric.yaml](configs/launch/science_evorubric.yaml) | [train_science_evo_rubric.sh](scripts/phase1/train_science_evo_rubric.sh) |

아래는 Science 예시입니다. Medicine은 해당 YAML/SH로 바꾸면 됩니다. 세 실험을 같은 GPU에서 동시에 시작하지 마세요.

## 2. 환경 준비

### 공통

제공된 GPU 환경은 **Linux x86_64 + Python 3.10** 기준입니다. GPU가 현재 Linux 실행 환경의 `nvidia-smi`에 보여야 합니다. `bash`, `git`, `patch`, `uv`, CUDA 런타임을 지원하는 NVIDIA 드라이버가 필요합니다. Python 3.11 이상이나 다른 GPU/OS 조합의 호환성을 보장하지 않습니다.

```bash
cd /path/to/dynamic_rubric
python3.10 -m venv .venv
.venv/bin/python -m pip install -e .
nvidia-smi -L
df -h .
```

`.venv`는 YAML 실행기용입니다. GPU 학습 환경은 아래처럼 **분리**합니다. 기존 Python을 쓰려면 YAML의 `launch.entry_python`, Online의 `launch.environment.RUNTIME_PYTHON`, 서비스의 `vllm_bin`/`python` 경로를 바꾸세요. 셸 진입점의 Python은 `LAUNCH_PYTHON`으로 바꿀 수 있습니다.

### Static / Online 학습 및 judge 환경

```bash
# 없을 때만 다운로드: 고정 commit + 현재 학습에 필요한 통합 패치를 준비
bash scripts/phase1/setup_verl_runtime.sh --prepare-source
# .venvs/verl 생성 및 패키지 설치 (네트워크/디스크 사용)
bash scripts/phase1/setup_verl_runtime.sh --install
bash scripts/phase1/setup_verl_runtime.sh --check
```

- 소스: veRL commit `890dfc3ebdd5647f7ea9730375414b1e3fb4e9a6` + [통합 패치](patches/verl_training_handoff.patch). **기존 두 veRL 패치를 중복 적용하지 마세요.** 이미 있는 소스 디렉터리는 덮어쓰지 않습니다.
- GPU 패키지: [verl-runtime-requirements.txt](environment/verl-runtime-requirements.txt). CUDA 12.9 계열 Torch/vLLM 환경이며 Evo 학습 환경과 다릅니다. CUDA wheel index가 필요한 환경에서는 설치 전에 `UV_EXTRA_INDEX_URL`을 지정하세요.
- 설치 경로 변경: `VERL_VENV`, `VERL_BASE_PYTHON`, `VERL_UV_BIN`, `VERL_SOURCE_ROOT`. 소스 위치를 바꿨다면 YAML `launch.environment.VERL_ROOT`도 지정합니다.
- 추론 머신에도 소스/환경/해당 모델이 필요합니다. YAML의 기본 judge 경로 `.venvs/judge`를 만들려면 같은 스크립트를 `VERL_VENV="$PWD/.venvs/judge" bash scripts/phase1/setup_verl_runtime.sh --install`로 실행합니다. 이미 `.venvs/verl`이 있다면 서비스 `vllm_bin`/`python`을 그 환경으로 지정해 재사용해도 됩니다.

### Evo 학습 환경

```bash
# docs/EvoRubrics-2155.zip이 필요. ZIP 검증, 압축 해제, 기록된 패치 적용
bash scripts/phase1/setup_evorubrics_runtime.sh --prepare-source
# .venvs/evorubrics 설치
bash scripts/phase1/setup_evorubrics_runtime.sh --install
bash scripts/phase1/setup_evorubrics_runtime.sh --check
```

Evo는 **Python 3.10 / Torch 2.6 / CUDA 12.4 / vLLM 0.8.5** 학습 환경을 사용합니다. [runtime lock](environment/evorubrics-runtime-lock.txt)과 [소스 패치 기록](environment/source-snapshots/EvoRubrics-2155-rq2-patch-manifest.json)을 함께 전달해야 합니다. 원본 ZIP만 두는 것으로는 충분하지 않습니다. 준비 스크립트가 `environment/upstream/EvoRubrics`를 만들며, 이미 수정된 소스는 덮어쓰지 않습니다. 경로 변경은 `EVORUBRICS_VENV`, `EVORUBRICS_BASE_PYTHON`, `EVORUBRICS_UV_BIN`을 사용합니다.

**GPT-OSS judge는 최신 judge 환경에서 별도로 실행합니다. Evo 학습용 vLLM 0.8.5로 GPT-OSS를 띄우지 마세요.**

환경 파일은 버전과 준비 절차를 제공합니다. 새 머신에서의 전체 패키지 설치·GPU 학습까지 자동 검증된 것은 아닙니다. 설치 시 의존성/드라이버 확인이 실패하면 학습을 시작하기 전에 해결하세요.

### 모델 파일

가중치는 GitHub에 포함하지 않습니다. 다음 모델의 YAML에 기록된 revision을 확보하고, 기본 경로에 두거나 `models.<역할>.local_snapshot`을 실제 경로로 수정하세요. 동일 머신에서는 기존 스냅샷 경로를 재사용할 수 있습니다.

| 역할 | 모델 | 기본 로컬 경로 |
| --- | --- | --- |
| 세 방법의 정책 | `Qwen/Qwen3-4B-Instruct-2507` | `models/Qwen3-4B-Instruct-2507` |
| Static / Online judge | `Qwen/Qwen3-32B` | `models/Qwen3-32B` |
| Online extractor / Evo judge | `openai/gpt-oss-120b` | `models/gpt-oss-120b` |

경로는 **저장소 루트 기준 상대경로**, 절대경로, `~`, `${환경변수}`를 지원합니다. 선언하지 않은 환경변수가 있으면 오류를 냅니다. `HF_HOME`은 사용자가 설정한 값을 존중하며 특정 사용자 디렉터리로 강제하지 않습니다. 토큰/비밀번호는 YAML에 기록하지 마세요.

## 3. GPU와 학습 파라미터 설정

여섯 YAML의 공통 구조는 `data → models → training → infrastructure → output → tracking → launch`입니다. `host`/`code_host`는 선택적 기록 필드라 기본값이 비어 있습니다. 자신의 호스트 라벨이 필요할 때만 채우고, GPU 번호·모델 경로·Python/vLLM 경로·서비스 URL은 실행 머신에 맞게 확인하세요. 컨테이너 이름과 이미지는 canonical 실행에 필요하지 않습니다.

| 바꿀 항목 | YAML 위치 |
| --- | --- |
| 학습 GPU | `infrastructure.optimizer.gpus` |
| judge / extractor GPU·TP·메모리·포트 | `infrastructure.services.<서비스>.instances[]` |
| Online 사전생성 GPU·TP·메모리·포트 | `infrastructure.pi0_control` |
| epochs / batch / 최대 step | `training.epochs`, `global_prompt_batch`, `expected_global_steps` |
| Static / Online LR·KL·sampling·길이 | `training` |
| Evo 두 LoRA의 LR·rank·reward 등 | `evorubrics` |
| 서버 접속 URL / 요청 동시성 | `launch.environment` |
| 출력 디스크 / 실험 이름 | `output.root`, `launch.run_id` |

예를 들어 **Static/Online**에서 학습 GPU 2·3번을 쓰고 batch/step을 바꾸려면 기존 YAML의 해당 값만 수정합니다.

```yaml
infrastructure:
  optimizer:
    host: ""                 # 선택적 기록값; 필요하면 자신의 호스트 라벨 입력
    gpus: [2, 3]             # 현재 실행 환경의 nvidia-smi 기준 GPU 인덱스
training:
  epochs: 4
  global_prompt_batch: 64
  ppo_mini_batch_size: 64
  expected_global_steps: 64
  rollout_tensor_parallel_size: 1
launch:
  tuning_mode: custom
```

이 코드는 부분 예시입니다. 다른 필드를 지우지 마세요. `CUDA_VISIBLE_DEVICES`와 GPU 개수는 실행기가 계산하므로 중복 export하지 않습니다. rollout TP는 학습 GPU 개수를 나누어야 합니다. 원하는 step까지 도달할 만큼 epochs와 데이터가 충분해야 합니다.

- `paper`: 기존 재현용 학습값 유지. **호스트 이름·GPU 배치·경로는 바꿀 수 있습니다.**
- `custom`: 지원되는 batch/epoch/step/seed/LR 등을 수정. Online/Evo의 step을 줄이면 `training.audit_checkpoints`, `training.reuse_anchors`, `analysis.reuse_anchors`도 새 범위에 맞춥니다. seed를 바꾸면 fixed-probe sample seed와 해당 manifest/cache 경로도 분리하세요.
- **Evo 학습은 선택한 GPU 한 장**을 사용합니다. 기존 dual-LoRA optimizer의 분산 저장·복원 문제 때문에 여러 장을 입력하면 사전에 거부합니다. Evo judge의 TP/여러 GPU 사용은 별개입니다.
- Online은 현재 reward backend의 **16 responses / 8 elicitation pairs**를 유지합니다. 알고리즘 자체를 바꾸는 필드까지 자유롭게 변경하는 것은 아닙니다. 상세 지원 범위는 [파라미터 안내](docs/phase1_training_entrypoints.md)를 참고하세요.

## 4. judge / extractor 시작

**추론 GPU가 있는 머신에서** 실행합니다. 서비스 실행기도 같은 YAML을 읽습니다. 호스트 이름은 기록용이며, 현재 머신에서 선택된 GPU에 서버를 띄웁니다. `--check`로 경로·GPU 목록·TP·포트·실행 명령을 먼저 확인하세요.

```bash
# Static: Qwen judge + 필수 점수 프록시를 함께 실행
.venv/bin/python scripts/phase1/serve_training.py \
  --config configs/launch/science_static_rubric.yaml --service judge --check
# 위 명령에서 --check를 빼면 실제 서버를 시작합니다.

# Online: 아래 두 서버를 각각 별도 터미널에서 시작 (--check 제거)
.venv/bin/python scripts/phase1/serve_training.py \
  --config configs/launch/science_online_rubric.yaml --service extractor --check
.venv/bin/python scripts/phase1/serve_training.py \
  --config configs/launch/science_online_rubric.yaml --service judge --check

# Evo: GPT-OSS judge
.venv/bin/python scripts/phase1/serve_training.py \
  --config configs/launch/science_evorubric.yaml --service judge --check
```

`instances`에 별도 GPU/포트의 두 번째 서버를 추가하면 `--instance 1`로 실행합니다. Online 접속 URL도 같은 순서의 쉼표 구분 목록으로 지정합니다. 한 인스턴스의 TP는 GPU 목록 길이와 같아야 합니다. **동일 GPU를 여러 서비스에 할당할 때 메모리 예산은 자동 조정되지 않습니다.** 기본 자원값은 기존 대용량 GPU 배치를 바탕으로 하므로 자신의 GPU 용량에 맞춰 조정하세요. 학습 GPU와 겹치지 않게 배치하는 것이 기본입니다.

| 방법 | 학습 프로세스가 접속할 기본 URL | 제공 서비스 |
| --- | --- | --- |
| Static | `http://127.0.0.1:28137` | Qwen 점수 프록시 (raw vLLM은 28136) |
| Online | `http://127.0.0.1:28011`, `http://127.0.0.1:28014` | GPT-OSS extractor, Qwen judge |
| Evo | `http://127.0.0.1:28011/v1` | GPT-OSS judge |

기본 bind는 `127.0.0.1`입니다. 원격 서버면 학습 환경에서 SSH 터널을 열거나 접근 가능한 네트워크 주소를 사용하고, `launch.environment`의 URL을 맞춥니다. 예: `ssh -N -L 28011:127.0.0.1:28011 -L 28014:127.0.0.1:28014 user@inference-host`. Docker를 선택적으로 사용하는 경우에만 컨테이너의 localhost와 호스트의 localhost가 다를 수 있습니다. 인증 없는 API를 공개 인터넷에 노출하지 마세요.

## 5. Online에서만: 고정 pi₀ 제어 응답 준비

```bash
bash scripts/phase1/precompute_science_pi0.sh --check
bash scripts/phase1/precompute_science_pi0.sh
# Medicine: precompute_medicine_pi0.sh
```

같은 Online YAML의 `infrastructure.pi0_control`을 사용합니다. GPU 1번 고정이 아닙니다. 지정한 GPU가 비어 있는지 확인한 뒤 임시 policy vLLM/identity proxy로 제어 응답을 생성하고, 자신이 시작한 서버만 종료합니다. 학습과 동시에 같은 GPU에서 실행하지 마세요.

제어 응답은 기본 `outputs/{domain}/shared/seed-11/pi0_control_cache`에 저장합니다. 학습은 `launch.environment.ONLINE_CONTROL_CACHE_DIR`에서 **정확히 하나의 sealed manifest**를 읽습니다. 새 seed·정책·데이터로 생성할 때는 별도 디렉터리를 사용하세요. 기존 데이터셋 캐시나 예전 실험의 제어 응답을 GitHub로 넘길 필요는 없습니다.

## 6. 실행 전 확인 → 학습

```bash
# 선택한 하나의 실험에 대해 실행
bash scripts/phase1/train_science_static_rubric.sh --check
bash scripts/phase1/train_science_static_rubric.sh --check-services
bash scripts/phase1/train_science_static_rubric.sh

# Online / Evo는 각각 해당 진입점을 사용
# bash scripts/phase1/train_science_online_rubric.sh --check
# bash scripts/phase1/train_science_evo_rubric.sh --check
```

- `--check`: 설정과 로컬 필수 경로 확인. 서버·학습을 시작하지 않고 GPU를 할당하지 않습니다.
- `--check-services`: 위 확인 + 실제 URL의 모델 ID 확인 후 종료. Static은 점수 프록시 identity도 확인합니다. 생성 요청은 보내지 않습니다.
- 두 검사 모두 VRAM 여유나 실제 학습 패키지의 완전한 호환성을 보장하지는 않습니다. 처음에는 `nvidia-smi`와 환경 설치 스크립트의 확인 결과를 함께 확인하세요.
- Online의 pi₀ manifest가 없으면 먼저 5단계를 진행합니다. Evo는 내부적으로 짧은 사전 검증과 재개 검증을 통과한 뒤 본 학습을 시작합니다.

## 7. 데이터셋과 저장 결과

현재 로컬에 준비된 입력은 다음과 같습니다. **데이터셋은 이 GitHub 저장소에 포함하지 않습니다.** 데이터 재배포 권한을 확인한 뒤 아래 입력과 manifest를 **동일 경로로 별도 전달**하세요. 저장소만 clone한 상태에서는 데이터가 없어 학습 전 검사가 실패하는 것이 정상입니다.

| 도메인 | JSONL 위치 | train / development / final | Static 입력 |
| --- | --- | --- | --- |
| Medicine | `data/rar/medicine/eval300/` | 1,500 / 150 / 300 | `data/rar/medicine/verl/{train,development}.parquet` |
| Science | `data/rar/science/public/` | 1,500 / 150 / 300 | `data/rar/science/verl/{train,development}.parquet` |

각 JSONL 디렉터리의 `split_manifest.json`, `rar_manifest.json`도 함께 전달합니다. Static development parquet의 2,400행은 150개 질문의 응답/seed 배치이며, 서로 다른 질문 2,400개가 아닙니다. Medicine의 예전 `public/final.jsonl`과 `eval300/final.jsonl`을 혼용하지 마세요. 예전 `horizon_science.yaml`로 다시 나눈 분할도 현재 준비된 300개 final 분할과 같다고 가정하면 안 됩니다.

외부 평가용 HealthBench / GPQA-Diamond 입력은 별도이며 여섯 학습의 필수 입력은 아닙니다. 이 저장소의 RaR Evo 설정은 원본 ZIP의 HealthBench 실험과 구별됩니다.

기본 결과 위치는 `outputs/{domain}/{method}/seed-{seed}/{run_id}/`입니다.

- Static/Online: `verl-run/checkpoints/global_step_*`, 매 step 모델 저장. 이전 step은 파라미터를 보존하고 **최신 완료 step만 optimizer 등 전체 재개 상태**를 유지합니다.
- Evo: `upstream-run` 아래 매 step policy/generator LoRA 및 실행 기록을 저장합니다.
- 학습 콘솔은 화면과 `.../seed-{seed}/_launch_logs/{run_id}-시간.log`에 함께 남습니다. 각 방법의 상세 reward/step 산출물도 해당 run 디렉터리에 저장됩니다.
- Static/Online 재개: 같은 YAML/run ID에 `--resume`. 오래된 파라미터-only checkpoint는 정확한 optimizer 재개용이 아닙니다. Evo 통합 진입점은 `--resume`을 지원하지 않습니다.

## 8. 코드를 읽는 순서

```text
configs/launch/{domain}_{method}.yaml     ← 사용자가 수정하는 실험 설정
scripts/phase1/train_{domain}_{method}.sh ← 최종 학습 진입점
  └─ scripts/phase1/launch_training.py    ← 경로/GPU/설정 검사, 환경 전달, 콘솔 로그
       ├─ Static: scripts/run_static_grpo.sh
       │    └─ veRL main_ppo + training/verl_reward.py
       ├─ Online: phase1/full_run.py → scripts/phase1/run_online_full.sh
       │    └─ veRL + training/verl_online_runtime.py 및 step hook
       └─ Evo: scripts/phase1/run_evorubrics.sh → phase1/evorubrics_run.py
            └─ 패치된 EvoRubrics shared-base trainer

src/dynamic_rubric/phase1/config.py       설정 검증, paper/custom 구분
src/dynamic_rubric/training/              학습 연결, reward, checkpoint 처리
src/dynamic_rubric/services/              vLLM 점수/정책 identity 프록시
scripts/phase1/serve_training.py          YAML 기반 judge/extractor 실행
scripts/phase1/precompute_pi0.py          YAML 기반 Online 제어 응답 준비
environment/, patches/                   환경 버전, upstream 재현 자료
```

`scripts/phase1`의 이전 운영·분석 스크립트는 위 여섯 학습을 시작할 때 직접 실행할 필요가 없습니다. 학습 설정 변경은 `configs/launch`에서 시작하세요. Medicine 이름의 예전 Evo runner는 호환용 wrapper이며, 실제 공통 구현은 `run_evorubrics.sh`입니다.

## GitHub 전달 범위

이 저장소는 **코드·설정(YAML/SH)·실행 문서·테스트·환경 requirements/lock·소스 패치/manifest만** 전달합니다. 동료는 컨테이너를 새로 만들 필요 없이 드라이버, Python 3.10 가상환경, 모델, 데이터를 준비해 실행할 수 있습니다. 데이터셋, 모델/체크포인트, 실행 결과·로그·그림, 가상환경, upstream checkout, 캐시, 비밀 설정, 개인 SSH 설정 및 논문 PDF/원본 ZIP은 포함하지 않습니다.

새 머신에서는 7절의 데이터와 manifest, 2절의 모델을 별도 준비하세요. Evo는 `docs/EvoRubrics-2155.zip`도 별도로 받아야 하며, 준비 스크립트가 기록된 SHA-256을 검증합니다. 저장소의 패치/manifest는 포함되지만 원본 ZIP을 대신하지는 않습니다.

`.gitignore`는 위 로컬 자료의 실수 업로드를 방지합니다. **이미 Git에 추적된 파일은 `.gitignore`로 제외되지 않으므로 업로드 전 `git status`와 스테이징 목록을 확인하세요.** 업로드에서 제외하는 것은 로컬 파일을 삭제한다는 뜻이 아닙니다.

## 코드 검증 범위

모델·데이터를 받기 전에도 아래 전달용 검사는 GPU나 컨테이너 없이 실행할 수 있습니다. Git으로 clone한 저장소 기준입니다.

```bash
.venv/bin/python -m pip install pytest
.venv/bin/python -m pytest -q \
  tests/phase1/test_public_handoff.py tests/phase1/test_public_scripts.py
```

전체 테스트에는 준비된 RaR 데이터, 패치된 EvoRubrics/veRL 소스 및 Torch가 필요한 항목이 포함됩니다. 해당 입력을 준비하지 않은 코드 전용 사본에서 전체 테스트를 바로 실행하면 관련 검사가 실패할 수 있습니다. 위 전달용 검사 통과는 실제 GPU 학습이나 새 머신의 패키지 호환성 검증을 대신하지 않습니다.
