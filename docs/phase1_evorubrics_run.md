# RQ2 EvoRubrics / RaR-Medicine 실행

이 구현은 RQ2의 데이터·학습량·분석 설계에 EvoRubrics의 dual-LoRA 목적함수와 공개 trainer를 적용한다. 원 논문의 HealthBench benchmark를 그대로 재현하는 실험은 아니다.

## 설정의 출처

| 항목 | 적용 설정 | 근거 |
|---|---|---|
| 초기 정책 | Qwen3-4B-Instruct-2507, non-thinking | RQ2 / 기존 OnlineRubrics와 동일 |
| 데이터 | train 1,500, held-out 300, seed 11 | RQ2 / 기존 immutable split |
| 학습량 | prompt batch 96, 3 epochs, 마지막 batch 60, 총 48 step | RQ2 |
| 학습 구조 | shared frozen backbone + θ(policy), ψ(rubric generator) | EvoRubrics 공개 코드 |
| LoRA | rank 32, alpha 64 | EvoRubrics 설정 |
| LR / KL / temperature | θ 2e-5, ψ 5e-6, KL 1e-4, temperature 0.7 | 결합 설정 |
| 학습 생성 | 문항당 답변 M=4, rubric N=4 | EvoRubrics |
| generator reward | similarity, discrimination, diversity, reflect 각 0.25 | golden/reference 경로 |
| 고정 reference | 기존 train R0 rubric | 독립 정답 label로 해석하지 않음 |
| judge | GPT-OSS-120B, snapshot b5c939de8f754692c1647ca79fbf85e8c1e70f8a | 우리 실험의 대체 judge / 현재 사용자 서버 |
| 분석 probe | train에서 고정 100문항, θ별 Pool-B=16, ψ별 rubric 4개 | RQ2 |

논문 §5.1과 공개 evaluator 기본값의 judge는 DeepSeek-V3.2다. GPT-OSS-120B는 우리 실험에서 교체한 모델이며 논문과 동일한 judge가 아니다.

공개 코드의 signed-weight 점수는 `(total - negative_sum) / (positive_sum - negative_sum)`을 [0,1]로 제한한다. 논문의 positive-weight denominator 식과 차이가 있어 공개 코드 동작을 유지했다. 초기 θ/ψ는 같은 backbone에서 시작하며 공개 배포의 별도 SFT warm start를 가져오지 않는다.

원본은 `docs/EvoRubrics-2155.zip`이며 SHA256은 `82bad1b31e6db0f0e9890431154a206c0f48881f8159ae04f81a45167b3025f6`이다. 원본 ZIP은 보존하고 `environment/upstream/EvoRubrics`에서 최소 패치를 적용한다. 829개 원본 파일 중 변경한 5개 파일의 전체 diff와 hash는 `environment/source-snapshots/EvoRubrics-2155-rq2.patch`, `EvoRubrics-2155-rq2-patch-manifest.json`에 있다.

## 환경과 실행

저장소 루트에서 직접 실행한다. Linux 가상환경만 있으면 되며 Docker 컨테이너는 필요하지 않다.

전용 환경은 저장소의 `.venvs/evorubrics`이다. 기존 OnlineRubrics 환경과 분리했다. Torch 2.6.0+cu124, vLLM 0.8.5, Transformers 4.57.6, PEFT 0.17.1, FlashAttention 2.7.4.post1을 사용한다. 전체 버전은 `environment/evorubrics-runtime-lock.txt`, 재설치 명령은 `scripts/phase1/setup_evorubrics_runtime.sh`에 있다. 공개 archive의 vLLM 버전과 설치 환경 차이는 lock으로 명시한다.

diversity 모델 `sentence-transformers/all-MiniLM-L6-v2`는 snapshot `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`을 캐시했다. RQ2에서는 diversity 계산 실패를 오류로 올려 reward 항목이 조용히 사라지지 않게 한다.

```bash
export PYTHONPATH=src
EVO_PY=.venvs/evorubrics/bin/python
EVO_JUDGE=http://127.0.0.1:28001/v1

# 데이터와 설정만 준비한다. 새 run-id를 사용한다.
$EVO_PY -m dynamic_rubric.phase1.evorubrics_run prepare \
  --config configs/phase1/medicine_evorubrics.yaml \
  --run-id evo-medicine-smoke --smoke

EVO_RUN=outputs/medicine/evorubrics/seed-11/evo-medicine-smoke
$EVO_PY scripts/phase1/preflight_evorubrics_judge.py \
  --run-root "$EVO_RUN" --judge-base-url "$EVO_JUDGE"
$EVO_PY -m dynamic_rubric.phase1.evorubrics_run launch \
  --run-root "$EVO_RUN" --judge-base-url "$EVO_JUDGE" --gpu 1

# 0:0, 0:1, 1:1 전체 셀. 1문항으로 실행 경로를 검증한다.
$EVO_PY -m dynamic_rubric.phase1.evorubrics_probe \
  --run-root "$EVO_RUN" --pairs all --max-prompts 1 --generate
$EVO_PY -m dynamic_rubric.phase1.evorubrics_probe \
  --run-root "$EVO_RUN" --pairs all --max-prompts 1 \
  --score --judge-base-url "$EVO_JUDGE"
$EVO_PY -m dynamic_rubric.phase1.evorubrics_probe \
  --run-root "$EVO_RUN" --pairs all --max-prompts 1 --kl

# 실제 checkpoint/optimizer reload를 검증한다.
$EVO_PY -m dynamic_rubric.phase1.evorubrics_run launch \
  --run-root "$EVO_RUN" --judge-base-url "$EVO_JUDGE" --gpu 1 --resume-step 1
$EVO_PY -m dynamic_rubric.phase1.evorubrics_probe \
  --run-root "$EVO_RUN" --max-prompts 1 --validate-smoke
```

smoke는 2문항·1 step만 학습한다. M/N, reward 구성, LoRA, optimizer 설정은 본 실험과 같다. 통과한 synthetic fixture는 실제 smoke 증명으로 인정하지 않는다. 실제 judge preflight, θ/ψ advantage, paired checkpoint, complete probe와 KL, reload 증거가 모두 있어야 `smoke_complete.json`이 생성된다.

```bash
$EVO_PY -m dynamic_rubric.phase1.evorubrics_run prepare \
  --config configs/phase1/medicine_evorubrics.yaml --run-id evo-medicine-full
EVO_FULL=outputs/medicine/evorubrics/seed-11/evo-medicine-full
$EVO_PY -m dynamic_rubric.phase1.evorubrics_run launch \
  --run-root "$EVO_FULL" --judge-base-url "$EVO_JUDGE" --gpu 1 \
  --smoke-proof "$EVO_RUN/smoke_complete.json"
```

본 학습은 위 마지막 명령을 명시적으로 실행할 때 시작한다. 검증된 smoke와 코드·데이터·runtime·judge가 달라지면 다시 검증한다.

## 저장과 재개

- `launch_spec.json`, `config.resolved.json`, `upstream.config.json`: 출처와 확정 설정.
- `data/`: 공개 코드용 JSON 배열. 원본 prompt ID/hash/source index와 R0를 보존하며 reference answer는 정책 입력에서 제외한다.
- `audit/train_batch/`: 학습 응답, rubric, reward, criterion별 원본 judge 판정, parse 실패.
- `audit/advantages/`: adapter별 토큰 advantage 증거와 tensor artifact.
- `metrics/`: 실제 누적 prompt 노출, completion, role별 response token 수.
- `checkpoints/committed/`: θ·ψ 파일이 모두 저장·검증된 checkpoint 쌍. 미완성 쌍은 분석 대상으로 인정하지 않는다.
- `audit/fixed_probe/`: 생성 cache, judge receipt, 분석, KL.

step 0은 업데이트 전 θ₀/ψ₀, step k는 k회 coevolution iteration을 마친 쌍이다. 각 iteration은 θ·ψ 두 optimizer 경로를 갖는다. 본 학습은 매 step과 epoch 끝에서 parameter 쌍을 보존하고 최신 optimizer만 유지한다. 따라서 0~48의 모든 θ/ψ LoRA 쌍을 분석할 수 있지만 optimizer가 제거된 과거 step에서 학습을 재개할 수는 없다. Python/NumPy/Torch/CUDA RNG를 저장하며 vLLM 생성까지 bitwise 동일한 재개는 보장하지 않는다.

## 분석과 해석

저장된 전체 쌍의 step 집합 C로부터 모든 `τ ≤ t` 셀을 구성한다. OnlineRubrics의 수동 checkpoint 목록을 재사용하지 않는다. θₜ 응답 cache 하나를 모든 ψτ가 채점하므로 fresh/stale 차이는 같은 응답에서 평가한다. Judge 호출은 학습 M=4와 같이 고정된 연속 4개 답변씩 나눠 실행하고, 순서를 유지해 16개 점수를 합친다. 공개 grader는 각 답변의 처음 2,000자만 채점한다. 이 동작을 유지하며 원래 전체 답변과 토큰은 cache에 남긴다. Pool-B는 고정 train 문항에 대해 별도로 생성하며 gradient에 쓰지 않는다.

- ZAR@4: 16개에서 동일한 4개 subset을 반복 추출해 fresh/stale을 paired 비교한다. 16개 전체의 동점·분리도는 별도 통계다.
- criterion 판정: 누락/parse 실패를 정상적인 0점이나 동점으로 세지 않는다. 학습 로그에서 grade details가 없고 parse failure도 false라면 judge 호출 실패를 확인한다.
- KL: θₜ가 생성한 동일 response token을 θₜ와 θτ에 teacher-force하여 `log pθₜ - log pθτ`를 계산한다. prompt token을 제외하고 token 가중/문항 균형 평균을 저장한다. 학습 metrics의 KL placeholder는 실제 계산 결과가 아니며, 저장된 checkpoint 쌍을 대상으로 별도 KL 명령을 실행해야 한다. 이는 temperature 0.7 rollout 표본에서 측정한 log-ratio 평균으로 음수도 가능하다. 원래 temperature 1 정책분포의 정확한 KL이나 불편 추정량으로 해석하면 안 되며, GRPO loss의 reference KL과도 다른 통계다.
- held-out 300 / HealthBench: 정책 성능 평가용이다. 이 구현의 trainer 안에서 자동 실행하지 않으며 evaluator update timing이나 정답 label로 쓰지 않는다.
- OnlineRubrics와 judge가 다르므로 reward/ZAR 절댓값으로 승자를 정하지 않는다. method 내부 paired 변화와 공통 정책 평가를 중심으로 비교한다.

1문항 smoke는 실행 경로 검증이다. 연구 결과는 고정 100문항 probe와 저장된 전체 checkpoint를 대상으로 별도 산출해야 한다.

## 현재 검증 및 준비 경로

2026-09-10 실제 smoke 검증을 통과했다. 2문항·1-step θ/ψ 업데이트와 checkpoint/optimizer 재개, 1문항의 3개 probe 셀(각 답변 16개 × rubric 4개), 같은 응답 16개의 log-probability 비교를 완료했다. Phase1 테스트는 190 passed, 0 failed, 0 skipped다. 본 학습 96문항 batch와 전체 48-step의 완료를 보장하는 검증은 아니다.

- 실제 검증 run: `outputs/medicine/evorubrics/seed-11/phase1-evo-medicine-live-smoke-20260910-v8`
- 본 학습 준비 run: `outputs/medicine/evorubrics/seed-11/phase1-evo-medicine-full-20260910`
- 두 run의 코드·데이터·환경·judge 설정을 묶은 semantic identity: `cdf3be076698075e813f4cd1928b6f1a276395f131d3665a7e57bef1c2435c74`

최종 검증 상태는 [검증 기록](phase1_evorubrics_validation.json)과 검증 run의 `smoke_complete.json`에서 확인한다. `training_complete.json`은 학습 종료 시점의 기록이므로 그 안의 `probe_audit_status`는 이후 probe 결과를 반영하지 않는다. 본 학습은 준비만 했으며 `training_started.json`이 없는 상태다.

## 2026-09-23 재실행 준비

현재 RaR-Medicine split으로 다음 run을 GPU 작업 없이 준비했다.

- live smoke: `outputs/medicine/evorubrics/seed-11/phase1-evo-medicine-live-smoke-dense-20260923-v1`
- full run: `outputs/medicine/evorubrics/seed-11/phase1-evo-medicine-full-dense-20260923-seed11`
- 실행 런처: `scripts/phase1/run_medicine_evorubrics.sh`

이전에 준비한 3-step 간격 run은 보존하지만 사용하지 않는다. 현재 dense config는 `checkpoint_interval_steps=1`이며 `trainer.save_freq`와 `trainer.save_lora_freq`를 모두 1로 설정한다. 매 step θ(policy)와 ψ(rubric generator) LoRA parameter 쌍을 보존하고 이전 optimizer state만 정리한다.
2026-09-10 proof와 config hash가 다르므로 fail-closed provenance 계약상 새 1-step live smoke를 통과해야 full run을 시작할 수 있다.

런처는 중간 실패 후 재개할 수 있도록 smoke를 단계별로 나눈다.

```bash
scripts/phase1/run_medicine_evorubrics.sh status
scripts/phase1/run_medicine_evorubrics.sh smoke-train
scripts/phase1/run_medicine_evorubrics.sh smoke-probe
scripts/phase1/run_medicine_evorubrics.sh smoke-resume
scripts/phase1/run_medicine_evorubrics.sh smoke-validate
scripts/phase1/run_medicine_evorubrics.sh full
```

`smoke` action은 네 smoke 단계를 순서대로 실행한다. 완료 마커가 있는 단계는 건너뛴다.
`full` action은 현재 config와 semantic provenance가 일치하는 `smoke_complete.json` 없이는 시작하지 않는다.
trainer 기본값은 Trainer GPU 1, judge 기본값은 `http://127.0.0.1:28011/v1`이다.
GPU 1에 다른 학습 프로세스가 있으면 런처가 겹쳐 실행하지 않고 종료한다.

## 설계 자료

- [RQ2 Notion 설계](https://app.notion.com/p/3d03e4146a15809788f2ce30c1b3aabd)
- [공개 EvoRubrics README](https://anonymous.4open.science/r/EvoRubrics-2155/README.md)
- 로컬 논문: `docs/2606.23038v1.pdf`
