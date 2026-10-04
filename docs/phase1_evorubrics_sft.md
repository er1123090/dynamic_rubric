# EvoRubrics용 GPT-OSS 교사 답변과 SFT

사용자 지정 교사는 `openai/gpt-oss-120b`이다. 논문의 DeepSeek-R1 교사와 다르며, SFT 세부 학습률·batch·epoch는 논문에서 공개하지 않은 우리 실험의 선택이다. 이 작업의 범위는 교사 데이터와 병합된 Qwen SFT 모델 확보까지이며 RL은 실행하지 않는다.

## 데이터와 모델

- 원본: `data/rar/medicine/eval300/train.jsonl` 1,500문항. prompt ID와 hash를 보존한다.
- 제외: `data/rar/medicine/eval300/final.jsonl` 300문항. 중복 검사만 수행하며 교사 요청이나 SFT target에 쓰지 않는다.
- 고정 train probe 100문항은 기존 설계대로 train에 포함된다. 일반화 평가 데이터가 아니다.
- 교사: GPT-OSS-120B, revision `b5c939de8f754692c1647ca79fbf85e8c1e70f8a`, MXFP4, medium reasoning, temperature 0.7, top-p 0.95.
- 학생: Qwen3-4B-Instruct-2507, revision `cdbee75f17c01a7cc42f958dc650907174af0554`.
- 교사 입력에는 원본 질문과 간결한 최종 답변 생성 지시만 전달한다. 기존 reference answer와 R0 rubric은 전달하지 않는다.
- 교사의 final/content만 SFT에 사용한다. reasoning은 원시 응답에 별도 보관하며 학습 target에 합치지 않는다.
- `finish_reason=stop`, 빈 답변/제어 토큰 없음, Qwen chat-template 기준 답변 target 1,024토큰 이내를 요구한다. 길이 초과나 생성 중단은 잘라서 사용하지 않고 재생성한다. 이는 형식·완료 검증이며 임상적 정확성을 보증하는 판정은 아니다.

## 실행

저장소 루트에서 Linux 가상환경으로 직접 실행한다. Docker는 필요하지 않다. GPU 1에 교사 서버를 실행한 뒤, 교사 생성이 완료되면 서버를 종료하고 같은 GPU에서 SFT를 수행한다.

실행 루트: `outputs/medicine/evorubrics_sft/seed-11/gptoss120b-teacher-sft-20260910`.

교사 환경은 사용자가 준비한 judge 가상환경이다. 소스 빌드가 필요한 경우 현재 Python 환경의 include 경로를 사용한다. 서버는 `127.0.0.1:28011`에만 바인딩하며 GPT-OSS reasoning parser를 사용한다. 실제 서버 로그와 preflight 응답은 실행 루트에 보존한다.

```bash
export PYTHONPATH=src
EVO_PY=.venvs/evorubrics/bin/python
EVO_SFT_RUN=outputs/medicine/evorubrics_sft/seed-11/gptoss120b-teacher-sft-20260910
EVO_BASE=models/Qwen3-4B-Instruct-2507

$EVO_PY scripts/phase1/generate_evorubrics_sft_teacher.py \
  --train data/rar/medicine/eval300/train.jsonl \
  --heldout data/rar/medicine/eval300/final.jsonl \
  --run-root "$EVO_SFT_RUN" --base-url http://127.0.0.1:28011/v1 \
  --reasoning-effort medium --concurrency 16

# 길이 초과 등으로 남은 문항만 재생성하고 원본/재생성 출처를 구분해 합친다.
$EVO_PY scripts/phase1/repair_evorubrics_sft_teacher.py \
  --train data/rar/medicine/eval300/train.jsonl \
  --heldout data/rar/medicine/eval300/final.jsonl \
  --run-root "$EVO_SFT_RUN" --base-url http://127.0.0.1:28011/v1 --concurrency 16

# 교사 서버 종료 및 GPU 반환 확인 후 실행한다.
$EVO_PY scripts/phase1/train_evorubrics_sft.py \
  --data "$EVO_SFT_RUN/teacher_final/train.jsonl" --run-root "$EVO_SFT_RUN/sft" \
  --base-model "$EVO_BASE" --epochs 1 --batch-size 32 \
  --learning-rate 2e-5 --gpu 1 --max-length 4608
```

최초 생성에서는 1,466문항이 통과했고 34문항이 길이 기준 등을 충족하지 못했다. 남은 34문항은 250단어 간결성 지시로 모두 재생성에 성공했다. 복구 도구는 필요 시 180/120단어 지시를 추가 적용하지만 이번 실행에서는 사용되지 않았다. 원본 답변과 재생성 답변의 profile을 최종 manifest에 구분하며, 답변을 잘라서 학습 데이터로 만들지 않는다.

최종 데이터는 원본 train 순서의 정확히 1,500문항이며 held-out 중복은 0개다. 실제 Qwen tokenizer로 전수 검사한 target은 총 175,940토큰, 중앙값 43토큰, 최대 1,024토큰이다. 최대 전체 sequence는 1,112토큰이며 truncation은 없다. 데이터 SHA-256은 `9a8448eafd5b36055e17ad47d70085adbecc9d07c66a64c8b4a02759b04c0419`이고, 각 응답의 원시 receipt hash와 profile 연결도 전수 검증했다.

교사 생성은 검증된 개별 receipt를 재사용하여 재개한다. 전체 1,500문항이 확보돼야 최종 학습 JSONL과 완료 manifest를 만든다. SFT는 새 run 디렉터리에 실행하며 실패한 실행을 덮어쓰지 않는다.

## SFT와 검증

LoRA rank 32, alpha 64, all-linear, dropout 0으로 1 epoch 학습한다. microbatch 1과 유효 batch 32를 사용하며 마지막 28문항까지 총 47회 optimizer update를 수행한다. prompt/history는 loss에서 제외하고 최종 assistant target만 학습한다. 각 유효 batch의 실제 target token 수로 gradient accumulation을 정규화한다.

adapter와 optimizer를 저장하고, SFT adapter를 원본 모델에 병합하여 별도 `merged_model`을 만든다. 실제 adapter 변경, 병합 전후 forward 수치 비교, 저장 모델의 재로딩과 생성 검사까지 통과해야 `sft/training_complete.json`이 생성된다. 이 검증은 모델 저장·학습 경로의 동작을 확인하며 held-out 성능 개선을 의미하지 않는다.

확보한 SFT 모델은 향후 EvoRubrics의 공통 backbone과 RL reference로 연결할 후보이다. RL 연결 시 모델 출처 검증을 추가하고 새 smoke를 수행해야 한다. 기존 OnlineRubrics에 같은 SFT를 적용하도록 변경하지 않는다.

## 이번 실행의 병합 복구

SFT 자체는 47/47 optimizer step, 1,500 exposures까지 완료했고 adapter와 optimizer가 저장됐다. 이후 BF16 adapter/merged logits 검사에서 최대 절대 차이 3.1875로 실패했다. 원본 `sft/training_failed.json`과 전체 로그를 보존한다. BF16 병합은 LoRA delta를 base weight의 BF16 표현으로 반올림하므로, 별도 adapter 연산과 수치적으로 같다고 보장할 수 없다.

재학습하거나 BF16 허용 오차를 완화하지 않고, 저장된 adapter와 원본 BF16 base 값을 FP32로 승격해 병합한다. 사전 GPU 진단에서 FP32 병합 전후 최대 절대 차이는 0.00023031234741210938, 평균 차이는 0.000006285958988883067이었고, rtol=1e-4/atol=1e-3 검사를 통과했으며 해당 probe의 top-1 token agreement는 100%였다. 이는 수치 검증이며 의료 답변 성능 평가는 아니다.

복구 export는 별도 실행 디렉터리에 저장한다. 최종 모델의 저장 dtype은 float32이며, 이후 BF16 RL/inference로 불러오면 추가 반올림이 발생한다. FP32 병합 모델을 BF16 live adapter와 수치적으로 동일하다고 해석해서는 안 된다.

복구 실행 명령:

```bash
CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src \
.venvs/evorubrics/bin/python scripts/phase1/export_evorubrics_sft_checkpoint.py \
  --source-run "$EVO_SFT_RUN/sft" \
  --training-log "$EVO_SFT_RUN/logs/sft-train.log" \
  --run-root "$EVO_SFT_RUN/sft_export_fp32" --gpu 1
```

이 명령은 원본 학습 로그·optimizer·adapter·teacher manifest·코드 hash를 검증하고, 추가 optimizer step 없이 저장된 학습 결과를 export한다. 실행한 디렉터리는 덮어쓰지 않으므로 다시 실행할 때는 새로운 `--run-root`를 지정한다.

## 최종 결과 — 2026-09-10

GPT-OSS-120B 교사 1,500문항 생성과 Qwen3-4B-Instruct-2507의 1 epoch SFT(47 optimizer steps, 1,500 exposures)를 완료했다. 추가 학습 없이 저장된 adapter로 FP32 병합 모델을 확보했다.

- 최종 모델: `outputs/medicine/evorubrics_sft/seed-11/gptoss120b-teacher-sft-20260910/sft_export_fp32/merged_model`
- 교사 데이터: `outputs/medicine/evorubrics_sft/seed-11/gptoss120b-teacher-sft-20260910/teacher_final/train.jsonl`
- 학습된 LoRA adapter: `outputs/medicine/evorubrics_sft/seed-11/gptoss120b-teacher-sft-20260910/sft/checkpoint/adapter`
- 최종 완료 요약: `outputs/medicine/evorubrics_sft/seed-11/gptoss120b-teacher-sft-20260910/completion.json`
- 병합 및 재로딩 증거: `outputs/medicine/evorubrics_sft/seed-11/gptoss120b-teacher-sft-20260910/sft_export_fp32/training_complete.json`
- 저장 dtype: float32. 전체 모델 파일 16,105,833,132 bytes.

252개 LoRA 레이어에서 FP32 병합 가중치가 `W + scaled(B@A)` 계산값과 hash 기준으로 일치했다. 세 고정 probe의 마지막 위치 logits는 rtol=1e-4, atol=1e-3를 모두 통과했고 최대 절대 차이는 0.000057220458984375였다. 저장 모델을 새로 불러온 후 전체 parameter와 probe logits가 정확히 일치했으며, 원본 질문 prefix에서 8토큰 생성까지 통과했다. 별도 사전 진단은 첫 probe의 최대 256개 모든 위치를 검사했다.

최종 파일 hash와 원본 학습 증거 hash를 다시 확인했다. Phase1 전체 테스트는 212개 통과했다. trainer GPU 1은 작업 종료 후 0 MiB를 확인했다. 의료 답변 성능 평가는 수행하지 않았고 RL은 시작하지 않았다. 원본 BF16 export 실패 기록은 감사용으로 그대로 남기며, 사용해야 하는 최종 모델은 위 `sft_export_fp32/merged_model`이다.
