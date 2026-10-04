# 학습 설정과 실행 경로 참고

처음 설치·실행하는 순서는 [README](../README.md)를 따릅니다. 지원 진입점은 `configs/launch/{medicine,science}_{static_rubric,online_rubric,evorubric}.yaml`과 대응하는 여섯 `scripts/phase1/train_*.sh`입니다. 이전 `configs/phase1`은 기존 실험 재현용이며 신규 인수인계 설정은 `configs/launch`에서 수정합니다.

## 공통 설정 원칙

- 상대경로 기준은 YAML 파일 디렉터리가 아닌 **dynamic_rubric 루트**입니다.
- `${VAR}`, `$VAR`, `~`를 지원합니다. 미설정 환경변수는 그대로 넘기지 않고 오류를 냅니다.
- `launch.entry_python`: 실제 학습/제어 Python. SH 자체의 Python은 `.venv/bin/python` 또는 `LAUNCH_PYTHON`입니다.
- 학습 GPU는 `infrastructure.optimizer.gpus`, 서비스 GPU는 `infrastructure.services.*.instances[].gpus`, 제어 응답 생성 GPU는 `infrastructure.pi0_control.gpus`입니다.
- host/code_host는 기록용입니다. 실제 서비스 실행 머신, SSH 터널, 컨테이너 GPU 노출은 사용자가 선택합니다. 서버 이름은 제한하지 않습니다.
- `output.root`에 절대경로를 지정해 다른 디스크를 사용할 수 있습니다. Online/Evo의 하위 layout은 `{domain}/{method}/seed-{seed}/{run_id}`를 유지합니다.
- 기존 결과 덮어쓰기를 막기 위해 새로운 run ID를 사용합니다. Online ID는 `phase1-online-rubrics-{domain}-full` 접두사를 유지합니다. Evo는 `launch.smoke_run_id`도 고유하게 지정합니다.

## 파라미터 지원 범위

`paper`는 기존 학습값을 고정합니다. `custom`은 아래 검증된 범위를 허용합니다. 배포 경로/호스트/GPU/TP 변경은 paper 모드에서도 가능합니다.

| 설정 | Static | Online custom | Evo custom |
| --- | --- | --- | --- |
| 학습 GPU | 한 머신의 1개 이상 GPU | 한 머신의 1개 이상 GPU | 선택한 1개 GPU |
| `training.epochs`, `global_prompt_batch`, `expected_global_steps` | 지원 | 지원 | 지원 |
| `seed` | 지원 | 지원 | 지원 |
| `data.train_prompt_count` | parquet 사용 | 지원, 실제 JSONL 행 수와 일치 | 지원, 실제 JSONL 행 수와 일치 |
| `training.ppo_mini_batch_size` | 지원 | 지원, global batch의 약수 | 공통 인터페이스 아님 |
| `training.max_prompt_length`, `max_response_length` | 지원 | 지원 | upstream recipe 사용 |
| `training.rollout_tensor_parallel_size` | 지원 | 지원 | 단일 학습 GPU |
| `training.learning_rate`, `warmup_ratio`, `kl_coefficient` | 지원 | 지원 | 아래 별도 Evo 필드 사용 |
| `training.rollout_temperature`, `rollout_top_p` | 지원 | 지원 | `evorubrics.generation_temperature` 사용 |
| response 수 | `training.rollouts_per_prompt` | 16 고정 | `evorubrics.policy_responses_m` |
| rubric 생성 수 | 해당 없음 | elicitation pairs 8 고정 | `evorubrics.rubric_sets_n` |
| LoRA | 해당 없음 | 해당 없음 | `evorubrics.lora_rank`, `lora_alpha` |
| 두 학습률 | 해당 없음 | 해당 없음 | `evorubrics.policy_learning_rate`, `rubric_generator_learning_rate` |
| Evo KL | 해당 없음 | 해당 없음 | `evorubrics.kl_loss_coefficient` |
| Evo reward | 해당 없음 | 해당 없음 | `evorubrics.reward_weights`, 비음수이며 합 1 |

`expected_global_steps`는 종료 상한입니다. epochs/데이터 batch가 그보다 먼저 소진되도록 설정하면 원하는 step에 도달하지 못할 수 있습니다. batch/mini-batch/응답 수는 선택한 GPU 수에 맞춰야 하며, 실제 VRAM 적합성은 GPU 실행에서 별도로 확인해야 합니다.

Online의 16 responses/8 pairs는 현재 reward runtime·pi₀ cache 구성의 전제이므로 custom에서도 거부됩니다. Evo의 M/N을 바꾸면 `pool_b_count = M*N`, `rubric_generation_seeds` 길이 N도 함께 맞춥니다. Evo 다중 GPU 학습은 upstream optimizer의 분산 checkpoint 재개를 안전하게 보장하지 못하므로 비활성화했습니다. judge의 GPU/TP와는 관계없습니다.

## 학습 길이·seed 변경

Online/Evo는 분석용 anchor도 기록하므로 짧게 학습할 때 다음 목록에서 종료 step을 초과하는 항목을 제거합니다. 목록은 0으로 시작하는 오름차순 고유 정수여야 합니다.

```yaml
training:
  epochs: 2
  global_prompt_batch: 96
  expected_global_steps: 32
  audit_checkpoints: [0, 3, 6, 9, 13, 16, 24, 32]
  reuse_anchors: [0, 9, 16, 32]
analysis:
  reuse_anchors: [0, 9, 16, 32]
launch:
  tuning_mode: custom
```

seed를 변경하면 `data.fixed_train_probe.sample_seed`도 맞추고 `data.fixed_train_probe.manifest`, Online `ONLINE_CONTROL_CACHE_DIR`를 새 seed 전용 경로로 지정합니다. 고정 train probe는 100개이므로 학습 데이터는 최소 그만큼 필요합니다. 데이터를 줄이려면 실제 JSONL을 별도로 준비하고 `train_path`와 `train_prompt_count`를 함께 수정하세요. 자동으로 일부 행만 잘라 쓰지 않습니다.

## 서비스와 사전 생성

- `serve_training.py --config ... --service judge|extractor --instance N --check`: 한 인스턴스의 로컬 실행 계획을 출력합니다. `--check`를 제거할 때만 서버를 시작합니다.
- 서비스 인스턴스에는 `gpus`, `tensor_parallel_size`, `vllm_bin`, `port`, `bind_host`, `gpu_memory_utilization`, `max_model_len`, `max_num_seqs`, `max_num_batched_tokens`를 설정합니다.
- Static은 `python`, `proxy_port`도 필요합니다. 점수 프록시가 제공하는 모델·revision identity를 학습에서 검증합니다. raw vLLM 포트를 Static judge URL로 쓰지 마세요.
- Online은 현재 각 모델당 1~2개의 서비스 인스턴스를 지원합니다. `PHASE1_GPT_OSS_BASE_URLS`/`PHASE1_QWEN32B_BASE_URLS`의 URL 개수를 `instances` 개수와 맞춥니다. 이전 `PHASE1_QWEN32B_EXPECTED_COUNT`보다 launch YAML의 인스턴스 개수가 우선합니다.
- 각 서비스는 foreground로 실행되며 종료 시 자신이 생성한 프로세스만 정리합니다. 다른 서버를 자동 종료/재배치하지 않습니다.
- `precompute_pi0.py --config ... --check`: 같은 YAML의 policy/data/seed와 `pi0_control` 자원값을 읽습니다. 실제 생성 시 지정 GPU가 사용 중이면 중단하며, 모델·데이터·seed에 묶인 immutable manifest를 저장합니다.

## 검사·로그·재개

`--check`는 설정과 로컬 경로를 검사하고 종료합니다. `--check-services`는 실제 서버의 모델 ID까지 읽기 전용으로 검사하고 종료합니다. 둘 다 모델을 GPU에 적재하거나 학습 요청을 보내지 않습니다. 서비스 helper의 `--check`는 원격 연결을 확인하지 않으며 학습 helper의 `--check-services`와 구별됩니다.

학습 실행 시 공통 콘솔 로그는 `run_root.parent/_launch_logs/`에 남습니다. Online이 새 run 디렉터리를 요구하므로 launch 로그가 run_root를 미리 생성하지 않도록 분리했습니다.

Static/Online은 매 step checkpoint를 저장하고 최신 완료 step에만 optimizer 등 전체 재개 상태를 유지합니다. 같은 설정/run ID에 `--resume`을 붙여 최신 checkpoint에서 재개합니다. Evo도 매 step LoRA를 저장하지만 통합 SH의 `--resume`은 아직 지원하지 않습니다. Evo의 내부 사전 검증에서 checkpoint 복구를 확인하는 것과 사용자가 중단된 본 학습을 자동 재개하는 것은 별개입니다.

CPU 회귀 검증은 인자 전달과 파일/서비스 프로토콜을 확인합니다. 새 머신에서 실제 GPU 학습이 성공했다는 증거를 대신하지 않습니다.
