# Static Rubric Discriminability Horizon

This lane implements the RaR Medicine and RaR Science experiment independently
from the HealthBench pilot. Policy training is static-`R0` only. Dynamic criteria
are generated post hoc for checkpoint audits and never enter the veRL reward
path.

## Frozen experiment contract

- Policy: `Qwen/Qwen3-1.7B` at revision
  `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`.
- Grader: `Qwen/Qwen3-32B` at revision
  `9216db5781bf21249d130ec9da846c4624c16137`.
- Extractor: `gpt-5-mini`, medium reasoning, structured JSON.
- Active trajectory seed: `11`; seeds `29` and `47` are optional replications and
  are not queued in the current run. Pool A: 8; Pool B / GRPO group: 16.
- Epoch checkpoints: `0, .2, .4, .6, .8, 1, 1.5, 2, 2.5, 3` mapped to
  global steps `0, 3, 6, 9, 13, 16, 24, 32, 40, 48`.
- Training: 3 epochs, batch 96, 16 rollouts, LR `5e-6`, warmup `.1`, KL
  `.01`, temperature `1`, response length 3584.

The typed config and launcher preflight reject changes to these values. A
finetuned policy endpoint must expose `/dynamic-rubric/identity` with the exact
checkpoint hash; only the immutable base snapshot can use the launch-spec
fallback.

## Dynamic rubric construction

For each nonzero seed/checkpoint/prompt, exactly eight current Pool-A responses
are blindly paired with the eight fixed `pi_0` controls. The A/B order is
randomized and source identity is omitted. The paper-style extractor runs once
per pair. A separate structured-output call deduplicates candidates but may only
copy source wording and source IDs; deterministic code then:

1. resolves importance/type/weight from source candidates;
2. rejects ungrounded evidence, `R0` duplicates, non-atomic/non-binary text,
   negative-form pitfalls, and policy/checkpoint identity leaks;
3. converts weights to 10/7/3/9 units and caps the extension at eight criteria;
4. retains only current-checkpoint lineage; prior extensions never accumulate;
5. selects an equal-count stale or sham control with the closest weight
   histogram.

At checkpoint zero, no `E0` is constructed for the primary rubric. A separate
sham extraction can be built from the two `pi_0` control pools solely to supply
the first count-matched control. Later checkpoints normally use the immediately
preceding extension as the stale candidate pool.

## Execution sequence

The examples use Medicine. Replace the config/domain/source for Science and run
an independent trajectory.

```bash
CONFIG=configs/horizon_medicine.yaml
RUN_ID=rar-horizon-v1-medicine
PROMPTS=data/rar/medicine/public/final.jsonl

uv run python -m dynamic_rubric validate-config --config "$CONFIG"
uv run python -m dynamic_rubric verify-horizon-models --config "$CONFIG"
uv run python -m dynamic_rubric estimate-horizon --config "$CONFIG"
uv run python -m dynamic_rubric prepare-rar-data --config "$CONFIG" \
  --source /path/to/rar-medicine.jsonl
uv run python -m dynamic_rubric export-rar-verl --config "$CONFIG" --run-id "$RUN_ID"

for seed in 11; do
  DOMAIN=medicine TRAINING_SEED="$seed" scripts/run_horizon_static_grpo.sh
done
```

For the current sequential run, Medicine's 100-prompt audit is a hard gate
before Science training:

```bash
MEDICINE_TRAIN_PID=<pid> scripts/supervise_medicine_audit_then_science.sh
```

Generate the fixed and sham controls once at `pi_0`. Generate Pool B for every
seed/checkpoint and Pool A for every nonzero seed/checkpoint. Each command writes
one immutable shard; the live endpoint and `--checkpoint-hash` must refer to the
same weights.

```bash
uv run python -m dynamic_rubric generate-horizon-pools --config "$CONFIG" \
  --run-id "$RUN_ID" --prompts "$PROMPTS" --pool-family fixed_control \
  --count 8 --policy-step 0 --checkpoint-hash 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e \
  --base-url http://127.0.0.1:8000 --output artifacts/horizon/medicine/pools/fixed.jsonl
```

Before extraction, validate the combined inventory. This checks the exact
prompt × seed × checkpoint grid, not merely the groups that happen to exist.

```bash
uv run python -m dynamic_rubric validate-horizon-inventory --config "$CONFIG" \
  --prompts "$PROMPTS" --pools artifacts/horizon/medicine/pools/*.jsonl
```

Build one nonzero checkpoint rubric after setting `OPENAI_API_KEY`. The current
Pool A and fixed pool are the blinded response sources. `--control-rubrics`
points to the sham extension at the first nonzero checkpoint and to the
immediately preceding checkpoint afterward.

```bash
uv run python -m dynamic_rubric build-horizon-rubrics --config "$CONFIG" \
  --run-id "$RUN_ID-seed11-step3" --prompts "$PROMPTS" \
  --current-pool artifacts/horizon/medicine/pools/seed-11-step-3-pool-a.jsonl \
  --control-pool artifacts/horizon/medicine/pools/fixed.jsonl \
  --control-rubrics artifacts/horizon/medicine/rubrics/sham.jsonl \
  --checkpoint-id step3 --output artifacts/horizon/medicine/rubrics/seed-11-step-3.jsonl
```

Grade Pool B with the exact full-sequence log probabilities of the continuations
`" YES"` and `" NO"`. Checkpoint zero omits `--rubrics`, which enforces `R0`
only. Nonzero checkpoints require a lineage-matched rubric.

```bash
uv run python -m dynamic_rubric grade-horizon --config "$CONFIG" \
  --run-id "$RUN_ID-seed11-step0" --prompts "$PROMPTS" \
  --pool-b artifacts/horizon/medicine/pools/seed-11-step-0-pool-b.jsonl \
  --checkpoint 0 --base-url http://127.0.0.1:8001 \
  --output-dir artifacts/horizon/medicine/scores/seed-11/epoch-0

uv run python -m dynamic_rubric grade-horizon --config "$CONFIG" \
  --run-id "$RUN_ID-seed11-step3" --prompts "$PROMPTS" \
  --pool-b artifacts/horizon/medicine/pools/seed-11-step-3-pool-b.jsonl \
  --rubrics artifacts/horizon/medicine/rubrics/seed-11-step-3.jsonl \
  --checkpoint 0.2 --base-url http://127.0.0.1:8001 \
  --output-dir artifacts/horizon/medicine/scores/seed-11/epoch-0.2
```

Each grading shard contains full criterion grades, rational variant scores,
prompt metrics, and `score_seal.json`. Build bootstrap observations only from
all 30 sealed summaries, then analyze them.

```bash
uv run python -m dynamic_rubric build-horizon-observations --config "$CONFIG" \
  --prompts "$PROMPTS" --summaries \
  artifacts/horizon/medicine/scores/seed-*/epoch-*/prompt_summary.jsonl \
  --output artifacts/horizon/medicine/observations.jsonl

uv run python -m dynamic_rubric analyze-horizon --config "$CONFIG" \
  --observations artifacts/horizon/medicine/observations.jsonl \
  --output results/horizon/medicine/horizon_report.json
```

`analyze-horizon` refuses an unsealed or modified observation file. An observed
`t*` requires a nonzero earlier equivalence checkpoint plus simultaneous-band
deterioration, refresh gain, count-control gain, and next-checkpoint
persistence. A final-only onset is reported as right-censored.

