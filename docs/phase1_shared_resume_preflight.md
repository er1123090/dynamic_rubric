# Shared-GPU resume preflight — 2026-09-09

User scope: load two TP=2 services together on Inference_A GPU0,1 and the latest
saved policy on Trainer GPU1; check whether training can resume. The initial
preflight did not execute an optimizer update or modify the training chain.
The user subsequently authorized training; see the resume section below.

## Verified deployment

| Service | Physical GPUs | Trainer-accessible URL | Settings |
| --- | --- | --- | --- |
| GPT-oss-120B | Inference_A 0,1, TP2 | `http://127.0.0.1:28011` → Inference_A 8001 | MXFP4/Marlin, vLLM0.20.1, memory fraction 0.48 |
| Qwen3-32B | Inference_A 0,1, TP2 | `http://127.0.0.1:28002` → Inference_A 8004 | BF16, vLLM0.19.1, memory fraction 0.46 |
| Checkpoint-45 policy | Trainer 1, TP1 | `http://127.0.0.1:28010` | BF16, vLLM0.19.1, memory fraction 0.20 |

Both Inference_A services share the **same** two GPUs; they are not one GPU per
model. They retain the pinned model revisions and 32,768-token context limit.
For memory headroom, these preflight servers use eager execution, 16 maximum
sequences and 2,048 batched tokens. These are serving changes, not reward or
sampling changes; do not claim bitwise equivalence or original throughput.
Measured combined use after the concurrent smoke was 77,937 MiB per Inference_A
GPU (81,920 MiB total). Qwen's KV cache supports 45,280 tokens in aggregate,
so long simultaneous requests may queue/preempt rather than run at the old
standalone concurrency. GPT-oss reports 185,705 KV-cache tokens.

Canonical launch sources are `scripts/phase1/serve_shared_inference_a_preflight.sh`
and `scripts/phase1/serve_checkpoint45_preflight.sh`, both on Trainer. The remote
source was streamed through SSH into a user-selected runtime environment.
Exact launch commands, identities and logs are in the journal below. Inference_A
GPU2,3, Trainer GPU0 and Inference_B were not repurposed.

## Resume state and boundary

The physical `global_step_45` checkpoint has optimizer steps **[45]** and
scheduler `last_epoch=45` (verified by actually reading the checkpoint).
There is no `global_step_47` or successful live-step47-save receipt. Logical
commits 46 and 47 exist without saved models. The read-only commit-chain and
checkpoint hash checks passed. Thus **updates 46, 47 and 48 remain on resume**.

The normal `prepare_full_resume` / `resolve_committed_resume` path preserves
the unsaved logical tail in `replay_archive/resume-from-000045/attempt-*` and
resumes from the sealed checkpoint. Do not relabel checkpoint45 as checkpoint47
or invoke `finalize_step47_pause.py`: its required live step47 snapshot is absent.
This preflight did not call the mutating resume function or archive the tail.

Before actual training, stop only this test policy server on Trainer GPU1, then
let the existing GRPO trainer restore checkpoint45 and manage its in-process
rollout engine. The standalone policy API is **not** the training optimizer and
must not remain alongside the trainer by accident. Endpoint environment:

```sh
CUDA_VISIBLE_DEVICES=1
PHASE1_GPT_OSS_BASE_URLS=http://127.0.0.1:28011
PHASE1_QWEN32B_BASE_URLS=http://127.0.0.1:28002
PHASE1_QWEN32B_EXPECTED_COUNT=1
```

## User-authorized training resume

The user subsequently authorized actual training and requested efficient GPU use.
`scripts/phase1/resume_medicine_shared_inference_a.sh` resumes this same run through
the existing `resume-online` CLI under an exclusive run lock. It preserves the
scientific configuration and explicitly retains log-probability prefetch,
128 rollout sequences, 16,384 batched rollout tokens and 0.55 rollout memory
fraction. Extractor/grader client concurrency remains the existing 32/64;
the shared Inference_A engines enforce their own memory-safe serving limits.

The standalone Trainer policy server was stopped before starting GRPO. The
training journal is `logs/shared-resume-training-20260909/` under the audit
artifact root below; `launch.json` and `training.log` identify the live run.
Updates 46–48 are the target. The normal three-step checkpoint schedule saves
checkpoint48 with optimizer/RNG/dataloader state; the retention worker only
removes older resume-only state after the replacement checkpoint is sealed.
Saved parameters and analysis responses/rubrics are unaffected.

## Preflight evidence and limits

Journal, relative to the repository:
`outputs/medicine/online_rubrics/seed-11/phase1-fixed-probe-regular-through45-20260908/logs/shared-resume-preflight-20260909/`.

- `concurrent-smoke.json`: 12 real simultaneous structured requests across the
  three services passed; longest latency 8.33 seconds. This is not a full-batch
  load test or a GRPO optimizer-step test.
- `topology-preflight.json`: existing training endpoint/data preflight passed.
  The configured HealthBench prompt file is absent; it is a separate downstream
  evaluation input, not a prerequisite to resume training.
- `checkpoint-state.json`: actual optimizer and scheduler values.
- `resume-chain.json`: read-only chain and sealed checkpoint hash verification.
- Old completed grading clients/judges were stopped with exact process identity
  checks; all scientific scoring artifacts remain intact.

Repeat the isolated smoke with a **new output path** (receipts are immutable):

```sh
PYTHONPATH=src:. .venv/bin/python \
  scripts/phase1/smoke_shared_resume_servers.py --output /tmp/new-resume-smoke.json
```
