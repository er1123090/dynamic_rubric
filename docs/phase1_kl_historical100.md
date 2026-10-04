# Fixed-train historical-matrix KL audit — 2026-09-09

The user paused training to prioritize policy-distance analysis on Trainer GPU1.
No training is resumed automatically by this workflow.

## Training pause

`scripts/phase1/pause_step46_for_kl.py` saved the actual live actor, optimizer,
RNG and scheduler after completed optimizer update46. It restored checkpoint45's
dataloader and verified the exact recorded batch46 and subsequent batches47/48
against the saved order oracle before publishing checkpoint46. The original
logical commit was backed up, then bound to the new parameter/full-state hashes.
The complete resume chain passed verification. Partial step47 artifacts were
preserved by the existing replay archive mechanism, not treated as an update.

Evidence: training run `logs/pause46-for-kl-20260909/{live-save,sealed,shutdown-complete}.json`.
The Ray driver subsequently exited on its keepalive watchdog after the controller
was held. This is not completion of step47 or48. Checkpoint46 was saved and sealed;
the training processes were verified absent and GPU1 was free before KL launch.

When the user requests training again, the existing
`bash scripts/phase1/resume_medicine_shared_inference_a.sh` restores the latest sealed
checkpoint46 and starts update47, provided the two recorded Inference_A services are
healthy. Keep checkpoint46 with optimizer/data/RNG; do not delete it for KL space.

## Parallel resume authorized later on 2026-09-09

The user subsequently requested concurrent KL and resumed training. Training
restores sealed46 and resumes update47 through48 using the same scientific
configuration. Inference_A0/1 retain GPT-OSS and Qwen judge;
Trainer1 shares training with a small KL server. No other GPU is used.
An initial startup using training rollout fraction0.55 reached142481MiB during
generation, so it was stopped before update47 and before any online-step47
artifacts. The resumed serving allocation uses `ROLLOUT_GPU_MEMORY=0.43` plus
explicit `ROLLOUT_KV_CACHE_MEMORY_BYTES=34359738368` (32GiB). The explicit cache
avoids cross-process auto-profiling and leaves room during later rollouts too.
Training batch96, n16, PPO mini/microbatch, token lengths, seeds, optimizer,
learning rate and reward construction are not changed.

KL serving-only overrides: `--share-gpu --gpu-memory-utilization 0.08
--max-num-seqs 4 --max-num-batched-tokens 1024 --kv-cache-memory-bytes 1610612736`.
The explicit1.5GiB KV allocation avoids the installed vLLM0.19.1 profiler's
cross-process free-memory assertion; it overrides automatic fraction-derived
KV sizing, so the fraction is not a hard total-process limit.
The scientific matrix, HTTP
batch64, responses, hashes, BF16 precision, max context8192 and estimator stay
unchanged. A distinct runtime receipt records these serving overrides rather
than altering the immutable analysis plan. Kernel scheduling can introduce
floating-point differences; this is not a promise of bitwise-identical scores.

Before loading each KL model, wait for the requested fraction plus2GiB of GPU
headroom; never stop training to reclaim memory. This is a startup guard, not a
hard isolation guarantee against later training peaks. Check both jobs' logs
before declaring coexistence verified. The previous dedicated KL runner was
interrupted during download; its batch caches and partial downloads were kept.
Latest launch records: training run `logs/parallel-kl-resume46-20260909/`.

Checkpoint45 was uploaded publicly, remotely hash-verified and removed locally
on 2026-09-09. Its archive receipt supplies pinned revision/hash identity on KL
restart; checkpoint46 model/optimizer/data were hash-verified unchanged.

## Analysis contract

- Source matrix: `configs/phase1/medicine_online_rubrics_historical100.json`.
- Same fixed100 RaR-Medicine training prompts, existing16 current-policy Pool-B
  responses per prompt. No new responses, held-out prompts, gradients or judges.
- 94 available model/pool cells through45, including21 diagonals;73 non-diagonal
  policy-distance comparisons. Six policy48 cells remain explicitly pending.
- For a cell `(tau,t)`, score both policy checkpoints on **the same Pool-B_t**.
  Aggregate current-minus-stale token log probabilities (K1) and clipped K3 at
  response, prompt and token-weighted levels. Initial policy is a reference, not GT.
- These are raw-softmax teacher-forced log-ratio proxies on retokenized,
  nucleus-sampled text. They are **not exact or unbiased distribution KL**.
  Prompt prefix is excluded; retokenization does not reconstruct omitted EOS.
  K1 may be negative. K3 log-ratio clipping at20 and clipped-token counts are saved.

## Execution, verification and eviction

`scripts/phase1/run_historical_kl.py` runs checkpoints in descending order so
adjacent recent-policy results become available first. One model scores all its
assigned pool columns before unloading. Batch logprobs and cell seals support
restart without repeating completed requests. Token identities and original
checkpoint hashes are checked across paired scores.

Historical inference exports are downloaded using `hf download` at the archive
receipt's exact commit, with the original per-file size and SHA256 checked before
model loading. Only the ~8GB HF inference export is downloaded, not original
FSDP shards or optimizers. Checkpoint45's original local export was used before
it was archived; restarts now recover its identity from the archive receipt.

After all assigned score files pass validation and the model server stops, only
the corresponding owned `temporary_models/global_step_N` tree is deleted.
Deletion receipts retain repo/commit identifiers for recovery. Canonical training
checkpoints, responses, rubrics, grades and KL outputs are never cleanup targets.

Output directory, under the existing audit artifact root:
`kl_historical100_20260909/` containing `plan.json`, `status.json`,
`batch-progress.json`, `scores/`, `batch_cache/`, `seals/`, `pairs/`, `downloads/`
and `runner.log`. A completed available matrix is not reported as100/100.

Restart with the same `argv` in `launch.json`, only after confirming its previous
runner is absent. An exclusive lock prevents duplicate runners.

## Model prefetch while training continues

The latest live KL invocation is recorded in `active-launch.json` (the original
`launch.json` is retained as history). `scripts/phase1/prefetch_historical_kl.py`
downloads pending model exports on CPU/network only with four model workers.
It uses the existing immutable KL plan, pinned archive revisions, and the same
owned `temporary_models/` directories; it excludes the current KL model whose
download remains owned by the scoring runner. No optimizer or FSDP archive is
downloaded. Free-space preflight includes all planned files plus100GiB reserve.

Downloader and scorer share persistent per-model locks outside the disposable
model trees. Both verify bytes/SHA256 before marking ready. Cleanup takes the
same lock; prefetch rechecks score seals while locked, so already consumed and
deleted models are not recreated. Completed files are used locally without
another HF request. Downloads retry up to three times and retain partial bytes.
Prefetch status and launch records are under `prefetch/`. A ready model may later
be consumed and deleted by KL; `verified_ready` means prefetch succeeded, not a
promise of permanent local retention.

## Full-context resume (2026-09-09)

An input in Pool-B32 needs10,133 context tokens including the discarded one-token
completion. The old8,192-token server limit caused HTTP400, not an OOM. A scan of
all33,600 existing responses is saved in `context-length-audit.json`, bound to
the immutable plan and tokenizer files. Restart uses `--max-model-len 10240`.
No response is truncated, skipped, or regenerated. Existing score seals and
token-bound batch caches remain valid; the scientific plan is unchanged.

The shared-GPU budget stays at1.5GiB explicit KV,4 sequences,1,024 batched tokens
and eager execution. Qwen3-4B BF16 KV needs147,456 bytes/token;10,240 tokens plus
one16-token block fit the existing KV allocation. The runner validates model
position/KV capacity before starting each server and checks the extra completion
token before sending each request. Training and CPU prefetch remain running.
