# Historical 100-cell OnlineRubrics Medicine audit

The user selected the earlier matrix on 2026-09-08. Its source of truth is
`configs/phase1/medicine_online_rubrics_historical100.json`, not a full triangle.

- 22 policy checkpoints: 0,3,6,9,12,13,15,16,18,21,24,27,30,32,33,34,36,39,40,42,45,48.
- 22 fresh cells + 21 immediately preceding saved-evaluator cells + 57 additional
  cells from anchors 0,9,16,32,48 = 100 unique evaluator–policy cells.
- Same fixed RaR-Medicine train 100 prompts and same 16 Pool-B responses within
  each policy column. Pool A remains independent. R0 is not ground truth.
- Checkpoint 48 does not exist. Its six cells remain pending; the other 94 cells
  can be completed with existing/pinned archived models. This does not authorize
  a training restart. Never substitute checkpoint 45 or a nonexistent 47 for 48.

## Runtime and cache preservation

The artifact root remains
`outputs/medicine/online_rubrics/seed-11/phase1-fixed-probe-regular-through45-20260908`.
Its historical name is retained to preserve absolute artifact/cache references;
`config.historical100.json` and the explicit cell plan define the new scope.
The directory `scores-adjacent/` is also retained for cache reuse. Only cells
listed in the new plan count toward completion; old off-plan cells are preserved.

Four tmux sessions handle the expanded run:

| Session | Responsibility |
|---|---|
| `phase1_matrix_restore` | Restore 13,16,32,34,40 from receipt-pinned public HF commits; verify original hashes; CPU-export for inference |
| `phase1_matrix100_extra_pools` | Wait for the exact existing regular-pool process to finish successfully, then generate missing checkpoint Pool A/B on Trainer GPU1 |
| `phase1_matrix100_fresh` | Generate fresh rubrics for ready Pool A checkpoints, skipping inputs still being prepared |
| `phase1_matrix100_scores` | Score exactly the explicit 100-cell plan on Inference_B GPU0,1, reusing all matching immutable receipts |

Trainer GPU1 continues sharing policy inference and gpt-oss-120b extraction.
Judge endpoint, model revisions, temperatures, seeds, response counts and
concurrency remain unchanged. Inference_A and Inference_B GPU2,3 are not used.

## Restart

Use the exact argv saved in `logs/matrix100-switch.json`, with the same resolved
config and artifact root. Stop only the owned client process before restarting
that client; keep shared vLLM servers running. The scorer has an exclusive lock.
Completed prompt/response receipts are validated and reused on restart.

For the extra-pool launcher, the `--wait-for-pid/--wait-for-starttime` pair is only
for the initial handoff from the old regular-pool process. After that handoff has
completed, omit both flags on a restart of the extra-pool launcher; first verify
no other policy generator owns the GPU/ports. A failed predecessor is not treated
as successful completion.

Status files are `logs/restore_matrix_status.json`, `logs/pool_a_status.json`,
`logs/fresh_queue_status.json`, and `scores-adjacent/status.json`. A 94/100 result
must be reported as partial, with six checkpoint-48 cells missing, not complete.
The scorer's configured seven-day idle deadline is a timeout, not a mechanism
for launching training or creating checkpoint 48.

HF restoration uses the HF CLI skill's revision-pinned download workflow.
Only original parameter and model metadata files are restored; optimizer state
is not restored for these historical checkpoints. Checkpoint 45's full resume
state is untouched.

## Deferred Trainer judge handoff (2026-09-09)

The user authorized repurposing Trainer GPU1 after all analysis-response and fresh
rubric generation finishes. `phase1_trainer_judge_transition` runs
`scripts/phase1/transition_trainer_to_judge.py` and persists its state under
`logs/trainer-judge-transition/status.json`. At installation it is waiting, not
already serving a second judge.

The gate requires all 21 available policy pools and 20 generated fresh-rubric
inventories to validate against this run. It then stops only the captured
extractor PID/starttime and its owned children, verifies GPU1 is free, and starts
the cached Qwen3-32B revision on localhost:28012: BF16, TP1, vLLM0.19.1,
32K context. GPU0, Inference_B's server, and checkpoints are untouched. The policy pool
launcher already shuts down its own Qwen3-4B server after completion.

A new-server-specific smoke directory must record 32 actual fresh/stale grades
before handoff. It is not included in the scientific score inventory and does
not claim bitwise equivalence between TP1 and TP2. If model loading or the smoke
fails, the existing Inference_B scorer has not been stopped.

After successful smoke, the 100-cell matrix is partitioned by **whole policy
column**, not individual rubric or response. Every already-started Inference_B column
stays Inference_B. Thus fresh/stale comparisons on identical responses always use the
same backend. Remaining columns are balanced by runnable cell count; the six
unavailable checkpoint-48 cells remain pending on Inference_B and contribute no load.

`logs/trainer-judge-transition/partition.json` identifies both cell plans, output
roots, exact restart commands, and validation receipt. Post-handoff coverage
must join `scores-adjacent/` (Inference_B) with `scores-trainer/` (Trainer), deduplicate by
(evaluator_step, policy_step), and verify their union against the original plan.
Neither backend's local completed count represents whole-experiment progress.

Restart a failed shard with its recorded command and original backend. Never
move partially graded Trainer policy columns to Inference_B without a separate documented
decision about backend consistency. No response, rubric, or score files are
deleted by the handoff.

## Throughput-based rebalancing (2026-09-09)

The user authorized moving additional untouched columns to Trainer so both judge
workers finish at approximately the same time. Measure recent saved-grade rates
separately for each backend; minimize the maximum estimated remaining duration,
not the difference in cell counts. Pin columns with any completed or partial
grade receipts, and also pin each worker's currently active policy column.
Keep all six unavailable checkpoint-48 cells visible and exclude them from ETA.

The rebalance journal is `logs/judge-rate-rebalance-20260909/`. Its preflight,
before/after partition, exact client argv, source hashes, and verification records
make the transition auditable. Original plan files and all grading artifacts
remain in place. Only scorer clients are restarted; judge servers, decoding
options, model revisions, seeds, Pool B, rubric inputs, and output roots remain
unchanged. Interrupted in-flight requests may be retried with the same identity;
saved immutable receipts are reused.

After a successful switch, use the current commands in
`logs/trainer-judge-transition/partition.json`, not the initial handoff commands.
The prior partition record is retained in the rebalance journal. Continue to
join both score roots and validate exact coverage against the original 100-cell
plan; a worker's local count is not the aggregate completion count.

## Inference_B-to-Inference_A physical judge migration (2026-09-09)

### Subsequent idle-Inference_A prompt sharing (2026-09-09)

The user later authorized assigning some remaining Trainer work to idle Inference_A.
The untouched policy-42 column is now split by **whole prompt across all six
evaluators**, an explicit exception to the earlier whole-column partition.
`logs/inference_a-prompt-share-20260909/assignment.json` fixes 40 prompts on Inference_A
GPU0,1 (3,840 response grades) and 60 on Trainer GPU1 (5,760 response grades).
Every fresh/stale comparison for a prompt stays on its assigned physical judge.
Both use the existing pinned model/version, grammar, seeds, responses and rubrics;
no bitwise equivalence between physical backends is claimed.

`scripts/phase1/share_probe_column.py` starts Inference_A's subset while a restricted
Trainer client finishes its already-started policy-40 column. Trainer then processes
its policy-42 complement. Both write disjoint response caches under the original
`scores-trainer/policy-000042/` tree; physical transport provenance remains in each
grade and in the helper's provider caches. No partial 40/60-prompt cell manifests
are published. Once both subsets finish, a cache-only pass of the original Trainer
plan assembles standard 100-prompt/1,600-response receipts; any attempted new
inference during finalization is a hard failure.

The canonical partition's `trainer_command` now restarts this coordinator. Its
`trainer_base_command` is the old full-column client and must not be launched
concurrently with the helpers. Keep `assignment.json` unchanged on resume.
Read `inference_a-status.json`, `trainer-status.json`, `coordinator.log`, and
`trainer-existing.log` in the sharing journal: the older `scores-adjacent/status.json`
continues to describe Inference_A's original shard, not this additional helper.
The original 100-cell union and six unavailable policy-48 cells are unchanged.

The user subsequently requested freeing Inference_B and running its judge on Inference_A
GPU0,1. `logs/inference_b-to-inference_a-20260909/migration.json` records migration state;
the shutdown, tunnel, saved-grade/provider inventories, server deployment, and
inference-smoke receipts in that directory identify the physical transition.
Trainer's judge and scorer remain running independently.

The logical shard name `inference_b`, its `scores-adjacent/` output root, and the client
URL `http://127.0.0.1:28002` remain stable for cache/provenance compatibility.
After the recorded tunnel switch, that URL routes to **Inference_A:8004**, not Inference_B.
The physical GPU changes from RTX A6000 TP2 to A100 80GB TP2; no bitwise decoding
equivalence is claimed. Preserve the pinned Qwen3-32B revision, vLLM0.19.1,
BF16, request seeds and grading constraints, and record physical placement
separately from the legacy logical shard label. Prior immutable records are
not rewritten to pretend they were generated on Inference_A.

Consult the migration state and active partition's physical-backend metadata
before restarting; do not restore the old Inference_B tunnel or model server implicitly.

## Historical policy offload (2026-09-09)

The user authorized public archival and temporary local removal of historical
parameters after response and rubric generation. This includes selected matrix
checkpoints, not only off-schedule checkpoints. The exact historical targets are
0,3,6,9,12,13,15,16,18,21,24,27,30,32,33,34,36,39,40,42. Protect the full latest
checkpoint 45, including its optimizer, extra state, data state, and export.

Saved-response scoring now uses `inspect_scoring_checkpoint`: a pinned public
archive receipt can establish the original model-file hash without opening local
weights. Training, response generation, export and KL retain strict local-model
requirements. This does not change the matrix, rubric grades, judge or metrics.
Responses, rubrics, grades, split manifests and archival receipts stay local.

Public models are named
`HYU-NLP-EVAL/qwen3-4b-rar-medicine-onlinerubrics-seed11-step-NNN`.
Root files provide the BF16 inference export; `original_checkpoint/` preserves
the exact original parameter bytes. Public releases exclude optimizer state,
responses, rubrics, credentials and infrastructure configuration. Upload success
alone never authorizes deletion: verify public visibility, pinned revision,
file inventory and SHA256 first. Consult `verl-run/checkpoint_archive_queue/`
and `verl-run/checkpoint_archives/` for actual completion, not this target list.

If later analysis requires policy weights, restore selected historical steps
with `scripts/phase1/restore_probe_checkpoints.py --run-dir RUN --artifact-root
AUDIT --steps STEP ...`. It downloads the receipt-pinned original files, verifies
their lineage and hashes, then recreates the inference export. It cannot replace
the latest resume checkpoint. Ordinary fresh/stale regrading needs no restore.

`logs/hf-checkpoint-offload-20260909/scorer-restarts.json` records the rollout of
archive-aware scorer clients. Keep using the exact current shard commands in
the active partition; model servers do not need restarting for this change.

## Trainer grading recovery (2026-09-09)

Trainer's judge exhausted 4,096 output tokens on repeated JSON whitespace and the
client exited after its request retries. Its previous command had no client
restart allowance. The same failed request passed an isolated real-judge smoke
with `--bounded-grading-whitespace`, retaining the binary grade vocabulary and
request inputs. The current Trainer command in the active partition now includes
that option plus `--max-client-restarts 3 --restart-delay 30`.

Use that recorded command for future restarts. Existing successful grades remain
immutable; only missing grades use the new serialization constraint. Transport
provenance distinguishes both formats, and bitwise equivalence is not claimed.
The recovery journal `logs/trainer-grading-recovery-20260909/` retains the failed
request reference, isolated smoke, previous partition, saved-grade hashes and
restart/progress evidence. No model server or experiment data was replaced.

## HF upload recovery (2026-09-09)

The step-3 Xet upload produced repeated request timeouts and then stopped doing
network/I/O work while its CLI status thread continued printing. A live process
or periodically printed file counter is not evidence of transfer progress.

Historical checkpoint uploads now default to `HF_HUB_DISABLE_XET=1`, selecting
the installed HF client's HTTP/LFS transfer path. Existing CLI upload metadata,
staged hardlinks, public repositories and verified receipts remain reusable.
`hf_upload_watchdog.py` supervises only the owned upload subprocess, with a
300-second inactivity limit and at most three retries after the first attempt.
Configure these with `HF_ARCHIVE_UPLOAD_IDLE_TIMEOUT_SECONDS` and
`HF_ARCHIVE_UPLOAD_MAX_RETRIES`. Repeated failure leaves the checkpoint local;
the unchanged public-visibility, revision, inventory and SHA256 checks remain
mandatory before cleanup. No checkpoint or score is deleted by the watchdog.

The recovery journal is `logs/hf-upload-recovery-20260909/`. Use its resumed
command/environment and the current queue status when assessing progress.
