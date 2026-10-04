# Paper-faithful OnlineRubrics training contract

This repository implements the same-step OnlineRubrics loop described in
*Online Rubrics Elicitation from Pairwise Comparisons* (arXiv:2510.07284v2).
This document fixes the paper-facing behavior that must not drift during refactors.

## One optimizer update

For every prompt occurrence in a training batch, the runtime must complete this order:

1. Generate 16 responses from the current actor.
2. Generate 8 responses from the configured control actor (`pi_ref` or `pi_old`).
3. Pair current rollout indexes `0..7` with the 8 control responses and blind A/B placement.
4. Run Figure 8 extraction exactly 8 times.
5. Run Figure 9 deduplication exactly once over all extracted candidates.
6. Form the prompt-and-step-local union `C_i ∪ C^e_(i,t)` without accumulation, filtering,
   admission, eviction, or a hard criterion cap.
7. Run Figure 10 grading once for each of the 16 current responses.
8. Compute Eq. 4 rewards, seal all receipts, and only then expose rewards to GRPO.

Any incomplete inventory, provider exception, returned-model drift, malformed JSON, rubric
collision, or missing receipt fails the step before the actor update. An empty but valid dedup
result is allowed and makes the union equal to the offline rubric.

## Prompt and output contracts

- Figure 8 receives the original prompt, existing rubric, and one blinded `Response A` /
  `Response B` pair. The model-facing request contains no runtime current/control identity.
- Figure 9 receives the original prompt, existing rubric, and candidates from all eight pairs.
  It may merge or clarify candidates but may not introduce an ungrounded criterion.
- Figure 10 receives the original prompt, one current response, and the numbered union rubric.
  Its output is a direct JSON object with keys `"1".."N"`; every value is exactly `PRESENT`
  or `NOT_PRESENT`. The paper example's spaced `NOT PRESENT` variant is treated as malformed
  because it violates the canonical label specified by the prompt.

The system-prompt SHA-256 values fixed by the snapshot tests are:

| Paper prompt | SHA-256 |
| --- | --- |
| Figure 8 extractor | `7258589ce47230e35ab2dcc02f808fe9157d8c5458c1876bce67d9a0b8098d4b` |
| Figure 9 dedup | `d258436acd99efe749faad233cbd8fff7a0ffb60b510653e74921c740c455b20` |
| Figure 10 grader | `70d79be861b32e4d9d1e4f6eec76d4eb722488b3d2341010a66876eaaf6a482d` |

Changing any prompt requires an explicit contract/version change rather than silently updating
the hashes.

## Eq. 4

For criterion weights `w_k` and binary grades `g_k`:

```text
numerator   = Σ_k w_k g_k
denominator = Σ_{k:w_k>0} w_k
reward      = numerator / denominator
```

Offline weights may be positive, negative, or zero. Negative weights remain signed in the
numerator; zero weights contribute to neither term. Online pairwise weights must be positive
integers. A non-positive denominator is a hard failure. Computation uses exact rational math
before conversion to the scalar consumed by training.

## Identity and sealing

Every response is bound to a run, optimizer update, batch, source row, prompt occurrence,
rollout index, and policy snapshot hash. A pre-update seal binds extraction, dedup, union rubric,
grader, and reward artifact hashes. Resume may reuse an identical immutable response cache, but
must reject any model, prompt, schema, response, policy, or rubric identity mismatch.

## Runtime readiness and launch

The production-ready control lane is currently `pi_ref`. It requires a separately served,
frozen copy of the exact initial actor `A0`. Preflight streams a canonical SHA-256 over the
local Hugging Face snapshot and requires `/dynamic-rubric/identity` on the control endpoint to
report that same hash. The runtime repeats the local tree-hash check before step 1. Matching
model and revision labels without matching weight bytes is rejected.

`pi_old` configuration and policy-ledger contracts exist, but live training deliberately fails
closed until the pinned sampler supports a bounded one-batch lookahead whose dataloader cursor
and control responses can be committed atomically. It must not be described as runnable yet.

The live `pi_ref` entry point is:

```bash
export OPENAI_API_KEY=...
export ONLINE_CONTROL_URL=http://127.0.0.1:8001
export ONLINE_CONTROL_CHECKPOINT_HASH=<canonical-local-A0-tree-sha256>
export ONLINE_CONTROL_LAUNCH_SPEC=/absolute/path/to/immutable-control-launch.json

.venv/bin/python -m dynamic_rubric train-online \
  --config configs/online_medicine_pi_ref.yaml \
  --run-id <new-immutable-run-id>
```

The control endpoint must serve `Qwen/Qwen3-4B-Instruct-2507` at the revision and tokenizer
revision pinned by the selected config, with thinking disabled and the exact A0 checkpoint hash.
The same requirements apply to `resume-online`; resume scans and validates the contiguous
immutable commit chain, repairs a missing or stale mutable latest pointer, and ignores newer
checkpoint directories that have no committed online-step manifest.

With 1,500 training rows, batch 96, drop-last, and three epochs, one online arm performs 45
optimizer updates. Its preflight estimate is 108,000 external rubric/grader calls and 34,560
local control generations. Run a live canary before authorizing that full cost.
