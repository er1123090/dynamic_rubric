# Phase-1: evaluator update value during GRPO

## Scope and experiment identity

This lane studies fresh versus stale evaluators **within the same method** on the
same current-policy Pool-B responses. It is not an OnlineRubrics-versus-EvoRubrics
leaderboard and does not use R0/static rubrics as ground truth.

Phase-1 is isolated from the previous static-rubric GRPO lane by all of the following:

| Surface | Phase-1 | Legacy static GRPO |
|---|---|---|
| experiment | `phase1_evaluator_update_value` | static R0 training |
| tracking project | `phase1_dynamic_evaluator_updates` | `dynamic_rubric_static_grpo` |
| method IDs | `online_rubrics`, `evorubrics` | static-specific IDs |
| output root | `outputs/{domain}/{method}/seed-{seed}/{run_id}` | legacy run layout |
| primary conclusion data | fixed Train Probe 100 | not reused for Phase-1 conclusions |
| GT / BoN | disabled | never imported into Phase-1 |

No cross-method absolute reward or absolute ZAR comparison is emitted.

## Short repository audit

Reusable code:

- `artifacts.py`: immutable, atomic JSON/JSONL writes and resume checks.
- `data/rar.py` and existing RaR JSONL splits: normalized prompt IDs and R0 payloads.
- `training/probe_export.py`: paired proof that probe-only work cannot change trainer state.
- `horizon/pools.py` and `horizon/metrics.py`: response identity and metric precedents.
- `training/online_step.py`: the existing OnlineRubrics extraction/dedup/grading barrier.
- Existing checkpoint-KL utilities can supply adjacent and cumulative policy KL later.

Not reused as the Phase-1 analysis driver:

- The legacy horizon pipeline is held-out/static-R0 oriented.
- BoN, GT regret, and private-GT modules are outside this phase.
- The previous OnlineRubrics run configs use a different backbone and checkpoint clock.

Missing before this change:

- A prompt-indexed Online rubric history for safe stale lookup.
- Evo theta/psi same-pool comparison invariants.
- Fixed-train-probe adjacent and full triangular analysis.
- Phase-1-specific configs, provenance schema, and restart-safe dry-run.

The full EvoRubrics theta/psi training runtime is still a separate implementation lane;
the current deliverable establishes its checkpoint and analysis contracts without
pretending that deterministic fixtures loaded real adapters.

## Added modules

| Path | Responsibility |
|---|---|
| `phase1/config.py` | fail-closed resolved config and topology validation |
| `phase1/provenance.py` | immutable Train Probe 100 and Pool A/B identities |
| `phase1/shadow.py` | same-prompt Online cache and Evo comparison invariants |
| `phase1/metrics.py` | ZAR, tie, separation, criteria, margin, Kendall metrics |
| `phase1/analysis.py` | adjacent update value and anchor-triangular reuse matrix |
| `phase1/smoke.py` | deterministic artifact/resume/provenance dry-run |
| `phase1/preflight.py` | trainer/data/model snapshot/remote model identity gate |

Configs:

- `configs/phase1/medicine_online_rubrics.yaml`
- `configs/phase1/science_online_rubrics.yaml`
- `configs/phase1/medicine_evorubrics.yaml`
- `configs/phase1/science_evorubrics.yaml`

All four resolve to Qwen3-4B with thinking disabled, seed 11, 3 epochs, global
prompt batch 96, 48 expected steps, the required audit schedule, and reuse anchors.

## Host and GPU topology

- Code, manifests, optimizer, checkpoints, and analysis are managed from `trainer`.
- The GRPO policy optimizer uses `trainer:GPU0`.
- The fixed pi0 control cache is precomputed on `trainer:GPU0` and sealed before
  the optimizer starts.
- gpt-oss-120b vLLM runs on `inference_a:GPU1,2` with tensor parallel size 2.
- Qwen3-32B vLLM runs on `inference_b:GPU0,1` with tensor parallel size 2.
- SSH credentials are neither stored nor printed by any config or launcher.

The optional remote helpers use a private SSH config supplied by the operator and never persist a password. Canonical training and inference run directly in Linux virtual environments; Docker is optional and no container name or image is required by the six public entry points.
Both launchers are idempotent: an existing ready or loading process is reported
and is never stopped or replaced.

## Commands and stop gates

Prepare the shared probe manifests:

```bash
scripts/phase1/prepare_fixed_probes.sh
```

Run the credential-free contract smoke:

```bash
scripts/phase1/run_contract_smoke.sh medicine online_rubrics contract-smoke-v1
scripts/phase1/run_contract_smoke.sh medicine evorubrics contract-smoke-v1
```

Start the two judge services in separate terminals, then export base URLs from
`configs/phase1/topology.env.example` and run endpoint/model preflight:

```bash
scripts/phase1/serve_gpt_oss_on_inference_a.sh
scripts/phase1/serve_qwen32b_on_inference_b.sh
scripts/phase1/manage_vllm_tunnels.sh start
scripts/phase1/vllm_status.sh
scripts/phase1/preflight_topology.sh medicine online_rubrics
```

Because direct service ports are blocked between hosts, the tunnel manager maps
`127.0.0.1:18001` to inference_a port 8001 and `127.0.0.1:18002` to inference_b port 8002.
It uses SSH control sockets under `/tmp`, so repeated `start` and `status` calls
are safe. `manage_vllm_tunnels.sh stop` only closes those local tunnels; it does
not stop either remote vLLM process. Remote logs are kept in
a writable log directory selected by the operator.

The deterministic smoke checks schemas, provenance, checkpoint artifacts, same-pool
shadow scoring, metrics, restart behavior, and the probe-exclusion contract. It explicitly records
`model_weights_loaded=false`, `remote_generation_and_grading_verified=false`,
`live_trainer_probe_side_effect_verified=false`, and
`full_training_authorized=false`. A live model smoke that loads the policy,
generates rubrics, and grades responses remains a required gate before any full run.
No expensive training command is provided or launched in this deliverable.

## Dry-run evidence

The Medicine contract smoke writes:

- `outputs/medicine/online_rubrics/seed-11/contract-smoke-v1/`
- `outputs/medicine/evorubrics/seed-11/contract-smoke-v1/`

Each run contains the resolved config, prompt/pool provenance, policy/evaluator
checkpoint metadata, criterion grades, rewards, advantages, pre-update state,
global-step counters, comparison metrics, and a `smoke_complete.json` gate.

The fixed Medicine probe manifest is shared across methods at
`outputs/medicine/shared/seed-11/manifests/fixed_train_probe.json`.
