# Rubric generation code versions

The two implementations are intentionally separated by filename and artifact namespace.

| Version | Prompt code | Batch code | Artifact folder prefix |
|---|---|---|---|
| Original experiment | `prompt_versions/initial_rubric_prompt.py` | `initial_batch_dynamic.py` | `initial_dynamic_*` |
| Paper-style OnlineRubrics | `prompt_versions/onlinerubric_prompt.py` | `onlinerubric_batch.py` | `onlinerubric_*` |

The historical `batch_dynamic.py` import remains compatible, but its prompt is now sourced
from `initial_rubric_prompt.py`. Existing historical folders such as `dynamic-prev-batch`
are not renamed or overwritten.

## What the `onlinerubric_*` version changes

For each prompt and policy step it:

1. includes the complete original multi-turn prompt;
2. includes the existing prompt-specific R0 with relative integer weights;
3. forms eight blinded current/control response pairs;
4. sends each pair through the Figure 8 extractor prompt independently;
5. requires response-grounded quotes, reward-hacking analysis, and positive integer weights;
6. sends all pair-level criteria through a separate Figure 9 LLM deduplication request;
7. preserves every request identity and writes the final criteria to
   `onlinerubric_rubrics.jsonl`.

The original `initial_*` version sends four pairs in one request, omits the original prompt
and existing R0, limits the result to three positive criteria, and uses embedding/admission
gates instead of the paper's LLM deduplication prompt.

## Commands

All commands use `scripts/run_rubric_generation_version.py`.

```bash
PYTHONPATH=src python scripts/run_rubric_generation_version.py \
  prepare-onlinerubric-extraction \
  --run-id RUN_ID --mode dynamic_prev_budgeted --max-step 50

PYTHONPATH=src python scripts/run_rubric_generation_version.py \
  submit-onlinerubric-extraction \
  --run-id RUN_ID --mode dynamic_prev_budgeted

PYTHONPATH=src python scripts/run_rubric_generation_version.py \
  collect-onlinerubric-extraction \
  --run-id RUN_ID --mode dynamic_prev_budgeted

PYTHONPATH=src python scripts/run_rubric_generation_version.py \
  prepare-onlinerubric-dedup \
  --run-id RUN_ID --mode dynamic_prev_budgeted
```

The corresponding `submit-onlinerubric-dedup`,
`status-onlinerubric-dedup`, and `collect-onlinerubric-dedup` commands complete the
second Batch stage. Replace `dynamic_prev_budgeted` with
`dynamic_fixed_budgeted` for the paper's current-policy versus reference-policy variant.

## Scope boundary

This code makes rubric **elicitation prompts and two-stage generation** match the paper.
With the already-completed static run, it operates as a counterfactual replay over stored
post-update responses. It does not claim to reproduce the paper's causal online-RL result:
a future training integration must generate these criteria from training-batch responses
before each optimizer update and include `R0 ∪ elicited criteria` in that update's reward.
