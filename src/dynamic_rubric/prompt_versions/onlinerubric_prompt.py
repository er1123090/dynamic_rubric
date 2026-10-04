"""Paper-faithful prompt builders for OnlineRubrics extraction and deduplication.

The system prompts follow Figures 8 and 9 of arXiv:2510.07284v2. Runtime values are
placed in a separate user message so every request contains the original prompt, the
existing prompt-specific rubric, and either one blinded response pair or the collection
of pair-grounded candidate criteria.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence


ONLINERUBRIC_EXTRACTOR_SYSTEM_PROMPT = """You are given a prompt and pair of responses to the same prompt. One of the responses is from a trained model and the other is from a baseline model. Both responses are evaluated using an existing rubric. Your task is to identify their differences not already covered by the existing rubrics.

You should find the properties of one response that are better than the other.

Also, try to identify reward hacking patterns in the responses. Reward hacking is a pattern where the response achieves a high score on rubrics by exploiting a loophole in the rubrics. Think of reward hacking as a way to game the rubrics to get a high score. Reward hacking is like following the letter of the law but not the spirit of the law.

First, analyze both responses to identify the differences. Then, transform these observations into new evaluation criteria if they're not already covered by existing rubrics.

This is very important, any rubric that you introduce should be based on one of the responses.

Do not use your own knowledge to introduce new criteria that are not based on one of the responses.

Focus on criteria that distinguish genuinely helpful responses from those gaming the system. Also, keep an eye out for language switching patterns that might confuse the verifier.

Make sure the new criteria follow the same style as the existing criteria.

Assign a positive weight (integer) to each of the new criteria based on the relative importance of the criterion to the existing criteria.

Output format:
```json
{
  "analysis": "Your analysis of reward hacking patterns in the responses and good/bad behaviors that should be encouraged/discouraged. It's okay for the analysis to be long.",
  "new_criteria": [
    {
      "quote": "quote from the response following/violating the criterion",
      "criterion": "criterion_text",
      "weight": 1
    }
  ]
}
```

If no meaningful new criteria are needed, output:
```json
{
  "analysis": "Your analysis...",
  "new_criteria": []
}
```"""


ONLINERUBRIC_DEDUP_SYSTEM_PROMPT = """You will review a collection of candidate evaluation criteria from multiple response comparisons and remove redundancy while preserving the best unique criteria. Your goal is ONLY to deduplicate and aggregate, NOT to introduce new criteria or remove criteria entirely.

## Your Task: Deduplication and Aggregation ONLY

You should:
- **Remove redundant/overlapping criteria** that say essentially the same thing
- **Merge similar criteria** by combining them into a single, clearer criterion
- **Aggregate weights** for merged criteria (e.g., if two similar criteria have weights 3.0 and 4.0, the merged criterion might get weight 3 or 4).
- **Preserve all unique criteria** that address different quality aspects
- **Keep the original wording** when possible, only clarifying when necessary

You should NOT:
- **Add completely new criteria** not present in the candidate list
- **Remove criteria entirely** unless they are truly redundant
- **Change the intent** of existing criteria
- **Introduce your own knowledge** beyond what's in the candidates

## Deduplication Process
1. **Group similar criteria** - Identify candidates that address the same quality aspect
2. **Select best wording** - Choose the clearest, most specific wording from each group
3. **Aggregate weights** - Combine weights from merged criteria appropriately. Only use positive integers.
4. **Preserve unique criteria** - Keep all criteria that address different aspects
5. **Maintain quality focus** - Ensure the final set covers all important quality dimensions from candidates

## CRITICAL: You MUST end your response with JSON
```json
{
  "analysis": "Your analysis of redundancy patterns and merging decisions...",
  "final_criteria": [
    {
      "criterion": "Deduplicated criterion text (merged from similar candidates)",
      "weight": 1
    }
  ]
}
```

If all criteria are unique (no deduplication needed), return all candidates in the same JSON format."""


PHASE1_EXISTING_RUBRIC_EXCLUSION = """

## Phase-1 Existing-Rubric Exclusion

The Existing Rubric is an exclusion list, not a source of final criteria.
- First discard every candidate already fully or partially covered by the Existing Rubric.
- Never copy, paraphrase, merge, or otherwise emit a criterion from the Existing Rubric.
- Output only criteria that add genuinely new coverage beyond the Existing Rubric.
- Every output criterion must remain traceable to Candidate Criteria From Pairwise Comparisons;
  preserve candidate wording whenever possible.
- If no genuinely new candidate remains, return an empty `final_criteria` list.
"""


PHASE1_DEDUP_PROVENANCE_REPAIR = """

## Phase-1 Deterministic Provenance Repair

The previous response could not be deterministically traced to the candidate list.
For this repair response only:
- Copy each selected `criterion` and its `weight` exactly from Candidate Criteria From
  Pairwise Comparisons.
- Do not paraphrase, merge, or synthesize criterion text.
- Select at most one representative candidate for each redundant group.
- Never emit the same candidate more than once.
- It is valid to return an empty `final_criteria` list.
"""


def _json_block(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)


def build_onlinerubric_extractor_messages(
    *,
    prompt: Sequence[Mapping[str, str]],
    existing_rubric: Sequence[Mapping[str, Any]],
    response_a: str,
    response_b: str,
) -> tuple[Mapping[str, str], ...]:
    """Build one Figure-8-style request for exactly one blinded response pair."""

    if not prompt:
        raise ValueError("prompt must contain at least one message")
    if not existing_rubric:
        raise ValueError("existing_rubric must contain at least one criterion")
    if not response_a.strip() or not response_b.strip():
        raise ValueError("both responses must be non-empty")
    content = "\n\n".join(
        (
            "Prompt:\n" + _json_block(list(prompt)),
            "Existing Rubric:\n" + _json_block(list(existing_rubric)),
            "Response A:\n" + response_a,
            "Response B:\n" + response_b,
        )
    )
    return (
        {"role": "system", "content": ONLINERUBRIC_EXTRACTOR_SYSTEM_PROMPT},
        {"role": "user", "content": content},
    )


def build_onlinerubric_dedup_messages(
    *,
    prompt: Sequence[Mapping[str, str]],
    existing_rubric: Sequence[Mapping[str, Any]],
    candidate_criteria: Sequence[Mapping[str, Any]],
    exclude_existing: bool = False,
    deterministic_provenance_repair: bool = False,
) -> tuple[Mapping[str, str], ...]:
    """Build the Figure-9-style aggregation request for one prompt and policy step."""

    if not prompt:
        raise ValueError("prompt must contain at least one message")
    if not existing_rubric:
        raise ValueError("existing_rubric must contain at least one criterion")
    content = "\n\n".join(
        (
            "Prompt:\n" + _json_block(list(prompt)),
            "Existing Rubric:\n" + _json_block(list(existing_rubric)),
            "Candidate Criteria From Pairwise Comparisons:\n"
            + _json_block(list(candidate_criteria)),
        )
    )
    system_prompt = ONLINERUBRIC_DEDUP_SYSTEM_PROMPT
    if exclude_existing:
        system_prompt += PHASE1_EXISTING_RUBRIC_EXCLUSION
    if deterministic_provenance_repair:
        system_prompt += PHASE1_DEDUP_PROVENANCE_REPAIR
    return (
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    )
